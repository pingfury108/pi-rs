//! Provider API abstraction.
//!
//! Mirrors pi's contract: `stream()` never fails synchronously (except auth
//! assertion); all request/runtime failures are encoded in the returned
//! stream as a terminal `Error` event carrying an assistant message with
//! `stop_reason` of `error`/`aborted`.

pub mod anthropic;
pub mod bedrock;
pub mod faux;
pub mod google;
pub mod mistral;
pub mod openai_completions;
pub mod openai_responses;
pub mod pi_messages;

pub use anthropic::AnthropicApi;
pub use bedrock::BedrockApi;
pub use faux::{FauxApi, FauxResponse};
pub use google::GoogleApi;
pub use mistral::MistralApi;
pub use openai_completions::OpenAICompletionsApi;
pub use openai_responses::OpenAIResponsesApi;
pub use pi_messages::PiMessagesApi;

use std::collections::BTreeMap;

use async_trait::async_trait;

use crate::events::AssistantMessageEventStream;
use crate::types::{Context, Message, Model, ModelCost, StreamOptions, SystemMessage, ToolDef, Usage};

/// A provider API implementation (anthropic-messages, openai-completions, ...).
#[async_trait]
pub trait LlmApi: Send + Sync {
    /// Protocol identifier, e.g. `"anthropic-messages"`.
    fn name(&self) -> &str;

    /// Start streaming an assistant message. Failures are reported inside the
    /// stream as terminal `Error` events.
    fn stream(&self, model: &Model, context: &Context, options: StreamOptions)
        -> AssistantMessageEventStream;
}

/// Fold `Context.systemPrompt`/`tools` and any system messages into a single
/// leading system message carrying the effective prompt and tool set.
///
/// This is the Rust equivalent of pi's `normalizeContext()`: providers receive
/// a transcript where the system prompt and tools live in system messages.
pub fn normalize_context(context: &Context) -> Vec<Message> {
    let mut system_text = context.system_prompt.clone().unwrap_or_default();
    let mut sections: BTreeMap<String, Option<String>> = BTreeMap::new();
    let mut tools: Vec<ToolDef> = context.tools.clone().unwrap_or_default();

    let mut rest = Vec::new();
    for msg in &context.messages {
        match msg {
            Message::System(sys) => {
                let text = sys.content.as_text();
                if !text.is_empty() {
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(&text);
                }
                if let Some(s) = &sys.sections {
                    for (k, v) in s {
                        sections.insert(k.clone(), v.clone());
                    }
                }
                if let Some(t) = &sys.tools_added {
                    for tool in t {
                        tools.retain(|x| x.name != tool.name);
                        tools.push(tool.clone());
                    }
                }
                if let Some(t) = &sys.tools_removed {
                    for name in t {
                        tools.retain(|x| &x.name != name);
                    }
                }
            }
            other => rest.push(other.clone()),
        }
    }

    let mut leading = SystemMessage::new(system_text);
    if !sections.is_empty() {
        leading.sections = Some(sections);
    }
    if !tools.is_empty() {
        leading.tools_added = Some(tools);
    }

    let mut out = Vec::with_capacity(rest.len() + 1);
    out.push(Message::System(leading));
    out.extend(rest);
    out
}

/// The effective system prompt text of a normalized transcript.
pub fn initial_system_text(messages: &[Message]) -> String {
    match messages.first() {
        Some(Message::System(sys)) => sys.content.as_text(),
        _ => String::new(),
    }
}

/// The effective tool set of a normalized transcript.
pub fn current_tools(messages: &[Message]) -> Vec<ToolDef> {
    match messages.first() {
        Some(Message::System(sys)) => sys.tools_added.clone().unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Fill `usage.cost` from model rates (USD per million tokens).
pub fn calculate_cost(cost: &ModelCost, usage: &mut Usage) {
    let m = 1_000_000f64;
    usage.cost.input = usage.input as f64 / m * cost.input;
    usage.cost.output = usage.output as f64 / m * cost.output;
    usage.cost.cache_read = usage.cache_read as f64 / m * cost.cache_read;
    usage.cost.cache_write = usage.cache_write as f64 / m * cost.cache_write;
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

/// Retry policy for the initial HTTP request.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 500,
            max_delay_ms: 30_000,
        }
    }
}

/// Whether a failed HTTP response is worth retrying.
pub fn is_retryable_status(status: u16) -> bool {
    status == 408 || status == 429 || status >= 500
}

/// Delay requested by the server via `Retry-After` (seconds), if parseable.
pub fn retry_after_delay(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let secs: u64 = value.trim().parse().ok()?;
    Some(std::time::Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, UserMessage};

    #[test]
    fn normalize_folds_system_prompt_and_tools() {
        let ctx = Context {
            system_prompt: Some("base".into()),
            messages: vec![
                Message::System(SystemMessage {
                    content: MessageContent::text("extra"),
                    sections: None,
                    tools_added: Some(vec![ToolDef::new("a", "", serde_json::json!({}))]),
                    tools_removed: None,
                    timestamp: 0,
                }),
                Message::User(UserMessage::new(MessageContent::text("hi"))),
                Message::System(SystemMessage {
                    content: MessageContent::text(""),
                    sections: None,
                    tools_added: Some(vec![ToolDef::new("b", "", serde_json::json!({}))]),
                    tools_removed: Some(vec!["a".into()]),
                    timestamp: 1,
                }),
            ],
            tools: Some(vec![ToolDef::new("c", "", serde_json::json!({}))]),
        };
        let normalized = normalize_context(&ctx);
        assert_eq!(normalized.len(), 2);
        let Message::System(leading) = &normalized[0] else {
            panic!("expected leading system message");
        };
        assert!(leading.content.as_text().contains("base"));
        assert!(leading.content.as_text().contains("extra"));
        let tools = current_tools(&normalized);
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        // context.tools acts as the leading system message's toolsAdded;
        // later system messages append (b) and remove (a).
        assert_eq!(names, ["c", "b"]);
    }
}


// ---------------------------------------------------------------------------
// Shared streaming-request plumbing used by the API implementations.
// ---------------------------------------------------------------------------

/// Outcome of opening an SSE stream: the byte stream to iterate.
pub(crate) type SseBody = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>,
>;

/// Issue the initial POST and return the response body stream, applying the
/// retry policy for transport errors and retryable statuses. On failure the
/// error message is stored in `error_slot` and `Err(reason)` returned.
pub(crate) async fn open_sse_post(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    url: &str,
    headers: Vec<(String, String)>,
    body: &serde_json::Value,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    error_slot: &mut String,
) -> Result<SseBody, crate::types::StopReason> {
    let aborted = || cancel.as_ref().is_some_and(|c| c.is_cancelled());

    let mut attempt = 0u32;
    loop {
        if aborted() {
            return Err(crate::types::StopReason::Aborted);
        }
        let mut request = http.post(url).json(body);
        for (name, value) in &headers {
            request = request.header(name, value);
        }
        match request.send().await {
            Ok(resp) if resp.status().is_success() => {
                return Ok(Box::pin(resp.bytes_stream()));
            }
            Ok(resp) => {
                let status = resp.status().as_u16();
                let text = resp.text().await.unwrap_or_default();
                if !is_retryable_status(status) || attempt >= retry.max_retries {
                    *error_slot = format!("HTTP {status}: {}", truncate_str(&text, 2048));
                    return Err(crate::types::StopReason::Error);
                }
                tracing::warn!(status, attempt, "request failed, retrying");
                attempt += 1;
                let delay = std::time::Duration::from_millis(
                    retry.base_delay_ms.saturating_mul(2u64.saturating_pow(attempt)),
                )
                .min(std::time::Duration::from_millis(retry.max_delay_ms));
                if !wait_or_abort(cancel, delay).await {
                    return Err(crate::types::StopReason::Aborted);
                }
            }
            Err(err) => {
                if attempt >= retry.max_retries {
                    *error_slot = format!("request failed: {err}");
                    return Err(crate::types::StopReason::Error);
                }
                tracing::warn!(%err, attempt, "request error, retrying");
                attempt += 1;
                let delay = std::time::Duration::from_millis(
                    retry.base_delay_ms.saturating_mul(2u64.saturating_pow(attempt)),
                )
                .min(std::time::Duration::from_millis(retry.max_delay_ms));
                if !wait_or_abort(cancel, delay).await {
                    return Err(crate::types::StopReason::Aborted);
                }
            }
        }
    }
}

/// Wait for `delay`, returning false when cancelled.
pub(crate) async fn wait_or_abort(
    cancel: &Option<tokio_util::sync::CancellationToken>,
    delay: std::time::Duration,
) -> bool {
    match cancel {
        Some(token) => {
            tokio::select! {
                _ = tokio::time::sleep(delay) => true,
                _ = token.cancelled() => false,
            }
        }
        None => {
            tokio::time::sleep(delay).await;
            true
        }
    }
}

pub(crate) fn truncate_str(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}
