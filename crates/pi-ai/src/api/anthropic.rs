//! Anthropic Messages API streaming implementation.
//!
//! Port of the essential path of `packages/ai/src/api/anthropic-messages.ts`:
//! request building, SSE event mapping, tool-argument accumulation via partial
//! JSON parsing, usage/cost accounting and abort handling.
//!
//! Not ported (yet): OAuth/Claude-Code identity, beta headers, native mid-
//! conversation tool changes, server-side fallbacks, adaptive thinking options.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use eventsource_stream::Eventsource;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{
    calculate_cost, current_tools, initial_system_text, is_retryable_status, normalize_context,
    LlmApi, RetryPolicy,
};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::partial_json::parse_partial_json;
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, StreamOptions,
    TextContent, ThinkingContent, ToolCall, Usage,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl AnthropicApi {
    pub fn new() -> Self {
        Self::with_http(reqwest::Client::new())
    }

    pub fn with_http(http: reqwest::Client) -> Self {
        Self {
            http,
            retry: RetryPolicy::default(),
        }
    }
}

impl Default for AnthropicApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for AnthropicApi {
    fn name(&self) -> &str {
        "anthropic-messages"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let model = model.clone();
        let messages = normalize_context(context);
        let http = self.http.clone();
        let retry = self.retry.clone();

        let (tx, stream) = AssistantMessageEventStream::channel(256);
        tokio::spawn(async move {
            run(&http, &retry, &model, &messages, &options, &tx).await;
            tx.close();
        });
        stream
    }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Build the `/v1/messages` request body from a normalized transcript.
pub fn build_body(
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
) -> Value {
    let system = initial_system_text(messages);
    let tools = current_tools(messages);

    let mut body = json!({
        "model": model.id,
        "max_tokens": options.max_tokens.unwrap_or(model.max_tokens),
        "messages": convert_messages(messages),
        "stream": true,
    });

    if !system.is_empty() {
        body["system"] = json!([{ "type": "text", "text": system, "cache_control": {"type": "ephemeral"} }]);
    }

    if !tools.is_empty() {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters}))
            .collect();
        body["tools"] = Value::Array(converted);
        // cache breakpoint on the last tool definition
        if let Some(last) = body["tools"].as_array_mut().and_then(|a| a.last_mut()) {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
    }

    if let Some(temperature) = options.temperature {
        body["temperature"] = json!(temperature);
    }

    // Thinking: budget-based extended thinking when a level is requested.
    if let Some(level) = options.reasoning {
        if model.reasoning {
            let budget = thinking_budget(level);
            let max_tokens = body["max_tokens"].as_u64().unwrap_or(4096);
            let budget = budget.min(max_tokens.saturating_sub(1024)).max(1024);
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
        }
    }

    body
}

fn thinking_budget(level: crate::types::ThinkingLevel) -> u64 {
    use crate::types::ThinkingLevel::*;
    match level {
        Minimal => 1024,
        Low => 4096,
        Medium => 10240,
        High => 16384,
        Xhigh => 24576,
        Max => 32768,
    }
}

/// Convert normalized transcript messages to the Anthropic wire format.
/// Consecutive tool results are merged into a single user turn (Anthropic
/// requires alternating roles).
pub fn convert_messages(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    // skip leading system message; its text goes to `system`
    for msg in messages.iter().skip(1) {
        match msg {
            Message::System(_) => {
                // Later system messages are folded by normalize_context; leftovers
                // are converted to user text for compatibility.
                if let Message::System(sys) = msg {
                    let text = sys.content.as_text();
                    if !text.is_empty() {
                        out.push(json!({"role": "user", "content": text}));
                    }
                }
            }
            Message::User(user) => {
                out.push(json!({"role": "user", "content": convert_user_content(&user.content)}));
            }
            Message::Assistant(assistant) => {
                let blocks: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(t) => {
                            Some(json!({"type": "text", "text": t.text}))
                        }
                        AssistantContent::Thinking(t) => {
                            let signature = t.thinking_signature.clone().unwrap_or_default();
                            if t.redacted {
                                Some(json!({"type": "redacted_thinking", "data": signature}))
                            } else {
                                Some(json!({"type": "thinking", "thinking": t.thinking, "signature": signature}))
                            }
                        }
                        AssistantContent::ToolCall(call) => Some(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        })),
                    })
                    .collect();
                if !blocks.is_empty() {
                    out.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            Message::ToolResult(result) => {
                // Merge consecutive tool results into one user message.
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": convert_tool_result_content(&result.content),
                    "is_error": result.is_error,
                });
                match out.last_mut() {
                    Some(last) if last["role"] == "user" && last["content"].is_array() => {
                        last["content"].as_array_mut().unwrap().push(block);
                    }
                    _ => {
                        out.push(json!({"role": "user", "content": [block]}));
                    }
                }
            }
        }
    }
    out
}

fn convert_user_content(content: &crate::types::MessageContent) -> Value {
    use crate::types::{MessageContent, UserContent};
    match content {
        MessageContent::Text(s) => json!(s),
        MessageContent::Blocks(blocks) => {
            let has_images = blocks.iter().any(|b| matches!(b, UserContent::Image(_)));
            if !has_images {
                // text-only: plain string
                let text = blocks
                    .iter()
                    .filter_map(|b| match b {
                        UserContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                json!(text)
            } else {
                let arr: Vec<Value> = blocks
                    .iter()
                    .map(|b| match b {
                        UserContent::Text(t) => json!({"type": "text", "text": t.text}),
                        UserContent::Image(img) => json!({
                            "type": "image",
                            "source": {"type": "base64", "media_type": img.mime_type, "data": img.data},
                        }),
                    })
                    .collect();
                Value::Array(arr)
            }
        }
    }
}

fn convert_tool_result_content(content: &[crate::types::ToolResultContent]) -> Value {
    use crate::types::ToolResultContent;
    let arr: Vec<Value> = content
        .iter()
        .map(|c| match c {
            ToolResultContent::Text(t) => json!({"type": "text", "text": t.text}),
            ToolResultContent::Image(img) => json!({
                "type": "image",
                "source": {"type": "base64", "media_type": img.mime_type, "data": img.data},
            }),
        })
        .collect();
    Value::Array(arr)
}

// ---------------------------------------------------------------------------
// SSE event mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartBody },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: usize, content_block: BlockStart },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: Delta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaBody, usage: Option<DeltaUsage> },
    #[serde(rename = "message_stop")]
    MessageStop {},
    #[serde(rename = "ping")]
    Ping {},
    #[serde(rename = "error")]
    Error { error: ErrorBody },
}

#[derive(Debug, Deserialize)]
struct MessageStartBody {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: StartUsage,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct StartUsage {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum BlockStart {
    #[serde(rename = "text")]
    Text { #[serde(default)] text: String },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { #[serde(default)] data: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct DeltaUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    message: String,
}

fn map_stop_reason(raw: &str) -> (StopReason, Option<String>) {
    match raw {
        "end_turn" | "stop_sequence" => (StopReason::Stop, None),
        "max_tokens" => (StopReason::Length, None),
        "tool_use" => (StopReason::ToolUse, None),
        "refusal" => (
            StopReason::Error,
            Some("Model refused to respond".to_string()),
        ),
        _ => (StopReason::Stop, None),
    }
}

/// Per-provider-index bookkeeping while assembling the output message.
struct Assembler {
    message: AssistantMessage,
    /// provider block index → position in `message.content`
    index_map: HashMap<usize, usize>,
    /// provider block index → accumulated partial tool JSON
    partial_json: HashMap<usize, String>,
}

impl Assembler {
    fn new(model: &Model) -> Self {
        Self {
            message: AssistantMessage {
                content: vec![],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: vec![],
                usage: Usage::default(),
                stop_reason: StopReason::Pending,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: crate::types::now_millis(),
            },
            index_map: HashMap::new(),
            partial_json: HashMap::new(),
        }
    }

    fn position(&self, provider_index: usize) -> Option<usize> {
        self.index_map.get(&provider_index).copied()
    }
}

/// Drive one streaming request to completion, pushing events into `tx`.
async fn run(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    tx: &EventStreamTx,
) {
    let mut assembler = Assembler::new(model);

    let result = execute(
        http, retry, model, messages, options, tx, &mut assembler,
    )
    .await;

    match result {
        Ok(()) => {
            let mut message = assembler.message;
            if message.stop_reason == StopReason::Pending {
                message.stop_reason = StopReason::Error;
                message.error_message = Some("Anthropic stream ended without a stop reason".into());
                tx.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: message,
                });
                return;
            }
            tx.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
        }
        Err(reason) => {
            let mut message = assembler.message;
            message.stop_reason = reason;
            message.error_message = Some(match reason {
                StopReason::Aborted => "Request was aborted".to_string(),
                _ => message.error_message.clone().unwrap_or_else(|| "Unknown error".into()),
            });
            tx.push(AssistantMessageEvent::Error { reason, error: message });
        }
    }
}

type ExecResult = Result<(), StopReason>;

async fn execute(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    tx: &EventStreamTx,
    asm: &mut Assembler,
) -> ExecResult {
    let cancel = options.cancel.clone();
    let aborted = || cancel.as_ref().is_some_and(|c| c.is_cancelled());

    let api_key = options
        .api_key
        .clone()
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .ok_or_else(|| {
            tracing::error!("missing API key for provider {}", model.provider);
            StopReason::Error
        })?;

    let base_url = options
        .base_url
        .clone()
        .unwrap_or_else(|| model.base_url.clone());
    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
    let body = build_body(model, messages, options);

    // Initial request with retries.
    let mut attempt = 0u32;
    let response = loop {
        if aborted() {
            return Err(StopReason::Aborted);
        }
        let request = http
            .post(&url)
            .header("x-api-key", &api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body);
        let send_result = request.send().await;

        match send_result {
            Ok(resp) if resp.status().is_success() => break resp,
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                let retryable = is_retryable_status(status) && attempt < retry.max_retries;
                if !retryable {
                    asm.message.error_message =
                        Some(format!("HTTP {status}: {}", truncate(&body_text, 2048)));
                    asm.message.raw_stop_reason = None;
                    return Err(StopReason::Error);
                }
                tracing::warn!(status, attempt, "anthropic request failed, retrying");
                let delay = std::time::Duration::from_millis(
                    retry.base_delay_ms.saturating_mul(2u64.saturating_pow(attempt)),
                )
                .min(Duration::from_millis(retry.max_delay_ms));
                attempt += 1;
                wait_or_abort(&cancel, delay).await.map_err(|()| StopReason::Aborted)?;
            }
            Err(err) => {
                if attempt < retry.max_retries {
                    tracing::warn!(%err, attempt, "anthropic request error, retrying");
                    let delay = std::time::Duration::from_millis(
                        retry.base_delay_ms.saturating_mul(2u64.saturating_pow(attempt)),
                    )
                    .min(Duration::from_millis(retry.max_delay_ms));
                    attempt += 1;
                    wait_or_abort(&cancel, delay).await.map_err(|()| StopReason::Aborted)?;
                    continue;
                }
                asm.message.error_message = Some(format!("request failed: {err}"));
                return Err(StopReason::Error);
            }
        }
    };

    // Stream the response body as SSE.
    let byte_stream = response.bytes_stream();
    let mut sse_stream = byte_stream.eventsource();

    // Emit start event with the pending message snapshot.
    tx.push(AssistantMessageEvent::Start {
        partial: asm.message.clone(),
    });

    while let Some(sse) = sse_stream.next().await {
        if aborted() {
            return Err(StopReason::Aborted);
        }
        let sse = match sse {
            Ok(ev) => ev,
            Err(err) => {
                asm.message.error_message = Some(format!("SSE error: {err}"));
                return Err(StopReason::Error);
            }
        };
        if sse.event == "ping" {
            continue;
        }
        let event: AnthropicEvent = match serde_json::from_str(&sse.data) {
            Ok(ev) => ev,
            Err(err) => {
                tracing::warn!(data = %sse.data, %err, "failed to parse anthropic SSE event");
                continue;
            }
        };
        let is_stop = matches!(event, AnthropicEvent::MessageStop { .. });
        handle_event(event, tx, asm, model)?;
        if is_stop {
            break;
        }
    }

    Ok(())
}

fn handle_event(
    event: AnthropicEvent,
    tx: &EventStreamTx,
    asm: &mut Assembler,
    model: &Model,
) -> ExecResult {
    match event {
        AnthropicEvent::MessageStart { message } => {
            asm.message.response_id = Some(message.id.clone());
            if message.model != model.id && !message.model.is_empty() {
                asm.message.response_model = Some(message.model);
            }
            let usage = &mut asm.message.usage;
            usage.input = message.usage.input_tokens;
            usage.output = message.usage.output_tokens;
            usage.cache_read = message.usage.cache_read_input_tokens.unwrap_or(0);
            usage.cache_write = message.usage.cache_creation_input_tokens.unwrap_or(0);
            usage.total_tokens =
                usage.input + usage.output + usage.cache_read + usage.cache_write;
            calculate_cost(&model.cost, usage);
        }
        AnthropicEvent::ContentBlockStart { index, content_block } => {
            let position = match content_block {
                BlockStart::Text { text } => {
                    let position = asm.message.content.len();
                    asm.message.content.push(AssistantContent::Text(TextContent {
                        text,
                        text_signature: None,
                    }));
                    tx.push(AssistantMessageEvent::TextStart {
                        content_index: position,
                        partial: asm.message.clone(),
                    });
                    Some(position)
                }
                BlockStart::Thinking { thinking, signature } => {
                    let position = asm.message.content.len();
                    asm.message
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking,
                            thinking_signature: Some(signature),
                            redacted: false,
                        }));
                    tx.push(AssistantMessageEvent::ThinkingStart {
                        content_index: position,
                        partial: asm.message.clone(),
                    });
                    Some(position)
                }
                BlockStart::RedactedThinking { data } => {
                    let position = asm.message.content.len();
                    asm.message
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking: "[Reasoning redacted]".into(),
                            thinking_signature: Some(data),
                            redacted: true,
                        }));
                    tx.push(AssistantMessageEvent::ThinkingStart {
                        content_index: position,
                        partial: asm.message.clone(),
                    });
                    Some(position)
                }
                BlockStart::ToolUse { id, name } => {
                    let position = asm.message.content.len();
                    asm.message
                        .content
                        .push(AssistantContent::ToolCall(ToolCall {
                            id,
                            name,
                            arguments: Default::default(),
                            thought_signature: None,
                            namespace: None,
                        }));
                    asm.partial_json.insert(index, String::new());
                    tx.push(AssistantMessageEvent::ToolcallStart {
                        content_index: position,
                        partial: asm.message.clone(),
                    });
                    Some(position)
                }
            };
            if let Some(position) = position {
                asm.index_map.insert(index, position);
            }
        }
        AnthropicEvent::ContentBlockDelta { index, delta } => {
            let Some(position) = asm.position(index) else {
                return Ok(());
            };
            match delta {
                Delta::TextDelta { text } => {
                    if let Some(AssistantContent::Text(t)) = asm.message.content.get_mut(position) {
                        t.text.push_str(&text);
                        tx.push(AssistantMessageEvent::TextDelta {
                            content_index: position,
                            delta: text,
                            partial: asm.message.clone(),
                        });
                    }
                }
                Delta::ThinkingDelta { thinking } => {
                    if let Some(AssistantContent::Thinking(t)) =
                        asm.message.content.get_mut(position)
                    {
                        t.thinking.push_str(&thinking);
                        tx.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: position,
                            delta: thinking,
                            partial: asm.message.clone(),
                        });
                    }
                }
                Delta::InputJsonDelta { partial_json } => {
                    let entry = asm.partial_json.entry(index).or_default();
                    entry.push_str(&partial_json);
                    let parsed = parse_partial_json(entry);
                    if let Some(AssistantContent::ToolCall(call)) =
                        asm.message.content.get_mut(position)
                    {
                        if let Value::Object(map) = parsed {
                            call.arguments = map;
                        }
                    }
                    tx.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: position,
                        delta: partial_json,
                        partial: asm.message.clone(),
                    });
                }
                Delta::SignatureDelta { signature } => {
                    if let Some(AssistantContent::Thinking(t)) =
                        asm.message.content.get_mut(position)
                    {
                        let sig = t.thinking_signature.get_or_insert_with(String::new);
                        sig.push_str(&signature);
                    }
                }
            }
        }
        AnthropicEvent::ContentBlockStop { index } => {
            let Some(position) = asm.position(index) else {
                return Ok(());
            };
            match asm.message.content.get(position).cloned() {
                Some(AssistantContent::Text(t)) => {
                    tx.push(AssistantMessageEvent::TextEnd {
                        content_index: position,
                        content: t.text,
                        partial: asm.message.clone(),
                    });
                }
                Some(AssistantContent::Thinking(t)) => {
                    tx.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: position,
                        content: t.thinking,
                        partial: asm.message.clone(),
                    });
                }
                Some(AssistantContent::ToolCall(call)) => {
                    // Final parse of the full accumulated JSON.
                    if let Some(json) = asm.partial_json.get(&index) {
                        if let Some(AssistantContent::ToolCall(target)) =
                            asm.message.content.get_mut(position)
                        {
                            if let Value::Object(map) = parse_partial_json(json) {
                                target.arguments = map;
                            }
                        }
                    }
                    asm.partial_json.remove(&index);
                    tx.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: position,
                        tool_call: AssistantContent::ToolCall(call),
                        partial: asm.message.clone(),
                    });
                }
                None => {}
            }
        }
        AnthropicEvent::MessageDelta { delta, usage } => {
            if let Some(raw) = &delta.stop_reason {
                asm.message.raw_stop_reason = Some(raw.clone());
                let (reason, error_message) = map_stop_reason(raw);
                asm.message.stop_reason = reason;
                if let Some(message) = error_message {
                    asm.message.error_message = Some(message);
                }
            }
            if let Some(usage) = usage {
                let u = &mut asm.message.usage;
                if let Some(v) = usage.input_tokens {
                    u.input = v;
                }
                if let Some(v) = usage.output_tokens {
                    u.output = v;
                }
                if let Some(v) = usage.cache_read_input_tokens {
                    u.cache_read = v;
                }
                if let Some(v) = usage.cache_creation_input_tokens {
                    u.cache_write = v;
                }
                u.total_tokens = u.input + u.output + u.cache_read + u.cache_write;
                calculate_cost(&model.cost, u);
            }
        }
        AnthropicEvent::MessageStop {} => {
            // handled by caller loop break
        }
        AnthropicEvent::Ping {} => {}
        AnthropicEvent::Error { error } => {
            asm.message.error_message = Some(error.message);
            return Err(StopReason::Error);
        }
    }
    Ok(())
}

async fn wait_or_abort(
    cancel: &Option<tokio_util::sync::CancellationToken>,
    delay: Duration,
) -> Result<(), ()> {
    match cancel {
        Some(token) => {
            tokio::select! {
                _ = tokio::time::sleep(delay) => Ok(()),
                _ = token.cancelled() => Err(()),
            }
        }
        None => {
            tokio::time::sleep(delay).await;
            Ok(())
        }
    }
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}
