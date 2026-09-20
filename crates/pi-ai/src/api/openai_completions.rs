//! OpenAI Chat Completions API streaming implementation.
//!
//! Port of the essential path of `packages/ai/src/api/openai-completions.ts`.
//! Also serves every OpenAI-compatible provider (deepseek, groq, openrouter,
//! moonshot, ollama, ...) via distinct `base_url` values.
//!
//! Not ported (yet): per-provider `compat` auto-detection, thinking-format
//! variants beyond `reasoning_effort`/`reasoning_content`, session affinity.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
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

pub struct OpenAICompletionsApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl OpenAICompletionsApi {
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

impl Default for OpenAICompletionsApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for OpenAICompletionsApi {
    fn name(&self) -> &str {
        "openai-completions"
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

/// Build the `/chat/completions` request body.
pub fn build_body(model: &Model, messages: &[Message], options: &StreamOptions) -> Value {
    let system = initial_system_text(messages);
    let tools = current_tools(messages);

    let mut wire_messages: Vec<Value> = Vec::new();
    if !system.is_empty() {
        wire_messages.push(json!({"role": "system", "content": system}));
    }
    wire_messages.extend(messages.iter().skip(1).filter_map(convert_message));

    let mut body = json!({
        "model": model.id,
        "messages": wire_messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });

    let max_tokens = options.max_tokens.unwrap_or(model.max_tokens);
    // Modern OpenAI models use max_completion_tokens; send both-safe default.
    body["max_tokens"] = json!(max_tokens);

    if let Some(temperature) = options.temperature {
        body["temperature"] = json!(temperature);
    }

    if !tools.is_empty() {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                })
            })
            .collect();
        body["tools"] = Value::Array(converted);
        if let Some(choice) = options.tool_choice {
            body["tool_choice"] = match choice {
                crate::types::ToolChoice::Auto => json!("auto"),
                crate::types::ToolChoice::None => json!("none"),
            };
        }
    }

    // Reasoning effort for reasoning-capable models (OpenAI o-series style).
    if model.reasoning {
        if let Some(level) = options.reasoning {
            let effort = match level {
                crate::types::ThinkingLevel::Minimal | crate::types::ThinkingLevel::Low => "low",
                crate::types::ThinkingLevel::Medium => "medium",
                _ => "high",
            };
            body["reasoning_effort"] = json!(effort);
        }
    }

    // Merge model default sampling params, then explicit overrides.
    if let Some(defaults) = &model.sampling_params {
        if let Some(target) = body.as_object_mut() {
            for (k, v) in defaults {
                target.insert(k.clone(), v.clone());
            }
        }
    }

    body
}

/// Convert a transcript message to the OpenAI wire format.
pub fn convert_message(msg: &Message) -> Option<Value> {
    match msg {
        Message::System(_) => None, // handled by caller (leading system)
        Message::User(user) => {
            use crate::types::{MessageContent, UserContent};
            let content = match &user.content {
                MessageContent::Text(s) => json!(s),
                MessageContent::Blocks(blocks) => {
                    let has_images = blocks.iter().any(|b| matches!(b, UserContent::Image(_)));
                    if !has_images {
                        json!(blocks
                            .iter()
                            .filter_map(|b| match b {
                                UserContent::Text(t) => Some(t.text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"))
                    } else {
                        let arr: Vec<Value> = blocks
                            .iter()
                            .map(|b| match b {
                                UserContent::Text(t) => {
                                    json!({"type": "text", "text": t.text})
                                }
                                UserContent::Image(img) => json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!("data:{};base64,{}", img.mime_type, img.data),
                                    },
                                }),
                            })
                            .collect();
                        Value::Array(arr)
                    }
                }
            };
            Some(json!({"role": "user", "content": content}))
        }
        Message::Assistant(assistant) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            for block in &assistant.content {
                match block {
                    AssistantContent::Text(t) => text_parts.push(t.text.clone()),
                    AssistantContent::Thinking(_) => {
                        // thinking content is not replayed on OpenAI-compatible APIs
                    }
                    AssistantContent::ToolCall(call) => {
                        tool_calls.push(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments).unwrap_or_default(),
                            },
                        }));
                    }
                }
            }
            let mut out = json!({
                "role": "assistant",
                "content": if text_parts.is_empty() {
                    Value::Null
                } else {
                    json!(text_parts.join("\n"))
                },
            });
            if !tool_calls.is_empty() {
                out["tool_calls"] = Value::Array(tool_calls);
            }
            Some(out)
        }
        Message::ToolResult(result) => {
            use crate::types::ToolResultContent;
            let text = result
                .content
                .iter()
                .map(|c| match c {
                    ToolResultContent::Text(t) => t.text.clone(),
                    ToolResultContent::Image(_) => "[image]".to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(json!({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": text,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// SSE mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Debug, Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Debug, Deserialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct WireUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// Per-tool-call-index bookkeeping.
struct Assembler {
    message: AssistantMessage,
    /// wire tool-call index → position in `message.content`
    tool_positions: HashMap<usize, usize>,
    /// wire tool-call index → accumulated partial JSON
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
            tool_positions: HashMap::new(),
            partial_json: HashMap::new(),
        }
    }
}

async fn run(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    tx: &EventStreamTx,
) {
    let mut asm = Assembler::new(model);
    let result = execute(http, retry, model, messages, options, tx, &mut asm).await;

    match result {
        Ok(()) => {
            let message = asm.message;
            if message.stop_reason == StopReason::Pending {
                let mut message = message;
                message.stop_reason = StopReason::Stop;
                tx.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
            } else {
                tx.push(AssistantMessageEvent::Done {
                    reason: message.stop_reason,
                    message,
                });
            }
        }
        Err(reason) => {
            let mut message = asm.message;
            message.stop_reason = reason;
            message.error_message = Some(match reason {
                StopReason::Aborted => "Request was aborted".to_string(),
                _ => message
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Unknown error".into()),
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
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| {
            tracing::error!("missing API key for provider {}", model.provider);
            StopReason::Error
        })?;

    let base_url = options
        .base_url
        .clone()
        .unwrap_or_else(|| model.base_url.clone());
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = build_body(model, messages, options);

    let mut attempt = 0u32;
    let response = loop {
        if aborted() {
            return Err(StopReason::Aborted);
        }
        let request = http
            .post(&url)
            .bearer_auth(&api_key)
            .header("content-type", "application/json")
            .json(&body);
        match request.send().await {
            Ok(resp) if resp.status().is_success() => break resp,
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                if !is_retryable_status(status) || attempt >= retry.max_retries {
                    asm.message.error_message =
                        Some(format!("HTTP {status}: {}", truncate(&body_text, 2048)));
                    return Err(StopReason::Error);
                }
                tracing::warn!(status, attempt, "openai request failed, retrying");
                attempt += 1;
                let delay = Duration::from_millis(
                    retry.base_delay_ms.saturating_mul(2u64.saturating_pow(attempt)),
                )
                .min(Duration::from_millis(retry.max_delay_ms));
                wait_or_abort(&cancel, delay).await.map_err(|()| StopReason::Aborted)?;
            }
            Err(err) => {
                if attempt >= retry.max_retries {
                    asm.message.error_message = Some(format!("request failed: {err}"));
                    return Err(StopReason::Error);
                }
                tracing::warn!(%err, attempt, "openai request error, retrying");
                attempt += 1;
                let delay = Duration::from_millis(
                    retry.base_delay_ms.saturating_mul(2u64.saturating_pow(attempt)),
                )
                .min(Duration::from_millis(retry.max_delay_ms));
                wait_or_abort(&cancel, delay).await.map_err(|()| StopReason::Aborted)?;
            }
        }
    };

    tx.push(AssistantMessageEvent::Start {
        partial: asm.message.clone(),
    });

    let mut sse_stream = response.bytes_stream().eventsource();
    let mut saw_finish_reason = false;
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
        let data = sse.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk: Chunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(data = %data, %err, "failed to parse openai SSE chunk");
                continue;
            }
        };

        if asm.message.response_id.is_none() && !chunk.id.is_empty() {
            asm.message.response_id = Some(chunk.id.clone());
        }
        if !chunk.model.is_empty() && chunk.model != model.id {
            asm.message.response_model = Some(chunk.model.clone());
        }

        for choice in &chunk.choices {
            handle_delta(choice, tx, asm, model)?;
            if choice.finish_reason.is_some() {
                saw_finish_reason = true;
                asm.message.raw_stop_reason = choice.finish_reason.clone();
                let (reason, error_message) = map_finish_reason(choice.finish_reason.as_deref());
                asm.message.stop_reason = reason;
                if let Some(message) = error_message {
                    asm.message.error_message = Some(message);
                }
            }
        }

        if let Some(usage) = chunk.usage {
            let u = &mut asm.message.usage;
            u.input = usage.prompt_tokens;
            u.output = usage.completion_tokens;
            u.total_tokens = u.input + u.output;
            calculate_cost(&model.cost, u);
        }
    }

    // Some compatible servers omit finish_reason entirely.
    if !saw_finish_reason && asm.message.stop_reason == StopReason::Pending {
        asm.message.stop_reason = StopReason::Stop;
    }
    Ok(())
}

fn handle_delta(
    choice: &ChunkChoice,
    tx: &EventStreamTx,
    asm: &mut Assembler,
    model: &Model,
) -> ExecResult {
    let _ = model;
    // reasoning content first (deepseek-r1 style)
    if let Some(thinking) = &choice.delta.reasoning_content {
        if !thinking.is_empty() {
            // one thinking block for the whole message; create lazily
            let position = match asm.message.content.first() {
                Some(AssistantContent::Thinking(_)) => 0,
                _ => {
                    asm.message
                        .content
                        .insert(0, AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                            redacted: false,
                        }));
                    // shift tool positions
                    for position in asm.tool_positions.values_mut() {
                        *position += 1;
                    }
                    tx.push(AssistantMessageEvent::ThinkingStart {
                        content_index: 0,
                        partial: asm.message.clone(),
                    });
                    0
                }
            };
            if let Some(AssistantContent::Thinking(t)) = asm.message.content.get_mut(position) {
                t.thinking.push_str(thinking);
            }
            tx.push(AssistantMessageEvent::ThinkingDelta {
                content_index: position,
                delta: thinking.clone(),
                partial: asm.message.clone(),
            });
        }
    }

    if let Some(text) = &choice.delta.content {
        if !text.is_empty() {
            let position = match asm.message.content.last() {
                Some(AssistantContent::Text(_)) => asm.message.content.len() - 1,
                _ => {
                    let position = asm.message.content.len();
                    asm.message
                        .content
                        .push(AssistantContent::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }));
                    tx.push(AssistantMessageEvent::TextStart {
                        content_index: position,
                        partial: asm.message.clone(),
                    });
                    position
                }
            };
            if let Some(AssistantContent::Text(t)) = asm.message.content.get_mut(position) {
                t.text.push_str(text);
            }
            tx.push(AssistantMessageEvent::TextDelta {
                content_index: position,
                delta: text.clone(),
                partial: asm.message.clone(),
            });
        }
    }

    for call in &choice.delta.tool_calls {
        let position = match asm.tool_positions.get(&call.index) {
            Some(position) => *position,
            None => {
                let position = asm.message.content.len();
                let id = call.id.clone().unwrap_or_default();
                let name = call
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();
                asm.message
                    .content
                    .push(AssistantContent::ToolCall(ToolCall {
                        id,
                        name,
                        arguments: Default::default(),
                        thought_signature: None,
                        namespace: None,
                    }));
                asm.partial_json.insert(call.index, String::new());
                asm.tool_positions.insert(call.index, position);
                tx.push(AssistantMessageEvent::ToolcallStart {
                    content_index: position,
                    partial: asm.message.clone(),
                });
                position
            }
        };

        if let Some(function) = &call.function {
            if let Some(name) = &function.name {
                if let Some(AssistantContent::ToolCall(target)) =
                    asm.message.content.get_mut(position)
                {
                    if target.name.is_empty() {
                        target.name = name.clone();
                    }
                }
            }
            if let Some(arguments) = &function.arguments {
                let entry = asm.partial_json.entry(call.index).or_default();
                entry.push_str(arguments);
                let parsed = parse_partial_json(entry);
                if let Some(AssistantContent::ToolCall(target)) =
                    asm.message.content.get_mut(position)
                {
                    if let Value::Object(map) = parsed {
                        target.arguments = map;
                    }
                    if let Some(id) = &call.id {
                        if target.id.is_empty() {
                            target.id = id.clone();
                        }
                    }
                }
                tx.push(AssistantMessageEvent::ToolcallDelta {
                    content_index: position,
                    delta: arguments.clone(),
                    partial: asm.message.clone(),
                });
            }
        }
    }
    Ok(())
}

fn map_finish_reason(raw: Option<&str>) -> (StopReason, Option<String>) {
    match raw {
        Some("stop") => (StopReason::Stop, None),
        Some("length") => (StopReason::Length, None),
        Some("tool_calls") | Some("function_call") => (StopReason::ToolUse, None),
        Some("content_filter") => (
            StopReason::Error,
            Some("Content filter triggered".to_string()),
        ),
        Some(other) => {
            tracing::debug!(reason = other, "unknown finish_reason, mapping to stop");
            (StopReason::Stop, None)
        }
        None => (StopReason::Pending, None),
    }
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
