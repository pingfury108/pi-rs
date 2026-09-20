//! OpenAI Responses API streaming (`openai-responses`), also covering the
//! Azure variant (different base URL + `api-key` header), ported from
//! `packages/ai/src/api/openai-responses.ts` + `openai-responses-shared.ts`.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{current_tools, initial_system_text, normalize_context, LlmApi, RetryPolicy};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::partial_json::parse_partial_json;
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, StreamOptions,
    TextContent, ThinkingContent, ToolCall, Usage,
};

pub struct OpenAIResponsesApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl OpenAIResponsesApi {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            retry: RetryPolicy::default(),
        }
    }
}

impl Default for OpenAIResponsesApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for OpenAIResponsesApi {
    fn name(&self) -> &str {
        "openai-responses"
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

/// Build the `/responses` request body.
pub fn build_body(model: &Model, messages: &[Message], options: &StreamOptions) -> Value {
    let system = initial_system_text(messages);
    let tools = current_tools(messages);

    let mut input: Vec<Value> = Vec::new();
    if !system.is_empty() {
        input.push(json!({"role": "developer", "content": [{"type": "input_text", "text": system}]}));
    }
    input.extend(messages.iter().skip(1).filter_map(convert_message));

    let mut body = json!({
        "model": model.id,
        "input": input,
        "stream": true,
        "store": false,
    });

    if let Some(max_tokens) = options.max_tokens {
        // OpenAI Responses rejects values below 16
        body["max_output_tokens"] = json!(max_tokens.max(16));
    }
    if let Some(temperature) = options.temperature {
        body["temperature"] = json!(temperature);
    }
    if !tools.is_empty() {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| {
                let parameters = crate::constrained_sampling::resolve_strict(&t.parameters, true)
                    .unwrap_or_else(|| t.parameters.clone());
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                    "strict": true,
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
    if model.reasoning {
        match options.reasoning {
            Some(level) => {
                let effort = match level {
                    crate::types::ThinkingLevel::Minimal | crate::types::ThinkingLevel::Low => "low",
                    crate::types::ThinkingLevel::Medium => "medium",
                    _ => "high",
                };
                body["reasoning"] = json!({"effort": effort, "summary": "auto"});
            }
            None => {
                body["reasoning"] = json!({"effort": "none"});
            }
        }
    }
    body
}

/// Convert a transcript message to the Responses `input` item format.
pub fn convert_message(msg: &Message) -> Option<Value> {
    match msg {
        Message::System(_) => None,
        Message::User(user) => {
            use crate::types::{MessageContent, UserContent};
            let content = match &user.content {
                MessageContent::Text(s) => json!([{ "type": "input_text", "text": s }]),
                MessageContent::Blocks(blocks) => Value::Array(
                    blocks
                        .iter()
                        .map(|b| match b {
                            UserContent::Text(t) => json!({"type": "input_text", "text": t.text}),
                            UserContent::Image(img) => json!({
                                "type": "input_image",
                                "image_url": format!("data:{};base64,{}", img.mime_type, img.data),
                            }),
                        })
                        .collect(),
                ),
            };
            Some(json!({"role": "user", "content": content}))
        }
        Message::Assistant(assistant) => {
            let mut items: Vec<Value> = Vec::new();
            let mut text_parts: Vec<Value> = Vec::new();
            for block in &assistant.content {
                match block {
                    AssistantContent::Text(t) => {
                        if !t.text.is_empty() {
                            text_parts.push(json!({"type": "output_text", "text": t.text, "annotations": []}));
                        }
                    }
                    AssistantContent::Thinking(_) => {}
                    AssistantContent::ToolCall(call) => {
                        if !text_parts.is_empty() {
                            items.push(json!({
                                "role": "assistant",
                                "content": text_parts,
                            }));
                            text_parts = Vec::new();
                        }
                        items.push(json!({
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).unwrap_or_default(),
                        }));
                    }
                }
            }
            if !text_parts.is_empty() {
                items.push(json!({"role": "assistant", "content": text_parts}));
            }
            if items.is_empty() {
                None
            } else {
                Some(Value::Array(items))
            }
        }
        Message::ToolResult(result) => {
            use crate::types::ToolResultContent;
            let text = result
                .content
                .iter()
                .map(|c| match c {
                    ToolResultContent::Text(t) => t.text.clone(),
                    ToolResultContent::Image(_) => "[image]".into(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(json!({
                "type": "function_call_output",
                "call_id": result.tool_call_id,
                "output": text,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// SSE mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    output_index: Option<usize>,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    item: Option<Value>,
    #[serde(default)]
    response: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

struct Assembler {
    message: AssistantMessage,
    /// wire output_index → (content position, kind, partial args json)
    slots: std::collections::HashMap<usize, (usize, SlotKind, String)>,
}

#[derive(Clone, Copy, PartialEq)]
enum SlotKind {
    Text,
    Thinking,
    ToolCall,
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
            slots: std::collections::HashMap::new(),
        }
    }
}

#[allow(clippy::too_many_lines)]
async fn run(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    tx: &EventStreamTx,
) {
    let mut asm = Assembler::new(model);
    match execute(http, retry, model, messages, options, tx, &mut asm).await {
        Ok(()) => {
            let mut message = asm.message;
            if message.stop_reason == StopReason::Pending {
                message.stop_reason = StopReason::Stop;
            }
            if message.stop_reason == StopReason::Stop
                && message
                    .content
                    .iter()
                    .any(|b| matches!(b, AssistantContent::ToolCall(_)))
            {
                message.stop_reason = StopReason::ToolUse;
            }
            tx.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
        }
        Err(reason) => {
            let mut message = asm.message;
            message.stop_reason = reason;
            message.error_message = Some(match reason {
                StopReason::Aborted => "Request was aborted".into(),
                _ => message
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Unknown error".into()),
            });
            tx.push(AssistantMessageEvent::Error { reason, error: message });
        }
    }
}

#[allow(clippy::too_many_lines)]
async fn execute(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    tx: &EventStreamTx,
    asm: &mut Assembler,
) -> Result<(), StopReason> {
    let cancel = options.cancel.clone();
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
    let is_azure = base_url.contains("openai.azure.com");
    let url = if is_azure {
        format!(
            "{}/openai/v1/responses?api-version=preview",
            base_url.trim_end_matches('/')
        )
    } else {
        format!("{}/responses", base_url.trim_end_matches('/'))
    };
    let auth_header = if is_azure {
        ("api-key".to_string(), api_key)
    } else {
        ("Authorization".to_string(), format!("Bearer {api_key}"))
    };

    let body = build_body(model, messages, options);
    let mut error_slot = String::new();
    let body_stream = match super::open_sse_post(
        http,
        retry,
        &url,
        vec![auth_header, ("content-type".into(), "application/json".into())],
        &body,
        &cancel,
        &mut error_slot,
    )
    .await
    {
        Ok(stream) => stream,
        Err(reason) => {
            if !error_slot.is_empty() {
                asm.message.error_message = Some(error_slot);
            }
            return Err(reason);
        }
    };

    tx.push(AssistantMessageEvent::Start {
        partial: asm.message.clone(),
    });

    let mut sse = body_stream.eventsource();
    let mut saw_terminal: bool = false;
    #[allow(unused_assignments)]
    while let Some(sse) = sse.next().await {
        if cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err(StopReason::Aborted);
        }
        let sse = match sse {
            Ok(ev) => ev,
            Err(err) => {
                asm.message.error_message = Some(format!("SSE error: {err}"));
                return Err(StopReason::Error);
            }
        };
        let Ok(event) = serde_json::from_str::<StreamEvent>(sse.data.trim()) else {
            continue;
        };

        match event.event_type.as_str() {
            "response.created" => {
                if let Some(response) = &event.response {
                    asm.message.response_id =
                        response.get("id").and_then(Value::as_str).map(String::from);
                }
            }
            "response.output_item.added" => {
                if let (Some(index), Some(item)) = (event.output_index, event.item.as_ref()) {
                    create_slot(asm, index, item, tx);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some((position, _, _)) = event.output_index.and_then(|i| asm.slots.get(&i)) {
                    let position = *position;
                    if let Some(AssistantContent::Thinking(t)) =
                        asm.message.content.get_mut(position)
                    {
                        if let Some(delta) = &event.delta {
                            t.thinking.push_str(delta);
                        }
                    }
                    tx.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: position,
                        delta: event.delta.clone().unwrap_or_default(),
                        partial: asm.message.clone(),
                    });
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                if let Some((position, _, _)) = event.output_index.and_then(|i| asm.slots.get(&i)) {
                    let position = *position;
                    if let Some(AssistantContent::Text(t)) = asm.message.content.get_mut(position) {
                        if let Some(delta) = &event.delta {
                            t.text.push_str(delta);
                        }
                    }
                    tx.push(AssistantMessageEvent::TextDelta {
                        content_index: position,
                        delta: event.delta.clone().unwrap_or_default(),
                        partial: asm.message.clone(),
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some((position, _, partial)) =
                    event.output_index.and_then(|i| asm.slots.get_mut(&i))
                {
                    let position = *position;
                    if let Some(delta) = &event.delta {
                        partial.push_str(delta);
                        if let Some(AssistantContent::ToolCall(call)) =
                            asm.message.content.get_mut(position)
                        {
                            if let Value::Object(map) = parse_partial_json(partial) {
                                call.arguments = map;
                            }
                        }
                        tx.push(AssistantMessageEvent::ToolcallDelta {
                            content_index: position,
                            delta: delta.clone(),
                            partial: asm.message.clone(),
                        });
                    }
                }
            }
            "response.output_item.done" => {
                if let (Some(index), Some(item)) = (event.output_index, event.item.as_ref()) {
                    finish_slot(asm, index, item, tx);
                }
            }
            "response.completed" | "response.incomplete" => {
                finalize_response(asm, event.response.as_ref());
                saw_terminal = true;
            }
            "response.failed" => {
                saw_terminal = true;
                asm.message.raw_stop_reason = event
                    .response
                    .as_ref()
                    .and_then(|r| r.get("status"))
                    .and_then(Value::as_str)
                    .map(String::from);
                let message = event
                    .response
                    .as_ref()
                    .and_then(|r| r.get("error").cloned())
                    .and_then(|e| e.get("message").and_then(Value::as_str).map(String::from));
                asm.message.error_message =
                    Some(message.unwrap_or_else(|| "response failed".into()));
                return Err(StopReason::Error);
            }
            "error" => {
                asm.message.error_message = Some(
                    event
                        .message
                        .clone()
                        .unwrap_or_else(|| "unknown error".into()),
                );
                return Err(StopReason::Error);
            }
            _ => {}
        }
    }

    if !saw_terminal {
        asm.message.error_message =
            Some("stream ended before a terminal response event".into());
        return Err(StopReason::Error);
    }
    Ok(())
}

fn create_slot(asm: &mut Assembler, index: usize, item: &Value, tx: &EventStreamTx) {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    let position = asm.message.content.len();
    match item_type {
        "message" => {
            asm.message
                .content
                .push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
            asm.slots.insert(index, (position, SlotKind::Text, String::new()));
            tx.push(AssistantMessageEvent::TextStart {
                content_index: position,
                partial: asm.message.clone(),
            });
        }
        "reasoning" => {
            asm.message
                .content
                .push(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: false,
                }));
            asm.slots
                .insert(index, (position, SlotKind::Thinking, String::new()));
            tx.push(AssistantMessageEvent::ThinkingStart {
                content_index: position,
                partial: asm.message.clone(),
            });
        }
        "function_call" => {
            let id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            asm.message
                .content
                .push(AssistantContent::ToolCall(ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: Default::default(),
                    thought_signature: None,
                    namespace: None,
                }));
            asm.slots.insert(
                index,
                (position, SlotKind::ToolCall, String::new()),
            );
            tx.push(AssistantMessageEvent::ToolcallStart {
                content_index: position,
                partial: asm.message.clone(),
            });
        }
        _ => {}
    }
}

fn finish_slot(asm: &mut Assembler, index: usize, item: &Value, tx: &EventStreamTx) {
    let Some((position, kind, partial)) = asm.slots.remove(&index) else {
        return;
    };
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match (item_type, kind) {
        ("message", SlotKind::Text) => {
            let final_text: String = item
                .get("content")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|c| c.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            if let Some(AssistantContent::Text(t)) = asm.message.content.get_mut(position) {
                if !final_text.is_empty() {
                    t.text = final_text.clone();
                }
            }
            tx.push(AssistantMessageEvent::TextEnd {
                content_index: position,
                content: final_text,
                partial: asm.message.clone(),
            });
        }
        ("reasoning", SlotKind::Thinking) => {
            let summary: String = item
                .get("summary")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|s| s.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default();
            if let Some(AssistantContent::Thinking(t)) = asm.message.content.get_mut(position) {
                if !summary.is_empty() {
                    t.thinking = summary.clone();
                }
                t.thinking_signature = serde_json::to_string(item).ok();
            }
            tx.push(AssistantMessageEvent::ThinkingEnd {
                content_index: position,
                content: summary,
                partial: asm.message.clone(),
            });
        }
        ("function_call", SlotKind::ToolCall) => {
            let final_args = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or(partial.as_str());
            if let Some(AssistantContent::ToolCall(call)) = asm.message.content.get_mut(position) {
                if let Value::Object(map) = parse_partial_json(final_args) {
                    call.arguments = map;
                }
            }
            asm.message.content.get(position).cloned().map(|call| {
                tx.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: position,
                    tool_call: call,
                    partial: asm.message.clone(),
                })
            });
        }
        _ => {}
    }
}

fn finalize_response(asm: &mut Assembler, response: Option<&Value>) {
    if let Some(response) = response {
        if let Some(usage) = response.get("usage") {
            let u = &mut asm.message.usage;
            u.input = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            u.output = usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            u.cache_read = usage
                .get("input_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            u.reasoning = usage
                .get("output_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(Value::as_u64);
            u.total_tokens = u.input + u.output;
        }
        let status = response.get("status").and_then(Value::as_str);
        let incomplete = response.get("incomplete_details").and_then(|d| {
            d.get("reason").and_then(Value::as_str).map(String::from)
        });
        asm.message.raw_stop_reason = status.map(String::from);
        asm.message.stop_reason = match (status, incomplete.as_deref()) {
            (Some("completed"), _) => StopReason::Stop,
            (Some("incomplete"), Some("max_output_tokens")) => StopReason::Length,
            (Some("incomplete"), reason) => {
                asm.message.error_message =
                    Some(format!("incomplete: {}", reason.unwrap_or("unknown")));
                StopReason::Error
            }
            _ => StopReason::Stop,
        };
        if asm
            .message
            .content
            .iter()
            .any(|b| matches!(b, AssistantContent::ToolCall(_)))
            && asm.message.stop_reason == StopReason::Stop
        {
            asm.message.stop_reason = StopReason::ToolUse;
        }
    }
}
