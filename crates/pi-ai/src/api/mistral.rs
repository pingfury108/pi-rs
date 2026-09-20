//! Mistral chat API streaming (`mistral-conversations`), port of the
//! essential path of `packages/ai/src/api/mistral-conversations.ts`.
//! Mistral's wire format is OpenAI-compatible with mistral-specific
//! reasoning fields (`prompt_mode`, `thinking` blocks).

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{calculate_cost, current_tools, initial_system_text, normalize_context, LlmApi, RetryPolicy};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::partial_json::parse_partial_json;
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, StreamOptions,
    TextContent, ThinkingContent, ToolCall, Usage,
};

pub struct MistralApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl MistralApi {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            retry: RetryPolicy::default(),
        }
    }
}

impl Default for MistralApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for MistralApi {
    fn name(&self) -> &str {
        "mistral-conversations"
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

fn build_body(model: &Model, messages: &[Message], options: &StreamOptions) -> Value {
    let system = initial_system_text(messages);
    let tools = current_tools(messages);

    let mut wire: Vec<Value> = Vec::new();
    if !system.is_empty() {
        wire.push(json!({"role": "system", "content": system}));
    }
    wire.extend(messages.iter().skip(1).filter_map(convert_message));

    let mut body = json!({
        "model": model.id,
        "messages": wire,
        "stream": true,
    });
    body["max_tokens"] = json!(options.max_tokens.unwrap_or(model.max_tokens));
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
    }
    if model.reasoning {
        if options.reasoning.is_some() {
            body["prompt_mode"] = json!("reasoning");
        }
    }
    body
}

fn convert_message(msg: &Message) -> Option<Value> {
    match msg {
        Message::System(_) => None,
        Message::User(user) => {
            let content = user.content.as_text();
            Some(json!({"role": "user", "content": content}))
        }
        Message::Assistant(assistant) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            for block in &assistant.content {
                match block {
                    AssistantContent::Text(t) => text_parts.push(t.text.clone()),
                    AssistantContent::Thinking(t) => {
                        // mistral expects thinking replay as separate content blocks
                        if !t.thinking.is_empty() {
                            text_parts.push(t.thinking.clone());
                        }
                    }
                    AssistantContent::ToolCall(call) => tool_calls.push(json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).unwrap_or_default(),
                        },
                    })),
                }
            }
            let mut out = json!({
                "role": "assistant",
                "content": if text_parts.is_empty() { Value::Null } else { json!(text_parts.join("\n")) },
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
                    ToolResultContent::Image(_) => "[image]".into(),
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

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    id: String,
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
    thinking: Option<Vec<ThinkingBlock>>,
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Debug, Deserialize)]
struct ThinkingBlock {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    index: usize,
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

struct Assembler {
    message: AssistantMessage,
    tool_positions: std::collections::HashMap<usize, usize>,
    partial_json: std::collections::HashMap<usize, String>,
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
            tool_positions: std::collections::HashMap::new(),
            partial_json: std::collections::HashMap::new(),
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
    match execute(http, retry, model, messages, options, tx, &mut asm).await {
        Ok(()) => {
            let mut message = asm.message;
            if message.stop_reason == StopReason::Pending {
                message.stop_reason = StopReason::Stop;
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
        .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
        .ok_or_else(|| {
            tracing::error!("missing API key for provider {}", model.provider);
            StopReason::Error
        })?;

    let url = format!(
        "{}/v1/chat/completions",
        model.base_url.trim_end_matches('/')
    );
    let body = build_body(model, messages, options);
    let mut error_slot = String::new();
    let body_stream = match super::open_sse_post(
        http,
        retry,
        &url,
        vec![
            ("Authorization".into(), format!("Bearer {api_key}")),
            ("content-type".into(), "application/json".into()),
        ],
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
        let data = sse.data.trim();
        if data.is_empty() || data == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Chunk>(data) else {
            continue;
        };

        if asm.message.response_id.is_none() && !chunk.id.is_empty() {
            asm.message.response_id = Some(chunk.id.clone());
        }

        for choice in &chunk.choices {
            // thinking blocks
            if let Some(blocks) = &choice.delta.thinking {
                for block in blocks {
                    let Some(text) = &block.text else { continue };
                    if text.is_empty() {
                        continue;
                    }
                    let position = match asm.message.content.last() {
                        Some(AssistantContent::Thinking(_)) => asm.message.content.len() - 1,
                        _ => {
                            asm.message
                                .content
                                .push(AssistantContent::Thinking(ThinkingContent {
                                    thinking: String::new(),
                                    thinking_signature: None,
                                    redacted: false,
                                }));
                            tx.push(AssistantMessageEvent::ThinkingStart {
                                content_index: asm.message.content.len() - 1,
                                partial: asm.message.clone(),
                            });
                            asm.message.content.len() - 1
                        }
                    };
                    if let Some(AssistantContent::Thinking(t)) =
                        asm.message.content.get_mut(position)
                    {
                        t.thinking.push_str(text);
                    }
                    tx.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: position,
                        delta: text.clone(),
                        partial: asm.message.clone(),
                    });
                }
            }

            // text content
            if let Some(text) = &choice.delta.content {
                if !text.is_empty() {
                    let position = match asm.message.content.last() {
                        Some(AssistantContent::Text(_)) => asm.message.content.len() - 1,
                        _ => {
                            asm.message
                                .content
                                .push(AssistantContent::Text(TextContent {
                                    text: String::new(),
                                    text_signature: None,
                                }));
                            tx.push(AssistantMessageEvent::TextStart {
                                content_index: asm.message.content.len() - 1,
                                partial: asm.message.clone(),
                            });
                            asm.message.content.len() - 1
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

            // tool calls
            for call in &choice.delta.tool_calls {
                let position = match asm.tool_positions.get(&call.index) {
                    Some(p) => *p,
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
                        if let Some(AssistantContent::ToolCall(target)) =
                            asm.message.content.get_mut(position)
                        {
                            if let Value::Object(map) = parse_partial_json(entry) {
                                target.arguments = map;
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

            if let Some(finish) = &choice.finish_reason {
                asm.message.raw_stop_reason = Some(finish.clone());
                asm.message.stop_reason = match finish.as_str() {
                    "stop" => StopReason::Stop,
                    "length" => StopReason::Length,
                    "tool_calls" => StopReason::ToolUse,
                    _ => StopReason::Stop,
                };
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

    if asm.message.stop_reason == StopReason::Stop
        && asm
            .message
            .content
            .iter()
            .any(|b| matches!(b, AssistantContent::ToolCall(_)))
    {
        asm.message.stop_reason = StopReason::ToolUse;
    }
    Ok(())
}
