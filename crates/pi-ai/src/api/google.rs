//! Google Generative Language API streaming (`google-generative-ai`) and the
//! Vertex AI variant (`google-vertex`, Bearer auth + project endpoint),
//! ported from `packages/ai/src/api/google-generative-ai.ts` +
//! `google-shared.ts`.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{calculate_cost, current_tools, initial_system_text, normalize_context, LlmApi, RetryPolicy};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, StreamOptions,
    TextContent, ThinkingContent, ToolCall, Usage,
};

/// Gemini streaming via Generative Language REST (+ Vertex when the model's
/// base_url points at `aiplatform.googleapis.com`).
pub struct GoogleApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl GoogleApi {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            retry: RetryPolicy::default(),
        }
    }
}

impl Default for GoogleApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for GoogleApi {
    fn name(&self) -> &str {
        "google-generative-ai"
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

/// Build the `streamGenerateContent` request body.
pub fn build_body(model: &Model, messages: &[Message], options: &StreamOptions) -> Value {
    let system = initial_system_text(messages);
    let tools = current_tools(messages);

    let mut config = json!({});
    if let Some(t) = options.temperature {
        config["temperature"] = json!(t);
    }
    config["maxOutputTokens"] = json!(options.max_tokens.unwrap_or(model.max_tokens));

    // thinking
    if model.reasoning {
        match options.reasoning {
            Some(level) => {
                let budget = match level {
                    crate::types::ThinkingLevel::Minimal => 512,
                    crate::types::ThinkingLevel::Low => 2048,
                    crate::types::ThinkingLevel::Medium => 8192,
                    crate::types::ThinkingLevel::High => 24576,
                    _ => 32768,
                };
                config["thinkingConfig"] =
                    json!({"includeThoughts": true, "thinkingBudget": budget});
            }
            None => {
                config["thinkingConfig"] = json!({"thinkingBudget": 0});
            }
        }
    }

    // systemInstruction / tools / toolConfig are top-level fields
    let mut body = json!({
        "contents": convert_contents(messages),
        "generationConfig": config,
    });
    if !system.is_empty() {
        body["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    if !tools.is_empty() {
        let decls: Vec<Value> = tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "parameters": t.parameters}))
            .collect();
        body["tools"] = json!([{"functionDeclarations": decls}]);
        if let Some(choice) = options.tool_choice {
            let mode = match choice {
                crate::types::ToolChoice::Auto => "AUTO",
                crate::types::ToolChoice::None => "NONE",
            };
            body["toolConfig"] = json!({"functionCallingConfig": {"mode": mode}});
        }
    }
    body
}

/// Gemini `contents` array; tool results become `functionResponse` parts in
/// user turns; assistant turns use role `"model"`.
pub fn convert_contents(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for msg in messages.iter().skip(1) {
        match msg {
            Message::System(_) => {} // folded by normalize_context
            Message::User(user) => {
                let parts = convert_user_parts(&user.content);
                if !parts.is_empty() {
                    out.push(json!({"role": "user", "parts": parts}));
                }
            }
            Message::Assistant(assistant) => {
                let mut parts: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(t) => {
                            if !t.text.trim().is_empty() {
                                parts.push(json!({"text": t.text}));
                            }
                        }
                        AssistantContent::Thinking(t) => {
                            if !t.thinking.trim().is_empty() {
                                parts.push(json!({"thought": true, "text": t.thinking}));
                            }
                        }
                        AssistantContent::ToolCall(call) => parts.push(json!({
                            "functionCall": {"name": call.name, "args": call.arguments},
                        })),
                    }
                }
                if !parts.is_empty() {
                    out.push(json!({"role": "model", "parts": parts}));
                }
            }
            Message::ToolResult(result) => {
                let text: String = result
                    .content
                    .iter()
                    .map(|c| match c {
                        crate::types::ToolResultContent::Text(t) => t.text.clone(),
                        crate::types::ToolResultContent::Image(_) => "[image]".into(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let part = json!({
                    "functionResponse": {
                        "name": result.tool_name,
                        "response": {"result": text},
                    }
                });
                // merge into previous user turn
                match out.last_mut() {
                    Some(last) if last["role"] == "user" => {
                        last["parts"].as_array_mut().unwrap().push(part);
                    }
                    _ => out.push(json!({"role": "user", "parts": [part]})),
                }
            }
        }
    }
    out
}

fn convert_user_parts(content: &crate::types::MessageContent) -> Vec<Value> {
    use crate::types::{MessageContent, UserContent};
    match content {
        MessageContent::Text(s) => vec![json!({"text": s})],
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| match b {
                UserContent::Text(t) => json!({"text": t.text}),
                UserContent::Image(img) => {
                    json!({"inlineData": {"mimeType": img.mime_type, "data": img.data}})
                }
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// SSE mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StreamChunk {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    error: Option<GoogleError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    #[serde(default)]
    content: Option<ContentBody>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBody {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Part {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
    #[serde(default)]
    function_call: Option<FunctionCall>,
}

#[derive(Debug, Deserialize)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    args: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    thoughts_token_count: Option<u64>,
    #[serde(default)]
    cached_content_token_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct GoogleError {
    #[serde(default)]
    message: Option<String>,
}

struct Assembler {
    message: AssistantMessage,
    /// function-call parts awaiting a stop-reason decision
    saw_function_call: bool,
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
            saw_function_call: false,
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
            if message.stop_reason == StopReason::Stop && asm.saw_function_call {
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
    let aborted = || cancel.as_ref().is_some_and(|c| c.is_cancelled());

    let is_vertex = model.base_url.contains("aiplatform.googleapis.com");
    let api_key = options
        .api_key
        .clone()
        .or_else(|| {
            std::env::var(if is_vertex {
                "GOOGLE_VERTEX_ACCESS_TOKEN"
            } else {
                "GEMINI_API_KEY"
            })
            .ok()
        })
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .ok_or_else(|| {
            tracing::error!("missing API key for provider {}", model.provider);
            StopReason::Error
        })?;

    let (url, headers) = if is_vertex {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            model.base_url.trim_end_matches('/'),
            model.id
        );
        (
            url,
            vec![("Authorization".into(), format!("Bearer {api_key}"))],
        )
    } else {
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
            model.base_url.trim_end_matches('/').trim_end_matches("/v1beta"),
            model.id
        );
        (url, vec![("x-goog-api-key".into(), api_key)])
    };

    let body = build_body(model, messages, options);
    let mut error_slot = String::new();
    let body_stream = match super::open_sse_post(
        http,
        retry,
        &url,
        headers,
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
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let chunk: StreamChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(data = %data, %err, "failed to parse google SSE chunk");
                continue;
            }
        };

        if let Some(error) = chunk.error {
            asm.message.error_message = error.message;
            return Err(StopReason::Error);
        }

        if let Some(usage) = chunk.usage_metadata {
            let u = &mut asm.message.usage;
            u.input = usage.prompt_token_count;
            u.output = usage.candidates_token_count;
            u.reasoning = usage.thoughts_token_count;
            u.cache_read = usage.cached_content_token_count.unwrap_or(0);
            u.total_tokens = u.input + u.output + u.cache_read;
            calculate_cost(&model.cost, u);
        }

        for candidate in chunk.candidates {
            if let Some(content) = candidate.content {
                for part in content.parts {
                    handle_part(part, tx, asm)?;
                }
            }
            if let Some(raw) = candidate.finish_reason {
                asm.message.raw_stop_reason = Some(raw.clone());
                asm.message.stop_reason = match raw.as_str() {
                    "STOP" => StopReason::Stop,
                    "MAX_TOKENS" => StopReason::Length,
                    other => {
                        asm.message.error_message =
                            Some(format!("finished with reason {other}"));
                        StopReason::Error
                    }
                };
            }
        }
    }
    Ok(())
}

fn handle_part(
    part: Part,
    tx: &EventStreamTx,
    asm: &mut Assembler,
) -> Result<(), StopReason> {
    if let Some(call) = part.function_call {
        let position = asm.message.content.len();
        let args = match call.args {
            Some(Value::Object(map)) => map,
            _ => Default::default(),
        };
        asm.message
            .content
            .push(AssistantContent::ToolCall(ToolCall {
                id: format!("google-call-{}", position),
                name: call.name,
                arguments: args,
                thought_signature: None,
                namespace: None,
            }));
        asm.saw_function_call = true;
        tx.push(AssistantMessageEvent::ToolcallStart {
            content_index: position,
            partial: asm.message.clone(),
        });
        tx.push(AssistantMessageEvent::ToolcallEnd {
            content_index: position,
            tool_call: asm.message.content[position].clone(),
            partial: asm.message.clone(),
        });
        return Ok(());
    }
    let Some(text) = part.text else {
        return Ok(());
    };
    if text.is_empty() {
        return Ok(());
    }
    if part.thought == Some(true) {
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
        if let Some(AssistantContent::Thinking(t)) = asm.message.content.get_mut(position) {
            t.thinking.push_str(&text);
        }
        tx.push(AssistantMessageEvent::ThinkingDelta {
            content_index: position,
            delta: text,
            partial: asm.message.clone(),
        });
    } else {
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
            t.text.push_str(&text);
        }
        tx.push(AssistantMessageEvent::TextDelta {
            content_index: position,
            delta: text,
            partial: asm.message.clone(),
        });
    }
    Ok(())
}

