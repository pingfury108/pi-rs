//! Amazon Bedrock ConverseStream (`bedrock-converse-stream`).
//!
//! Hand-rolled AWS SigV4 signing and `vnd.amazon.eventstream` binary frame
//! decoding (no AWS SDK dependency). Credentials resolve from options,
//! then `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`
//! (+ optional `AWS_SESSION_TOKEN`).

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{current_tools, initial_system_text, normalize_context, LlmApi, RetryPolicy};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::partial_json::parse_partial_json;
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, StreamOptions,
    TextContent, ThinkingContent, ToolCall, Usage,
};

pub struct BedrockApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl BedrockApi {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            retry: RetryPolicy::default(),
        }
    }
}

impl Default for BedrockApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for BedrockApi {
    fn name(&self) -> &str {
        "bedrock-converse-stream"
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

struct AwsCredentials {
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    region: String,
}

fn resolve_credentials(model: &Model, options: &StreamOptions) -> Result<AwsCredentials, StopReason> {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let access_key = options
        .api_key
        .clone()
        .or_else(|| get("AWS_ACCESS_KEY_ID"))
        .ok_or_else(|| {
            tracing::error!("missing AWS credentials for bedrock");
            StopReason::Error
        })?;
    let secret_key = get("AWS_SECRET_ACCESS_KEY").ok_or_else(|| {
        tracing::error!("missing AWS_SECRET_ACCESS_KEY for bedrock");
        StopReason::Error
    })?;
    let region = model
        .base_url
        .split('.')
        .nth(1)
        .unwrap_or("us-east-1")
        .to_string();
    Ok(AwsCredentials {
        access_key,
        secret_key,
        session_token: get("AWS_SESSION_TOKEN"),
        region,
    })
}

// ---------------------------------------------------------------------------
// SigV4
// ---------------------------------------------------------------------------

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    // HMAC-SHA256 (RFC 2104)
    let block_size = 64;
    let mut key = key.to_vec();
    if key.len() > block_size {
        key = Sha256::digest(&key).to_vec();
    }
    key.resize(block_size, 0);
    let ipad: Vec<u8> = key.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = key.iter().map(|b| b ^ 0x5c).collect();
    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(data);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(inner);
    outer.finalize().to_vec()
}

fn sigv4_headers(
    creds: &AwsCredentials,
    method: &str,
    host: &str,
    path: &str,
    body_hash: &str,
    payload: &[u8],
) -> Vec<(String, String)> {
    let now = chrono::Utc::now();
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date_stamp = now.format("%Y%m%d").to_string();
    let service = "bedrock";
    let content_type = "application/json";

    let mut canonical_headers = format!(
        "content-type:{content_type}\nhost:{host}\nx-amz-date:{amz_date}\n"
    );
    let mut signed_headers = "content-type;host;x-amz-date".to_string();
    if creds.session_token.is_some() {
        canonical_headers = format!(
            "content-type:{content_type}\nhost:{host}\nx-amz-date:{amz_date}\nx-amz-security-token:{}\n",
            creds.session_token.as_deref().unwrap_or("")
        );
        signed_headers = "content-type;host;x-amz-date;x-amz-security-token".to_string();
    }

    let canonical_request = format!(
        "{method}\n{path}\n\n{canonical_headers}\n{signed_headers}\n{body_hash}"
    );
    let scope = format!("{date_stamp}/{region}/{service}/aws4_request", region = creds.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex(Sha256::digest(canonical_request.as_bytes()).as_slice())
    );

    let k_date = hmac_sha256(format!("AWS4{}", creds.secret_key).as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, creds.region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex(&hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        creds.access_key, scope, signed_headers, signature
    );

    let mut headers = vec![
        ("Content-Type".into(), content_type.into()),
        ("x-amz-date".into(), amz_date),
        ("Authorization".into(), authorization),
    ];
    if let Some(token) = &creds.session_token {
        headers.push(("x-amz-security-token".into(), token.clone()));
    }
    let _ = payload;
    headers
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(data: &[u8]) -> String {
    hex(Sha256::digest(data).as_slice())
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

fn build_body(model: &Model, messages: &[Message], options: &StreamOptions) -> Value {
    let system = initial_system_text(messages);
    let tools = current_tools(messages);

    let mut converse: Vec<Value> = Vec::new();
    for msg in messages.iter().skip(1) {
        match msg {
            Message::System(_) => {}
            Message::User(user) => {
                let mut parts = vec![json!({"text": user.content.as_text()})];
                if let crate::types::MessageContent::Blocks(blocks) = &user.content {
                    for block in blocks {
                        if let crate::types::UserContent::Image(img) = block {
                            parts.push(json!({
                                "image": {
                                    "format": img.mime_type.trim_start_matches("image/"),
                                    "source": {"bytes": img.data},
                                }
                            }));
                        }
                    }
                }
                converse.push(json!({"role": "user", "content": parts}));
            }
            Message::Assistant(assistant) => {
                let mut parts: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(t) => {
                            if !t.text.is_empty() {
                                parts.push(json!({"text": t.text}));
                            }
                        }
                        AssistantContent::Thinking(_) => {}
                        AssistantContent::ToolCall(call) => parts.push(json!({
                            "toolUse": {
                                "toolUseId": call.id,
                                "name": call.name,
                                "input": call.arguments,
                            }
                        })),
                    }
                }
                if !parts.is_empty() {
                    converse.push(json!({"role": "assistant", "content": parts}));
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
                converse.push(json!({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": result.tool_call_id,
                            "content": [{"json": {"result": text}}],
                            "status": if result.is_error { "error" } else { "success" },
                        }
                    }],
                }));
            }
        }
    }

    let mut body = json!({
        "messages": converse,
        // streaming hint; ConverseStream accepts inferenceConfig
        "inferenceConfig": {
            "maxTokens": options.max_tokens.map(|t| t as i64).unwrap_or(model.max_tokens as i64),
        },
    });
    if !system.is_empty() {
        body["system"] = json!([{"text": system}]);
    }
    if !tools.is_empty() {
        let specs: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.parameters},
                    }
                })
            })
            .collect();
        body["toolConfig"] = json!({"tools": specs});
    }
    body
}

// ---------------------------------------------------------------------------
// AWS event-stream decoding
// ---------------------------------------------------------------------------

/// Decode one `vnd.amazon.eventstream` frame: [total_len: u32][headers_len: u32]
/// [prelude_crc: u32][headers][payload][message_crc: u32].
fn decode_frames(buf: &mut Vec<u8>) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        if buf.len() < 12 {
            break;
        }
        let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        if buf.len() < total_len {
            break;
        }
        let frame = buf.drain(..total_len).collect::<Vec<u8>>();
        let payload = &frame[12 + headers_len..total_len - 4];
        let message_type = header_value(&frame[12..12 + headers_len], ":message-type");
        let event_type = header_value(&frame[12..12 + headers_len], ":event-type");
        match message_type.as_deref() {
            Some("event") => {
                if let Some(json_payload) = std::str::from_utf8(payload)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                {
                    let _ = event_type;
                    events.push(json_payload);
                }
            }
            Some("exception") => {
                if let Ok(json_payload) = std::str::from_utf8(payload) {
                    if let Ok(v) = serde_json::from_str::<Value>(json_payload) {
                        events.push(json!({"exception": event_type.clone().unwrap_or_default(), "body": v}));
                    }
                }
            }
            _ => {}
        }
    }
    events
}

/// Minimal event-stream header reader: find a string header by name.
fn header_value(headers: &[u8], name: &str) -> Option<String> {
    let mut i = 0usize;
    while i + 4 <= headers.len() {
        let name_len = u16::from_be_bytes([headers[i], headers[i + 1]]) as usize;
        i += 2;
        if i + name_len > headers.len() {
            return None;
        }
        let header_name = std::str::from_utf8(&headers[i..i + name_len]).ok()?;
        i += name_len;
        if i + 3 > headers.len() {
            return None;
        }
        let value_type = headers[i];
        let value_len_bytes = u16::from_be_bytes([headers[i + 1], headers[i + 2]]) as usize;
        i += 3;
        if i + value_len_bytes > headers.len() {
            return None;
        }
        let value = &headers[i..i + value_len_bytes];
        i += value_len_bytes;
        if header_name == name && value_type == 7 {
            // 7 = string header
            return std::str::from_utf8(value).ok().map(String::from);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Event mapping
// ---------------------------------------------------------------------------

struct Assembler {
    message: AssistantMessage,
    tool_positions: std::collections::HashMap<String, usize>,
    partial_json: std::collections::HashMap<String, String>,
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
    let creds = resolve_credentials(model, options)?;

    // model.base_url format: https://bedrock-runtime.{region}.amazonaws.com
    let host = model
        .base_url
        .trim_start_matches("https://")
        .trim_end_matches('/')
        .to_string();
    let path = format!(
        "/model/{}:converse-stream",
        urlencode(model.id.as_bytes())
    );
    let url = format!("https://{host}{path}");

    let body = build_body(model, messages, options);
    let payload = serde_json::to_vec(&body).map_err(|e| {
        asm.message.error_message = Some(e.to_string());
        StopReason::Error
    })?;
    let body_hash = sha256_hex(&payload);
    let headers = sigv4_headers(&creds, "POST", &host, &path, &body_hash, &payload);

    let mut error_slot = String::new();
    let mut body_stream = match super::open_sse_post(http, retry, &url, headers, &body, &cancel, &mut error_slot).await {
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

    // AWS event-stream is binary; buffer bytes and decode frames.
    let mut buffer: Vec<u8> = Vec::new();
    while let Some(chunk) = body_stream.next().await {
        if cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err(StopReason::Aborted);
        }
        let bytes = match chunk {
            Ok(b) => b,
            Err(err) => {
                asm.message.error_message = Some(format!("stream error: {err}"));
                return Err(StopReason::Error);
            }
        };
        buffer.extend_from_slice(&bytes);
        for event in decode_frames(&mut buffer) {
            handle_event(event, tx, asm, model)?;
        }
    }
    Ok(())
}

fn urlencode(bytes: &[u8]) -> String {
    let mut out = String::new();
    for b in bytes {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(*b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[allow(clippy::too_many_lines)]
fn handle_event(
    event: Value,
    tx: &EventStreamTx,
    asm: &mut Assembler,
    model: &Model,
) -> Result<(), StopReason> {
    // exception frames
    if let Some(name) = event.get("exception").and_then(Value::as_str) {
        asm.message.error_message =
            Some(format!("bedrock exception: {name}"));
        return Err(StopReason::Error);
    }

    let kind = event
        .as_object()
        .and_then(|o| o.keys().next().cloned())
        .unwrap_or_default();
    match kind.as_str() {
        "messageStart" => {
            // role info; nothing to do
        }
        "contentBlockStart" => {
            let start = &event["contentBlockStart"];
            let index = start
                .get("contentBlockIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if let Some(tool_use) = start
                .get("start")
                .and_then(|s| s.get("toolUse"))
            {
                let id = tool_use
                    .get("toolUseId")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let name = tool_use
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let position = asm.message.content.len();
                asm.message
                    .content
                    .push(AssistantContent::ToolCall(ToolCall {
                        id: id.into(),
                        name: name.into(),
                        arguments: Default::default(),
                        thought_signature: None,
                        namespace: None,
                    }));
                asm.partial_json.insert(id.into(), String::new());
                asm.tool_positions.insert(id.into(), position);
                tx.push(AssistantMessageEvent::ToolcallStart {
                    content_index: position,
                    partial: asm.message.clone(),
                });
            }
            let _ = index;
        }
        "contentBlockDelta" => {
            let delta = &event["contentBlockDelta"]["delta"];
            let index = event["contentBlockDelta"]
                .get("contentBlockIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if let Some(text) = delta.get("text").and_then(Value::as_str) {
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
                    delta: text.into(),
                    partial: asm.message.clone(),
                });
            } else if let Some(thinking) = delta
                .get("reasoningContent")
                .and_then(|r| r.get("text"))
                .and_then(Value::as_str)
            {
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
                    t.thinking.push_str(thinking);
                }
                tx.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: position,
                    delta: thinking.into(),
                    partial: asm.message.clone(),
                });
            } else if let Some(input) = delta
                .get("toolUse")
                .and_then(|t| t.get("input"))
                .and_then(Value::as_str)
            {
                if let Some(tool_use_id) = event["contentBlockDelta"]["delta"]["toolUse"]
                    .get("toolUseId")
                    .and_then(Value::as_str)
                {
                    let position =
                        asm.tool_positions.get(tool_use_id).copied().unwrap_or(0);
                    let entry = asm.partial_json.entry(tool_use_id.into()).or_default();
                    entry.push_str(input);
                    if let Some(AssistantContent::ToolCall(call)) =
                        asm.message.content.get_mut(position)
                    {
                        if let Value::Object(map) = parse_partial_json(entry) {
                            call.arguments = map;
                        }
                    }
                    tx.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: position,
                        delta: input.into(),
                        partial: asm.message.clone(),
                    });
                }
                let _ = index;
            }
        }
        "contentBlockStop" => {
            let stop = &event["contentBlockStop"];
            let index = stop
                .get("contentBlockIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            // close matching tool call if any (search by block index mapping)
            let tool_id = asm
                .partial_json
                .keys()
                .find(|_| true)
                .cloned();
            if let Some(tool_id) = tool_id {
                if let Some(position) = asm.tool_positions.get(&tool_id).copied() {
                    if let Some(json) = asm.partial_json.get(&tool_id) {
                        if let Some(AssistantContent::ToolCall(call)) =
                            asm.message.content.get(position).cloned()
                        {
                            if let Some(AssistantContent::ToolCall(target)) =
                                asm.message.content.get_mut(position)
                            {
                                if let Value::Object(map) = parse_partial_json(json) {
                                    target.arguments = map;
                                }
                            }
                            asm.partial_json.remove(&tool_id);
                            tx.push(AssistantMessageEvent::ToolcallEnd {
                                content_index: position,
                                tool_call: AssistantContent::ToolCall(call),
                                partial: asm.message.clone(),
                            });
                        }
                    }
                }
            }
            let _ = index;
        }
        "messageStop" => {
            let stop = &event["messageStop"];
            if let Some(reason) = stop.get("stopReason").and_then(Value::as_str) {
                asm.message.raw_stop_reason = Some(reason.into());
                asm.message.stop_reason = match reason {
                    "end_turn" | "stop_sequence" => StopReason::Stop,
                    "max_tokens" => StopReason::Length,
                    "tool_use" => StopReason::ToolUse,
                    _ => StopReason::Stop,
                };
            }
        }
        "metadata" => {
            if let Some(usage) = event["metadata"].get("usage") {
                let u = &mut asm.message.usage;
                u.input = usage
                    .get("inputTokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                u.output = usage
                    .get("outputTokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                u.cache_read = usage
                    .get("cacheReadInputTokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                u.cache_write = usage
                    .get("cacheWriteInputTokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                u.total_tokens = u.input + u.output + u.cache_read + u.cache_write;
                super::calculate_cost(&model.cost, u);
            }
        }
        _ => {}
    }
    Ok(())
}

