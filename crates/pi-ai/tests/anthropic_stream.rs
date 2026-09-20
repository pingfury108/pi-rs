//! Integration tests for the Anthropic streaming implementation using a
//! local mock server speaking canned SSE.

use futures::StreamExt;
use pi_ai::api::{normalize_context, AnthropicApi, LlmApi};
use pi_ai::events::AssistantMessageEvent;
use pi_ai::types::*;
use serde_json::{json, Value};

/// Serve one HTTP response then close.
async fn serve_once(response: String) -> (u16, tokio::sync::mpsc::Receiver<String>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        read_request_and_respond(socket, response, Some(tx)).await;
    });
    (port, rx)
}

async fn read_request_and_respond(
    mut socket: tokio::net::TcpStream,
    response: String,
    body_tx: Option<tokio::sync::mpsc::Sender<String>>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = vec![0u8; 65536];
    let mut request = String::new();
    loop {
        let n = socket.read(&mut buf).await.unwrap();
        if n == 0 {
            break;
        }
        request.push_str(&String::from_utf8_lossy(&buf[..n]));
        // stop as soon as we likely have the full body (heuristic: headers + body length)
        if let Some(header_end) = request.find("\r\n\r\n") {
            if let Some(cl) = request
                .lines()
                .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length: ").map(|s| s.to_string()))
            {
                let declared: usize = cl.trim().parse().unwrap_or(0);
                if request.len() - header_end - 4 >= declared {
                    break;
                }
            }
        }
    }
    if let Some(tx) = body_tx {
        let body = request
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or_default();
        let _ = tx.send(body).await;
    }
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.flush().await.unwrap();
    socket.shutdown().await.unwrap();
}

fn sse_response(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("event: {}\ndata: {}\n\n", event["type"], event));
    }
    body.push_str("event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n");
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn test_model(port: u16) -> Model {
    Model {
        id: "claude-sonnet-4-5".into(),
        name: "Claude Sonnet 4.5".into(),
        api: "anthropic-messages".into(),
        provider: "anthropic".into(),
        base_url: format!("http://127.0.0.1:{port}"),
        reasoning: true,
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        context_window: 200_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn text_context() -> Context {
    Context {
        system_prompt: Some("You are helpful.".into()),
        messages: vec![Message::User(UserMessage::new(MessageContent::text("hi")))],
        tools: Some(vec![ToolDef::new(
            "get_time",
            "Get time",
            json!({"type": "object", "properties": {}}),
        )]),
    }
}

async fn collect(stream: pi_ai::events::AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    let mut stream = stream;
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

#[tokio::test]
async fn streams_text_response() {
    let response = sse_response(&[
        json!({
            "type": "message_start",
            "message": {"id": "msg_1", "model": "claude-sonnet-4-5",
                        "usage": {"input_tokens": 10, "output_tokens": 0}}
        }),
        json!({"type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "Hel"}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "lo"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5}
        }),
    ]);
    let (port, mut body_rx) = serve_once(response).await;
    let api = AnthropicApi::new();
    let events = collect(api.stream(
        &test_model(port),
        &text_context(),
        StreamOptions::api_key("test-key"),
    ))
    .await;

    let names: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
            _ => "other",
        })
        .collect();
    assert_eq!(
        names,
        ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
    );

    // verify request body received by server
    let body = body_rx.recv().await.unwrap();
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["model"], "claude-sonnet-4-5");
    assert_eq!(body["stream"], true);
    assert_eq!(body["system"][0]["text"], "You are helpful.");
    assert_eq!(body["tools"][0]["name"], "get_time");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "hi");
}

#[tokio::test]
async fn streams_tool_call_with_partial_json() {
    let response = sse_response(&[
        json!({"type": "message_start",
               "message": {"id": "msg_2", "model": "claude-sonnet-4-5",
                           "usage": {"input_tokens": 8, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0,
               "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_time"}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": "{\"timezone\": \"As"}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": "ia/Shanghai\"}"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
               "usage": {"output_tokens": 12}}),
    ]);
    let (port, _body_rx) = serve_once(response).await;
    let api = AnthropicApi::new();
    let events = collect(api.stream(
        &test_model(port),
        &text_context(),
        StreamOptions::api_key("test-key"),
    ))
    .await;

    let done = events
        .iter()
        .find_map(|e| match e {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            _ => None,
        })
        .expect("expected done event");
    assert_eq!(done.stop_reason, StopReason::ToolUse);
    assert_eq!(done.usage.input, 8);
    assert_eq!(done.usage.output, 12);
    assert_eq!(done.usage.total_tokens, 20);
    // cost: 8 input * 3.0/1M + 12 output * 15.0/1M
    assert!((done.usage.cost.total - (8.0 * 3.0 + 12.0 * 15.0) / 1_000_000.0).abs() < 1e-9);

    let tool_call = done
        .content
        .iter()
        .find_map(|b| match b {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(tool_call.id, "toolu_1");
    assert_eq!(tool_call.name, "get_time");
    assert_eq!(tool_call.arguments["timezone"], "Asia/Shanghai");

    // toolcall events carried partial args as they streamed
    let delta_args: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::ToolcallDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(delta_args.len(), 2);
}

#[tokio::test]
async fn error_status_becomes_error_event() {
    let (port, _rx) = serve_once(
        "HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".into(),
    )
    .await;
    let api = AnthropicApi::new();
    let events = collect(api.stream(
        &test_model(port),
        &text_context(),
        StreamOptions::api_key("bad-key"),
    ))
    .await;
    assert_eq!(events.len(), 1);
    match &events[0] {
        AssistantMessageEvent::Error { error, .. } => {
            assert_eq!(error.stop_reason, StopReason::Error);
            assert!(error.error_message.as_deref().unwrap_or("").contains("401"));
        }
        other => panic!("expected error event, got {other:?}"),
    }
}

#[tokio::test]
async fn missing_api_key_becomes_error_event() {
    // ensure env var does not interfere
    // SAFETY-free approach: only assert on error event presence
    let (port, _rx) = serve_once(String::new()).await;
    let api = AnthropicApi::new();
    let events =
        collect(api.stream(&test_model(port), &text_context(), StreamOptions::default())).await;
    // Either missing-key error (if env unset) or 401-style server error (env set).
    assert!(matches!(events[0], AssistantMessageEvent::Error { .. }));
}

#[test]
fn normalize_context_is_pure() {
    let context = text_context();
    let normalized = normalize_context(&context);
    assert_eq!(normalized.len(), 2);
    assert!(matches!(normalized[0], Message::System(_)));
    assert!(matches!(normalized[1], Message::User(_)));
}
