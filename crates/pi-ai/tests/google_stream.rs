//! Integration test for the Google Generative Language streaming
//! implementation using a local mock server.

use futures::StreamExt;
use pi_ai::api::{normalize_context, GoogleApi, LlmApi};
use pi_ai::events::AssistantMessageEvent;
use pi_ai::types::*;
use serde_json::{json, Value};

fn sse_response(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {}\n\n", event));
    }
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn test_model(port: u16) -> Model {
    Model {
        id: "gemini-2.5-pro".into(),
        name: "Gemini".into(),
        api: "google-generative-ai".into(),
        provider: "google".into(),
        base_url: format!("http://127.0.0.1:{port}"),
        reasoning: true,
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 1.25,
            output: 10.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 1_000_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn text_context() -> Context {
    Context {
        system_prompt: Some("be brief".into()),
        messages: vec![Message::User(UserMessage::new(MessageContent::text("hi")))],
        tools: Some(vec![ToolDef::new("get_time", "Get time", json!({}))]),
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
async fn streams_text_thinking_and_tool_call() {
    // mock server: capture request, respond with SSE
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let response = sse_response(&[
        json!({
            "candidates": [{"content": {"parts": [
                {"text": "thinking hard", "thought": true},
                {"text": "Hello!"}
            ], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 9, "totalTokenCount": 16}
        }),
        json!({
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "get_time", "args": {"tz": "UTC"}}}
            ], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 12, "totalTokenCount": 19}
        }),
    ]);
    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(1);
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut buf = vec![0u8; 65536];
        let mut request = String::new();
        loop {
            let n = socket.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            request.push_str(&String::from_utf8_lossy(&buf[..n]));
            if let Some(header_end) = request.find("\r\n\r\n") {
                if let Some(cl) = request.lines().find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|s| s.trim().to_string())
                }) {
                    if request.len() - header_end - 4 >= cl.parse().unwrap_or(0) {
                        break;
                    }
                }
            }
        }
        let _ = request_tx.send(request).await;
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });

    let api = GoogleApi::new();
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        reasoning: Some(ThinkingLevel::High),
        ..Default::default()
    };
    let events = collect(api.stream(&test_model(port), &text_context(), options)).await;

    // verify final message
    let done = events
        .iter()
        .find_map(|e| match e {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            _ => None,
        })
        .expect("done event");
    assert_eq!(done.stop_reason, StopReason::ToolUse);
    assert_eq!(done.usage.input, 7);
    assert_eq!(done.usage.output, 12);

    // thinking + text + toolCall blocks
    assert!(done.content.iter().any(|b| matches!(
        b,
        AssistantContent::Thinking(t) if t.thinking == "thinking hard"
    )));
    assert!(done
        .content
        .iter()
        .any(|b| matches!(b, AssistantContent::Text(t) if t.text == "Hello!")));
    let call = done
        .content
        .iter()
        .find_map(|b| match b {
            AssistantContent::ToolCall(c) => Some(c.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(call.name, "get_time");
    assert_eq!(call.arguments["tz"], "UTC");

    // request assertions: URL path + api key header + body shape
    let request = request_rx.recv().await.unwrap();
    assert!(request.contains("POST /v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"), "{request}");
    assert!(request.contains("x-goog-api-key: test-key"), "{request}");
    let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body["systemInstruction"]["parts"][0]["text"],
        "be brief"
    );
    assert_eq!(body["tools"][0]["functionDeclarations"][0]["name"], "get_time");
    assert_eq!(body["contents"][0]["role"], "user");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["includeThoughts"],
        json!(true)
    );

    let _ = normalize_context(&text_context());
}
