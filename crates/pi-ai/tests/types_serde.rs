//! Verifies that serialized JSON shapes match pi's session format.

use pi_ai::*;
use serde_json::json;

#[test]
fn user_message_serializes_like_pi() {
    let msg = Message::User(UserMessage {
        content: MessageContent::text("hello"),
        timestamp: 1700000000000,
    });
    let v = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        v,
        json!({"role": "user", "content": "hello", "timestamp": 1700000000000_i64})
    );
}

#[test]
fn assistant_message_serializes_like_pi() {
    let msg = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "hmm".into(),
                thinking_signature: Some("sig".into()),
                redacted: false,
            }),
            AssistantContent::Text(TextContent {
                text: "hi".into(),
                text_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: serde_json::from_value(json!({"path": "a.rs"})).unwrap(),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: "anthropic-messages".into(),
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5".into(),
        response_model: None,
        response_id: Some("msg_01".into()),
        provider_thinking_level: None,
        diagnostics: vec![],
        usage: Usage {
            input: 10,
            output: 5,
            total_tokens: 15,
            ..Default::default()
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1700000000000,
    };
    let v = serde_json::to_value(&Message::Assistant(msg.clone())).unwrap();
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"][0]["type"], "thinking");
    assert_eq!(v["content"][0]["thinkingSignature"], "sig");
    assert_eq!(v["content"][1]["type"], "text");
    assert_eq!(v["content"][2]["type"], "toolCall");
    assert_eq!(v["content"][2]["arguments"]["path"], "a.rs");
    assert_eq!(v["stopReason"], "toolUse");
    assert_eq!(v["usage"]["cacheRead"], 0);
    assert_eq!(v["usage"]["totalTokens"], 15);
    // omitted optional fields must be absent
    assert!(v.get("errorMessage").is_none());
    assert!(v.get("diagnostics").is_none());

    // round-trip
    let back: AssistantMessage = serde_json::from_value(v).unwrap();
    assert_eq!(back, msg);
}

#[test]
fn tool_result_message_serializes_like_pi() {
    let msg = Message::ToolResult(ToolResultMessage {
        tool_call_id: "call_1".into(),
        tool_name: "read".into(),
        content: vec![ToolResultContent::Text(TextContent {
            text: "file content".into(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        is_error: false,
        timestamp: 1700000000000,
    });
    let v = serde_json::to_value(&msg).unwrap();
    assert_eq!(v["role"], "toolResult");
    assert_eq!(v["toolCallId"], "call_1");
    assert_eq!(v["toolName"], "read");
    assert_eq!(v["isError"], false);
    assert_eq!(v["content"][0]["type"], "text");
}

#[test]
fn user_content_blocks_roundtrip() {
    let msg = Message::User(UserMessage {
        content: MessageContent::Blocks(vec![
            UserContent::Text(TextContent {
                text: "look at this".into(),
                text_signature: None,
            }),
            UserContent::Image(ImageContent {
                data: "aGVsbG8=".into(),
                mime_type: "image/png".into(),
            }),
        ]),
        timestamp: 1700000000000,
    });
    let v = serde_json::to_value(&msg).unwrap();
    assert!(v["content"].is_array());
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][1]["type"], "image");
    assert_eq!(v["content"][1]["mimeType"], "image/png");

    let back: Message = serde_json::from_value(v).unwrap();
    assert_eq!(back, msg);
}

#[test]
fn stop_reason_naming() {
    assert_eq!(
        serde_json::to_value(StopReason::ToolUse).unwrap(),
        json!("toolUse")
    );
    assert_eq!(
        serde_json::to_value(StopReason::Aborted).unwrap(),
        json!("aborted")
    );
}

#[test]
fn event_stream_terminal_tracking() {
    let (tx, stream) = AssistantMessageEventStream::channel(16);
    let partial = AssistantMessage {
        content: vec![],
        api: "test".into(),
        provider: "faux".into(),
        model: "faux-1".into(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: vec![],
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    };
    let mut done = partial.clone();
    done.stop_reason = StopReason::Stop;
    tx.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    tx.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: done.clone(),
    });
    drop(tx);

    let final_msg = futures::executor::block_on(stream.result());
    assert_eq!(final_msg, done);
}
