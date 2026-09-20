//! Faux provider for tests, port of `packages/ai/src/providers/faux.ts`.
//!
//! Replays scripted assistant responses as streaming events without any
//! network I/O. Used by agent-loop tests (Phase 3).

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::LlmApi;
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::types::{
    AssistantContent, AssistantMessage, Context, Model, StopReason, StreamOptions, TextContent,
    ThinkingContent, ToolCall, Usage,
};

/// One scripted response.
#[derive(Debug, Clone)]
pub enum FauxResponse {
    /// A successful assistant message (streamed block by block).
    Message {
        content: Vec<AssistantContent>,
        usage: Option<Usage>,
        stop_reason: StopReason,
    },
    /// A failed request.
    Error { message: String },
}

impl FauxResponse {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Message {
            content: vec![AssistantContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            usage: None,
            stop_reason: StopReason::Stop,
        }
    }

    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self::Message {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: arguments.as_object().cloned().unwrap_or_default(),
                thought_signature: None,
                namespace: None,
            })],
            usage: None,
            stop_reason: StopReason::ToolUse,
        }
    }
}

/// Scripted provider: pops one `FauxResponse` per `stream()` call.
#[derive(Default)]
pub struct FauxApi {
    script: Arc<Mutex<Vec<FauxResponse>>>,
}

impl FauxApi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_script(responses: Vec<FauxResponse>) -> Self {
        Self {
            script: Arc::new(Mutex::new(responses)),
        }
    }

    /// Append a response to the script.
    pub async fn push(&self, response: FauxResponse) {
        self.script.lock().await.push(response);
    }
}

#[async_trait]
impl LlmApi for FauxApi {
    fn name(&self) -> &str {
        "faux"
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let model = model.clone();
        let script = self.script.clone();

        let (tx, stream) = AssistantMessageEventStream::channel(64);
        tokio::spawn(async move {
            let next = {
                let mut script = script.lock().await;
                if script.is_empty() {
                    None
                } else {
                    Some(script.remove(0))
                }
            };
            match next {
                Some(FauxResponse::Error { message }) => {
                    tx.push(error_event(&model, &message));
                }
                Some(FauxResponse::Message {
                    content,
                    usage,
                    stop_reason,
                }) => {
                    replay(&tx, &model, content, usage, stop_reason);
                }
                None => {
                    tx.push(error_event(&model, "faux: script exhausted"));
                }
            }
            tx.close();
        });
        stream
    }
}

fn base_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: Some("faux-msg".into()),
        provider_thinking_level: None,
        diagnostics: vec![],
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: crate::types::now_millis(),
    }
}

fn error_event(model: &Model, msg: &str) -> AssistantMessageEvent {
    let mut message = base_message(model);
    message.stop_reason = StopReason::Error;
    message.error_message = Some(msg.to_string());
    AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message,
    }
}

fn replay(
    tx: &EventStreamTx,
    model: &Model,
    content: Vec<AssistantContent>,
    usage: Option<Usage>,
    stop_reason: StopReason,
) {
    let mut message = base_message(model);
    tx.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });

    for (index, block) in content.into_iter().enumerate() {
        match block {
            AssistantContent::Text(t) => {
                tx.push(AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: message.clone(),
                });
                // stream in two chunks to exercise delta handling
                let (head, tail) = split_at_half(&t.text);
                if !head.is_empty() {
                    message.content.push(AssistantContent::Text(TextContent {
                        text: head.to_string(),
                        text_signature: None,
                    }));
                }
                tx.push(AssistantMessageEvent::TextDelta {
                    content_index: index,
                    delta: head.to_string(),
                    partial: message.clone(),
                });
                if !tail.is_empty() {
                    if let Some(AssistantContent::Text(last)) = message.content.last_mut() {
                        last.text.push_str(tail);
                    } else {
                        message
                            .content
                            .push(AssistantContent::Text(TextContent {
                                text: tail.to_string(),
                                text_signature: None,
                            }));
                    }
                }
                tx.push(AssistantMessageEvent::TextDelta {
                    content_index: index,
                    delta: tail.to_string(),
                    partial: message.clone(),
                });
                tx.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content: t.text.clone(),
                    partial: message.clone(),
                });
                // ensure the final block matches exactly
                if let Some(AssistantContent::Text(last)) = message.content.last_mut() {
                    last.text = t.text;
                }
            }
            AssistantContent::Thinking(t) => {
                tx.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: message.clone(),
                });
                tx.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: index,
                    delta: t.thinking.clone(),
                    partial: message.clone(),
                });
                message
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: t.thinking.clone(),
                        thinking_signature: t.thinking_signature.clone(),
                        redacted: t.redacted,
                    }));
                tx.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content: t.thinking,
                    partial: message.clone(),
                });
            }
            AssistantContent::ToolCall(call) => {
                tx.push(AssistantMessageEvent::ToolcallStart {
                    content_index: index,
                    partial: message.clone(),
                });
                // stream arguments as a single delta via partial JSON
                let args_json = serde_json::to_string(&call.arguments).unwrap_or_default();
                tx.push(AssistantMessageEvent::ToolcallDelta {
                    content_index: index,
                    delta: args_json,
                    partial: message.clone(),
                });
                message
                    .content
                    .push(AssistantContent::ToolCall(ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                        thought_signature: None,
                        namespace: None,
                    }));
                tx.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: index,
                    tool_call: AssistantContent::ToolCall(call),
                    partial: message.clone(),
                });
            }
        }
    }

    message.stop_reason = stop_reason;
    if let Some(usage) = usage {
        message.usage = usage;
    }
    tx.push(AssistantMessageEvent::Done {
        reason: message.stop_reason,
        message,
    });
}

fn split_at_half(s: &str) -> (&str, &str) {
    match s.char_indices().nth(s.chars().count() / 2) {
        Some((idx, _)) => s.split_at(idx),
        None => (s, ""),
    }
}
