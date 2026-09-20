//! Streaming event protocol, ported from `packages/ai/src/types.ts` +
//! `packages/ai/src/utils/event-stream.ts`.
//!
//! Contract (mirrors pi):
//! - Successful streams emit `Start` first and terminate with `Done`.
//! - Failures after `Start` terminate with `Error`; request setup failures may
//!   emit only `Error`.
//! - `Done`/`Error` carry the final `AssistantMessage` (stopReason
//!   `error`/`aborted` + `errorMessage` on failures).

use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::types::{AssistantContent, AssistantMessage, StopReason};

/// Streaming events emitted while an assistant message is generated.
///
/// `partial` carries a snapshot of the response-so-far at event time (pi shares
/// a live object; we clone, which is semantically equivalent for consumers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolcallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolcallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolcallEnd {
        content_index: usize,
        tool_call: AssistantContent,
        partial: AssistantMessage,
    },
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Terminal events carry the final message.
    pub fn final_message(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Done { message, .. } | Self::Error { error: message, .. } => Some(message),
            _ => None,
        }
    }
}

/// Boxed, sendable stream of assistant message events.
pub type BoxedEventStream = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send>>;

/// Stream of assistant message events that tracks the final message.
///
/// Producer side: [`AssistantMessageEventStream::channel`] or build from any
/// stream with [`AssistantMessageEventStream::new`].
pub struct AssistantMessageEventStream {
    inner: BoxedEventStream,
    result: Option<AssistantMessage>,
}

impl AssistantMessageEventStream {
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = AssistantMessageEvent> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
            result: None,
        }
    }

    /// Create a channel-backed stream plus a sender for the producer.
    pub fn channel(buffer: usize) -> (EventStreamTx, Self) {
        let (tx, rx) = mpsc::channel(buffer);
        (EventStreamTx { tx }, Self::new(receiver_stream(rx)))
    }

    /// The final assistant message, once a terminal event has been observed.
    pub fn take_result(&mut self) -> Option<AssistantMessage> {
        self.result.take()
    }

    /// Poll the stream until the terminal event, then return the final message.
    pub async fn result(mut self) -> AssistantMessage {
        use futures::StreamExt;
        while let Some(event) = self.inner.next().await {
            if let Some(message) = event.final_message() {
                self.result = Some(message.clone());
            }
        }
        self.result.expect("stream ended without a terminal event")
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(event)) => {
                if let Some(message) = event.final_message() {
                    self.result = Some(message.clone());
                }
                Poll::Ready(Some(event))
            }
            other => other,
        }
    }
}

fn receiver_stream(
    mut rx: mpsc::Receiver<AssistantMessageEvent>,
) -> impl Stream<Item = AssistantMessageEvent> {
    async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield event;
        }
    }
}

/// Producer handle for a channel-backed [`AssistantMessageEventStream`].
#[derive(Clone)]
pub struct EventStreamTx {
    tx: mpsc::Sender<AssistantMessageEvent>,
}

impl EventStreamTx {
    /// Push an event; no-op after the stream is closed.
    pub fn push(&self, event: AssistantMessageEvent) {
        // Use try_send first to stay sync; fall back to blocking send from
        // async contexts via try_send only (buffer should be sized generously).
        let _ = self.tx.try_send(event);
    }

    /// Push an event, waiting for buffer capacity (async context).
    pub async fn send(&self, event: AssistantMessageEvent) {
        let _ = self.tx.send(event).await;
    }

    /// Close the stream by dropping the sender; consumers see the stream end
    /// after buffered events. Idiomatic alternative to pi's `end()`.
    pub fn close(self) {
        drop(self.tx);
    }
}
