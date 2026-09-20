//! pi-messages API: pi's own gateway protocol. Request is a single POST of
//! `{ model, context, options }` to `<baseUrl>/messages`; the response is an
//! SSE stream of serialized assistant-message events plus a terminal
//! `done`/`error` event.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{normalize_context, LlmApi, RetryPolicy};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, EventStreamTx};
use crate::types::{AssistantMessage, Context, Message, Model, StopReason, StreamOptions};

pub struct PiMessagesApi {
    http: reqwest::Client,
    retry: RetryPolicy,
}

impl PiMessagesApi {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            retry: RetryPolicy::default(),
        }
    }
}

impl Default for PiMessagesApi {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmApi for PiMessagesApi {
    fn name(&self) -> &str {
        "pi-messages"
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

async fn run(
    http: &reqwest::Client,
    retry: &RetryPolicy,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    tx: &EventStreamTx,
) {
    let cancel = options.cancel.clone();
    let api_key = options.api_key.clone().or_else(|| {
        std::env::var("PI_MESSAGES_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
    });

    let url = format!("{}/messages", model.base_url.trim_end_matches('/'));
    let mut headers = vec![
        ("content-type".to_string(), "application/json".to_string()),
    ];
    if let Some(key) = &api_key {
        headers.push(("Authorization".into(), format!("Bearer {key}")));
    }

    let body = json!({
        "model": {
            "id": model.id,
            "api": model.api,
            "provider": model.provider,
            "baseUrl": model.base_url,
            "reasoning": model.reasoning,
            "contextWindow": model.context_window,
            "maxTokens": model.max_tokens,
        },
        "context": {"messages": messages},
        "options": {
            "apiKey": api_key,
            "maxTokens": options.max_tokens,
            "temperature": options.temperature,
            "reasoning": options.reasoning,
        },
    });

    let mut error_slot = String::new();
    let body_stream = match super::open_sse_post(http, retry, &url, headers, &body, &cancel, &mut error_slot).await {
        Ok(stream) => stream,
        Err(reason) => {
            let mut message = AssistantMessage {
                content: vec![],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: vec![],
                usage: Default::default(),
                stop_reason: reason,
                error_message: if error_slot.is_empty() {
                    None
                } else {
                    Some(error_slot)
                },
                raw_stop_reason: None,
                end_turn: None,
                timestamp: crate::types::now_millis(),
            };
            if message.error_message.is_none() && reason == StopReason::Error {
                message.error_message = Some("request failed".into());
            }
            tx.push(AssistantMessageEvent::Error {
                reason,
                error: message,
            });
            return;
        }
    };

    let mut sse = body_stream.eventsource();
    while let Some(sse) = sse.next().await {
        if cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            let mut message = AssistantMessage {
                content: vec![],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: vec![],
                usage: Default::default(),
                stop_reason: StopReason::Aborted,
                error_message: Some("Request was aborted".into()),
                raw_stop_reason: None,
                end_turn: None,
                timestamp: crate::types::now_millis(),
            };
            let _ = &mut message;
            tx.push(AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                error: message,
            });
            return;
        }
        let Ok(sse) = sse else { continue };
        let Ok(event) = serde_json::from_str::<WireEvent>(sse.data.trim()) else {
            continue;
        };
        match event.event {
            WireEventKind::Done => {
                let message = event.message.unwrap_or_else(|| AssistantMessage {
                    content: vec![],
                    api: model.api.clone(),
                    provider: model.provider.clone(),
                    model: model.id.clone(),
                    response_model: None,
                    response_id: None,
                    provider_thinking_level: None,
                    diagnostics: vec![],
                    usage: Default::default(),
                    stop_reason: StopReason::Stop,
                    error_message: None,
                    raw_stop_reason: None,
                    end_turn: None,
                    timestamp: crate::types::now_millis(),
                });
                tx.push(AssistantMessageEvent::Done {
                    reason: event.reason.unwrap_or(StopReason::Stop),
                    message,
                });
                return;
            }
            WireEventKind::Error => {
                let error = event.error.unwrap_or_else(|| AssistantMessage {
                    content: vec![],
                    api: model.api.clone(),
                    provider: model.provider.clone(),
                    model: model.id.clone(),
                    response_model: None,
                    response_id: None,
                    provider_thinking_level: None,
                    diagnostics: vec![],
                    usage: Default::default(),
                    stop_reason: StopReason::Error,
                    error_message: Some("gateway error".into()),
                    raw_stop_reason: None,
                    end_turn: None,
                    timestamp: crate::types::now_millis(),
                });
                tx.push(AssistantMessageEvent::Error {
                    reason: event.reason.unwrap_or(StopReason::Error),
                    error,
                });
                return;
            }
            WireEventKind::Event => {
                if let Some(mut e) = event.event_payload {
                    // patch partial snapshots with api/provider identity
                    if let Some(partial) = e.get_mut("partial") {
                        patch_identity(partial, model);
                    }
                    if let Some(message) = e.get_mut("message") {
                        patch_identity(message, model);
                    }
                    if let Ok(parsed) = serde_json::from_value::<AssistantMessageEvent>(e) {
                        tx.push(parsed);
                    }
                }
            }
        }
    }
}

fn patch_identity(value: &mut Value, model: &Model) {
    if let Some(obj) = value.as_object_mut() {
        obj.entry("api").or_insert_with(|| json!(model.api));
        obj.entry("provider").or_insert_with(|| json!(model.provider));
        obj.entry("model").or_insert_with(|| json!(model.id));
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum WireEventKind {
    Event,
    Done,
    Error,
}

#[derive(Debug, Deserialize)]
struct WireEvent {
    event: WireEventKind,
    #[serde(default)]
    reason: Option<StopReason>,
    #[serde(default)]
    message: Option<AssistantMessage>,
    #[serde(default)]
    error: Option<AssistantMessage>,
    /// for `event` variant: the full AssistantMessageEvent
    #[serde(flatten)]
    event_payload: Option<Value>,
}
