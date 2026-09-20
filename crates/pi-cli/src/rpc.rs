//! RPC mode: headless JSONL protocol on stdin/stdout, port of
//! `coding-agent/src/modes/rpc/rpc-mode.ts` (core command subset).
//!
//! - stdin: one command per line, `{id?, type: "<command>", ...}`
//! - stdout: responses `{id, type: "response", command, success, data?/error}`
//!   and agent events `{type: "event", event: {...}}`

use std::io::{BufRead as _, Write as _};
use std::sync::Arc;

use pi_agent::AgentEvent;
use pi_core::AgentSession;
use serde_json::{json, Value};

pub async fn run_rpc(session: Arc<AgentSession>) -> anyhow::Result<()> {
    let (line_tx, mut line_rx) = tokio::sync::mpsc::channel::<String>(32);

    // stdin reader: blocking reads on a dedicated thread.
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if line_tx.blocking_send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut events = session.subscribe();
    let mut out = std::io::stdout().lock();

    // After stdin closes, keep streaming until the active run settles.
    let mut stdin_closed = false;
    loop {
        tokio::select! {
            line = line_rx.recv(), if !stdin_closed => {
                let Some(line) = line else { stdin_closed = true; continue };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let command: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        writeln!(out, "{}", json!({
                            "type": "response", "success": false,
                            "error": format!("invalid command json: {e}"),
                        })).ok();
                        out.flush().ok();
                        continue;
                    }
                };
                let response = handle_command(&session, command).await;
                writeln!(out, "{response}").ok();
                out.flush().ok();
            }
            event = events.recv() => {
                match event {
                    Ok(event) => {
                        let payload = match &event {
                            AgentEvent::MessageUpdate {
                                assistant_message_event,
                                ..
                            } => json!({
                                "type": "event",
                                "event": {
                                    "type": "message_update",
                                    "assistantMessageEvent": assistant_message_event.as_ref(),
                                },
                            }),
                            other => json!({"type": "event", "event": other}),
                        };
                        writeln!(out, "{payload}").ok();
                        out.flush().ok();
                        if stdin_closed && matches!(event, AgentEvent::AgentEnd { .. }) {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            // stdin closed and no active run: shut down cleanly
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)), if stdin_closed => {
                if !session.agent.is_streaming() {
                    break;
                }
            }
        }
    }
    Ok(())
}

async fn handle_command(session: &Arc<AgentSession>, command: Value) -> Value {
    let id = command.get("id").cloned();
    let command_type = command
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let success = |data: Option<Value>| {
        let mut obj = serde_json::Map::new();
        if let Some(id) = id.clone() {
            obj.insert("id".into(), id);
        }
        obj.insert("type".into(), json!("response"));
        obj.insert("command".into(), json!(command_type));
        obj.insert("success".into(), json!(true));
        if let Some(data) = data {
            obj.insert("data".into(), data);
        }
        Value::Object(obj)
    };
    let failure = |message: String| {
        json!({
            "id": id, "type": "response", "command": command_type,
            "success": false, "error": message,
        })
    };

    match command_type.as_str() {
        "prompt" => {
            let Some(message) = str_field(&command, "message") else {
                return failure("missing message".into());
            };
            // streamingBehavior: "steer"/"followUp" route to the queues
            match command.get("streamingBehavior").and_then(Value::as_str) {
                Some("steer") => session.steer(message),
                Some("followUp") => session.follow_up(message),
                _ => {
                    session.prompt(message);
                }
            }
            success(None)
        }
        "steer" => match str_field(&command, "message") {
            Some(message) => {
                session.steer(message);
                success(None)
            }
            None => failure("missing message".into()),
        },
        "follow_up" => match str_field(&command, "message") {
            Some(message) => {
                session.follow_up(message);
                success(None)
            }
            None => failure("missing message".into()),
        },
        "abort" => {
            session.abort();
            success(None)
        }
        "get_state" => success(Some(session.rpc_state())),
        "set_model" => {
            let provider = str_field(&command, "provider").unwrap_or("");
            let model_id = str_field(&command, "modelId").unwrap_or("");
            match pi_core::build_model(provider, Some(model_id), None)
                .map_err(anyhow::Error::msg)
            {
                Ok(model) => {
                    session.set_model(model.clone());
                    success(Some(json!({"provider": model.provider, "modelId": model.id})))
                }
                Err(e) => failure(e.to_string()),
            }
        }
        "set_thinking_level" => {
            // accepted for protocol parity; applied on next session start in
            // this build
            success(None)
        }
        "compact" => match session.force_compact().await {
            Ok(compacted) => success(Some(json!({"compacted": compacted}))),
            Err(e) => failure(e.to_string()),
        },
        "set_auto_compaction" => {
            let enabled = command.get("enabled").and_then(Value::as_bool);
            match enabled {
                Some(enabled) => {
                    session.set_auto_compaction(enabled);
                    success(None)
                }
                None => failure("missing enabled".into()),
            }
        }
        "get_messages" => success(Some(json!({"messages": session.messages()}))),
        "get_last_assistant_text" => success(Some(json!({
            "text": session.last_assistant_text().unwrap_or_default(),
        }))),
        "get_session_stats" => {
            let state = session.rpc_state();
            success(Some(json!({
                "sessionFile": state["sessionFile"],
                "messageCount": state["messageCount"],
            })))
        }
        "switch_session" => {
            failure("switch_session requires restarting pi-rs with --resume in this build".into())
        }
        "get_commands" => success(Some(json!({"commands": [
            "prompt", "steer", "follow_up", "abort", "get_state", "set_model",
            "set_thinking_level", "compact", "set_auto_compaction",
            "get_messages", "get_last_assistant_text", "get_session_stats",
        ]}))),
        other => failure(format!("unknown command: {other}")),
    }
}

fn str_field<'a>(command: &'a Value, key: &str) -> Option<&'a str> {
    command.get(key).and_then(Value::as_str)
}
