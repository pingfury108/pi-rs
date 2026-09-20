//! The agent loop, ported from `packages/agent/src/agent-loop.ts`.
//!
//! Works with `AgentMessage`s throughout; converts to LLM messages only at
//! the LLM call boundary via `config.convert_to_llm`.
//!
//! Event ordering guarantees (mirroring pi):
//! - Sequential mode: prepare → execute → finalize per call, one by one.
//! - Parallel mode: preflight (start + hooks) sequentially; execution runs
//!   concurrently with `tool_execution_end` emitted in completion order;
//!   toolResult `message_start/end` events are emitted afterwards in
//!   assistant source order.

use std::sync::Arc;

use pi_ai::types::now_millis;
use pi_ai::types::{
    AssistantMessage, Context, Message, StopReason, SystemMessage, ToolCall, ToolResultMessage,
};

use crate::types::{
    AfterToolCallContext, AfterToolCallResult, AgentContext, AgentEvent, AgentLoopConfig,
    AgentMessage, AgentToolResult, BeforeToolCallContext, EventSink, MessageQueueFn,
    ShouldStopAfterTurnContext, ToolExecutionMode,
};

fn aborted(cancel: &Option<tokio_util::sync::CancellationToken>) -> bool {
    cancel.as_ref().is_some_and(|c| c.is_cancelled())
}

/// Run the agent loop with a new prompt. The prompt is added to the context.
/// Returns the new messages produced by this run.
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    cancel: Option<tokio_util::sync::CancellationToken>,
    emit: EventSink,
) -> Vec<AgentMessage> {
    let initial_messages = declare_tool_changes(context, Vec::new(), prompts);
    let mut new_messages = initial_messages.clone();
    for message in &initial_messages {
        context.messages.push(message.clone());
    }

    emit(AgentEvent::AgentStart).await;
    emit(AgentEvent::TurnStart).await;
    for message in &initial_messages {
        emit(AgentEvent::MessageStart { message: Box::new(message.clone()) }).await;
        emit(AgentEvent::MessageEnd { message: Box::new(message.clone()) }).await;
    }

    run_loop(context, &mut new_messages, config, &cancel, &emit).await;
    new_messages
}

/// Continue the loop from the current context without adding a new message
/// (used for retries). The last message must convert to `user`/`toolResult`.
pub async fn run_agent_loop_continue(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    cancel: Option<tokio_util::sync::CancellationToken>,
    emit: EventSink,
) -> Vec<AgentMessage> {
    assert!(!context.messages.is_empty(), "cannot continue: no messages in context");
    assert!(
        !matches!(context.messages.last(), Some(AgentMessage::Message(Message::Assistant(_)))),
        "cannot continue from assistant message"
    );

    let mut new_messages = Vec::new();
    emit(AgentEvent::AgentStart).await;
    emit(AgentEvent::TurnStart).await;
    run_loop(context, &mut new_messages, config, &cancel, &emit).await;
    new_messages
}

#[allow(clippy::too_many_lines)]
async fn run_loop(
    context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) {
    let mut last_completed_turn: Option<(AssistantMessage, Vec<ToolResultMessage>)> = None;
    let mut pending_messages = poll_queue(&config.get_steering_messages).await;

    // Outer loop: continues when queued follow-up messages arrive.
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages.
        while has_more_tool_calls || !pending_messages.is_empty() {
            let mut prepared_messages: Vec<AgentMessage> = Vec::new();
            if let Some((message, tool_results)) = &last_completed_turn {
                let snapshot = ShouldStopAfterTurnContext {
                    message,
                    tool_results,
                    new_messages,
                };
                if let Some(hook) = &config.prepare_next_turn {
                    if let Some(update) = hook(&snapshot).await {
                        if let Some(replacement) = update.context {
                            *context = replacement;
                        }
                        prepared_messages = update.messages;
                    }
                }
                // Preparation can be long-running (e.g. compaction); pick up
                // steering queued while it ran.
                if pending_messages.is_empty() {
                    pending_messages = poll_queue(&config.get_steering_messages).await;
                }
                emit(AgentEvent::TurnStart).await;
            }

            // Process prepared and queued messages before the next response.
            let to_announce = declare_tool_changes(context, prepared_messages, pending_messages);
            #[allow(unused_assignments)]
            {
                pending_messages = Vec::new();
            }
            for message in to_announce {
                emit(AgentEvent::MessageStart { message: Box::new(message.clone()) }).await;
                emit(AgentEvent::MessageEnd { message: Box::new(message.clone()) }).await;
                context.messages.push(message.clone());
                new_messages.push(message);
            }

            // Stream assistant response.
            let message = stream_assistant_response(context, config, cancel, emit).await;
            new_messages.push(AgentMessage::Message(Message::Assistant(message.clone())));

            if message.stop_reason == StopReason::Error || message.stop_reason == StopReason::Aborted
            {
                emit(AgentEvent::TurnEnd {
                    message: Box::new(AgentMessage::Message(Message::Assistant(message.clone()))),
                    tool_results: vec![],
                })
                .await;
                break;
            }

            // Execute tool calls.
            let tool_calls: Vec<ToolCall> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    pi_ai::types::AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                // A "length" stop means output was cut off: fail all tool calls
                // rather than execute potentially truncated arguments.
                let batch = if message.stop_reason == StopReason::Length {
                    fail_tool_calls_from_truncated_message(&tool_calls, emit).await
                } else {
                    execute_tool_calls(context, &message, config, cancel, emit).await
                };
                has_more_tool_calls = !batch.terminate;
                tool_results = batch.messages;
                for result in &tool_results {
                    let msg = AgentMessage::Message(Message::ToolResult(result.clone()));
                    context.messages.push(msg.clone());
                    new_messages.push(msg);
                }
            }

            emit(AgentEvent::TurnEnd {
                message: Box::new(AgentMessage::Message(Message::Assistant(message.clone()))),
                tool_results: tool_results.clone(),
            })
            .await;

            last_completed_turn = Some((message.clone(), tool_results.clone()));

            if let Some(hook) = &config.should_stop_after_turn {
                let snapshot = ShouldStopAfterTurnContext {
                    message: &message,
                    tool_results: &tool_results,
                    new_messages,
                };
                if hook(&snapshot) {
                    return;
                }
            }

            pending_messages = poll_queue(&config.get_steering_messages).await;
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_ups = poll_queue(&config.get_follow_up_messages).await;
        if !follow_ups.is_empty() {
            pending_messages = follow_ups;
            continue;
        }
        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await;
}

async fn poll_queue(queue: &Option<MessageQueueFn>) -> Vec<AgentMessage> {
    match queue {
        Some(f) => f().await,
        None => vec![],
    }
}

/// Declare tool loadout changes to the model: diff the transcript's declared
/// tools against `context.tools` and surface the delta as a system message.
pub fn declare_tool_changes(
    context: &AgentContext,
    mut prepared: Vec<AgentMessage>,
    mut pending: Vec<AgentMessage>,
) -> Vec<AgentMessage> {
    let system_index = pending
        .iter()
        .rposition(|m| matches!(m, AgentMessage::Message(Message::System(_))));

    let declared = declared_tools(&context.messages);
    let target: Vec<String> = context.tools.iter().map(|t| t.name().to_string()).collect();
    let added: Vec<pi_ai::types::ToolDef> = context
        .tools
        .iter()
        .filter(|t| !declared.contains(&t.name().to_string()))
        .map(|t| pi_ai::types::ToolDef::new(t.name(), t.description(), t.parameters()))
        .collect();
    let removed: Vec<String> = declared
        .iter()
        .filter(|name| !target.contains(name))
        .cloned()
        .collect();

    if added.is_empty() && removed.is_empty() {
        prepared.append(&mut pending);
        return prepared;
    }

    if let Some(index) = system_index {
        if let Some(AgentMessage::Message(Message::System(sys))) = pending.get_mut(index) {
            sys.tools_added = if added.is_empty() { None } else { Some(added) };
            sys.tools_removed = if removed.is_empty() { None } else { Some(removed) };
        }
        prepared.append(&mut pending);
        return prepared;
    }

    let mut update = SystemMessage::new("");
    update.tools_added = if added.is_empty() { None } else { Some(added) };
    update.tools_removed = if removed.is_empty() { None } else { Some(removed) };
    let update = AgentMessage::Message(Message::System(update));

    let insert_index = pending
        .iter()
        .position(|m| !matches!(m, AgentMessage::Message(Message::System(_))))
        .unwrap_or(pending.len());
    pending.insert(insert_index, update);
    prepared.append(&mut pending);
    prepared
}

/// Tool names declared by the transcript's system messages.
fn declared_tools(messages: &[AgentMessage]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for message in messages {
        if let AgentMessage::Message(Message::System(sys)) = message {
            if let Some(added) = &sys.tools_added {
                for tool in added {
                    names.retain(|n| n != &tool.name);
                    names.push(tool.name.clone());
                }
            }
            if let Some(removed) = &sys.tools_removed {
                for name in removed {
                    names.retain(|n| n != name);
                }
            }
        }
    }
    names
}

/// Stream one assistant response; pushes the live partial into
/// `context.messages` (replacing it as updates arrive) and emits events.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) -> AssistantMessage {
    // AgentMessage[] → transformContext → convertToLlm → Message[]
    let mut transcript = context.messages.clone();
    if let Some(transform) = &config.transform_context {
        transcript = transform(transcript).await;
    }
    let llm_messages = (config.convert_to_llm)(&transcript);
    let llm_context = Context {
        system_prompt: None,
        messages: llm_messages,
        tools: None,
    };

    let mut options = config.stream_options.clone();
    options.cancel = cancel.clone();
    if let Some(resolve) = &config.get_api_key {
        if let Some(key) = resolve(&config.model.provider).await {
            options.api_key = Some(key);
        }
    }

    let mut stream = (config.stream_fn)(&config.model, &llm_context, &options);

    use futures::StreamExt;
    let mut added_partial = false;
    let mut final_message: Option<AssistantMessage> = None;

    while let Some(event) = stream.next().await {
        match &event {
            pi_ai::events::AssistantMessageEvent::Start { partial } => {
                let message = AgentMessage::Message(Message::Assistant(partial.clone()));
                context.messages.push(message.clone());
                added_partial = true;
                emit(AgentEvent::MessageStart { message: Box::new(message) }).await;
            }
            pi_ai::events::AssistantMessageEvent::Done { message, .. }
            | pi_ai::events::AssistantMessageEvent::Error { error: message, .. } => {
                let final_msg = message.clone();
                let msg = AgentMessage::Message(Message::Assistant(final_msg.clone()));
                if added_partial {
                    context.messages.pop();
                }
                context.messages.push(msg.clone());
                if !added_partial {
                    emit(AgentEvent::MessageStart { message: Box::new(msg.clone()) }).await;
                }
                emit(AgentEvent::MessageEnd { message: Box::new(msg) }).await;
                final_message = Some(final_msg);
                break;
            }
            other => {
                // text/thinking/toolcall lifecycle events
                if let Some(partial) = snapshot_partial(other) {
                    let message = AgentMessage::Message(Message::Assistant(partial));
                    if added_partial {
                        context.messages.pop();
                    }
                    context.messages.push(message.clone());
                    emit(AgentEvent::MessageUpdate {
                        message: Box::new(message),
                        assistant_message_event: Box::new(other.clone()),
                    })
                    .await;
                }
            }
        }
    }

    if let Some(final_msg) = final_message {
        return final_msg;
    }

    // Stream ended without a terminal event; synthesize an error message.
    let message = AssistantMessage {
        content: vec![],
        api: config.model.api.clone(),
        provider: config.model.provider.clone(),
        model: config.model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: vec![],
        usage: Default::default(),
        stop_reason: StopReason::Error,
        error_message: Some("stream ended without a terminal event".into()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_millis(),
    };
    if added_partial {
        context.messages.pop();
    }
    let msg = AgentMessage::Message(Message::Assistant(message.clone()));
    context.messages.push(msg.clone());
    emit(AgentEvent::MessageEnd { message: Box::new(msg) }).await;
    message
}

/// Extract the response-so-far snapshot from non-terminal events.
fn snapshot_partial(event: &pi_ai::events::AssistantMessageEvent) -> Option<AssistantMessage> {
    use pi_ai::events::AssistantMessageEvent as E;
    match event {
        E::TextStart { partial, .. }
        | E::TextDelta { partial, .. }
        | E::TextEnd { partial, .. }
        | E::ThinkingStart { partial, .. }
        | E::ThinkingDelta { partial, .. }
        | E::ThinkingEnd { partial, .. }
        | E::ToolcallStart { partial, .. }
        | E::ToolcallDelta { partial, .. }
        | E::ToolcallEnd { partial, .. } => Some(partial.clone()),
        _ => None,
    }
}

struct ExecutedBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

/// Fail all tool calls from a message truncated by the output token limit.
async fn fail_tool_calls_from_truncated_message(
    tool_calls: &[ToolCall],
    emit: &EventSink,
) -> ExecutedBatch {
    let mut messages = Vec::new();
    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: args_value(&tool_call.arguments),
        })
        .await;
        let result = AgentToolResult::error(format!(
            "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
            tool_call.name
        ));
        emit(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            result: Box::new(result.clone()),
            is_error: true,
        })
        .await;
        let message = tool_result_message(tool_call, &result, true);
        emit_tool_result_message(&message, emit).await;
        messages.push(message);
    }
    ExecutedBatch {
        messages,
        terminate: false,
    }
}

/// Execute the tool calls of an assistant message (sequential or parallel).
async fn execute_tool_calls(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) -> ExecutedBatch {
    let tool_calls: Vec<ToolCall> = assistant_message
        .content
        .iter()
        .filter_map(|block| match block {
            pi_ai::types::AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect();

    let has_sequential = tool_calls.iter().any(|call| {
        context
            .find_tool(&call.name)
            .is_some_and(|tool| tool.execution_mode() == ToolExecutionMode::Sequential)
    });

    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential {
        execute_tool_calls_sequential(context, assistant_message, &tool_calls, config, cancel, emit)
            .await
    } else {
        execute_tool_calls_parallel(context, assistant_message, &tool_calls, config, cancel, emit)
            .await
    }
}

type FinalizedOutcome = (ToolCall, AgentToolResult, bool);

fn should_terminate_batch(finalized: &[FinalizedOutcome]) -> bool {
    !finalized.is_empty() && finalized.iter().all(|(_, result, _)| result.terminate)
}

fn tool_result_message(
    tool_call: &ToolCall,
    result: &AgentToolResult,
    is_error: bool,
) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content: result.content.clone(),
        details: Some(result.details.clone()),
        usage: result.usage.clone(),
        is_error,
        timestamp: now_millis(),
    }
}

async fn emit_tool_execution_end(finalized: &FinalizedOutcome, emit: &EventSink) {
    let (tool_call, result, is_error) = finalized;
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        result: Box::new(result.clone()),
        is_error: *is_error,
    })
    .await;
}

async fn emit_tool_result_message(message: &ToolResultMessage, emit: &EventSink) {
    let msg = AgentMessage::Message(Message::ToolResult(message.clone()));
    emit(AgentEvent::MessageStart { message: Box::new(msg.clone()) }).await;
    emit(AgentEvent::MessageEnd { message: Box::new(msg) }).await;
}

async fn emit_tool_execution_start(tool_call: &ToolCall, emit: &EventSink) {
    emit(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        args: args_value(&tool_call.arguments),
    })
    .await;
}

fn args_value(arguments: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    serde_json::Value::Object(arguments.clone())
}

async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) -> ExecutedBatch {
    let mut finalized_calls: Vec<FinalizedOutcome> = Vec::new();
    let mut messages = Vec::new();

    for tool_call in tool_calls {
        emit_tool_execution_start(tool_call, emit).await;

        let finalized =
            prepare_and_execute(context, assistant_message, tool_call, config, cancel, emit).await;
        emit_tool_execution_end(&finalized, emit).await;
        let message = tool_result_message(tool_call, &finalized.1, finalized.2);
        emit_tool_result_message(&message, emit).await;
        messages.push(message);
        finalized_calls.push(finalized);

        if aborted(cancel) {
            break;
        }
    }

    ExecutedBatch {
        messages,
        terminate: should_terminate_batch(&finalized_calls),
    }
}

enum Entry {
    Ready(FinalizedOutcome),
    Async(std::pin::Pin<Box<dyn futures::Future<Output = FinalizedOutcome> + Send>>),
}

async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) -> ExecutedBatch {
    let mut entries: Vec<Entry> = Vec::new();

    // Preflight sequentially, emitting tool_execution_start.
    for tool_call in tool_calls {
        emit_tool_execution_start(tool_call, emit).await;

        match prepare_tool_call(context, assistant_message, tool_call, config, cancel).await {
            PreparedToolCall::Immediate { result, is_error } => {
                entries.push(Entry::Ready((tool_call.clone(), result, is_error)));
                if aborted(cancel) {
                    break;
                }
            }
            PreparedToolCall::Prepared { tool, args } => {
                let tool = tool.clone();
                let tool_call = tool_call.clone();
                let args = args.clone();
                let cancel_inner = cancel.clone();
                let emit = emit.clone();
                let after_hook = config.after_tool_call.clone();
                let assistant_snapshot = assistant_message.clone();
                entries.push(Entry::Async(Box::pin(async move {
                    let finalized = if aborted(&cancel_inner) {
                        (tool_call.clone(), AgentToolResult::error("Operation aborted"), true)
                    } else {
                        let (mut result, mut is_error) =
                            execute_tool(&tool, &tool_call, &args, &cancel_inner, &emit).await;
                        if let Some(hook) = &after_hook {
                            if let Some(after) = hook(AfterToolCallContext {
                                assistant_message: &assistant_snapshot,
                                tool_call: &tool_call,
                                args: args.clone(),
                                result: &result,
                                is_error,
                            })
                            .await
                            {
                                apply_after_tool_call(&mut result, &mut is_error, after);
                            }
                        }
                        (tool_call.clone(), result, is_error)
                    };
                    // tool_execution_end in completion order
                    emit_tool_execution_end(&finalized, &emit).await;
                    finalized
                })));
                if aborted(&cancel) {
                    break;
                }
            }
        }
    }

    // Run concurrently; preserves source order in results.
    let order_preserved = futures::future::join_all(entries.into_iter().map(|entry| async move {
        match entry {
            Entry::Ready(outcome) => {
                emit_tool_execution_end(&outcome, emit).await;
                outcome
            }
            Entry::Async(fut) => fut.await,
        }
    }))
    .await;

    // Emit toolResult messages in assistant source order.
    let mut messages = Vec::new();
    for (tool_call, result, is_error) in &order_preserved {
        let message = tool_result_message(tool_call, result, *is_error);
        emit_tool_result_message(&message, emit).await;
        messages.push(message);
    }

    ExecutedBatch {
        messages,
        terminate: should_terminate_batch(&order_preserved),
    }
}

fn apply_after_tool_call(
    result: &mut AgentToolResult,
    is_error: &mut bool,
    after: AfterToolCallResult,
) {
    if let Some(content) = after.content {
        result.content = content;
    }
    if let Some(details) = after.details {
        result.details = details;
    }
    if let Some(usage) = after.usage {
        result.usage = Some(usage);
    }
    if let Some(terminate) = after.terminate {
        result.terminate = terminate;
    }
    if let Some(err) = after.is_error {
        *is_error = err;
    }
}

enum PreparedToolCall {
    Immediate { result: AgentToolResult, is_error: bool },
    Prepared { tool: Arc<dyn crate::types::AgentTool>, args: serde_json::Value },
}

/// Validate + run beforeToolCall hook. Failures become immediate error results.
async fn prepare_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
) -> PreparedToolCall {
    let Some(tool) = context.find_tool(&tool_call.name) else {
        return PreparedToolCall::Immediate {
            result: AgentToolResult::error(format!("Tool {} not found", tool_call.name)),
            is_error: true,
        };
    };

    // Arguments arrive as a JSON object map from the provider; per-tool
    // validation is deferred to the tools themselves.

    let args = serde_json::Value::Object(tool_call.arguments.clone());
    if let Some(hook) = &config.before_tool_call {
        let before = hook(BeforeToolCallContext {
            assistant_message,
            tool_call,
            args: args.clone(),
        })
        .await;
        if aborted(cancel) {
            return PreparedToolCall::Immediate {
                result: AgentToolResult::error("Operation aborted"),
                is_error: true,
            };
        }
        if let Some(before) = before {
            if before.block {
                let mut result = AgentToolResult::error(
                    before
                        .reason
                        .unwrap_or_else(|| "Tool execution was blocked".to_string()),
                );
                result.terminate = before.terminate;
                return PreparedToolCall::Immediate { result, is_error: true };
            }
        }
    }
    if aborted(cancel) {
        return PreparedToolCall::Immediate {
            result: AgentToolResult::error("Operation aborted"),
            is_error: true,
        };
    }

    PreparedToolCall::Prepared { tool: tool.clone(), args }
}

/// prepare + execute + afterToolCall (used by sequential mode).
async fn prepare_and_execute(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    config: &AgentLoopConfig,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) -> FinalizedOutcome {
    match prepare_tool_call(context, assistant_message, tool_call, config, cancel).await {
        PreparedToolCall::Immediate { result, is_error } => (tool_call.clone(), result, is_error),
        PreparedToolCall::Prepared { tool, args } => {
            let (mut result, mut is_error) =
                execute_tool(&tool, tool_call, &args, cancel, emit).await;
            if let Some(hook) = &config.after_tool_call {
                if let Some(after) = hook(AfterToolCallContext {
                    assistant_message,
                    tool_call,
                    args: args.clone(),
                    result: &result,
                    is_error,
                })
                .await
                {
                    apply_after_tool_call(&mut result, &mut is_error, after);
                }
            }
            (tool_call.clone(), result, is_error)
        }
    }
}

/// Execute a prepared tool call, forwarding updates as tool_execution_update.
async fn execute_tool(
    tool: &Arc<dyn crate::types::AgentTool>,
    tool_call: &ToolCall,
    args: &serde_json::Value,
    cancel: &Option<tokio_util::sync::CancellationToken>,
    emit: &EventSink,
) -> (AgentToolResult, bool) {
    let emit_for_updates = emit.clone();
    let tool_call_id = tool_call.id.clone();
    let tool_name = tool_call.name.clone();
    let args_for_updates = args.clone();

    let on_update: crate::types::ToolUpdateFn = Arc::new(move |partial: AgentToolResult| {
        let sink = emit_for_updates.clone();
        let id = tool_call_id.clone();
        let name = tool_name.clone();
        let args = args_for_updates.clone();
        tokio::spawn(async move {
            sink(AgentEvent::ToolExecutionUpdate {
                tool_call_id: id,
                tool_name: name,
                args,
                partial_result: Box::new(partial),
            })
            .await;
        });
    });

    let ctx = crate::types::ToolContext { cancel: cancel.clone() };
    let fut = tool.execute(&tool_call.id, args.clone(), &ctx, on_update);

    let result = match cancel {
        Some(token) => {
            tokio::select! {
                _ = token.cancelled() => Err("Operation aborted".to_string()),
                result = fut => result,
            }
        }
        None => fut.await,
    };

    match result {
        Ok(result) => (result, false),
        Err(message) => (AgentToolResult::error(message), true),
    }
}

// Keep BoxFuture import used (type aliases below).
#[allow(unused_imports)]
use crate::types::BoxFuture as _BoxFutureUsed;
