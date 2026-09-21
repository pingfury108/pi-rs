//! The `Agent` struct, ported from `packages/agent/src/agent.ts` (subset):
//! state, steering/follow-up queues, subscription, prompt/abort.
//!
//! Design note: `prompt`/`retry` are async and borrow `&self`; the transcript
//! is snapshotted into the loop and written back after the run. Events stream
//! live via [`Agent::subscribe`] (broadcast channel).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::loop_::{run_agent_loop, run_agent_loop_continue};
use crate::types::*;

/// Stateful agent with tool execution and event streaming.
pub struct Agent {
    state: Mutex<StateInner>,
    tools: Mutex<Vec<Arc<dyn AgentTool>>>,
    stream_fn: StreamFn,
    tool_execution: ToolExecutionMode,
    steering: Arc<Mutex<VecDeque<AgentMessage>>>,
    follow_up: Arc<Mutex<VecDeque<AgentMessage>>>,
    event_tx: broadcast::Sender<AgentEvent>,
    is_streaming: AtomicBool,
    cancel: Mutex<Option<CancellationToken>>,
    hooks: Mutex<Option<LoopHooks>>,
    api_key_resolver: Mutex<Option<crate::types::ApiKeyResolver>>,
}

/// Tool hooks installed on an agent (bridged into the loop config per run).
#[derive(Default, Clone)]
pub struct LoopHooks {
    pub before_tool_call: Option<
        Arc<dyn Fn(crate::types::BeforeToolCallContext<'_>) -> BoxFuture<Option<crate::types::BeforeToolCallResult>> + Send + Sync>,
    >,
    pub after_tool_call: Option<
        Arc<dyn Fn(crate::types::AfterToolCallContext<'_>) -> BoxFuture<Option<crate::types::AfterToolCallResult>> + Send + Sync>,
    >,
}

struct StateInner {
    messages: Vec<AgentMessage>,
    model: pi_ai::types::Model,
    thinking_level: ThinkingLevel,
    error_message: Option<String>,
}

enum RunKind {
    Prompt(AgentMessage),
    Retry,
}

/// Builder for [`Agent`].
pub struct AgentBuilder {
    system_prompt: String,
    model: pi_ai::types::Model,
    thinking_level: ThinkingLevel,
    stream_fn: StreamFn,
    tool_execution: ToolExecutionMode,
}

impl AgentBuilder {
    pub fn new(model: pi_ai::types::Model, stream_fn: StreamFn) -> Self {
        Self {
            system_prompt: String::new(),
            model,
            thinking_level: ThinkingLevel::Off,
            stream_fn,
            tool_execution: ToolExecutionMode::Parallel,
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn thinking_level(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = level;
        self
    }

    pub fn tool_execution(mut self, mode: ToolExecutionMode) -> Self {
        self.tool_execution = mode;
        self
    }

    pub fn build(self) -> Agent {
        let (event_tx, _) = broadcast::channel(1024);
        let messages = if self.system_prompt.is_empty() {
            Vec::new()
        } else {
            vec![AgentMessage::Message(pi_ai::types::Message::System(
                pi_ai::types::SystemMessage::new(self.system_prompt.clone()),
            ))]
        };
        Agent {
            state: Mutex::new(StateInner {
                messages,
                model: self.model,
                thinking_level: self.thinking_level,
                error_message: None,
            }),
            tools: Mutex::new(Vec::new()),
            stream_fn: self.stream_fn,
            tool_execution: self.tool_execution,
            steering: Arc::new(Mutex::new(VecDeque::new())),
            follow_up: Arc::new(Mutex::new(VecDeque::new())),
            event_tx,
            is_streaming: AtomicBool::new(false),
            cancel: Mutex::new(None),
            hooks: Mutex::new(None),
            api_key_resolver: Mutex::new(None),
        }
    }
}

impl Agent {
    /// Subscribe to agent events (broadcast).
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Register an executable tool.
    pub fn add_tool(&self, tool: Arc<dyn AgentTool>) {
        self.tools.lock().unwrap().push(tool);
    }

    pub fn tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.tools.lock().unwrap().clone()
    }

    pub fn messages(&self) -> Vec<AgentMessage> {
        self.state.lock().unwrap().messages.clone()
    }

    pub fn set_messages(&self, messages: Vec<AgentMessage>) {
        self.state.lock().unwrap().messages = messages;
    }

    pub fn is_streaming(&self) -> bool {
        self.is_streaming.load(Ordering::SeqCst)
    }

    pub fn model(&self) -> pi_ai::types::Model {
        self.state.lock().unwrap().model.clone()
    }

    pub fn set_model(&self, model: pi_ai::types::Model) {
        self.state.lock().unwrap().model = model;
    }

    pub fn thinking_level(&self) -> ThinkingLevel {
        self.state.lock().unwrap().thinking_level
    }

    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.state.lock().unwrap().thinking_level = level;
    }

    pub fn error_message(&self) -> Option<String> {
        self.state.lock().unwrap().error_message.clone()
    }

    /// Queue a steering message (injected when the current tool batch ends).
    pub fn steer(&self, message: AgentMessage) {
        self.steering.lock().unwrap().push_back(message);
    }

    /// Queue a follow-up message (processed when the agent would stop).
    pub fn follow_up(&self, message: AgentMessage) {
        self.follow_up.lock().unwrap().push_back(message);
    }


    /// Install tool hooks (bridged into every subsequent run).
    pub fn set_hooks(&self, hooks: LoopHooks) {
        *self.hooks.lock().unwrap() = Some(hooks);
    }

    /// Install a dynamic API-key resolver, called before every LLM call
    /// (supports short-lived tokens; pi's getApiKey contract).
    pub fn set_api_key_resolver(&self, resolver: crate::types::ApiKeyResolver) {
        *self.api_key_resolver.lock().unwrap() = Some(resolver);
    }

    /// Emit a session-level event (e.g. auto-retry notifications).
    pub fn emit(&self, event: AgentEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Remove and return the trailing assistant message (auto-retry keeps
    /// the failed message in the session file but drops it from context).
    pub fn pop_last_assistant_message(&self) -> Option<AgentMessage> {
        let mut state = self.state.lock().unwrap();
        if matches!(state.messages.last(), Some(AgentMessage::Message(pi_ai::types::Message::Assistant(_)))) {
            state.messages.pop()
        } else {
            None
        }
    }

    /// Abort the current run.
    pub fn abort(&self) {
        if let Some(cancel) = self.cancel.lock().unwrap().as_ref() {
            cancel.cancel();
        }
    }

    /// Send a prompt: runs the agent loop to completion.
    /// Events flow via [`Agent::subscribe`].
    pub async fn prompt(&self, message: AgentMessage) -> Vec<AgentMessage> {
        assert!(
            !self.is_streaming.swap(true, Ordering::SeqCst),
            "agent is already streaming"
        );
        self.run(RunKind::Prompt(message)).await
    }

    /// Continue the loop from the current context (retry).
    pub async fn retry(&self) -> Vec<AgentMessage> {
        assert!(
            !self.is_streaming.swap(true, Ordering::SeqCst),
            "agent is already streaming"
        );
        self.run(RunKind::Retry).await
    }

    async fn run(&self, kind: RunKind) -> Vec<AgentMessage> {
        // Snapshot state into run inputs.
        let (messages, model, thinking_level) = {
            let state = self.state.lock().unwrap();
            (state.messages.clone(), state.model.clone(), state.thinking_level)
        };
        let mut context = AgentContext {
            messages,
            tools: self.tools(),
        };
        let config = self.build_run_config(model, thinking_level);
        let cancel = CancellationToken::new();
        *self.cancel.lock().unwrap() = Some(cancel.clone());
        let event_tx = self.event_tx.clone();
        let emit: EventSink = Arc::new(move |event| {
            let tx = event_tx.clone();
            Box::pin(async move {
                let _ = tx.send(event);
            })
        });

        let new_messages = match kind {
            RunKind::Prompt(message) => {
                run_agent_loop(vec![message], &mut context, &config, Some(cancel), emit).await
            }
            RunKind::Retry => {
                run_agent_loop_continue(&mut context, &config, Some(cancel), emit).await
            }
        };

        // Write transcript back and clear streaming state.
        {
            let mut state = self.state.lock().unwrap();
            state.messages = context.messages;
            state.error_message = None;
        }
        self.is_streaming.store(false, Ordering::SeqCst);
        *self.cancel.lock().unwrap() = None;
        new_messages
    }

    fn build_run_config(
        &self,
        model: pi_ai::types::Model,
        thinking_level: ThinkingLevel,
    ) -> AgentLoopConfig {
        // NOTE: clone hooks first — two `self.hooks.lock()` calls inside one
        // struct expression would deadlock (temporary guard lives to the end
        // of the expression).
        let hooks = self.hooks.lock().unwrap().clone();
        let api_key_resolver = self.api_key_resolver.lock().unwrap().clone();
        let get_steering: MessageQueueFn = {
            let deque = self.steering.clone();
            Arc::new(move || {
                let deque = deque.clone();
                Box::pin(async move { drain(&deque) })
            })
        };
        let get_follow_up: MessageQueueFn = {
            let deque = self.follow_up.clone();
            Arc::new(move || {
                let deque = deque.clone();
                Box::pin(async move { drain(&deque) })
            })
        };

        AgentLoopConfig {
            model,
            stream_options: pi_ai::types::StreamOptions {
                reasoning: thinking_to_reasoning(thinking_level),
                ..Default::default()
            },
            stream_fn: self.stream_fn.clone(),
            convert_to_llm: Arc::new(convert_to_llm_default),
            transform_context: None,
            get_api_key: api_key_resolver,
            get_steering_messages: Some(get_steering),
            get_follow_up_messages: Some(get_follow_up),
            should_stop_after_turn: None,
            prepare_next_turn: None,
            tool_execution: self.tool_execution,
            before_tool_call: hooks.as_ref().and_then(|h| h.before_tool_call.clone()),
            after_tool_call: hooks.as_ref().and_then(|h| h.after_tool_call.clone()),
        }
    }
}

/// Drain a shared queue (QueueMode "all").
fn drain(deque: &Arc<Mutex<VecDeque<AgentMessage>>>) -> Vec<AgentMessage> {
    let mut deque = deque.lock().unwrap();
    deque.drain(..).collect()
}

/// Default convertToLlm: pass standard messages through; custom messages
/// become user text (apps may install their own converter at the loop level).
pub fn convert_to_llm_default(messages: &[AgentMessage]) -> Vec<pi_ai::types::Message> {
    messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Message(message) => Some(message.clone()),
            AgentMessage::Custom(custom) => Some(pi_ai::types::Message::User(
                pi_ai::types::UserMessage::new(pi_ai::types::MessageContent::text(format!(
                    "[{}] {}",
                    custom.custom_type, custom.content
                ))),
            )),
        })
        .collect()
}

fn thinking_to_reasoning(level: ThinkingLevel) -> Option<pi_ai::types::ThinkingLevel> {
    use pi_ai::types::ThinkingLevel as P;
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some(P::Minimal),
        ThinkingLevel::Low => Some(P::Low),
        ThinkingLevel::Medium => Some(P::Medium),
        ThinkingLevel::High => Some(P::High),
        ThinkingLevel::Xhigh => Some(P::Xhigh),
        ThinkingLevel::Max => Some(P::Max),
    }
}
