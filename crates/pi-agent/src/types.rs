//! Core agent runtime types, ported from `packages/agent/src/types.ts`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use pi_ai::events::AssistantMessageEvent;
use pi_ai::types::{
    AssistantMessage, Context, Message, Model, StreamOptions, TextContent, Timestamp,
    ToolResultContent, ToolResultMessage, Usage,
};

/// Agent thinking level (superset of the provider-neutral level with `off`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// A message in the agent transcript: either an LLM message or an app-specific
/// custom message (pi's `AgentMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentMessage {
    Message(Message),
    Custom(CustomAgentMessage),
}

impl AgentMessage {
    pub fn timestamp(&self) -> Timestamp {
        match self {
            Self::Message(m) => m.timestamp(),
            Self::Custom(m) => m.timestamp,
        }
    }
}

impl From<Message> for AgentMessage {
    fn from(m: Message) -> Self {
        Self::Message(m)
    }
}

/// App-specific message that is not sent to the LLM unless converted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentMessage {
    /// Must be `"custom"` for session compatibility.
    pub role: String,
    /// App-defined subtype, e.g. `"notification"`.
    pub custom_type: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub timestamp: Timestamp,
}

impl CustomAgentMessage {
    pub fn new(custom_type: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "custom".into(),
            custom_type: custom_type.into(),
            content: content.into(),
            details: None,
            timestamp: pi_ai::types::now_millis(),
        }
    }
}

/// Result produced by a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResult {
    pub content: Vec<ToolResultContent>,
    /// Arbitrary structured details for logs or UI rendering.
    #[serde(default)]
    pub details: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Hint that the agent should stop after the current tool batch.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub terminate: bool,
}

impl AgentToolResult {
    /// Text-only result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: serde_json::Value::Object(Default::default()),
            usage: None,
            terminate: false,
        }
    }

    /// Error result carrying a message.
    pub fn error(text: impl Into<String>) -> Self {
        Self::text(text)
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }
}

/// Handle for streaming partial updates while a tool executes.
pub type ToolUpdateFn = Arc<dyn Fn(AgentToolResult) + Send + Sync>;

/// Execution context passed to tools.
#[derive(Clone, Default)]
pub struct ToolContext {
    /// Cooperative cancellation for the current run.
    pub cancel: Option<tokio_util::sync::CancellationToken>,
}

/// A tool the agent can execute.
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the tool arguments.
    fn parameters(&self) -> Value;
    /// Human-readable label for UI display.
    fn label(&self, args: &Value) -> String;

    /// Execute the tool. Return `Err` to produce an error tool result (pi:
    /// tools throw instead of encoding errors in content).
    async fn execute(
        &self,
        tool_call_id: &str,
        args: Value,
        ctx: &ToolContext,
        on_update: ToolUpdateFn,
    ) -> Result<AgentToolResult, String>;

    /// Per-tool execution mode override.
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }
}

/// How tool calls from one assistant message are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

/// Context passed to `before_tool_call`.
pub struct BeforeToolCallContext<'a> {
    pub assistant_message: &'a AssistantMessage,
    pub tool_call: &'a pi_ai::types::ToolCall,
    pub args: Value,
}

/// Returned by `before_tool_call`; `block: true` prevents execution.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
    pub terminate: bool,
}

/// Context passed to `after_tool_call`.
pub struct AfterToolCallContext<'a> {
    pub assistant_message: &'a AssistantMessage,
    pub tool_call: &'a pi_ai::types::ToolCall,
    pub args: Value,
    pub result: &'a AgentToolResult,
    pub is_error: bool,
}

/// Partial override returned by `after_tool_call`.
#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ToolResultContent>>,
    pub details: Option<Value>,
    pub is_error: Option<bool>,
    pub usage: Option<Usage>,
    pub terminate: Option<bool>,
}

/// Context snapshot passed into the low-level agent loop.
#[derive(Clone, Default)]
pub struct AgentContext {
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl AgentContext {
    pub fn find_tool(&self, name: &str) -> Option<&Arc<dyn AgentTool>> {
        self.tools.iter().find(|t| t.name() == name)
    }
}

/// Boxed sendable future.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Stream function: called at each LLM boundary with the converted context.
pub type StreamFn = Arc<dyn Fn(&Model, &Context, &StreamOptions) -> pi_ai::events::AssistantMessageEventStream + Send + Sync>;

/// Queue accessor returning steering/follow-up messages.
pub type MessageQueueFn = Arc<dyn Fn() -> BoxFuture<Vec<AgentMessage>> + Send + Sync>;

/// Event sink receiving agent events.
pub type EventSink = Arc<dyn Fn(AgentEvent) -> BoxFuture<()> + Send + Sync>;

/// Lifecycle hooks + queue accessors + LLM boundary transforms.
pub struct AgentLoopConfig {
    pub model: Model,
    pub stream_options: StreamOptions,
    pub stream_fn: StreamFn,

    /// Converts `AgentMessage`s to LLM messages before each LLM call.
    pub convert_to_llm: Arc<dyn Fn(&[AgentMessage]) -> Vec<Message> + Send + Sync>,
    /// Optional transform applied before `convert_to_llm` (context pruning).
    pub transform_context:
        Option<Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<Vec<AgentMessage>> + Send + Sync>>,
    /// Resolves the API key dynamically per LLM call.
    pub get_api_key: Option<Arc<dyn Fn(&str) -> BoxFuture<Option<String>> + Send + Sync>>,

    /// Steering messages injected mid-run.
    pub get_steering_messages: Option<MessageQueueFn>,
    /// Follow-up messages processed when the agent would stop.
    pub get_follow_up_messages: Option<MessageQueueFn>,

    /// Graceful stop request after a completed turn.
    pub should_stop_after_turn: Option<Arc<dyn Fn(&ShouldStopAfterTurnContext) -> bool + Send + Sync>>,
    /// Called before the next turn starts (compaction hook point).
    pub prepare_next_turn: Option<
        Arc<dyn Fn(&PrepareNextTurnContext) -> BoxFuture<Option<AgentLoopTurnUpdate>> + Send + Sync>,
    >,

    /// Tool execution mode for this run.
    pub tool_execution: ToolExecutionMode,

    pub before_tool_call: Option<
        Arc<dyn Fn(BeforeToolCallContext<'_>) -> BoxFuture<Option<BeforeToolCallResult>> + Send + Sync>,
    >,
    pub after_tool_call:
        Option<Arc<dyn Fn(AfterToolCallContext<'_>) -> BoxFuture<Option<AfterToolCallResult>> + Send + Sync>>,
}

/// State that may be replaced before the next provider request.
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub messages: Vec<AgentMessage>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
}

/// Context passed to `should_stop_after_turn` / `prepare_next_turn`.
pub struct ShouldStopAfterTurnContext<'a> {
    pub message: &'a AssistantMessage,
    pub tool_results: &'a [ToolResultMessage],
    pub new_messages: &'a [AgentMessage],
}

pub type PrepareNextTurnContext<'a> = ShouldStopAfterTurnContext<'a>;

/// Events emitted by the agent loop / agent for UI updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    AgentStart,
    AgentEnd { messages: Vec<AgentMessage> },
    TurnStart,
    TurnEnd { message: Box<AgentMessage>, tool_results: Vec<ToolResultMessage> },
    MessageStart { message: Box<AgentMessage> },
    MessageUpdate { message: Box<AgentMessage>, assistant_message_event: Box<AssistantMessageEvent> },
    MessageEnd { message: Box<AgentMessage> },
    ToolExecutionStart { tool_call_id: String, tool_name: String, args: Value },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: Box<AgentToolResult>,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: Box<AgentToolResult>,
        is_error: bool,
    },
}

/// Build an error tool result (pi's `createErrorToolResult`).
pub fn error_tool_result(message: impl Into<String>) -> AgentToolResult {
    AgentToolResult::error(message)
}

/// Render a tool result's text content (helper used by UIs).
pub fn content_text(content: &[ToolResultContent]) -> String {
    content
        .iter()
        .map(|c| match c {
            ToolResultContent::Text(t) => t.text.clone(),
            ToolResultContent::Image(i) => format!("[image: {}]", i.mime_type),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
