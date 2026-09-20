//! Core LLM types, ported from `packages/ai/src/types.ts`.
//!
//! JSON field names match pi's session format (camelCase) so session files
//! stay interoperable with the TypeScript implementation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Provider API protocol identifier (subset of pi `KnownApi`).
pub type Api = String;

/// Provider identifier (openai, anthropic, deepseek, ...).
pub type ProviderId = String;

pub mod content {
    use super::*;

    /// Text content block.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct TextContent {
        pub text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub text_signature: Option<String>,
    }

    /// Thinking/reasoning content block.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThinkingContent {
        pub thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub thinking_signature: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        pub redacted: bool,
    }

    /// Image content block (base64-encoded data).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ImageContent {
        pub data: String,
        pub mime_type: String,
    }

    /// Tool call emitted by an assistant message.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ToolCall {
        pub id: String,
        pub name: String,
        pub arguments: serde_json::Map<String, Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub thought_signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub namespace: Option<String>,
    }

    /// Content blocks allowed in user messages.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "camelCase")]
    pub enum UserContent {
        Text(TextContent),
        Image(ImageContent),
    }

    /// Content blocks allowed in tool results.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "camelCase")]
    pub enum ToolResultContent {
        Text(TextContent),
        Image(ImageContent),
    }

    /// Content blocks emitted by assistant messages.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "camelCase")]
    pub enum AssistantContent {
        Text(TextContent),
        Thinking(ThinkingContent),
        ToolCall(ToolCall),
    }

    /// Message content: either a plain string or an array of blocks
    /// (`string | (TextContent | ImageContent)[]` in pi).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum MessageContent {
        Text(String),
        Blocks(Vec<UserContent>),
    }

    impl MessageContent {
        pub fn text(s: impl Into<String>) -> Self {
            Self::Text(s.into())
        }

        /// Concatenated text of all text blocks (or the string itself).
        pub fn as_text(&self) -> String {
            match self {
                Self::Text(s) => s.clone(),
                Self::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        UserContent::Text(t) => Some(t.text.as_str()),
                        UserContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            }
        }
    }
}

pub use content::*;

/// Token usage and cost accounting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(default)]
    pub input: u64,
    #[serde(default)]
    pub output: u64,
    #[serde(default)]
    pub cache_read: u64,
    #[serde(default)]
    pub cache_write: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: Cost,
}

/// Cost breakdown in USD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
    #[serde(default)]
    pub total: f64,
}

/// Why the assistant message stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

/// Provider-neutral thinking level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// Tool selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
}

/// Tool declaration sent to the LLM (`parameters` is a JSON Schema value).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDef {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Unix timestamp in milliseconds.
pub type Timestamp = i64;

/// Current wall clock in milliseconds.
pub fn now_millis() -> Timestamp {
    chrono::Utc::now().timestamp_millis()
}

/// System instructions and tool declarations at one point in the transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
    pub content: MessageContent,
    /// Named prompt sections; `None` value removes a section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<BTreeMap<String, Option<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_added: Option<Vec<ToolDef>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_removed: Option<Vec<String>>,
    pub timestamp: Timestamp,
}

/// User message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: MessageContent,
    pub timestamp: Timestamp,
}

/// Assistant response message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Value>,
    #[serde(default)]
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    pub timestamp: Timestamp,
}

/// Tool result message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ToolResultContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub is_error: bool,
    pub timestamp: Timestamp,
}

/// One message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn timestamp(&self) -> Timestamp {
        match self {
            Self::System(m) => m.timestamp,
            Self::User(m) => m.timestamp,
            Self::Assistant(m) => m.timestamp,
            Self::ToolResult(m) => m.timestamp,
        }
    }
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: MessageContent::text(content),
            sections: None,
            tools_added: None,
            tools_removed: None,
            timestamp: now_millis(),
        }
    }
}

impl UserMessage {
    pub fn new(content: MessageContent) -> Self {
        Self {
            content,
            timestamp: now_millis(),
        }
    }
}

impl ToolResultMessage {
    pub fn text(tool_call_id: impl Into<String>, tool_name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            is_error: false,
            timestamp: now_millis(),
        }
    }
}

/// Request input: system prompt + tools are shorthand for a leading system message.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
}

/// Pricing for a model, USD per million tokens.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
}

/// Model input/output modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
}

/// A model catalog entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    pub base_url: String,
    /// Whether the model supports reasoning/thinking.
    pub reasoning: bool,
    pub input: Vec<Modality>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    /// Default sampling parameters; per-request keys override these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<serde_json::Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Provider compat overrides (opaque in the core; interpreted by API impls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

/// Options for a streaming request.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    pub api_key: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub reasoning: Option<ThinkingLevel>,
    pub tool_choice: Option<ToolChoice>,
    /// Extra headers merged over provider defaults.
    pub headers: Option<BTreeMap<String, String>>,
    /// Optional custom base URL override.
    pub base_url: Option<String>,
    /// Cooperative cancellation.
    pub cancel: Option<tokio_util::sync::CancellationToken>,
}

impl StreamOptions {
    pub fn api_key(key: impl Into<String>) -> Self {
        Self {
            api_key: Some(key.into()),
            ..Default::default()
        }
    }
}
