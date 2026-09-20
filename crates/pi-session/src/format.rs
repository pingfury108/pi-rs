//! Session JSONL format, port of the types in
//! `coding-agent/src/core/session-manager.ts`.
//!
//! A session file is JSONL: line 1 is a [`SessionHeader`], every following
//! line is a [`SessionEntry`]. Entries form a tree via `id`/`parentId`,
//! enabling branching.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use pi_ai::types::{
    Message, SystemMessage, Timestamp, ToolDef, ToolResultContent, Usage,
};

pub const CURRENT_SESSION_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    pub id: String,
    /// RFC 3339 timestamp.
    pub timestamp: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

/// Base fields shared by all entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryBase {
    pub id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub message: Message,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelChangeEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub thinking_level: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChangeEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub provider: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    /// Arbitrary usage category, such as "cache_warm".
    pub kind: String,
    pub provider: String,
    pub model: String,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub from_hook: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_message: Option<SystemMessage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub from_id: String,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub from_hook: bool,
}

/// Extension state entry; does not participate in LLM context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub custom_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Extension-injected message entry; participates in LLM context as user text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub custom_type: String,
    pub content: CustomMessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub display: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageContent {
    Text(String),
    Blocks(Vec<ToolResultContent>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    pub target_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoEntry {
    #[serde(flatten)]
    pub base: EntryBase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// One session entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    Message(MessageEntry),
    ThinkingLevelChange(ThinkingLevelChangeEntry),
    ModelChange(ModelChangeEntry),
    Usage(UsageEntry),
    Compaction(CompactionEntry),
    BranchSummary(BranchSummaryEntry),
    Custom(CustomEntry),
    CustomMessage(CustomMessageEntry),
    Label(LabelEntry),
    SessionInfo(SessionInfoEntry),
}

impl SessionEntry {
    pub fn base(&self) -> &EntryBase {
        match self {
            Self::Message(e) => &e.base,
            Self::ThinkingLevelChange(e) => &e.base,
            Self::ModelChange(e) => &e.base,
            Self::Usage(e) => &e.base,
            Self::Compaction(e) => &e.base,
            Self::BranchSummary(e) => &e.base,
            Self::Custom(e) => &e.base,
            Self::CustomMessage(e) => &e.base,
            Self::Label(e) => &e.base,
            Self::SessionInfo(e) => &e.base,
        }
    }

    pub fn id(&self) -> &str {
        &self.base().id
    }

    pub fn parent_id(&self) -> Option<&str> {
        self.base().parent_id.as_deref()
    }

    /// Build a new entry inheriting `parent` as its parent.
    pub fn set_parent(&mut self, parent_id: Option<String>) {
        match self {
            Self::Message(e) => e.base.parent_id = parent_id,
            Self::ThinkingLevelChange(e) => e.base.parent_id = parent_id,
            Self::ModelChange(e) => e.base.parent_id = parent_id,
            Self::Usage(e) => e.base.parent_id = parent_id,
            Self::Compaction(e) => e.base.parent_id = parent_id,
            Self::BranchSummary(e) => e.base.parent_id = parent_id,
            Self::Custom(e) => e.base.parent_id = parent_id,
            Self::CustomMessage(e) => e.base.parent_id = parent_id,
            Self::Label(e) => e.base.parent_id = parent_id,
            Self::SessionInfo(e) => e.base.parent_id = parent_id,
        }
    }
}

/// Raw file line: header or entry. Serialization matches pi's format
/// (discriminated by the `type` field: "session" vs entry tags).
#[derive(Debug, Clone, PartialEq)]
pub enum FileEntry {
    Session(SessionHeader),
    Entry(SessionEntry),
}

/// Serialize a file entry as one JSONL line.
pub fn entry_to_line(entry: &FileEntry) -> serde_json::Result<String> {
    match entry {
        FileEntry::Session(header) => {
            let mut value = serde_json::to_value(header)?;
            value["type"] = Value::String("session".into());
            serde_json::to_string(&value)
        }
        FileEntry::Entry(e) => serde_json::to_string(e),
    }
}

/// Parse one JSONL line by its `type` discriminator.
pub fn line_to_entry(line: &str) -> serde_json::Result<FileEntry> {
    let value: Value = serde_json::from_str(line)?;
    let is_header = value.get("type").and_then(Value::as_str) == Some("session");
    if is_header {
        Ok(FileEntry::Session(serde_json::from_value(value)?))
    } else {
        Ok(FileEntry::Entry(serde_json::from_value(value)?))
    }
}

/// Build a session header for a new session.
pub fn new_header(id: String, cwd: &str, version: u32) -> SessionHeader {
    SessionHeader {
        id,
        timestamp: pi_ai::types::now_millis().to_string(),
        cwd: cwd.to_string(),
        version: Some(version),
        parent_session: None,
    }
}

/// Tool declaration helper (kept for API parity; unused in core format).
pub type ToolDecls = Vec<ToolDef>;

/// Convert a [`Timestamp`] (ms) to an RFC 3339 string.
pub fn timestamp_to_rfc3339(ts: Timestamp) -> String {
    chrono::DateTime::from_timestamp_millis(ts)
        .unwrap_or_default()
        .to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_entry_serializes_like_pi() {
        let entry = FileEntry::Entry(SessionEntry::Message(MessageEntry {
            base: EntryBase {
                id: "e1".into(),
                parent_id: Some("e0".into()),
                timestamp: "2026-01-01T00:00:00Z".into(),
            },
            message: Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::MessageContent::text("hi"),
                timestamp: 1700000000000,
            }),
        }));
        let line = entry_to_line(&entry).unwrap();
        assert!(line.contains("\"type\":\"message\""), "{line}");
        assert!(line.contains("\"parentId\":\"e0\""), "{line}");
        assert!(line.contains("\"role\":\"user\""), "{line}");
        let back: FileEntry = line_to_entry(&line).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn header_serializes_like_pi() {
        let header = FileEntry::Session(SessionHeader {
            id: "s1".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            cwd: "/work".into(),
            version: Some(3),
            parent_session: None,
        });
        let line = entry_to_line(&header).unwrap();
        assert!(line.contains("\"type\":\"session\""), "{line}");
        let back: FileEntry = line_to_entry(&line).unwrap();
        assert_eq!(back, header);
    }
}
