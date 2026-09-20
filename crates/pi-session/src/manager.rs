//! Session manager: append/branch/read over a JSONL session file.
//! Port of the core (non-UI) parts of `session-manager.ts`.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use pi_agent::AgentMessage;
use pi_ai::types::Message;

use crate::format::{
    entry_to_line, line_to_entry, timestamp_to_rfc3339, EntryBase, FileEntry, MessageEntry,
    SessionEntry, SessionHeader, CURRENT_SESSION_VERSION,
};

/// Read-write handle over one session file.
pub struct SessionManager {
    file: PathBuf,
    header: SessionHeader,
    entries: Vec<SessionEntry>,
    /// Last entry id (tree leaf).
    leaf_id: Option<String>,
}

impl SessionManager {
    /// Create a new session file.
    pub fn create(dir: &Path, cwd: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let id = uuid::Uuid::now_v7().to_string();
        let header = SessionHeader {
            id: id.clone(),
            timestamp: timestamp_to_rfc3339(pi_ai::types::now_millis()),
            cwd: cwd.to_string_lossy().into_owned(),
            version: Some(CURRENT_SESSION_VERSION),
            parent_session: None,
        };
        let file = dir.join(format!("{}.jsonl", id));
        let mut f = std::fs::File::create(&file)?;
        writeln!(f, "{}", entry_to_line(&FileEntry::Session(header.clone()))?)?;

        Ok(Self {
            file,
            header,
            entries: Vec::new(),
            leaf_id: None,
        })
    }

    /// Open an existing session file.
    pub fn open(file: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(file)?;
        let mut header: Option<SessionHeader> = None;
        let mut entries = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match line_to_entry(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            {
                FileEntry::Session(h) => header = Some(h),
                FileEntry::Entry(e) => entries.push(e),
            }
        }

        let header = header.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing session header")
        })?;
        let leaf_id = entries.last().map(|e| e.id().to_string());

        Ok(Self {
            file: file.to_path_buf(),
            header,
            entries,
            leaf_id,
        })
    }

    pub fn file(&self) -> &Path {
        &self.file
    }

    pub fn id(&self) -> &str {
        &self.header.id
    }

    pub fn header(&self) -> &SessionHeader {
        &self.header
    }

    pub fn entries(&self) -> &[SessionEntry] {
        &self.entries
    }

    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn set_leaf_id(&mut self, leaf_id: Option<String>) {
        self.leaf_id = leaf_id;
    }

    /// Append an entry at the current leaf (branching: parentId = leaf).
    pub fn append(&mut self, mut entry: SessionEntry) -> std::io::Result<String> {
        entry.set_parent(self.leaf_id.clone());
        let id = entry.id().to_string();
        let f = std::fs::OpenOptions::new().append(true).open(&self.file)?;
        let mut f = f;
        writeln!(f, "{}", entry_to_line(&FileEntry::Entry(entry.clone()))?)?;
        self.entries.push(entry);
        self.leaf_id = Some(id);
        Ok(self.leaf_id.clone().unwrap())
    }

    /// Append an agent message.
    pub fn append_message(&mut self, message: AgentMessage) -> std::io::Result<String> {
        let entry = SessionEntry::Message(MessageEntry {
            base: self.new_base(),
            message: agent_to_llm_message(message),
        });
        self.append(entry)
    }

    fn new_base(&self) -> EntryBase {
        EntryBase {
            id: uuid::Uuid::now_v7().to_string(),
            parent_id: None, // set by append()
            timestamp: timestamp_to_rfc3339(pi_ai::types::now_millis()),
        }
    }

    /// Path from root to the given leaf (defaults to the current leaf).
    pub fn build_session_path(&self, leaf_id: Option<&str>) -> Vec<&SessionEntry> {
        let index: std::collections::HashMap<&str, &SessionEntry> = self
            .entries
            .iter()
            .map(|e| (e.id(), e))
            .collect();

        let leaf = leaf_id
            .and_then(|id| index.get(id).copied())
            .or_else(|| self.entries.last());

        let Some(mut current) = leaf else {
            return Vec::new();
        };

        let mut path = vec![current];
        while let Some(parent_id) = current.parent_id() {
            match index.get(parent_id) {
                Some(parent) => {
                    path.push(parent);
                    current = parent;
                }
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Context entries for the LLM: session path with the most recent
    /// compaction applied (pi's `buildContextEntries`).
    pub fn build_context_entries(&self, leaf_id: Option<&str>) -> Vec<&SessionEntry> {
        let path = self.build_session_path(leaf_id);
        let Some(compaction) = path.iter().rev().find(|e| matches!(e, SessionEntry::Compaction(_)))
        else {
            return path;
        };
        let compaction_idx = path.iter().position(|e| e.id() == compaction.id()).unwrap();

        let compaction_entry = match compaction {
            SessionEntry::Compaction(c) => c,
            _ => unreachable!(),
        };

        let mut context_entries: Vec<&SessionEntry> = vec![compaction];
        let mut found_first_kept = false;
        for entry in &path[..compaction_idx] {
            if entry.id() == compaction_entry.first_kept_entry_id {
                found_first_kept = true;
            }
            if found_first_kept && !matches!(entry, SessionEntry::Message(MessageEntry { message: Message::System(_), .. }))
            {
                context_entries.push(entry);
            }
        }
        context_entries.extend(&path[compaction_idx + 1..]);
        context_entries
    }

    /// Resolved agent messages for the LLM (pi's `sessionEntryToContextMessages`
    /// applied over `buildContextEntries`).
    pub fn build_context_messages(&self, leaf_id: Option<&str>) -> Vec<AgentMessage> {
        self.build_context_entries(leaf_id)
            .iter()
            .flat_map(|e| entry_to_context_messages(e))
            .collect()
    }

    /// Resolved (thinkingLevel, model) settings along the path
    /// (pi's `getSessionContextSettings`).
    pub fn context_settings(&self, leaf_id: Option<&str>) -> (String, Option<(String, String)>) {
        let mut thinking_level = "off".to_string();
        let mut model: Option<(String, String)> = None;
        for entry in self.build_session_path(leaf_id) {
            match entry {
                SessionEntry::ThinkingLevelChange(e) => thinking_level = e.thinking_level.clone(),
                SessionEntry::ModelChange(e) => {
                    model = Some((e.provider.clone(), e.model_id.clone()))
                }
                SessionEntry::Message(MessageEntry {
                    message: Message::Assistant(a),
                    ..
                }) => model = Some((a.provider.clone(), a.model.clone())),
                _ => {}
            }
        }
        (thinking_level, model)
    }

    /// Branch from an earlier entry: subsequent appends extend that node.
    pub fn branch(&mut self, from_id: Option<String>) {
        self.leaf_id = from_id;
    }
}

/// Normalize an [`AgentMessage`] into its LLM [`Message`] representation for
/// session storage (session files store LLM-shaped messages).
fn agent_to_llm_message(message: AgentMessage) -> Message {
    match message {
        AgentMessage::Message(m) => m,
        AgentMessage::Custom(custom) => Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::MessageContent::text(format!(
                "[{}] {}",
                custom.custom_type, custom.content
            )),
            timestamp: custom.timestamp,
        }),
    }
}

/// Project one entry into agent messages (pi's `sessionEntryToContextMessages`).
pub fn entry_to_context_messages(entry: &SessionEntry) -> Vec<AgentMessage> {
    match entry {
        SessionEntry::Message(m) => vec![AgentMessage::Message(m.message.clone())],
        SessionEntry::CustomMessage(cm) => {
            let text = match &cm.content {
                crate::format::CustomMessageContent::Text(t) => t.clone(),
                crate::format::CustomMessageContent::Blocks(_) => String::new(),
            };
            vec![AgentMessage::Custom(pi_agent::CustomAgentMessage {
                role: "custom".into(),
                custom_type: cm.custom_type.clone(),
                content: text,
                details: cm.details.clone(),
                timestamp: pi_ai::types::now_millis(),
            })]
        }
        SessionEntry::BranchSummary(bs) => {
            vec![AgentMessage::Custom(pi_agent::CustomAgentMessage {
                role: "custom".into(),
                custom_type: "branch_summary".into(),
                content: format!(
                    "This session was continued from a previous conversation. Summary of the prior conversation:\n\n{}",
                    bs.summary
                ),
                details: Some(serde_json::json!({ "fromId": bs.from_id })),
                timestamp: pi_ai::types::now_millis(),
            })]
        }
        SessionEntry::Compaction(c) => {
            let mut out = Vec::new();
            if let Some(system) = &c.system_message {
                out.push(AgentMessage::Message(Message::System(system.clone())));
            }
            out.push(AgentMessage::Custom(pi_agent::CustomAgentMessage {
                role: "custom".into(),
                custom_type: "compaction_summary".into(),
                content: format!(
                    "[Conversation summary for earlier messages ({} tokens before compaction)]\n\n{}",
                    c.tokens_before, c.summary
                ),
                details: None,
                timestamp: pi_ai::types::now_millis(),
            }));
            out
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::CustomMessageContent;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pi-rs-session-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_msg(text: &str) -> AgentMessage {
        AgentMessage::Message(Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::MessageContent::text(text),
            timestamp: pi_ai::types::now_millis(),
        }))
    }

    #[test]
    fn create_append_and_reload() {
        let dir = tmp_dir();
        let mut manager = SessionManager::create(&dir, Path::new("/work")).unwrap();

        manager.append_message(user_msg("hello")).unwrap();
        manager.append_message(user_msg("world")).unwrap();

        assert_eq!(manager.entries().len(), 2);
        let reloaded = SessionManager::open(manager.file()).unwrap();
        assert_eq!(reloaded.entries().len(), 2);
        assert_eq!(reloaded.id(), manager.id());
        assert_eq!(reloaded.header().cwd, "/work");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn branching_and_context() {
        let dir = tmp_dir();
        let mut manager = SessionManager::create(&dir, Path::new("/work")).unwrap();
        manager.append_message(user_msg("a")).unwrap();
        let first_leaf = manager.leaf_id().unwrap().to_string();
        manager.append_message(user_msg("b1")).unwrap();

        // branch from first_leaf: path = [a, b2]
        manager.branch(Some(first_leaf.clone()));
        manager.append_message(user_msg("b2")).unwrap();

        let path = manager.build_session_path(None);
        let texts: Vec<String> = path
            .iter()
            .filter_map(|e| match e {
                SessionEntry::Message(m) => match &m.message {
                    Message::User(u) => Some(u.content.as_text()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["a", "b2"]);

        // old leaf path still reachable
        let old_path = manager.build_session_path(Some(&first_leaf));
        assert_eq!(old_path.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn custom_message_participates_in_context() {
        let dir = tmp_dir();
        let mut manager = SessionManager::create(&dir, Path::new("/work")).unwrap();
        manager
            .append(SessionEntry::CustomMessage(crate::format::CustomMessageEntry {
                base: manager_new_base(&manager),
                custom_type: "note".into(),
                content: CustomMessageContent::Text("remember this".into()),
                details: None,
                display: true,
            }))
            .unwrap();

        let messages = manager.build_context_messages(None);
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            AgentMessage::Custom(c) => assert_eq!(c.content, "remember this"),
            other => panic!("expected custom message, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn manager_new_base(manager: &SessionManager) -> EntryBase {
        EntryBase {
            id: uuid::Uuid::now_v7().to_string(),
            parent_id: None,
            timestamp: timestamp_to_rfc3339(pi_ai::types::now_millis()),
        }
    }
}
