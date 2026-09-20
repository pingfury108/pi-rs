//! pi-session: JSONL session persistence with entry tree, branching and compaction.

pub mod format;
pub mod manager;

pub use format::*;
pub use manager::{entry_to_context_messages, SessionManager};
