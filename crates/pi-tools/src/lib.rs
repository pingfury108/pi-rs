//! pi-tools: Built-in coding agent tools (read/bash/edit/write/grep/find/ls).
//!
//! Port of `coding-agent/src/core/tools/*`. Tool implementations follow pi's
//! semantics: truncation limits (2000 lines / 50KB), actionable continuation
//! notices, per-path mutation serialization and .gitignore-aware search.

pub mod bash;
pub mod diff;
pub mod edit;
pub mod edit_diff;
pub mod find;
pub mod grep;
pub mod ls;
pub mod mutation_queue;
pub mod path_utils;
pub mod read;
pub mod truncate;
pub mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use find::FindTool;
pub use grep::GrepTool;
pub use ls::LsTool;
pub use read::ReadTool;
pub use truncate::{format_size, truncate_head, truncate_line, truncate_tail};
pub use write::WriteTool;

use std::sync::Arc;

/// Register all built-in tools on an agent against the given cwd.
pub fn register_all(agent: &pi_agent::Agent, cwd: &std::path::Path) {
    agent.add_tool(Arc::new(ReadTool::new(cwd)));
    agent.add_tool(Arc::new(BashTool::new(cwd)));
    agent.add_tool(Arc::new(EditTool::new(cwd)));
    agent.add_tool(Arc::new(WriteTool::new(cwd)));
    agent.add_tool(Arc::new(GrepTool::new(cwd)));
    agent.add_tool(Arc::new(FindTool::new(cwd)));
    agent.add_tool(Arc::new(LsTool::new(cwd)));
}
