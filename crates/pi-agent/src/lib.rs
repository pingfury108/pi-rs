//! pi-agent: Agent runtime with tool execution and event streaming.
//!
//! Rust port of `@earendil-works/pi-agent-core`.

pub mod agent;
pub mod loop_;
pub mod types;

pub use agent::{convert_to_llm_default, Agent, AgentBuilder};
pub use loop_::{declare_tool_changes, run_agent_loop, run_agent_loop_continue};
pub use types::*;
