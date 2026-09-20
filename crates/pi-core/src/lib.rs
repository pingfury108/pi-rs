//! pi-core: AgentSession — lifecycle, system prompt, compaction, retry.

pub mod agent_session;
pub mod model_registry;
pub mod system_prompt;

pub use agent_session::{AgentSession, SessionOptions, SUMMARIZATION_SYSTEM_PROMPT};
pub use model_registry::{
    build_api, build_model, provider_defaults, resolve_api_key, PROVIDERS,
};
pub use system_prompt::{build_system_prompt, load_context_files};
