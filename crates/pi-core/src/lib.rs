//! pi-core: AgentSession — lifecycle, system prompt, compaction, retry.

pub mod agent_session;
pub mod extensions;
pub mod model_catalog;
pub mod model_registry;
pub mod prompt_templates;
pub mod settings;
pub mod skills;
pub mod system_prompt;

pub use agent_session::{AgentSession, SessionOptions, SUMMARIZATION_SYSTEM_PROMPT};
pub use extensions::ExtensionHost;
pub use model_registry::{
    build_api, build_model, provider_defaults, resolve_api_key, PROVIDERS,
};
pub use prompt_templates::{expand_file_references, expand_template, load_prompt_templates};
pub use settings::{load_settings, load_custom_providers, resolve_custom_model, Settings};
pub use skills::{format_skills_for_prompt, load_skills};
pub use system_prompt::{build_system_prompt, load_context_files};
