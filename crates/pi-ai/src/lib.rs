//! pi-ai: Unified multi-provider LLM API.
//!
//! Rust port of `@earendil-works/pi-ai`. Provides message/context types,
//! streaming events, provider abstraction and model catalog.

pub mod types;
pub mod events;
pub mod partial_json;
pub mod api;

pub use api::*;
pub use events::*;
pub use partial_json::*;
pub use types::*;
