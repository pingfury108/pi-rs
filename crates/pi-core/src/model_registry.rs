//! Minimal model registry: provider defaults + env key resolution.
//! (pi's full generated catalog is Phase 2 backlog; this covers common
//! providers for end-to-end use.)

use pi_ai::types::{Model, ModelCost, Modality};

pub struct ProviderDefaults {
    pub provider: &'static str,
    pub api: &'static str,
    pub base_url: &'static str,
    pub api_key_env: &'static str,
}

/// Built-in provider defaults.
pub const PROVIDERS: &[ProviderDefaults] = &[
    ProviderDefaults {
        provider: "anthropic",
        api: "anthropic-messages",
        base_url: "https://api.anthropic.com",
        api_key_env: "ANTHROPIC_API_KEY",
    },
    ProviderDefaults {
        provider: "openai",
        api: "openai-completions",
        base_url: "https://api.openai.com/v1",
        api_key_env: "OPENAI_API_KEY",
    },
    ProviderDefaults {
        provider: "deepseek",
        api: "openai-completions",
        base_url: "https://api.deepseek.com",
        api_key_env: "DEEPSEEK_API_KEY",
    },
    ProviderDefaults {
        provider: "openrouter",
        api: "openai-completions",
        base_url: "https://openrouter.ai/api/v1",
        api_key_env: "OPENROUTER_API_KEY",
    },
    ProviderDefaults {
        provider: "moonshotai",
        api: "openai-completions",
        base_url: "https://api.moonshot.ai/v1",
        api_key_env: "MOONSHOT_API_KEY",
    },
    ProviderDefaults {
        provider: "kimi-coding",
        api: "anthropic-messages",
        base_url: "https://api.kimi.com/coding",
        api_key_env: "KIMI_API_KEY",
    },
    ProviderDefaults {
        provider: "groq",
        api: "openai-completions",
        base_url: "https://api.groq.com/openai/v1",
        api_key_env: "GROQ_API_KEY",
    },
    ProviderDefaults {
        provider: "xai",
        api: "openai-completions",
        base_url: "https://api.x.ai/v1",
        api_key_env: "XAI_API_KEY",
    },
];

/// Resolve a provider's defaults by id.
pub fn provider_defaults(provider: &str) -> Option<&'static ProviderDefaults> {
    PROVIDERS.iter().find(|p| p.provider == provider)
}

/// Construct a model entry from CLI options. `model_id` defaults to a
/// provider-known default.
pub fn build_model(
    provider: &str,
    model_id: Option<&str>,
    base_url_override: Option<&str>,
) -> Result<Model, String> {
    let defaults = provider_defaults(provider)
        .ok_or_else(|| format!("unknown provider {provider}; known: {}", known_providers()))?;
    let model_id = model_id.unwrap_or(match defaults.provider {
        "anthropic" => "claude-sonnet-4-5",
        "openai" => "gpt-5",
        "deepseek" => "deepseek-chat",
        "openrouter" => "anthropic/claude-sonnet-4.5",
        "moonshotai" => "kimi-k2-0905-preview",
        "kimi-coding" => "kimi-latest",
        "groq" => "llama-3.3-70b-versatile",
        "xai" => "grok-4",
        _ => "default",
    });
    Ok(Model {
        id: model_id.into(),
        name: model_id.into(),
        api: defaults.api.into(),
        provider: defaults.provider.into(),
        base_url: base_url_override.unwrap_or(defaults.base_url).into(),
        reasoning: true,
        input: vec![Modality::Text, Modality::Image],
        cost: ModelCost::default(),
        context_window: 200_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    })
}

/// Resolve the API key: explicit option → provider env var → auth.json
/// (`~/.pi-rs/agent/auth.json`, pi-compatible `{"provider": {"key": ...}}`).
pub fn resolve_api_key(provider: &str, explicit: Option<&str>) -> Option<String> {
    if let Some(key) = explicit {
        return Some(key.to_string());
    }
    if let Ok(key) = std::env::var(format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"))) {
        if !key.is_empty() {
            return Some(key);
        }
    }
    if let Some(defaults) = provider_defaults(provider) {
        if let Ok(key) = std::env::var(defaults.api_key_env) {
            if !key.is_empty() {
                return Some(key);
            }
        }
    }
    read_auth_json(provider)
}

fn read_auth_json(provider: &str) -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let path = std::path::Path::new(&home).join(".pi-rs/agent/auth.json");
    let content = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&content).ok()?;
    let entry = value.get(provider)?;
    entry
        .get("key")
        .or_else(|| entry.get("api_key"))
        .and_then(|k| k.as_str())
        .map(str::to_string)
        .filter(|k| !k.is_empty())
}

/// Construct the LlmApi implementation for an api identifier.
pub fn build_api(api: &str) -> Result<std::sync::Arc<dyn pi_ai::api::LlmApi>, String> {
    match api {
        "anthropic-messages" => Ok(std::sync::Arc::new(pi_ai::api::AnthropicApi::new())),
        "openai-completions" => Ok(std::sync::Arc::new(pi_ai::api::OpenAICompletionsApi::new())),
        other => Err(format!("unsupported api: {other}")),
    }
}

fn known_providers() -> String {
    PROVIDERS
        .iter()
        .map(|p| p.provider)
        .collect::<Vec<_>>()
        .join(", ")
}
