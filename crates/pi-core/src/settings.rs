//! settings.json + models.json support (subset of `settings-manager.ts` and
//! pi's custom provider/model definitions).

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// User settings (`~/.pi-rs/agent/settings.json`, pi-compatible subset).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
    pub append_system_prompt: Option<String>,
    pub compaction_enabled: Option<bool>,
    /// Arbitrary passthrough (pi stores many more fields; we keep them intact).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Custom model definition from `models.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomModel {
    pub id: String,
    pub name: Option<String>,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default = "default_context_window")]
    pub context_window: u64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
}

fn default_context_window() -> u64 {
    128_000
}

fn default_max_tokens() -> u64 {
    8192
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CustomProvider {
    pub base_url: String,
    pub api: Option<String>,
    pub api_key_env: Option<String>,
    pub models: Vec<CustomModel>,
}

fn settings_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        Path::new(&home)
            .join(".pi-rs/agent/settings.json")
    })
}

/// Load settings; falls back to pi's own settings.json for defaults we
/// understand (best-effort).
pub fn load_settings() -> Settings {
    let Some(path) = settings_path() else {
        return Settings::default();
    };
    let Ok(content) = std::fs::read_to_string(path) else {
        return Settings::default();
    };
    serde_json::from_str(&content).unwrap_or_default()
}

fn models_json_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| Path::new(&home).join(".pi-rs/agent/models.json"))
}

/// Load custom providers from `~/.pi-rs/agent/models.json`
/// (also accepts pi's `~/.pi/agent/models.json` when ours is absent).
pub fn load_custom_providers() -> std::collections::HashMap<String, CustomProvider> {
    let mut out = std::collections::HashMap::new();
    let mut paths = Vec::new();
    if let Some(p) = models_json_path() {
        paths.push(p);
    }
    if let Some(home) = std::env::var_os("HOME") {
        paths.push(PathBuf::from(home).join(".pi/agent/models.json"));
    }
    for path in paths {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(map) = serde_json::from_str::<
            std::collections::HashMap<String, CustomProvider>,
        >(&content) else {
            continue;
        };
        for (name, provider) in map {
            out.entry(name).or_insert(provider);
        }
    }
    out
}

/// Resolve a model from custom providers; returns (model, api_key_env).
pub fn resolve_custom_model(
    providers: &std::collections::HashMap<String, CustomProvider>,
    provider: &str,
    model_id: Option<&str>,
) -> Option<(pi_ai::types::Model, Option<String>)> {
    let custom = providers.get(provider)?;
    let model = custom.models.iter().find(|m| {
        model_id.is_none_or(|id| m.id == id)
    })?;
    Some((
        pi_ai::types::Model {
            id: model.id.clone(),
            name: model.name.clone().unwrap_or_else(|| model.id.clone()),
            api: model.api.clone(),
            provider: provider.to_string(),
            base_url: custom.base_url.clone(),
            reasoning: model.reasoning,
            input: vec![pi_ai::types::Modality::Text],
            cost: Default::default(),
            context_window: model.context_window,
            max_tokens: model.max_tokens,
            sampling_params: None,
            headers: None,
            compat: None,
        },
        custom.api_key_env.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_settings_subset() {
        let settings: Settings = serde_json::from_str(
            r#"{"defaultProvider": "kimi-coding", "defaultModel": "kimi-latest",
                "unknownFutureField": {"x": 1}}"#,
        )
        .unwrap();
        assert_eq!(settings.default_provider.as_deref(), Some("kimi-coding"));
        assert!(settings.extra.contains_key("unknownFutureField"));
    }

    #[test]
    fn resolves_custom_model() {
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "my-relay".to_string(),
            CustomProvider {
                base_url: "http://localhost:8000/v1".into(),
                api: Some("openai-completions".into()),
                api_key_env: Some("MY_KEY".into()),
                models: vec![CustomModel {
                    id: "mini".into(),
                    name: None,
                    api: "openai-completions".into(),
                    provider: "my-relay".into(),
                    base_url: "http://localhost:8000/v1".into(),
                    reasoning: false,
                    context_window: 32_000,
                    max_tokens: 4096,
                }],
            },
        );
        let (model, key_env) = resolve_custom_model(&providers, "my-relay", Some("mini")).unwrap();
        assert_eq!(model.id, "mini");
        assert_eq!(model.base_url, "http://localhost:8000/v1");
        assert_eq!(key_env.as_deref(), Some("MY_KEY"));
    }
}
