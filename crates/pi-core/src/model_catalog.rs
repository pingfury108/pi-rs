//! Model catalog: hydrate from models.dev with a local cache, falling back to
//! the built-in static registry (pi's `hydrate:model-data` flow, simplified).

use std::collections::BTreeMap;
use std::path::PathBuf;

use pi_ai::types::{Model, ModelCost, Modality};

use crate::model_registry::PROVIDERS;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const CACHE_TTL_SECS: u64 = 24 * 60 * 60;

fn cache_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        Path::new(&home)
            .join(".pi-rs/agent/models.dev.json")
    })
}

/// models.dev entry (subset of fields we consume).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelsDevProvider {
    pub id: String,
    pub name: String,
    pub env: Option<Vec<String>>,
    pub npm: Option<String>,
    pub models: BTreeMap<String, ModelsDevModel>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelsDevModel {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: bool,
    pub context_window: u64,
    pub max_output_tokens: Option<u64>,
    pub cost: Option<ModelsDevCost>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct ModelsDevCost {
    pub input: Option<f64>,
    pub output: Option<f64>,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
}

type Catalog = BTreeMap<String, ModelsDevProvider>;

/// Load the catalog: cache if fresh, else fetch (best-effort), else empty.
pub async fn load_catalog() -> Catalog {
    if let Some(cached) = read_cache() {
        return cached;
    }
    match fetch_catalog().await {
        Ok(catalog) => {
            write_cache(&catalog);
            catalog
        }
        Err(e) => {
            tracing::warn!("models.dev fetch failed: {e}");
            BTreeMap::new()
        }
    }
}

fn read_cache() -> Option<Catalog> {
    let path = cache_path()?;
    let meta = std::fs::metadata(&path).ok()?;
    // future mtime (clock skew) counts as fresh
    let age = meta
        .modified()
        .ok()
        .and_then(|m| std::time::SystemTime::now().duration_since(m).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if age > CACHE_TTL_SECS {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn write_cache(catalog: &Catalog) {
    if let Some(path) = cache_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string(catalog) {
            let _ = std::fs::write(path, json);
        }
    }
}

async fn fetch_catalog() -> Result<Catalog, String> {
    tracing::info!("hydrating model catalog from models.dev");
    let body = reqwest::Client::new()
        .get(MODELS_DEV_URL)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    serde_json::from_str(&body).map_err(|e| e.to_string())
}

/// Map a models.dev provider id to our known api + base url via the built-in
/// registry; models.dev `npm` field hints the SDK family.
fn infer_api(provider_id: &str, dev: &ModelsDevProvider) -> Option<(&'static str, &'static str)> {
    let _ = provider_id;
    // npm SDK hints from models.dev
    match dev.npm.as_deref() {
        Some("@ai-sdk/anthropic") => Some(("anthropic-messages", "https://api.anthropic.com")),
        Some("@ai-sdk/openai-compatible") | Some("@ai-sdk/openai") => {
            Some(("openai-completions", ""))
        }
        _ => {
            // fall back to our static registry by id
            PROVIDERS
                .iter()
                .find(|p| p.provider == dev.id)
                .map(|p| (p.api, p.base_url))
        }
    }
}

/// Convert a models.dev entry into our Model. Returns None when the provider
/// family is unknown (e.g. bedrock without credentials config).
pub fn convert_dev_model(dev_provider: &ModelsDevProvider, model: &ModelsDevModel) -> Option<Model> {
    let (api, default_base) = infer_api(&dev_provider.id, dev_provider)?;
    let base_url = if default_base.is_empty() {
        // openai-compatible providers carry their own base url in models.dev;
        // we cannot know it, so skip unless static registry knows it
        PROVIDERS
            .iter()
            .find(|p| p.provider == dev_provider.id)
            .map(|p| p.base_url)?
    } else {
        default_base
    };
    let cost = model.cost.clone().unwrap_or_default();
    Some(Model {
        id: model.id.clone(),
        name: model.name.clone().unwrap_or_else(|| model.id.clone()),
        api: api.into(),
        provider: dev_provider.id.clone(),
        base_url: base_url.into(),
        reasoning: model.reasoning,
        input: vec![Modality::Text],
        cost: ModelCost {
            input: cost.input.unwrap_or(0.0),
            output: cost.output.unwrap_or(0.0),
            cache_read: cost.cache_read.unwrap_or(0.0),
            cache_write: cost.cache_write.unwrap_or(0.0),
        },
        context_window: model.context_window.max(4096),
        max_tokens: model.max_output_tokens.unwrap_or(8192),
        sampling_params: None,
        headers: None,
        compat: None,
    })
}

/// Look up a model in the hydrated catalog.
pub fn catalog_lookup(catalog: &Catalog, provider: &str, model_id: &str) -> Option<Model> {
    let dev = catalog.get(provider)?;
    let model = if model_id.contains('*') {
        // simple glob: first match
        let pattern = globset::Glob::new(model_id).ok()?.compile_matcher();
        dev.models
            .values()
            .find(|m| pattern.is_match(&m.id) || pattern.is_match(&m.name.clone().unwrap_or_default()))?
    } else {
        dev.models.get(model_id)?
    };
    convert_dev_model(dev, model)
}

/// List "provider/model" ids matching an optional glob (for --list-models).
pub fn catalog_list(catalog: &Catalog, pattern: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let matcher = pattern.and_then(|p| globset::Glob::new(p).ok()).map(|g| g.compile_matcher());
    for provider in catalog.values() {
        for model in provider.models.values() {
            let full = format!("{}/{}", provider.id, model.id);
            let matches = match &matcher {
                Some(m) => m.is_match(&full) || m.is_match(&model.id),
                None => true,
            };
            if matches {
                out.push(full);
            }
        }
    }
    out.sort();
    out
}

use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_anthropic_entry() {
        let provider = ModelsDevProvider {
            id: "anthropic".into(),
            name: "Anthropic".into(),
            env: None,
            npm: Some("@ai-sdk/anthropic".into()),
            models: BTreeMap::from([(
                "claude-sonnet-4-5".into(),
                ModelsDevModel {
                    id: "claude-sonnet-4-5".into(),
                    name: Some("Claude Sonnet 4.5".into()),
                    reasoning: true,
                    context_window: 200_000,
                    max_output_tokens: Some(64_000),
                    cost: Some(ModelsDevCost {
                        input: Some(3.0),
                        output: Some(15.0),
                        cache_read: Some(0.3),
                        cache_write: Some(3.75),
                    }),
                },
            )]),
        };
        let model = convert_dev_model(&provider, &provider.models["claude-sonnet-4-5"]).unwrap();
        assert_eq!(model.api, "anthropic-messages");
        assert_eq!(model.base_url, "https://api.anthropic.com");
        assert_eq!(model.cost.input, 3.0);
        assert_eq!(model.context_window, 200_000);
    }

    #[test]
    fn catalog_lookup_and_list() {
        let catalog = Catalog::from([(
            "openai".into(),
            ModelsDevProvider {
                id: "openai".into(),
                name: "OpenAI".into(),
                env: None,
                npm: Some("@ai-sdk/openai".into()),
                models: BTreeMap::from([(
                    "gpt-5".into(),
                    ModelsDevModel {
                        id: "gpt-5".into(),
                        name: None,
                        reasoning: true,
                        context_window: 400_000,
                        max_output_tokens: None,
                        cost: None,
                    },
                )]),
            },
        )]);
        assert!(catalog_lookup(&catalog, "openai", "gpt-5").is_some());
        let list = catalog_list(&catalog, Some("openai/*"));
        assert_eq!(list, ["openai/gpt-5"]);
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    #[ignore = "requires local models.dev cache; run with --ignored"]
    fn parses_local_cache() {
        let path = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".pi-rs/agent/models.dev.json");
        let content = std::fs::read_to_string(path).unwrap();
        match serde_json::from_str::<Catalog>(&content) {
            Ok(c) => println!("ok, providers: {}", c.len()),
            Err(e) => panic!("parse error: {e}"),
        }
    }
}
