//! Extension host: rhai-scripted extensions, the Rust counterpart of pi's
//! TypeScript extension system (subset).
//!
//! Script locations: `~/.pi-rs/agent/extensions/*.rhai` and
//! `.pi/extensions/*.rhai` (project).
//!
//! Conventions (script-defined functions, all optional):
//! - `fn on_event(event_json)` — receives agent lifecycle events as a JSON
//!   string; return value ignored.
//! - `fn before_tool_call(tool_name, args_json)` — return `#{ "block": true,
//!   "reason": "..." }` to block the call, or `()` to allow.
//! - `fn after_tool_call(tool_name, result_text)` — return a string to
//!   replace the tool result text, or `()` to keep.
//! - `register_tool(name, description, schema_json, execute_fn_name)` —
//!   registers a custom tool whose execute function receives the args JSON
//!   and returns the result text.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pi_agent::{AgentTool, ToolContext, ToolUpdateFn};
use rhai::{AST, Dynamic, Engine};

/// A loaded extension script (shared so scripted tools can call back in).
pub struct Extension {
    pub name: String,
    engine: Engine,
    ast: AST,
    scope: Mutex<rhai::Scope<'static>>,
}

impl Extension {
    fn compile(name: String, source: &str) -> Result<Self, String> {
        let mut engine = Engine::new();
        engine.set_max_operations(1_000_000);
        engine.set_max_string_size(1_000_000);
        let ast = engine
            .compile(source)
            .map_err(|e| format!("compile error: {e}"))?;
        Ok(Self {
            name,
            engine,
            ast,
            scope: Mutex::new(rhai::Scope::new()),
        })
    }

    fn has_fn(&self, name: &str) -> bool {
        self.ast.iter_functions().any(|f| f.name == name)
    }

    fn call0(&self, fn_name: &str) -> Option<Dynamic> {
        let mut scope = self.scope.lock().unwrap();
        self.engine
            .call_fn(&mut scope, &self.ast, fn_name, ())
            .ok()
    }

    fn call2(&self, fn_name: &str, a: String, b: String) -> Option<Dynamic> {
        let mut scope = self.scope.lock().unwrap();
        self.engine
            .call_fn(&mut scope, &self.ast, fn_name, (a, b))
            .ok()
    }
}

/// Tools registered by scripts via `register_tool`.
#[derive(Debug, Clone)]
pub struct RegisteredTool {
    pub name: String,
    pub description: String,
    pub schema_json: String,
    pub execute_fn: String,
}

/// A custom tool backed by a rhai function `<execute_fn>(args_json) -> String`.
pub struct ScriptedTool {
    extension: Arc<Mutex<Extension>>,
    inner: RegisteredTool,
}

impl ScriptedTool {
    fn new(extension: Arc<Mutex<Extension>>, inner: RegisteredTool) -> Arc<Self> {
        Arc::new(Self {
            extension,
            inner,
        })
    }
}

#[async_trait::async_trait]
impl AgentTool for ScriptedTool {
    fn name(&self) -> &str {
        &self.inner.name
    }
    fn description(&self) -> &str {
        &self.inner.description
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::from_str(&self.inner.schema_json)
            .unwrap_or_else(|_| serde_json::json!({"type": "object"}))
    }
    fn label(&self, args: &serde_json::Value) -> String {
        format!("{}({args})", self.inner.name)
    }
    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
        _on_update: ToolUpdateFn,
    ) -> Result<pi_agent::AgentToolResult, String> {
        // rhai execution is sync + quick; keep it off the async thread.
        let extension = self.extension.clone();
        let fn_name = self.inner.execute_fn.clone();
        let args_json = serde_json::to_string(&args).unwrap_or_default();
        let cancelled = ctx.cancel.as_ref().map(|c| c.clone());
        let result = tokio::task::spawn_blocking(move || {
            if let Some(token) = &cancelled {
                if token.is_cancelled() {
                    return Err("Operation aborted".to_string());
                }
            }
            let extension = extension.lock().unwrap();
            extension
                .call2(&fn_name, args_json, String::new())
                .and_then(|d| d.into_string().ok())
                .ok_or_else(|| "extension tool returned no string".to_string())
        })
        .await
        .map_err(|e| format!("extension task failed: {e}"))??;
        Ok(pi_agent::AgentToolResult::text(result))
    }
}

/// Host for all loaded extensions.
pub struct ExtensionHost {
    extensions: Vec<Arc<Mutex<Extension>>>,
    pub tool_registrations: Vec<(Arc<Mutex<Extension>>, RegisteredTool)>,
}

impl Default for ExtensionHost {
    fn default() -> Self {
        Self {
            extensions: Vec::new(),
            tool_registrations: Vec::new(),
        }
    }
}

impl ExtensionHost {
    /// Discover and compile extension scripts from global + project dirs.
    pub fn load(cwd: &Path) -> Self {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            dirs.push(PathBuf::from(&home).join(".pi-rs/agent/extensions"));
            dirs.push(PathBuf::from(home).join(".pi/agent/extensions"));
        }
        dirs.push(cwd.join(".pi/extensions"));

        let mut extensions: Vec<Arc<Mutex<Extension>>> = Vec::new();
        let mut registrations = Vec::new();

        for dir in dirs {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e != "rhai") {
                    continue;
                }
                let Ok(source) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let mut extension = match Extension::compile(name.clone(), &source) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!("extension {name}: {e}");
                        continue;
                    }
                };

                // expose register_tool during top-level evaluation
                let sink: Arc<Mutex<Vec<RegisteredTool>>> = Arc::default();
                let sink2 = sink.clone();
                extension.engine.register_fn(
                    "register_tool",
                    move |name: &str, description: &str, schema: &str, execute_fn: &str| {
                        sink2.lock().unwrap().push(RegisteredTool {
                            name: name.to_string(),
                            description: description.to_string(),
                            schema_json: schema.to_string(),
                            execute_fn: execute_fn.to_string(),
                        });
                    },
                );

                let mut scope = extension.scope.lock().unwrap();
                if let Err(e) = extension.engine.run_ast_with_scope(&mut scope, &extension.ast) {
                    tracing::warn!("extension {name} init failed: {e}");
                    continue;
                }
                drop(scope);

                let shared = Arc::new(Mutex::new(extension));
                for tool in sink.lock().unwrap().drain(..) {
                    registrations.push((shared.clone(), tool));
                }
                tracing::info!("loaded extension: {name}");
                extensions.push(shared);
            }
        }

        Self {
            extensions,
            tool_registrations: registrations,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }

    /// Instantiate scripted tools registered by the loaded extensions.
    pub fn build_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.tool_registrations
            .iter()
            .map(|(extension, inner)| ScriptedTool::new(extension.clone(), inner.clone()))
            .map(|tool| tool as Arc<dyn AgentTool>)
            .collect()
    }

    /// Fan an event out to all extensions defining `on_event`.
    pub fn emit_event(&self, event_json: &str) {
        for extension in &self.extensions {
            let extension = extension.lock().unwrap();
            if extension.has_fn("on_event") {
                extension.call2("on_event", event_json.to_string(), String::new());
            }
        }
    }

    /// Query extensions for a block decision (first blocking result wins).
    pub fn before_tool_call(&self, tool_name: &str, args_json: &str) -> Option<String> {
        for extension in &self.extensions {
            let extension = extension.lock().unwrap();
            if extension.has_fn("before_tool_call") {
                if let Some(result) =
                    extension.call2("before_tool_call", tool_name.into(), args_json.into())
                {
                    // result is a rhai Map: #{ "block": true, "reason": "..." }
                    let map = result.try_cast::<rhai::Map>().unwrap_or_default();
                    let block = map
                        .get("block")
                        .and_then(|b| b.as_bool().ok())
                        .unwrap_or(false);
                    if block {
                        let reason = map
                            .get("reason")
                            .and_then(|r| r.clone().into_string().ok())
                            .unwrap_or_else(|| "blocked by extension".into());
                        return Some(reason);
                    }
                }
            }
        }
        None
    }

    /// Query extensions for a result override (first string result wins).
    pub fn after_tool_call(&self, tool_name: &str, result_text: &str) -> Option<String> {
        for extension in &self.extensions {
            let extension = extension.lock().unwrap();
            if extension.has_fn("after_tool_call") {
                if let Some(result) =
                    extension.call2("after_tool_call", tool_name.into(), result_text.into())
                {
                    if let Ok(text) = result.into_string() {
                        return Some(text);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_extension(source: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ext-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test-ext.rhai");
        std::fs::write(&path, source).unwrap();
        dir
    }

    #[test]
    fn loads_and_registers_tools() {
        let dir = write_extension(
            r#"
            register_tool("greet", "Greets a person", `{"type":"object"}`, "greet_exec");
            fn greet_exec(args_json) {
                let name = "world";
                `Hello, ` + name + `!`;
            }
            "#,
        );
        // write into a cwd-scoped location the loader scans
        let cwd = dir.clone();
        let host = ExtensionHost::load(&cwd);
        // loader only scans <cwd>/.pi/extensions; move file there
        let target = cwd.join(".pi/extensions");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::rename(dir.join("test-ext.rhai"), target.join("test-ext.rhai")).unwrap();
        let host = ExtensionHost::load(&cwd);
        assert!(!host.is_empty());
        let tools = host.build_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "greet");
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn blocks_tool_calls() {
        let cwd = std::env::temp_dir().join(format!("ext-{}", uuid::Uuid::now_v7()));
        let target = cwd.join(".pi/extensions");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(
            target.join("guard.rhai"),
            r#"
            fn before_tool_call(tool_name, args_json) {
                if tool_name == "bash" {
                    #{ "block": true, "reason": "no shell for you" }
                } else {
                    ()
                }
            }
            "#,
        )
        .unwrap();

        let host = ExtensionHost::load(&cwd);
        let blocked = host.before_tool_call("bash", "{}");
        assert_eq!(blocked.as_deref(), Some("no shell for you"));
        assert!(host.before_tool_call("read", "{}").is_none());
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn rewrites_tool_results() {
        let cwd = std::env::temp_dir().join(format!("ext-{}", uuid::Uuid::now_v7()));
        let target = cwd.join(".pi/extensions");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(
            target.join("redact.rhai"),
            r#"
            fn after_tool_call(tool_name, result_text) {
                `REDACTED(len=` + result_text.len() + `)`;
            }
            "#,
        )
        .unwrap();

        let host = ExtensionHost::load(&cwd);
        let rewritten = host.after_tool_call("read", "secret content");
        assert_eq!(rewritten.as_deref(), Some("REDACTED(len=14)"));
        std::fs::remove_dir_all(&cwd).ok();
    }
}
