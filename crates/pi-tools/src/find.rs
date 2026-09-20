//! find tool, port of `core/tools/find.ts`.
//!
//! Uses `ignore` + `globset` (the fd/ripgrep core) instead of the `fd` binary.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use serde_json::json;

use crate::truncate::{format_size, truncate_head, DEFAULT_MAX_BYTES};

const DEFAULT_LIMIT: usize = 1000;

pub struct FindTool {
    cwd: PathBuf,
}

impl FindTool {
    pub fn new(cwd: &Path) -> Self {
        Self { cwd: cwd.to_path_buf() }
    }
}

#[async_trait]
impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files by glob pattern (respects .gitignore)"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"},
                "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
            },
            "required": ["pattern"],
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("find {}", args["pattern"].as_str().unwrap_or(""))
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
        _on_update: ToolUpdateFn,
    ) -> Result<AgentToolResult, String> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| "missing required argument: pattern".to_string())?;
        let search_dir = args["path"].as_str().unwrap_or(".");
        let effective_limit = (args["limit"].as_u64().unwrap_or(DEFAULT_LIMIT as u64) as usize).max(1);

        let search_path = crate::path_utils::resolve_to_cwd(search_dir, &self.cwd);
        if !search_path.exists() {
            return Err(format!("Path not found: {}", search_path.display()));
        }

        let glob = globset::GlobBuilder::new(pattern)
            .literal_separator(pattern.contains("**"))
            .build()
            .map_err(|e| format!("Invalid glob: {e}"))?
            .compile_matcher();

        let mut walker = ignore::WalkBuilder::new(&search_path);
        walker.hidden(true);
        if let Some(token) = ctx.cancel.as_ref() {
            let token = token.clone();
            walker.filter_entry(move |_| !token.is_cancelled());
        }

        let mut results: Vec<String> = Vec::new();
        let mut limit_reached = false;
        for entry in walker.build().flatten() {
            let path = entry.path();
            if results.len() >= effective_limit {
                limit_reached = true;
                break;
            }
            let rel = path.strip_prefix(&search_path).unwrap_or(path);
            let rel_str = crate::path_utils::to_posix(rel);
            if rel_str.is_empty() {
                continue;
            }
            let is_dir = path.is_dir();
            let candidate = if is_dir { format!("{rel_str}/") } else { rel_str.clone() };
            if glob.is_match(&rel_str) || glob.is_match(&candidate) || basename_matches(&glob, path) {
                results.push(candidate);
            }
        }

        if results.is_empty() {
            return Ok(AgentToolResult::text("No files found"));
        }

        let truncation =
            truncate_head(&results.join("\n"), crate::truncate::TruncationOptions {
                max_lines: usize::MAX,
                ..Default::default()
            });
        let mut output = truncation.content;
        let mut notices: Vec<String> = Vec::new();
        if limit_reached {
            notices.push(format!(
                "{effective_limit} results limit reached. Use limit={} for more",
                effective_limit * 2
            ));
        }
        if truncation.truncated {
            notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
        }
        if !notices.is_empty() {
            output.push_str(&format!("\n\n[{}]", notices.join(". ")));
        }

        Ok(AgentToolResult {
            content: vec![ToolResultContent::Text(TextContent {
                text: output,
                text_signature: None,
            })],
            details: serde_json::Value::Null,
            usage: None,
            terminate: false,
        })
    }
}

fn basename_matches(glob: &globset::GlobMatcher, path: &Path) -> bool {
    // Patterns like "*.ts" should match at any depth (fd behavior).
    path.file_name()
        .map(|n| glob.is_match(n.to_string_lossy().as_ref()))
        .unwrap_or(false)
}
