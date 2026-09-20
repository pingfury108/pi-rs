//! ls tool, port of `core/tools/ls.ts`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use serde_json::json;

use crate::truncate::{format_size, truncate_head, DEFAULT_MAX_BYTES};

const DEFAULT_LIMIT: usize = 500;

pub struct LsTool {
    cwd: PathBuf,
}

impl LsTool {
    pub fn new(cwd: &Path) -> Self {
        Self { cwd: cwd.to_path_buf() }
    }
}

#[async_trait]
impl AgentTool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first)."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
            },
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("ls {}", args["path"].as_str().unwrap_or(""))
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        _ctx: &ToolContext,
        _on_update: ToolUpdateFn,
    ) -> Result<AgentToolResult, String> {
        let dir_arg = args["path"].as_str().unwrap_or(".");
        let effective_limit = (args["limit"].as_u64().unwrap_or(DEFAULT_LIMIT as u64) as usize).max(1);

        let dir_path = crate::path_utils::resolve_to_cwd(dir_arg, &self.cwd);
        if !dir_path.exists() {
            return Err(format!("Path not found: {}", dir_path.display()));
        }
        if !dir_path.is_dir() {
            return Err(format!("Not a directory: {}", dir_path.display()));
        }

        let mut reader = tokio::fs::read_dir(&dir_path)
            .await
            .map_err(|e| format!("Cannot read directory: {e}"))?;
        let mut entries: Vec<String> = Vec::new();
        while let Some(entry) = reader
            .next_entry()
            .await
            .map_err(|e| format!("Cannot read directory: {e}"))?
        {
            entries.push(entry.file_name().to_string_lossy().into_owned());
        }

        entries.sort_by_key(|a| a.to_lowercase());

        let mut results: Vec<String> = Vec::new();
        let mut entry_limit_reached = false;
        for entry in entries {
            if results.len() >= effective_limit {
                entry_limit_reached = true;
                break;
            }
            let full_path = dir_path.join(&entry);
            match tokio::fs::metadata(&full_path).await {
                Ok(meta) if meta.is_dir() => results.push(format!("{entry}/")),
                Ok(_) => results.push(entry),
                Err(_) => continue, // skip entries we cannot stat
            }
        }

        if results.is_empty() {
            return Ok(AgentToolResult::text("(empty directory)"));
        }

        let truncation =
            truncate_head(&results.join("\n"), crate::truncate::TruncationOptions {
                max_lines: usize::MAX,
                ..Default::default()
            });
        let mut output = truncation.content;
        let mut notices: Vec<String> = Vec::new();
        if entry_limit_reached {
            notices.push(format!(
                "{effective_limit} entries limit reached. Use limit={} for more",
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
