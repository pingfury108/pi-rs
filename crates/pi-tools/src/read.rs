//! read tool, port of `core/tools/read.ts`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use serde_json::json;

use crate::truncate::{format_size, truncate_head, DEFAULT_MAX_BYTES};

pub struct ReadTool {
    cwd: PathBuf,
}

impl ReadTool {
    pub fn new(cwd: &Path) -> Self {
        Self { cwd: cwd.to_path_buf() }
    }
}

const DESCRIPTION: &str = "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.";

#[async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("read {}", args["path"].as_str().unwrap_or(""))
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        _ctx: &ToolContext,
        _on_update: ToolUpdateFn,
    ) -> Result<AgentToolResult, String> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| "missing required argument: path".to_string())?;
        let offset = args["offset"].as_u64().map(|v| v as usize);
        let limit = args["limit"].as_u64().map(|v| v as usize);

        let absolute_path = crate::path_utils::resolve_to_cwd(path, &self.cwd);
        let bytes = tokio::fs::read(&absolute_path)
            .await
            .map_err(|e| format!("Could not read file: {path}. {e}."))?;

        // Image support: Phase 6 (image pipeline); text note for now.
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let all_lines: Vec<&str> = text.split('\n').collect();
        let total_file_lines = all_lines.len();

        let start_line = offset.map(|o| o.saturating_sub(1)).unwrap_or(0);
        let start_line_display = start_line + 1;
        if start_line >= all_lines.len() {
            return Err(format!(
                "Offset {offset:?} is beyond end of file ({total_file_lines} lines total)"
            ));
        }

        let selected_content = match limit {
            Some(l) => all_lines[start_line..(start_line + l).min(all_lines.len())].join("\n"),
            None => all_lines[start_line..].join("\n"),
        };

        let truncation = truncate_head(&selected_content, None::<crate::truncate::TruncationOptions>);
        let output_text = if truncation.first_line_exceeds_limit {
            let first_line_size = format_size(all_lines[start_line].len());
            format!(
                "[Line {start_line_display} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]",
                format_size(DEFAULT_MAX_BYTES)
            )
        } else if truncation.truncated {
            let end_line_display = start_line_display + truncation.output_lines - 1;
            let next_offset = end_line_display + 1;
            let reason = if truncation.truncated_by == Some("lines") {
                format!(
                    "[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
                )
            } else {
                format!(
                    "[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} ({} limit). Use offset={next_offset} to continue.]",
                    format_size(DEFAULT_MAX_BYTES)
                )
            };
            format!("{}\n\n{}", truncation.content, reason)
        } else if let Some(l) = limit {
            let read = l.min(all_lines.len() - start_line);
            if start_line + read < all_lines.len() {
                let remaining = all_lines.len() - (start_line + read);
                let next_offset = start_line + read + 1;
                format!(
                    "{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
                    truncation.content
                )
            } else {
                truncation.content.clone()
            }
        } else {
            truncation.content.clone()
        };

        let details = if truncation.truncated {
            serde_json::to_value(&truncation).unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };

        Ok(AgentToolResult {
            content: vec![ToolResultContent::Text(TextContent {
                text: output_text,
                text_signature: None,
            })],
            details,
            usage: None,
            terminate: false,
        })
    }
}
