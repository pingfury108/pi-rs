//! grep tool, port of `core/tools/grep.ts`.
//!
//! Uses the `ignore` crate (ripgrep's library) instead of shelling out to the
//! `rg` binary: same regex engine family, same .gitignore/hidden handling.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use regex::RegexBuilder;
use serde_json::json;

use crate::truncate::{format_size, truncate_head, truncate_line, DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH};

const DEFAULT_LIMIT: usize = 100;

pub struct GrepTool {
    cwd: PathBuf,
}

impl GrepTool {
    pub fn new(cwd: &Path) -> Self {
        Self { cwd: cwd.to_path_buf() }
    }
}

#[async_trait]
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
                "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"},
                "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                "literal": {"type": "boolean", "description": "Treat pattern as literal string instead of regex (default: false)"},
                "context": {"type": "number", "description": "Number of lines to show before and after each match (default: 0)"},
                "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
            },
            "required": ["pattern"],
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("grep {}", args["pattern"].as_str().unwrap_or(""))
    }

    #[allow(clippy::too_many_lines)]
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
        let glob = args["glob"].as_str();
        let ignore_case = args["ignoreCase"].as_bool().unwrap_or(false);
        let literal = args["literal"].as_bool().unwrap_or(false);
        let context_value = args["context"].as_u64().unwrap_or(0).max(0) as usize;
        let effective_limit = (args["limit"].as_u64().unwrap_or(DEFAULT_LIMIT as u64) as usize).max(1);

        let search_path = crate::path_utils::resolve_to_cwd(search_dir, &self.cwd);
        if !search_path.exists() {
            return Err(format!("Path not found: {}", search_path.display()));
        }
        let is_directory = search_path.is_dir();

        let pattern_source = if literal {
            regex::escape(pattern)
        } else {
            pattern.to_string()
        };
        let mut regex_builder = RegexBuilder::new(&pattern_source);
        regex_builder.case_insensitive(ignore_case);
        let regex = regex_builder
            .build()
            .map_err(|e| format!("Invalid pattern: {e}"))?;

        let glob_matcher = match glob {
            Some(g) => Some(
                globset::GlobBuilder::new(g)
                    .literal_separator(g.contains("**"))
                    .build()
                    .map_err(|e| format!("Invalid glob: {e}"))?
                    .compile_matcher(),
            ),
            None => None,
        };

        // Walk with .gitignore + hidden-file handling (ripgrep defaults).
        let mut walker = ignore::WalkBuilder::new(&search_path);
        walker.hidden(true);
        if let Some(token) = ctx.cancel.as_ref() {
            let token = token.clone();
            walker.filter_entry(move |_| !token.is_cancelled());
        }

        let mut output_lines: Vec<String> = Vec::new();
        let mut match_count = 0usize;
        let mut match_limit_reached = false;
        let mut lines_truncated = false;

        for entry in walker.build().flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(matcher) = &glob_matcher {
                let rel = path.strip_prefix(&search_path).unwrap_or(path);
                let rel_str = crate::path_utils::to_posix(rel);
                if !matcher.is_match(&rel_str) && !matcher.is_match(path) {
                    continue;
                }
            }
            let Ok(content) = tokio::fs::read(path).await else {
                continue;
            };
            // Binary heuristic: skip files containing NUL.
            if content.contains(&0u8) {
                continue;
            }
            let Ok(mut text) = String::from_utf8(content) else {
                continue;
            };
            text = text.replace("\r\n", "\n").replace('\r', "\n");
            let lines: Vec<&str> = text.split('\n').collect();

            for (i, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    match_count += 1;
                    let relative = display_path(path, &search_path, is_directory);
                    if context_value == 0 {
                        let (truncated_text, was) = truncate_line(line, None);
                        lines_truncated |= was;
                        output_lines.push(format!("{relative}:{}: {truncated_text}", i + 1));
                    } else {
                        let start = (i + 1).saturating_sub(context_value).max(1);
                        let end = (i + 1 + context_value).min(lines.len());
                        for current in start..=end {
                            let line_text = lines.get(current - 1).copied().unwrap_or("");
                            let (truncated_text, was) = truncate_line(line_text, None);
                            lines_truncated |= was;
                            if current == i + 1 {
                                output_lines.push(format!("{relative}:{current}: {truncated_text}"));
                            } else {
                                output_lines.push(format!("{relative}-{current}- {truncated_text}"));
                            }
                        }
                    }
                    if match_count >= effective_limit {
                        match_limit_reached = true;
                        break;
                    }
                }
            }
            if match_limit_reached {
                break;
            }
        }

        if match_count == 0 {
            return Ok(AgentToolResult::text("No matches found"));
        }

        let truncation =
            truncate_head(&output_lines.join("\n"), crate::truncate::TruncationOptions {
                max_lines: usize::MAX,
                ..Default::default()
            });
        let mut output = truncation.content;
        let mut notices: Vec<String> = Vec::new();
        if match_limit_reached {
            notices.push(format!(
                "{effective_limit} matches limit reached. Use limit={} for more, or refine pattern",
                effective_limit * 2
            ));
        }
        if truncation.truncated {
            notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
        }
        if lines_truncated {
            notices.push(format!(
                "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
            ));
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

/// Relative path against the search root when searching a directory,
/// basename for single files (pi's `formatPath`).
fn display_path(file: &Path, search_root: &Path, is_directory: bool) -> String {
    if is_directory {
        if let Ok(rel) = file.strip_prefix(search_root) {
            let s = crate::path_utils::to_posix(rel);
            if !s.is_empty() && !s.starts_with("..") {
                return s;
            }
        }
    }
    file.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}
