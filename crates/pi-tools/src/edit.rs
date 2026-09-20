//! edit tool, port of `core/tools/edit.ts`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use serde_json::json;

use crate::edit_diff::{apply_edits_to_normalized_content, detect_line_ending, normalize_to_lf, restore_line_endings, split_bom, Edit};

pub struct EditTool {
    cwd: PathBuf,
}

impl EditTool {
    pub fn new(cwd: &Path) -> Self {
        Self { cwd: cwd.to_path_buf() }
    }

    /// Parse the `edits` argument, tolerating a JSON string or single-object
    /// form (pi's `prepareEditArguments`).
    fn parse_edits(args: &serde_json::Value) -> Result<Vec<Edit>, String> {
        let mut edits: Vec<Edit> = Vec::new();
        let edits_value = &args["edits"];
        match edits_value {
            serde_json::Value::Array(items) => {
                for item in items {
                    edits.push(Edit {
                        old_text: item["oldText"]
                            .as_str()
                            .ok_or("each edits[] entry needs string oldText")?
                            .to_string(),
                        new_text: item["newText"]
                            .as_str()
                            .ok_or("each edits[] entry needs string newText")?
                            .to_string(),
                    });
                }
            }
            serde_json::Value::String(raw) => {
                let parsed: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|_| "edits is a string but not valid JSON".to_string())?;
                return Self::parse_edits(&json!({ "edits": parsed }));
            }
            serde_json::Value::Object(_) => {
                edits.push(Edit {
                    old_text: edits_value["oldText"]
                        .as_str()
                        .ok_or("edits entry needs string oldText")?
                        .to_string(),
                    new_text: edits_value["newText"]
                        .as_str()
                        .ok_or("edits entry needs string newText")?
                        .to_string(),
                });
            }
            serde_json::Value::Null => {}
            _ => return Err("edits must be an array".into()),
        }

        // Legacy flat form: {path, oldText, newText}
        if edits.is_empty() {
            if let (Some(old_text), Some(new_text)) =
                (args["oldText"].as_str(), args["newText"].as_str())
            {
                edits.push(Edit {
                    old_text: old_text.to_string(),
                    new_text: new_text.to_string(),
                });
            }
        }

        if edits.is_empty() {
            return Err(
                "Edit tool input is invalid. edits must contain at least one replacement.".into(),
            );
        }
        Ok(edits)
    }
}

#[async_trait]
impl AgentTool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file."},
                            "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("edit {}", args["path"].as_str().unwrap_or(""))
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
        let edits = Self::parse_edits(&args)?;

        let absolute_path = crate::path_utils::resolve_to_cwd(path, &self.cwd);

        crate::mutation_queue::with_file_mutation_queue(&absolute_path, async {
            let bytes = tokio::fs::read(&absolute_path).await.map_err(|e| {
                format!("Could not edit file: {path}. {e}.")
            })?;
            let raw_content = String::from_utf8_lossy(&bytes).into_owned();

            // Strip BOM before matching; preserve it on write.
            let (bom, content) = split_bom(&raw_content);
            let bom = bom.to_string();
            let original_ending = detect_line_ending(content);
            let normalized = normalize_to_lf(content);

            let applied = apply_edits_to_normalized_content(&normalized, &edits, path)?;
            let final_content = format!(
                "{bom}{}",
                restore_line_endings(&applied.new_content, original_ending)
            );
            tokio::fs::write(&absolute_path, final_content)
                .await
                .map_err(|e| format!("Cannot write file {path}: {e}"))?;

            let (diff, first_changed_line) =
                crate::diff::generate_diff_string(&applied.base_content, &applied.new_content, 4);
            let patch =
                crate::diff::generate_unified_patch(path, &applied.base_content, &applied.new_content);

            Ok(AgentToolResult {
                content: vec![ToolResultContent::Text(TextContent {
                    text: format!("Successfully replaced {} block(s) in {path}.", edits.len()),
                    text_signature: None,
                })],
                details: json!({
                    "diff": diff,
                    "patch": patch,
                    "firstChangedLine": first_changed_line,
                }),
                usage: None,
                terminate: false,
            })
        })
        .await
    }
}
