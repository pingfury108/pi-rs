//! write tool, port of `core/tools/write.ts`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use serde_json::json;

pub struct WriteTool {
    cwd: PathBuf,
}

impl WriteTool {
    pub fn new(cwd: &Path) -> Self {
        Self { cwd: cwd.to_path_buf() }
    }
}

#[async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("write {}", args["path"].as_str().unwrap_or(""))
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
        let content = args["content"]
            .as_str()
            .ok_or_else(|| "missing required argument: content".to_string())?;

        let absolute_path = crate::path_utils::resolve_to_cwd(path, &self.cwd);
        let dir = absolute_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        crate::mutation_queue::with_file_mutation_queue(&absolute_path, async {
            tokio::fs::create_dir_all(&dir)
                .await
                .map_err(|e| format!("Cannot create directory {}: {e}", dir.display()))?;
            tokio::fs::write(&absolute_path, content)
                .await
                .map_err(|e| format!("Cannot write file {path}: {e}"))?;
            Ok(AgentToolResult {
                content: vec![ToolResultContent::Text(TextContent {
                    text: format!("Successfully wrote to {path}"),
                    text_signature: None,
                })],
                details: serde_json::Value::Null,
                usage: None,
                terminate: false,
            })
        })
        .await
    }
}
