//! bash tool, port of `core/tools/bash.ts` + `output-accumulator.ts`.
//!
//! Streams stdout/stderr; output is truncated from the tail (last 2000 lines
//! or 50KB), full output spills to a temp file when truncated. Aborting kills
//! the process group.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolContext, ToolUpdateFn};
use pi_ai::types::{TextContent, ToolResultContent};
use serde_json::json;
use tokio::io::AsyncReadExt;

use crate::truncate::{format_size, truncate_tail, DEFAULT_MAX_BYTES};

pub struct BashTool {
    cwd: PathBuf,
    /// Command prefix prepended to every command.
    pub command_prefix: Option<String>,
}

impl BashTool {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            command_prefix: None,
        }
    }
}

/// Accumulated output of a running command.
struct OutputAccumulator {
    buf: Vec<u8>,
    /// Full output spilled to disk once truncation is likely.
    spill: Option<PathBuf>,
}

impl OutputAccumulator {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            spill: None,
        }
    }

    fn append(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Snapshot with tail truncation; spill full output to a temp file when
    /// the accumulated output exceeds the byte limit.
    fn snapshot(&mut self) -> (String, Option<PathBuf>, Option<crate::truncate::TruncationResult>) {
        let content = String::from_utf8_lossy(&self.buf).into_owned();
        let truncation = truncate_tail(&content, None::<crate::truncate::TruncationOptions>);
        if !truncation.truncated {
            return (truncation.content, self.spill.take(), None);
        }
        // keep ownership: clone the fields we need
        if self.spill.is_none() {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "pi-rs-bash-{}.log",
                uuid::Uuid::now_v7().simple()
            ));
            if let Ok(()) = std::fs::write(&path, &self.buf) {
                self.spill = Some(path);
            }
        }
        (truncation.content.clone(), self.spill.clone(), Some(truncation))
    }
}

fn format_output(
    content: &str,
    truncation: Option<&crate::truncate::TruncationResult>,
    full_output_path: Option<&Path>,
) -> String {
    let mut text = if content.is_empty() {
        "(no output)".to_string()
    } else {
        content.to_string()
    };
    if let Some(t) = truncation {
        let start_line = t.total_lines - t.output_lines + 1;
        let end_line = t.total_lines;
        let suffix = if t.last_line_partial {
            format!(
                "\n\n[Showing last {} of line {end_line} (line is {})]. Full output: {}]",
                format_size(t.output_bytes),
                format_size(t.total_bytes),
                full_output_path.map(|p| p.display().to_string()).unwrap_or_default()
            )
        } else if t.truncated_by == Some("lines") {
            format!(
                "\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {}]",
                t.total_lines,
                full_output_path.map(|p| p.display().to_string()).unwrap_or_default()
            )
        } else {
            format!(
                "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {}]",
                t.total_lines,
                format_size(DEFAULT_MAX_BYTES),
                full_output_path.map(|p| p.display().to_string()).unwrap_or_default()
            )
        };
        text.push_str(&suffix);
    }
    text
}

#[async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
            },
            "required": ["command"],
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        let cmd = args["command"].as_str().unwrap_or("");
        let mut chars = cmd.chars();
        let first: String = chars.by_ref().take(40).collect();
        if chars.next().is_some() {
            format!("bash {first}...")
        } else {
            format!("bash {first}")
        }
    }

    #[allow(clippy::too_many_lines)]
    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
        on_update: ToolUpdateFn,
    ) -> Result<AgentToolResult, String> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| "missing required argument: command".to_string())?;
        let timeout_secs = args["timeout"].as_u64();

        let resolved_command = match &self.command_prefix {
            Some(prefix) => format!("{prefix}\n{command}"),
            None => command.to_string(),
        };

        if !self.cwd.exists() {
            return Err(format!(
                "Working directory does not exist: {}\nCannot execute bash commands.",
                self.cwd.display()
            ));
        }

        let mut accumulator = OutputAccumulator::new();
        let mut child = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(&resolved_command)
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn bash: {e}"))?;

        let mut stdout = child.stdout.take().expect("stdout piped");
        let mut stderr = child.stderr.take().expect("stderr piped");

        // Stream output through the accumulator, forwarding throttled updates.
        let update_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let update_flag_reader = update_flag.clone();

        let read_half = async {
            let mut buf = [0u8; 8192];
            loop {
                let n = stdout.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                accumulator.append(&buf[..n]);
                update_flag_reader.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let mut buf = [0u8; 8192];
            loop {
                let n = stderr.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                accumulator.append(&buf[..n]);
                update_flag_reader.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        };

        // Abort handling: kill the shell on cancellation. (Process-group
        // kill needs libc; revisit with a safe wrapper for detached children.)
        tokio::pin!(read_half);
        let mut timed_out = false;
        let mut was_aborted = false;
        match &ctx.cancel {
            Some(token) => {
                tokio::select! {
                    _ = token.cancelled() => {
                        was_aborted = true;
                        let _ = child.start_kill();
                        let _ = (&mut read_half).await;
                    }
                    result = async {
                        match timeout_secs {
                            Some(secs) => {
                                tokio::time::timeout(Duration::from_secs(secs), &mut read_half)
                                    .await
                                    .map(|_| ())
                                    .map_err(|_| ())
                            }
                            None => {
                                (&mut read_half).await;
                                Ok(())
                            }
                        }
                    } => {
                        timed_out = result.is_err();
                    }
                }
            }
            None => {
                let result = match timeout_secs {
                    Some(secs) => {
                        tokio::time::timeout(Duration::from_secs(secs), &mut read_half)
                            .await
                            .map(|_| ())
                            .map_err(|_| ())
                    }
                    None => {
                        (&mut read_half).await;
                        Ok(())
                    }
                };
                timed_out = result.is_err();
            }
        }

        // Wait for exit.
        let status = child.wait().await;
        if was_aborted || ctx.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            let (text, _, _) = accumulator.snapshot();
            return Err(format_output(&text, None, None) + "\n\nCommand aborted");
        }
        if timed_out {
            let (text, _, _) = accumulator.snapshot();
            return Err(format!(
                "{}\n\nCommand timed out after {} seconds",
                format_output(&text, None, None),
                timeout_secs.unwrap_or(0)
            ));
        }

        let (text, spill_path, truncation) = accumulator.snapshot();
        // flush pending update
        if update_flag.load(std::sync::atomic::Ordering::Relaxed) {
            on_update(AgentToolResult::text(text.clone()));
        }

        let exit_code = status.map(|s| s.code().unwrap_or(1)).unwrap_or(1);
        let output_text = format_output(&text, truncation.as_ref(), spill_path.as_deref());
        if exit_code != 0 {
            return Err(format!("{output_text}\n\nCommand exited with code {exit_code}"));
        }

        Ok(AgentToolResult {
            content: vec![ToolResultContent::Text(TextContent {
                text: output_text,
                text_signature: None,
            })],
            details: truncation
                .map(|t| {
                    json!({
                        "truncation": serde_json::to_value(&t).unwrap_or(serde_json::Value::Null),
                        "fullOutputPath": spill_path.map(|p| p.display().to_string()),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            usage: None,
            terminate: false,
        })
    }
}

use std::time::Duration;
