//! System prompt construction, port of the essential parts of
//! `core/system-prompt.ts`: ordered sections (preamble/tools/rules + project
//! context + cwd), rendered as `<name>...</name>` blocks.

use std::path::Path;

use pi_agent::AgentTool;

pub struct SystemPromptInput<'a> {
    pub cwd: &'a Path,
    /// Loaded AGENTS.md-style context files (path, content).
    pub context_files: Vec<(String, String)>,
    pub tools: &'a [std::sync::Arc<dyn AgentTool>],
    /// Loaded skills for the skills prompt section.
    pub skills: &'a [crate::skills::Skill],
    /// User append from settings.
    pub append_system_prompt: Option<String>,
    /// Full replacement (pi's forceSystemPrompt).
    pub force_system_prompt: Option<String>,
}

const PREAMBLE: &str = "You are an expert coding assistant operating inside pi-rs, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.";

/// Per-tool prompt snippets/guidelines contributed to the prompt
/// (pi's toolSnippet/promptGuidelines contributions).
fn tool_contribution(name: &str) -> (&'static str, &'static [&'static str]) {
    match name {
        "read" => (
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            &["Use read to examine files instead of cat or sed."],
        ),
        "bash" => (
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            &["Use bash for running shell commands and build tools.", "Don't use bash echo/printf to write files unless asked - use write."],
        ),
        "edit" => (
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
            &[
                "Use edit for precise changes (edits[].oldText must match exactly)",
                "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
                "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
                "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
            ],
        ),
        "write" => (
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
            &["Use write only for new files or complete rewrites."],
        ),
        "grep" => (
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars.",
            &[],
        ),
        "find" => (
            "Find files by glob pattern (respects .gitignore)",
            &[],
        ),
        "ls" => (
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).",
            &[],
        ),
        _ => ("", &[]),
    }
}

/// Build the system prompt text (pi's `buildSystemPrompt`).
#[allow(clippy::vec_init_then_push)]
pub fn build_system_prompt(input: &SystemPromptInput<'_>) -> String {
    if let Some(forced) = &input.force_system_prompt {
        return forced.clone();
    }

    let mut sections: Vec<(String, String)> = Vec::new();

    // preamble
    sections.push(("preamble".to_string(), PREAMBLE.to_string()));

    // tools
    let tools_list: Vec<String> = input
        .tools
        .iter()
        .map(|t| {
            let (snippet, _) = tool_contribution(t.name());
            format!("- {}: {}", t.name(), snippet)
        })
        .collect();
    sections.push((
        "tools".to_string(),
        format!(
            "{}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.",
            tools_list.join("\n")
        ),
    ));

    // rules
    let mut rules: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut add_rule = |rule: &str, rules: &mut Vec<String>| {
        let normalized = rule.trim();
        if normalized.is_empty() || !seen.insert(normalized.to_string()) {
            return;
        }
        rules.push(format!("- {normalized}"));
    };
    for tool in input.tools {
        for guideline in tool_contribution(tool.name()).1 {
            add_rule(guideline, &mut rules);
        }
    }
    add_rule("Be concise in your responses", &mut rules);
    add_rule("Show file paths clearly when working with files", &mut rules);
    sections.push(("rules".to_string(), rules.join("\n")));

    // addendum
    if let Some(append) = &input.append_system_prompt {
        if !append.is_empty() {
            sections.push(("addendum".to_string(), append.clone()));
        }
    }

    // project context (AGENTS.md files)
    if !input.context_files.is_empty() {
        let rendered: Vec<String> = input
            .context_files
            .iter()
            .map(|(path, content)| {
                format!("<project_instructions path=\"{path}\">\n{content}\n</project_instructions>")
            })
            .collect();
        sections.push((
            "project_context".to_string(),
            format!(
                "Project-specific instructions and guidelines:\n\n{}",
                rendered.join("\n\n")
            ),
        ));
    }

    // skills (pi: formatSkillsForPrompt with a file-reading tool)
    if !input.skills.is_empty() {
        let rendered = crate::skills::format_skills_for_prompt(input.skills, "read");
        if !rendered.is_empty() {
            sections.push(("skills".to_string(), rendered));
        }
    }

    // cwd
    sections.push((
        "cwd".to_string(),
        input.cwd.to_string_lossy().replace('\\', "/"),
    ));

    // render: preamble plain, others wrapped in <name> tags
    let mut out = String::new();
    for (i, (name, content)) in sections.iter().enumerate() {
        if i == 0 && name == "preamble" {
            out.push_str(content);
        } else {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&format!("<{name}>\n{content}\n</{name}>"));
        }
    }
    out
}

/// Load AGENTS.md-style context files: global (`~/.pi/agent/AGENTS.md`) plus
/// `AGENTS.md`/`CLAUDE.md` from the working directory and its ancestors up to
/// the filesystem root (nearest first).
pub fn load_context_files(cwd: &Path) -> Vec<(String, String)> {
    let mut files: Vec<(String, String)> = Vec::new();

    if let Some(home) = std::env::var_os("HOME") {
        let global = Path::new(&home).join(".pi/agent/AGENTS.md");
        if let Some(content) = read_if_exists(&global) {
            files.push((global.to_string_lossy().into_owned(), content));
        }
    }

    let mut ancestors: Vec<PathBuf> = cwd.ancestors().map(Path::to_path_buf).collect();
    ancestors.reverse(); // farthest first so nearest overrides display order later
    for dir in ancestors {
        for name in ["AGENTS.md", "CLAUDE.md"] {
            let candidate = dir.join(name);
            if let Some(content) = read_if_exists(&candidate) {
                files.push((candidate.to_string_lossy().into_owned(), content));
            }
        }
    }
    files
}

fn read_if_exists(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|content| content.trim_end().to_string())
}

use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool;
    #[async_trait::async_trait]
    impl AgentTool for DummyTool {
        fn name(&self) -> &str {
            "bash"
        }
        fn description(&self) -> &str {
            ""
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn label(&self, _args: &serde_json::Value) -> String {
            String::new()
        }
        async fn execute(
            &self,
            _id: &str,
            _args: serde_json::Value,
            _ctx: &pi_agent::ToolContext,
            _on_update: pi_agent::ToolUpdateFn,
        ) -> Result<pi_agent::AgentToolResult, String> {
            Ok(pi_agent::AgentToolResult::text(""))
        }
    }

    #[test]
    fn sections_are_wrapped_and_ordered() {
        let tools = vec![std::sync::Arc::new(DummyTool) as std::sync::Arc<dyn AgentTool>];
        let prompt = build_system_prompt(&SystemPromptInput {
            cwd: Path::new("/work"),
            context_files: vec![("/work/AGENTS.md".into(), "Be nice.".into())],
            tools: &tools,
            skills: &[],
            append_system_prompt: None,
            force_system_prompt: None,
        });
        assert!(prompt.starts_with("You are an expert coding assistant"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("<rules>"));
        assert!(prompt.contains(
            "<project_instructions path=\"/work/AGENTS.md\">\nBe nice.\n</project_instructions>"
        ));
        assert!(prompt.contains("<cwd>\n/work\n</cwd>"));
        // order: preamble plain (unwrapped), then tags
        let preamble_end = prompt.find("writing new files.").unwrap();
        let rest = &prompt[preamble_end..];
        assert!(rest.starts_with("writing new files.\n\n<tools>"), "{rest:?}");
    }

    #[test]
    fn forced_prompt_replaces_everything() {
        let tools: Vec<std::sync::Arc<dyn AgentTool>> = vec![];
        let prompt = build_system_prompt(&SystemPromptInput {
            cwd: Path::new("/work"),
            context_files: vec![],
            tools: &tools,
            skills: &[],
            append_system_prompt: None,
            force_system_prompt: Some("custom only".into()),
        });
        assert_eq!(prompt, "custom only");
    }
}
