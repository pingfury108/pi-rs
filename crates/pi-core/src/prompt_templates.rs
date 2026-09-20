//! Prompt templates, port of `core/prompt-templates.ts` (subset): markdown
//! files with frontmatter discovered from global + project directories.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct PromptTemplate {
    pub name: String,
    pub description: Option<String>,
    pub file_path: PathBuf,
    /// Body after frontmatter.
    pub content: String,
}

/// Minimal frontmatter parse (name/description).
fn parse(content: &str) -> (Option<String>, Option<String>, String) {
    let mut name = None;
    let mut description = None;
    let mut rest = content.to_string();
    if let Some(stripped) = content.strip_prefix("---\n") {
        if let Some(end) = stripped.find("\n---") {
            for line in stripped[..end].lines() {
                if let Some((k, v)) = line.split_once(':') {
                    match k.trim() {
                        "name" => name = Some(v.trim().to_string()),
                        "description" => description = Some(v.trim().to_string()),
                        _ => {}
                    }
                }
            }
            rest = stripped[end + 4..].trim_start_matches('\n').to_string();
        }
    }
    (name, description, rest)
}

/// Load templates from global (`~/.pi-rs/agent/prompts`, `~/.pi/agent/prompts`)
/// and project (`.pi/prompts`) directories.
pub fn load_prompt_templates(cwd: &Path) -> Vec<PromptTemplate> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".pi-rs/agent/prompts"));
        dirs.push(home.join(".pi/agent/prompts"));
    }
    dirs.push(cwd.join(".pi/prompts"));

    let mut templates = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e != "md") {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let (name, description, body) = parse(&content);
            let name = name.unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            if seen.insert(name.clone()) {
                templates.push(PromptTemplate {
                    name,
                    description,
                    file_path: path,
                    content: body,
                });
            }
        }
    }
    templates
}

/// Expand `$1`..`$9`/`$@` argument substitution (pi's `expandPromptTemplate`).
pub fn expand_template(content: &str, args: &[String]) -> String {
    let mut out = content.to_string();
    if content.contains("$@") {
        out = out.replace("$@", &args.join(" "));
    }
    for (i, arg) in args.iter().enumerate().take(9) {
        out = out.replace(&format!("${}", i + 1), arg);
    }
    out
}

/// Expand `@path` file references in a user prompt into file contents
/// (pi's `@file` handling in cli/file-processor.ts, simplified: text files).
pub fn expand_file_references(prompt: &str, cwd: &Path) -> String {
    let mut replaced = String::new();
    let mut rest = prompt;
    while let Some(at) = rest.find('@') {
        let after = &rest[at + 1..];
        let token_len = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || matches!(c, '/' | '.' | '-' | '_' | '~'))
            .map(char::len_utf8)
            .sum::<usize>();
        if token_len == 0 {
            replaced.push_str(&rest[..=at]);
            rest = &rest[at + 1..];
            continue;
        }
        let token = &after[..token_len];
        replaced.push_str(&rest[..at]);
        let path = if token.starts_with('~') {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(&token[1..]))
                .unwrap_or_else(|| PathBuf::from(token))
        } else {
            cwd.join(token)
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                replaced.push_str(&format!(
                    "<file path=\"{}\">\n{}\n</file>",
                    path.display(),
                    content.trim_end()
                ));
            }
            Err(_) => {
                replaced.push('@');
                replaced.push_str(token);
            }
        }
        rest = &after[token_len..];
    }
    replaced.push_str(rest);
    replaced
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_args() {
        assert_eq!(expand_template("hi $1 and $2", &["a".into(), "b".into()]), "hi a and b");
        assert_eq!(expand_template("all: $@", &["a".into(), "b".into()]), "all: a b");
    }

    #[test]
    fn expands_file_references() {
        let dir = std::env::temp_dir().join(format!("pt-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "FILE CONTENT").unwrap();
        let out = expand_file_references("look at @f.txt please", &dir);
        assert!(out.contains("<file path="));
        assert!(out.contains("FILE CONTENT"));
        // unknown file stays as-is
        let out = expand_file_references("look at @missing.txt", &dir);
        assert_eq!(out, "look at @missing.txt");
        std::fs::remove_dir_all(&dir).ok();
    }
}
