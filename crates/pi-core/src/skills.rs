//! Skills loading, port of the essential path of `core/skills.ts`:
//! SKILL.md discovery (global + project), frontmatter parsing and prompt
//! formatting.

use std::path::{Path, PathBuf};

/// Parsed skill.
#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub base_dir: PathBuf,
    pub disable_model_invocation: bool,
}

/// Minimal frontmatter parser for the three fields pi uses (name, description,
/// disable-model-invocation). Full YAML is not required for these simple
/// flat mappings.
fn unquote(value: &str) -> String {
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn parse_frontmatter(content: &str) -> (Option<String>, Option<String>, bool, String) {
    let mut name = None;
    let mut description = None;
    let mut disable = false;
    let mut rest = content;
    if let Some(stripped) = content.strip_prefix("---\n") {
        if let Some(end) = stripped.find("\n---") {
            let fm = &stripped[..end];
            rest = &stripped[end + 4..].trim_start_matches('\n');
            for line in fm.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let key = key.trim();
                let value = unquote(value.trim());
                match key {
                    "name" => name = Some(value.to_string()),
                    "description" => description = Some(value.to_string()),
                    "disable-model-invocation" => disable = value == "true",
                    _ => {}
                }
            }
        }
    }
    (name, description, disable, rest.to_string())
}

/// Validate a skill name per the Agent Skills spec (pi's `validateName`).
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
}

/// Load skills from one directory (non-recursive: `<dir>/<name>/SKILL.md` and
/// `<dir>/<name>.md`).
pub fn load_skills_from_dir(dir: &Path) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let skill_md = if path.is_dir() {
            path.join("SKILL.md")
        } else if path.extension().is_some_and(|e| e == "md") {
            path.clone()
        } else {
            continue;
        };
        let Ok(content) = std::fs::read_to_string(&skill_md) else {
            continue;
        };
        let (fm_name, description, disable, _body) = parse_frontmatter(&content);
        let name = fm_name.unwrap_or_else(|| {
            // directory name or file stem
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        if !valid_name(&name) {
            tracing::warn!("skipping skill with invalid name: {name}");
            continue;
        }
        skills.push(Skill {
            name,
            description: description.unwrap_or_default(),
            file_path: skill_md.clone(),
            base_dir: skill_md
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
            disable_model_invocation: disable,
        });
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Load skills: global (`~/.pi-rs/agent/skills`), pi-compatible
/// (`~/.pi/agent/skills`), then project (`.pi/skills`, `.claude/skills`).
pub fn load_skills(cwd: &Path) -> Vec<Skill> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".pi-rs/agent/skills"));
        dirs.push(home.join(".pi/agent/skills"));
    }
    for ancestor in cwd.ancestors() {
        dirs.push(ancestor.join(".pi/skills"));
        dirs.push(ancestor.join(".claude/skills"));
    }

    let mut skills: Vec<Skill> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dir in dirs {
        for skill in load_skills_from_dir(&dir) {
            if seen.insert(skill.name.clone()) {
                skills.push(skill);
            }
        }
    }
    skills
}

/// Render the skills section for the system prompt (pi's `formatSkillsForPrompt`).
pub fn format_skills_for_prompt(skills: &[Skill], file_read_tool: &str) -> String {
    let mut out = String::from(
        "The following skills provide specialized instructions for specific tasks.\nUse the read tool to load a skill's file when the task matches its description.\nWhen a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md) and use that absolute path in tool commands.\n\n",
    );
    out.push_str("<available_skills>\n");
    for skill in skills.iter().filter(|s| !s.disable_model_invocation) {
        out.push_str(&format!(
            "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    <location>{}</location>\n  </skill>\n",
            skill.name,
            html_escape(&skill.description),
            skill.file_path.display()
        ));
    }
    out.push_str("</available_skills>");
    let _ = file_read_tool;
    out
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter() {
        let content = "---\nname: my-skill\ndescription: Does things \"quoted\"\ndisable-model-invocation: false\n---\n\nBody here.\n";
        let (name, description, disable, body) = parse_frontmatter(content);
        assert_eq!(name.as_deref(), Some("my-skill"));
        assert_eq!(description.as_deref(), Some("Does things \"quoted\""));
        assert!(!disable);
        assert!(body.starts_with("Body here."));
    }

    #[test]
    fn loads_skill_from_dir_layout() {
        let dir = std::env::temp_dir().join(format!("skills-{}", uuid::Uuid::now_v7()));
        let skill_dir = dir.join("greet");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: greet\ndescription: Greets people\n---\nBody",
        )
        .unwrap();

        let skills = load_skills_from_dir(&dir);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "greet");
        assert_eq!(skills[0].description, "Greets people");
        assert!(skills[0].file_path.ends_with("SKILL.md"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prompt_format_lists_skills() {
        let skills = vec![Skill {
            name: "greet".into(),
            description: "Greets <people>".into(),
            file_path: PathBuf::from("/skills/greet/SKILL.md"),
            base_dir: PathBuf::from("/skills/greet"),
            disable_model_invocation: false,
        }];
        let prompt = format_skills_for_prompt(&skills, "read");
        assert!(prompt.contains("<name>greet</name>"));
        assert!(prompt.contains("Greets &lt;people&gt;"));
    }
}
