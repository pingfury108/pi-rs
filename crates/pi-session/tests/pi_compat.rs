//! Compatibility test: open a real pi session file when one exists on this
//! machine; skipped otherwise (CI-safe).

use pi_session::SessionManager;

fn find_real_pi_session() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let sessions_dir = PathBuf::from(home).join(".pi/agent/sessions");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(sessions_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    candidates.sort();

    for dir in candidates.into_iter().rev() {
        if let Ok(files) = std::fs::read_dir(&dir) {
            let mut files: Vec<PathBuf> = files
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
                .collect();
            files.sort();
            if let Some(f) = files.last() {
                return Some(f.to_path_buf());
            }
        }
    }
    None
}

use std::path::PathBuf;

#[test]
fn opens_real_pi_session_and_builds_context() {
    let Some(path) = find_real_pi_session() else {
        eprintln!("no real pi session found; skipping");
        return;
    };
    eprintln!("testing against {path:?}");

    let manager = SessionManager::open(&path).expect("should open pi session file");
    assert_eq!(manager.header().version, Some(3));
    assert!(!manager.entries().is_empty(), "session should have entries");

    // context messages must build without errors
    let messages = manager.build_context_messages(None);
    eprintln!("entries: {}, context messages: {}", manager.entries().len(), messages.len());
    assert!(!messages.is_empty(), "expected context messages from a real session");

    // every message should be a known role
    for message in &messages {
        match message {
            pi_agent::AgentMessage::Message(m) => match m {
                pi_ai::types::Message::System(_)
                | pi_ai::types::Message::User(_)
                | pi_ai::types::Message::Assistant(_)
                | pi_ai::types::Message::ToolResult(_) => {}
            },
            pi_agent::AgentMessage::Custom(_) => {}
        }
    }

    // settings resolve
    let (thinking, model) = manager.context_settings(None);
    eprintln!("thinking={thinking}, model={model:?}");
}
