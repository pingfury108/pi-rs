//! Full-featured embedding example: AgentSession with tools, persistent
//! session, auto-compaction, skills and extension support — the whole
//! pi-core stack in a few lines.
//!
//! Run: cargo run -p pi-core --example full_session
//!
//! This uses the faux provider; point SessionOptions at a real model + api
//! (see pi_core::model_registry) for production use.

use std::path::PathBuf;
use std::sync::Arc;

use pi_ai::api::{FauxApi, FauxResponse, LlmApi};
use pi_core::{AgentSession, SessionOptions};

fn faux_model() -> pi_ai::types::Model {
    pi_ai::types::Model {
        id: "demo-1".into(),
        name: "Demo".into(),
        api: "faux".into(),
        provider: "faux".into(),
        base_url: "http://localhost".into(),
        reasoning: false,
        input: vec![pi_ai::types::Modality::Text],
        cost: Default::default(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let faux = Arc::new(FauxApi::with_script(vec![
        FauxResponse::text("All done: created example.txt."),
    ]));

    let api: Arc<dyn LlmApi> = faux;
    let workdir = std::env::temp_dir().join(format!("pi-rs-demo-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir_all(&workdir)?;

    let session = AgentSession::new(SessionOptions {
        cwd: workdir.clone(),
        model: faux_model(),
        api,
        api_key: None, // faux needs none; real providers resolve via env/auth.json
        force_system_prompt: None,
        append_system_prompt: None,
        resume_file: None, // or Some(path) to resume a pi-compatible session
        compaction_enabled: true,
        auto_retry: None, // defaults: 3 attempts, exponential backoff
    })?;

    println!("session file: {}", session.session_file().display());

    // Events stream live; the returned handle yields the final messages.
    let mut events = session.subscribe();
    let runner = session.prompt("create example.txt with 'hello'");

    while let Ok(event) = events.recv().await {
        use pi_agent::AgentEvent as E;
        match &event {
            E::MessageUpdate {
                assistant_message_event,
                ..
            } => {
                use pi_ai::events::AssistantMessageEvent as A;
                if let A::TextDelta { delta, .. } = assistant_message_event.as_ref() {
                    print!("{delta}");
                    use std::io::Write as _;
                    std::io::stdout().flush().ok();
                }
            }
            E::ToolExecutionStart { tool_name, .. } => println!("\n[tool: {tool_name}]"),
            E::AutoRetryStart {
                attempt,
                max_attempts,
                ..
            } => println!("\n[retry {attempt}/{max_attempts}]"),
            E::AgentEnd { .. } => break,
            _ => {}
        }
    }
    println!();

    // Steering while idle queues the next prompt; abort() cancels a run.
    let _ = session.last_assistant_text();
    let _ = session.messages().len();

    // Later: resume with SessionOptions { resume_file: Some(session_file), .. }
    let _ = PathBuf::from(session.session_file());
    std::fs::remove_dir_all(&workdir).ok();
    Ok(())
}
