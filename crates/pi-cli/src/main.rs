//! pi-cli: Headless coding agent CLI (print / json / repl modes).

mod repl;
mod rpc;

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use pi_agent::AgentEvent;
use pi_core::{
    build_api, build_model, load_custom_providers, load_settings, resolve_api_key,
    resolve_custom_model, AgentSession, SessionOptions,
};


/// Headless coding agent (Rust port of pi's core, no TUI).
#[derive(Parser, Debug)]
#[command(name = "pi-rs", version, about)]
struct Cli {
    /// Prompt to send (single-shot). Reads stdin if omitted.
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,

    /// Output mode: text (default) or json (agent event stream as JSONL).
    #[arg(long, value_parser = ["text", "json"], default_value = "text")]
    mode: String,

    /// Provider id (anthropic, openai, deepseek, openrouter, kimi-coding, ...).
    #[arg(long)]
    provider: Option<String>,

    /// Model id within the provider.
    #[arg(long)]
    model: Option<String>,

    /// Override base URL (OpenAI-compatible endpoints).
    #[arg(long)]
    base_url: Option<String>,

    /// API key (defaults to the provider's env var or PI_RS_API_KEY).
    #[arg(long)]
    api_key: Option<String>,

    /// Resume an existing session file instead of creating a new one.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Working directory (default: current directory).
    #[arg(long)]
    cwd: Option<PathBuf>,

    /// Full system prompt replacement.
    #[arg(long)]
    system_prompt: Option<String>,

    /// Disable automatic context compaction.
    #[arg(long, default_value_t = false)]
    no_compact: bool,

    /// Disable automatic retry of transient LLM errors.
    #[arg(long, default_value_t = false)]
    no_retry: bool,

    /// Print the session file path and exit.
    #[arg(long, default_value_t = false)]
    print_session: bool,

    /// Enter interactive REPL mode (default when no prompt is given on a TTY).
    #[arg(long, default_value_t = false)]
    repl: bool,

    /// Serve the JSONL RPC protocol on stdin/stdout.
    #[arg(long, default_value_t = false)]
    rpc: bool,

    /// List saved sessions and exit.
    #[arg(long, default_value_t = false)]
    list_sessions: bool,

    /// Continue the most recent session.
    #[arg(long, default_value_t = false)]
    continue_last: bool,

    /// List models (optional glob, e.g. 'anthropic/*') and exit.
    #[arg(long, num_args = 0..=1, default_missing_value = "*")]
    list_models: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let cli = Cli::parse();
    let cwd = cli
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    if cli.list_sessions {
        for f in AgentSession::list_sessions() {
            println!("{}", f.display());
        }
        return Ok(());
    }
    if let Some(pattern) = &cli.list_models {
        let catalog = pi_core::model_catalog::load_catalog().await;
        for entry in pi_core::model_catalog::catalog_list(&catalog, Some(pattern)) {
            println!("{entry}");
        }
        return Ok(());
    }

    let provider = cli
        .provider
        .clone()
        .or_else(|| std::env::var("PI_RS_PROVIDER").ok())
        .or_else(|| load_settings().default_provider)
        .unwrap_or_else(|| "anthropic".to_string());
    let api_key_opt = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("PI_RS_API_KEY").ok());

    // custom providers first, then static registry
    let model_arg = cli.model.as_deref();
    let (model, custom_key_env) =
        match resolve_custom_model(&load_custom_providers(), &provider, model_arg) {
            Some((model, env)) => (model, env),
            None => (
                build_model(&provider, model_arg, cli.base_url.as_deref())
                    .map_err(anyhow::Error::msg)
                    .context("resolving model")?,
                None,
            ),
        };
    // models.json apiKeyEnv names an environment variable; resolve it first
    let custom_key = custom_key_env
        .as_deref()
        .and_then(|name| std::env::var(name).ok())
        .filter(|k| !k.is_empty());
    let api_key = resolve_api_key(
        &provider,
        api_key_opt.as_deref().or(custom_key.as_deref()),
    )
    .context("no API key found (use --api-key, the provider env var or auth.json)")?;
    let api = build_api(&model.api)
        .map_err(anyhow::Error::msg)
        .context("building api")?;

    let resume_path: Option<PathBuf> = if cli.continue_last {
        Some(
            AgentSession::list_sessions()
                .pop()
                .context("no saved sessions to continue")?,
        )
    } else {
        cli.resume.clone()
    };
    let settings = load_settings();

    let session = AgentSession::new(SessionOptions {
        cwd: cwd.clone(),
        model: model.clone(),
        api,
        api_key: Some(api_key),
        force_system_prompt: cli.system_prompt.clone(),
        append_system_prompt: settings.append_system_prompt.clone(),
        resume_file: resume_path,
        compaction_enabled: !cli.no_compact && settings.compaction_enabled.unwrap_or(true),
        auto_retry: if cli.no_retry {
            None
        } else {
            Some(pi_core::agent_session::RetrySettings::default())
        },
    })?;

    if cli.print_session {
        println!("{}", session.session_file().display());
        return Ok(());
    }

    if cli.rpc {
        return Box::pin(rpc::run_rpc(session)).await;
    }

    if cli.repl {
        let interactive = repl::Repl {
            session,
            provider,
            cwd,
        };
        return Box::pin(interactive.run()).await;
    }

    let prompt = match cli.prompt {
        Some(p) => p,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_line(&mut buf)
                .context("reading prompt from stdin")?;
            buf.trim().to_string()
        }
    };
    if prompt.is_empty() {
        anyhow::bail!("empty prompt");
    }

    let mut events = session.subscribe();
    let mut runner = Box::pin(session.prompt(&prompt));
    // captured when the runner branch wins the select; awaiting a finished
    // JoinHandle again would panic
    let mut prompt_result: Option<anyhow::Result<Vec<pi_agent::AgentMessage>>> = None;

    let json_mode = cli.mode == "json";
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if !json_mode {
        // text mode: stream assistant text deltas; annotate tool activity
        let mut current_tool: Option<String> = None;
        let mut runner_done = false;
        loop {
            if runner_done {
                // drain briefly; the final agent_end may still be buffered
                match tokio::time::timeout(
                    std::time::Duration::from_millis(500),
                    events.recv(),
                )
                .await
                {
                    Ok(Ok(event)) => {
                        if matches!(event, AgentEvent::AgentEnd { .. }) {
                            break;
                        }
                        continue;
                    }
                    _ => break,
                }
            }
            let event = tokio::select! {
                event = events.recv(), if !runner_done => event,
                res = &mut runner, if !runner_done => {
                    prompt_result = Some(res.unwrap_or_else(|e| {
                        anyhow::bail!(e.to_string())
                    }));
                    runner_done = true;
                    continue;
                }
            };
            let Ok(event) = event else { break };
            match &event {
                AgentEvent::MessageUpdate { assistant_message_event, .. } => {
                    use pi_ai::events::AssistantMessageEvent as E;
                    if let Some(delta) = match assistant_message_event.as_ref() {
                        E::TextDelta { delta, .. } => Some(delta.clone()),
                        _ => None,
                    } {
                        write!(out, "{delta}").ok();
                        out.flush().ok();
                    }
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    current_tool = Some(tool_name.clone());
                    writeln!(out, "\n[{tool_name}]", tool_name = tool_name).ok();
                    out.flush().ok();
                }
                AgentEvent::ToolExecutionEnd { .. } => {
                    current_tool = None;
                }
                AgentEvent::TurnEnd { .. } => {
                    if current_tool.is_none() {
                        writeln!(out).ok();
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    if runner_done {
                        break;
                    }
                }
                AgentEvent::AutoRetryStart { attempt, max_attempts, delay_ms, error_message } => {
                    writeln!(
                        out,
                        "\n[auto retry {attempt}/{max_attempts} in {delay_ms}ms: {error_message}]"
                    )
                    .ok();
                }
                AgentEvent::AutoRetryEnd { success, .. } => {
                    if !success {
                        writeln!(out, "\n[auto retry exhausted]").ok();
                    }
                }
                _ => {}
            }
        }
        out.flush().ok();
    }

    if json_mode {
        // json mode: every event as one JSON line
        let mut runner_done = false;
        loop {
            if runner_done {
                match tokio::time::timeout(
                    std::time::Duration::from_millis(500),
                    events.recv(),
                )
                .await
                {
                    Ok(Ok(event)) => {
                        if matches!(event, AgentEvent::AgentEnd { .. }) {
                            break;
                        }
                        if let Ok(line) = serde_json::to_string(&event) {
                            writeln!(out, "{line}").ok();
                        }
                        continue;
                    }
                    _ => break,
                }
            }
            let event = tokio::select! {
                event = events.recv(), if !runner_done => event,
                res = &mut runner, if !runner_done => {
                    prompt_result = Some(res.unwrap_or_else(|e| {
                        anyhow::bail!(e.to_string())
                    }));
                    runner_done = true;
                    continue;
                }
            };
            let Ok(event) = event else { break };
            if matches!(&event, AgentEvent::AgentEnd { .. }) {
                if let Ok(line) = serde_json::to_string(&event) {
                    writeln!(out, "{line}").ok();
                }
                continue;
            }
            // expand message_update into the raw assistant stream event for parity
            let value = match &event {
                AgentEvent::MessageUpdate {
                    message,
                    assistant_message_event,
                } => serde_json::json!({
                    "type": "message_update",
                    "message": message.as_ref(),
                    "assistantMessageEvent": assistant_message_event.as_ref(),
                }),
                other => serde_json::to_value(other).unwrap_or(serde_json::json!({})),
            };
            writeln!(out, "{value}").ok();
            out.flush().ok();
        }
    }

    let new_messages = match prompt_result {
        Some(result) => result?,
        None => runner.await??,
    };

    // non-zero exit when the final assistant message errored
    if let Some(last) = new_messages.iter().rev().find_map(|m| match m {
        pi_agent::AgentMessage::Message(pi_ai::types::Message::Assistant(a)) => Some(a),
        _ => None,
    }) {
        if last.stop_reason == pi_ai::types::StopReason::Error
            || last.stop_reason == pi_ai::types::StopReason::Aborted
        {
            eprintln!(
                "error: {}",
                last.error_message.as_deref().unwrap_or("unknown")
            );
            std::process::exit(1);
        }
    }

    if json_mode {
        // emit session summary for consumers
        let summary = serde_json::json!({
            "sessionFile": session.session_file(),
            "newMessages": new_messages.len(),
        });
        println!("{summary}");
    }
    Ok(())
}
