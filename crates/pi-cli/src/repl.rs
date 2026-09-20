//! Line-based interactive mode: REPL loop with slash commands
//! (no TUI; pi's slash-commands surface for the headless port).

use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;

use pi_agent::AgentEvent;
use pi_core::load_prompt_templates;
use pi_core::AgentSession;

pub struct Repl {
    pub session: Arc<AgentSession>,
    pub provider: String,
    pub cwd: PathBuf,
}

impl Repl {
    pub async fn run(&self) -> anyhow::Result<()> {
        let stdin = std::io::stdin();
        let templates = load_prompt_templates(&self.cwd);
        println!(
            "pi-rs REPL — provider={}, session={}\ncommands: /help /quit /branch [entryId] /sessions /templates /template <name> [args] /steer <text>",
            self.provider,
            self.session.session_file().display()
        );

        loop {
            print!("\n> ");
            use std::io::Write as _;
            std::io::stdout().flush().ok();

            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break; // EOF
            }
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            match line.split_whitespace().next().unwrap_or("") {
                "/quit" | "/exit" | "/q" => break,
                "/help" => println!(
                    "/help /quit /branch [entryId] /sessions /templates /template <name> [args...] /steer <text>"
                ),
                "/sessions" => {
                    for f in AgentSession::list_sessions() {
                        println!("{}", f.display());
                    }
                }
                "/templates" => {
                    for t in &templates {
                        println!(
                            "{}{}",
                            t.name,
                            t.description
                                .as_deref()
                                .map(|d| format!(" — {d}"))
                                .unwrap_or_default()
                        );
                    }
                }
                "/template" => {
                    let mut parts = line.splitn(3, ' ');
                    let _cmd = parts.next();
                    match (parts.next(), parts.next()) {
                        (Some(name), rest) => {
                            let args: Vec<String> = rest
                                .unwrap_or("")
                                .split_whitespace()
                                .map(str::to_string)
                                .collect();
                            match templates.iter().find(|t| t.name == name) {
                                Some(t) => {
                                    let expanded = pi_core::expand_template(&t.content, &args);
                                    self.send(&expanded).await;
                                }
                                None => println!("no template named {name}"),
                            }
                        }
                        _ => println!("usage: /template <name> [args...]"),
                    }
                }
                "/branch" => {
                    let target = line.split_whitespace().nth(1).map(str::to_string);
                    match self.session.branch(target).await {
                        Ok(()) => println!("branched"),
                        Err(e) => println!("branch failed: {e}"),
                    }
                }
                "/steer" => {
                    let text = line.trim_start_matches("/steer").trim();
                    if text.is_empty() {
                        println!("usage: /steer <text>");
                    } else {
                        // queued as the next prompt when idle (steering applies
                        // during an active run; REPL runs are synchronous)
                        self.send(text).await;
                    }
                }
                cmd if cmd.starts_with('/') => println!("unknown command {cmd} (try /help)"),
                _ => {
                    let expanded = pi_core::expand_file_references(&line, &self.cwd);
                    self.send(&expanded).await;
                }
            }
        }
        Ok(())
    }

    /// Send one prompt and stream the response to stdout.
    async fn send(&self, text: &str) {
        let mut events = self.session.subscribe();
        let runner = self.session.prompt(text);

        while let Ok(event) = events.recv().await {
            match &event {
                AgentEvent::MessageUpdate { assistant_message_event, .. } => {
                    use pi_ai::events::AssistantMessageEvent as E;
                    if let E::TextDelta { delta, .. } = assistant_message_event.as_ref() {
                        print!("{delta}");
                        use std::io::Write as _;
                        std::io::stdout().flush().ok();
                    }
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    println!("\n[{tool_name}]");
                }
                AgentEvent::AgentEnd { .. } => break,
                _ => {}
            }
        }
        println!();

        if let Err(e) = runner.await {
            eprintln!("error: {e}");
        }
    }
}
