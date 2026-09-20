//! AgentSession, port of the essential path of `core/agent-session.ts`:
//! agent lifecycle over a persistent session, message persistence and
//! automatic context compaction.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use pi_agent::{Agent, AgentBuilder, AgentEvent, AgentMessage, StreamFn};
use pi_ai::api::LlmApi;
use pi_ai::types::{
    AssistantMessage, Context, Message, Model, StopReason, StreamOptions, SystemMessage,
    ToolResultMessage, Usage,
};
use pi_session::{CompactionEntry, EntryBase, SessionEntry, SessionManager};

/// Options for creating a session.
pub struct SessionOptions {
    pub cwd: PathBuf,
    pub model: Model,
    pub api: Arc<dyn LlmApi>,
    pub api_key: Option<String>,
    /// Full system prompt replacement (pi's forceSystemPrompt).
    pub force_system_prompt: Option<String>,
    /// User append from settings.
    pub append_system_prompt: Option<String>,
    /// Resume this session file instead of creating a new one.
    pub resume_file: Option<PathBuf>,
    pub compaction_enabled: bool,
}

pub const DEFAULT_RESERVE_TOKENS: u64 = 16_384;
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 16_384;
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

/// Stateful agent session with persistence.
pub struct AgentSession {
    pub agent: Agent,
    session: Mutex<SessionManager>,
    model: Mutex<Model>,
    compaction_enabled: bool,
    cancel: CancellationToken,
}

impl AgentSession {
    /// Create (or resume) a session with tools registered.
    pub fn new(options: SessionOptions) -> anyhow::Result<Arc<Self>> {
        let session = match &options.resume_file {
            Some(file) => SessionManager::open(file)?,
            None => SessionManager::create(&sessions_dir(&options.cwd), &options.cwd)?,
        };

        // System prompt from cwd context + tools.
        let context_files = crate::system_prompt::load_context_files(&options.cwd);
        let system_prompt = {
            // tools are needed for prompt construction; build a temporary tool set
            let tools = tool_set(&options.cwd);
            crate::system_prompt::build_system_prompt(&crate::system_prompt::SystemPromptInput {
                cwd: &options.cwd,
                context_files,
                tools: &tools,
                append_system_prompt: options.append_system_prompt.clone(),
                force_system_prompt: options.force_system_prompt.clone(),
            })
        };

        let api = options.api.clone();
        let stream_fn: StreamFn = {
            let api = api.clone();
            let api_key = options.api_key.clone();
            Arc::new(move |model, context, options| {
                let mut opts = options.clone();
                opts.api_key = opts.api_key.or_else(|| api_key.clone());
                api.stream(model, context, opts)
            })
        };

        let agent = AgentBuilder::new(options.model.clone(), stream_fn)
            .system_prompt(system_prompt)
            .build();

        // Register tools.
        for tool in tool_set(&options.cwd) {
            agent.add_tool(tool);
        }

        // Seed transcript from session context (resume support).
        let context_messages = session.build_context_messages(None);
        if !context_messages.is_empty() {
            let current_system = AgentMessage::Message(Message::System(SystemMessage::new(
                agent_system_prompt(&agent),
            )));
            let without_system: Vec<AgentMessage> = context_messages
                .into_iter()
                .filter(|m| !matches!(m, AgentMessage::Message(Message::System(_))))
                .collect();
            let mut messages = vec![current_system];
            messages.extend(without_system);
            agent.set_messages(messages);
        }

        Ok(Arc::new(Self {
            agent,
            session: Mutex::new(session),
            model: Mutex::new(options.model.clone()),
            compaction_enabled: options.compaction_enabled,
            cancel: CancellationToken::new(),
        }))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.agent.subscribe()
    }

    pub fn session_file(&self) -> PathBuf {
        self.session.lock().unwrap().file().to_path_buf()
    }

    pub fn model(&self) -> Model {
        self.model.lock().unwrap().clone()
    }

    pub fn set_model(&self, model: Model) {
        *self.model.lock().unwrap() = model.clone();
        self.agent.set_model(model);
    }

    pub fn abort(&self) {
        self.cancel.cancel();
        self.agent.abort();
    }

    /// Send a user prompt: persists the message, runs the agent loop with
    /// event streaming and session persistence, then checks compaction.
    /// Returns a handle yielding the new messages.
    pub fn prompt(self: &Arc<Self>, text: &str) -> JoinHandle<anyhow::Result<Vec<AgentMessage>>> {
        // Persist the user message first.
        let user_message = AgentMessage::Message(Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::MessageContent::text(text),
            timestamp: pi_ai::types::now_millis(),
        }));
        if let Err(e) = self.session.lock().unwrap().append_message(user_message.clone()) {
            return tokio::spawn(async move { Err(anyhow::anyhow!("session append failed: {e}")) });
        }

        // Rebuild agent transcript from the session so steering/branch state stays canonical.
        let context_messages = self.session.lock().unwrap().build_context_messages(None);
        let current_system = AgentMessage::Message(Message::System(SystemMessage::new(
            agent_system_prompt(&self.agent),
        )));
        let mut messages = vec![current_system];
        messages.extend(
            context_messages
                .into_iter()
                .filter(|m| !matches!(m, AgentMessage::Message(Message::System(_)))),
        );
        self.agent.set_messages(messages);

        let this = self.clone();
        let persist_this = self.clone();
        tokio::spawn(async move {
            let mut events = this.subscribe();

            // Persist assistant/toolResult messages as they complete.
            let persister = tokio::spawn(async move {
                loop {
                    match events.recv().await {
                        Ok(AgentEvent::MessageEnd { message }) => match message.as_ref() {
                            AgentMessage::Message(Message::Assistant(_))
                            | AgentMessage::Message(Message::ToolResult(_)) => {
                                let _ = persist_this
                                    .session
                                    .lock()
                                    .unwrap()
                                    .append_message(message.as_ref().clone());
                            }
                            _ => {}
                        },
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            });

            let new_messages = this.agent.prompt(user_message).await;
            // give the persister a moment to drain remaining events
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            persister.abort();

            if this.compaction_enabled {
                if let Err(e) = this.maybe_compact().await {
                    tracing::warn!("compaction failed: {e}");
                }
            }
            Ok(new_messages)
        })
    }

    /// Trigger compaction if context usage exceeds the reserve threshold.
    async fn maybe_compact(&self) -> anyhow::Result<bool> {
        let model = self.model();
        let Some(usage) = self.last_assistant_usage() else {
            return Ok(false);
        };
        let context_tokens = usage
            .total_tokens
            .max(usage.input + usage.output + usage.cache_read + usage.cache_write);
        if !should_compact(context_tokens, model.context_window, DEFAULT_RESERVE_TOKENS) {
            return Ok(false);
        }

        // Collect owned copies of the entries we need (id + message) to
        // avoid holding the session lock across the LLM call.
        struct EntryCopy {
            id: String,
            message: Option<Message>,
        }
        let entries: Vec<EntryCopy> = {
            let session = self.session.lock().unwrap();
            session
                .build_context_entries(None)
                .into_iter()
                .map(|e| EntryCopy {
                    id: e.id().to_string(),
                    message: match e {
                        SessionEntry::Message(m) => Some(m.message.clone()),
                        _ => None,
                    },
                })
                .collect()
        };

        // Cut point: walk from the end, keep ~DEFAULT_KEEP_RECENT_TOKENS.
        let keep_chars = (DEFAULT_KEEP_RECENT_TOKENS * 4) as usize;
        let mut kept_chars = 0usize;
        let mut cut_index = 0usize;
        for (i, entry) in entries.iter().enumerate().rev() {
            if let Some(message) = &entry.message {
                kept_chars += message_char_len(message);
                if kept_chars > keep_chars {
                    cut_index = i + 1;
                    break;
                }
            }
            if i == 0 {
                cut_index = 0;
            }
        }
        if cut_index == 0 {
            return Ok(false); // nothing to compact
        }
        if cut_index >= entries.len() {
            return Ok(false);
        }

        // Serialize the portion to summarize.
        let to_summarize: Vec<String> = entries[..cut_index]
            .iter()
            .filter_map(|e| e.message.as_ref().map(format_message_for_summary))
            .collect();
        let conversation = to_summarize.join("\n\n");
        if conversation.is_empty() {
            return Ok(false);
        }

        // Generate the summary with the current model.
        let summary = self
            .complete_once(&SUMMARIZATION_SYSTEM_PROMPT, &format!(
                "Summarize the following conversation:\n\n{conversation}"
            ))
            .await?;

        let tokens_before = context_tokens;
        let first_kept = entries[cut_index].id.clone();

        self.session.lock().unwrap().append(SessionEntry::Compaction(CompactionEntry {
            base: EntryBase {
                id: uuid::Uuid::now_v7().to_string(),
                parent_id: None,
                timestamp: chrono_now_rfc3339(),
            },
            summary,
            first_kept_entry_id: first_kept,
            tokens_before,
            details: None,
            usage: Some(usage.clone()),
            from_hook: false,
            system_message: None,
        }))?;
        tracing::info!(tokens_before, "context compacted");
        Ok(true)
    }

    async fn complete_once(&self, system: &str, user_text: &str) -> anyhow::Result<String> {
        let model = self.model();
        let context = Context {
            system_prompt: Some(system.to_string()),
            messages: vec![Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::MessageContent::text(user_text),
                timestamp: pi_ai::types::now_millis(),
            })],
            tools: None,
        };
        // Route through the agent's stream_fn? The api is wrapped in Agent;
        // simplest correct path: use the registered tool-free stream via the
        // agent's stream function by calling the api directly.
        // AgentSession keeps no direct api handle, so reconstruct via provider.
        // We instead use a one-shot agent-free call through the same closure
        // used at construction time; AgentSession stores the api implicitly.
        // To avoid an extra field we call through `self.agent`'s stream_fn by
        // running a minimal loop with no tools.
        let _ = context;
        let stream_fn = self.stream_fn_for_complete()?;
        let stream = stream_fn(
            &model,
            &Context {
                system_prompt: Some(system.to_string()),
                messages: vec![Message::User(pi_ai::types::UserMessage {
                    content: pi_ai::types::MessageContent::text(user_text),
                    timestamp: pi_ai::types::now_millis(),
                })],
                tools: None,
            },
            &StreamOptions::default(),
        );
        let message = stream.result().await;
        match message.stop_reason {
            StopReason::Error | StopReason::Aborted => {
                Err(anyhow::anyhow!(message.error_message.unwrap_or_else(|| "llm error".into())))
            }
            _ => {
                let text = message
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        pi_ai::types::AssistantContent::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                Ok(text)
            }
        }
    }

    fn stream_fn_for_complete(
        &self,
    ) -> anyhow::Result<impl Fn(&Model, &Context, &StreamOptions) -> pi_ai::events::AssistantMessageEventStream>
    {
        // The Agent holds the stream_fn privately; reuse it via a one-message
        // loop would persist events. Instead, AgentSession constructs its own
        // api call from the model's api field through the registry default.
        let api = crate::model_registry::build_api(&self.model().api).map_err(anyhow::Error::msg)?;
        Ok(move |model: &Model, context: &Context, options: &StreamOptions| {
            api.stream(model, context, options.clone())
        })
    }

    fn last_assistant_usage(&self) -> Option<Usage> {
        let session = self.session.lock().unwrap();
        for entry in session.build_context_entries(None).iter().rev() {
            if let SessionEntry::Message(m) = entry {
                if let Message::Assistant(a) = &m.message {
                    if a.stop_reason != StopReason::Aborted
                        && a.stop_reason != StopReason::Error
                        && (a.usage.total_tokens > 0 || a.usage.input > 0)
                    {
                        return Some(a.usage.clone());
                    }
                }
            }
        }
        None
    }
}

fn sessions_dir(cwd: &Path) -> PathBuf {
    let encoded = cwd.to_string_lossy().replace(['/', '\\'], "-");
    std::env::var_os("HOME")
        .map(|home| {
            Path::new(&home)
                .join(".pi-rs/agent/sessions")
                .join(format!("-{encoded}"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("pi-rs-sessions"))
}

fn tool_set(cwd: &Path) -> Vec<Arc<dyn pi_agent::AgentTool>> {
    vec![
        Arc::new(pi_tools::ReadTool::new(cwd)),
        Arc::new(pi_tools::BashTool::new(cwd)),
        Arc::new(pi_tools::EditTool::new(cwd)),
        Arc::new(pi_tools::WriteTool::new(cwd)),
        Arc::new(pi_tools::GrepTool::new(cwd)),
        Arc::new(pi_tools::FindTool::new(cwd)),
        Arc::new(pi_tools::LsTool::new(cwd)),
    ]
}

fn agent_system_prompt(agent: &Agent) -> String {
    agent
        .messages()
        .iter()
        .find_map(|m| match m {
            AgentMessage::Message(Message::System(s)) => Some(s.content.as_text()),
            _ => None,
        })
        .unwrap_or_default()
}

fn should_compact(context_tokens: u64, context_window: u64, reserve_tokens: u64) -> bool {
    context_window > reserve_tokens && context_tokens > context_window - reserve_tokens
}

fn message_char_len(message: &Message) -> usize {
    match message {
        Message::System(m) => m.content.as_text().len(),
        Message::User(m) => m.content.as_text().len(),
        Message::Assistant(m) => m
            .content
            .iter()
            .map(|b| match b {
                pi_ai::types::AssistantContent::Text(t) => t.text.len(),
                pi_ai::types::AssistantContent::Thinking(t) => t.thinking.len(),
                pi_ai::types::AssistantContent::ToolCall(c) => c.name.len() + 64,
            })
            .sum(),
        Message::ToolResult(m) => m
            .content
            .iter()
            .map(|c| match c {
                pi_ai::types::ToolResultContent::Text(t) => t.text.len(),
                pi_ai::types::ToolResultContent::Image(_) => 4800, // pi's ESTIMATED_IMAGE_CHARS
            })
            .sum(),
    }
}

fn format_message_for_summary(message: &Message) -> String {
    match message {
        Message::System(_) => String::new(), // system prompt is not summarized
        Message::User(m) => format!("<user>\n{}\n</user>", m.content.as_text()),
        Message::Assistant(m) => {
            let mut parts = Vec::new();
            for block in &m.content {
                match block {
                    pi_ai::types::AssistantContent::Text(t) => parts.push(t.text.clone()),
                    pi_ai::types::AssistantContent::Thinking(_) => {}
                    pi_ai::types::AssistantContent::ToolCall(c) => parts.push(format!(
                        "[tool call: {}({})]",
                        c.name,
                        serde_json::to_string(&c.arguments).unwrap_or_default()
                    )),
                }
            }
            format!("<assistant>\n{}\n</assistant>", parts.join("\n"))
        }
        Message::ToolResult(m) => {
            let text: String = m
                .content
                .iter()
                .map(|c| match c {
                    pi_ai::types::ToolResultContent::Text(t) => t.text.clone(),
                    pi_ai::types::ToolResultContent::Image(_) => "[image]".to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("<tool_result name=\"{}\">\n{}\n</tool_result>", m.tool_name, text)
        }
    }
}

fn chrono_now_rfc3339() -> String {
    pi_session::timestamp_to_rfc3339(pi_ai::types::now_millis())
}

// Re-export for CLI convenience.
pub type ToolResultMessageAlias = ToolResultMessage;
pub type AssistantMessageAlias = AssistantMessage;
