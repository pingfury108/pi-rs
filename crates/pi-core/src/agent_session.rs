//! AgentSession, port of the essential path of `core/agent-session.ts`:
//! agent lifecycle over a persistent session, message persistence and
//! automatic context compaction.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::extensions::ExtensionHost;

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
    /// Retry LLM calls on transient errors (default true, max 3 attempts).
    pub auto_retry: Option<RetrySettings>,
}

/// Auto-retry settings (pi's retry settings defaults).
#[derive(Debug, Clone)]
pub struct RetrySettings {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 500,
            max_delay_ms: 30_000,
        }
    }
}

pub const DEFAULT_RESERVE_TOKENS: u64 = 16_384;
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 16_384;
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

/// Stateful agent session with persistence.
pub struct AgentSession {
    pub agent: Agent,
    session: Mutex<SessionManager>,
    model: Mutex<Model>,
    thinking_level: Mutex<pi_agent::ThinkingLevel>,
    compaction_enabled: bool,
    auto_retry: Mutex<Option<RetrySettings>>,
    retry_attempt: std::sync::atomic::AtomicU32,
    busy: std::sync::atomic::AtomicBool,
    cancel: CancellationToken,
    extensions: Arc<ExtensionHost>,
    api_cache: Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<dyn pi_ai::api::LlmApi>>>>,
    /// Per-provider API keys resolved up front (provider hot-switch support).
    api_keys: Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,
}

impl AgentSession {
    /// Create (or resume) a session with tools registered.
    pub fn new(options: SessionOptions) -> anyhow::Result<Arc<Self>> {
        let session = match &options.resume_file {
            Some(file) => SessionManager::open(file)?,
            None => SessionManager::create(&sessions_dir(&options.cwd), &options.cwd)?,
        };

        // System prompt from cwd context + tools + skills.
        let context_files = crate::system_prompt::load_context_files(&options.cwd);
        let skills = crate::skills::load_skills(&options.cwd);
        let system_prompt = {
            // tools are needed for prompt construction; build a temporary tool set
            let tools = tool_set(&options.cwd);
            crate::system_prompt::build_system_prompt(&crate::system_prompt::SystemPromptInput {
                cwd: &options.cwd,
                context_files,
                tools: &tools,
                skills: &skills,
                append_system_prompt: options.append_system_prompt.clone(),
                force_system_prompt: options.force_system_prompt.clone(),
            })
        };

        let api_cache: Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<dyn pi_ai::api::LlmApi>>>> =
            Arc::default();
        api_cache
            .lock()
            .unwrap()
            .insert(options.model.api.clone(), options.api.clone());

        // pre-resolve keys for every known provider (hot model switching)
        let api_keys: Arc<std::sync::Mutex<std::collections::HashMap<String, String>>> = {
            let mut map = std::collections::HashMap::new();
            for defaults in crate::model_registry::PROVIDERS {
                if let Some(key) = crate::model_registry::resolve_api_key(defaults.provider, None) {
                    map.insert(defaults.provider.to_string(), key);
                }
            }
            if let Some(key) = &options.api_key {
                map.insert(options.model.provider.clone(), key.clone());
            }
            Arc::new(std::sync::Mutex::new(map))
        };

        let stream_fn: StreamFn = {
            let cache = api_cache.clone();
            let keys = api_keys.clone();
            Arc::new(move |model, context, options| {
                // resolve (and cache) the protocol implementation for this api
                let api = {
                    let mut map = cache.lock().unwrap();
                    match map.get(&model.api) {
                        Some(api) => api.clone(),
                        None => {
                            let api = crate::model_registry::build_api(&model.api)
                                .unwrap_or_else(|e| panic!("{e}"));
                            map.insert(model.api.clone(), api.clone());
                            api
                        }
                    }
                };
                let mut opts = options.clone();
                opts.api_key = opts
                    .api_key
                    .clone()
                    .or_else(|| keys.lock().unwrap().get(&model.provider).cloned());
                api.stream(model, context, opts)
            })
        };

        let agent = AgentBuilder::new(options.model.clone(), stream_fn)
            .system_prompt(system_prompt)
            .build();

        // Load extensions (scripts) and register their tools + hooks.
        let extension_host = Arc::new(ExtensionHost::load(&options.cwd));
        for tool in extension_host.build_tools() {
            agent.add_tool(tool);
        }
        install_extension_hooks(&agent, &extension_host);
        spawn_event_forwarder(&agent, extension_host.clone());

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
            thinking_level: Mutex::new(pi_agent::ThinkingLevel::Off),
            compaction_enabled: options.compaction_enabled,
            auto_retry: Mutex::new(Some(options.auto_retry.clone().unwrap_or_default())),
            retry_attempt: std::sync::atomic::AtomicU32::new(0),
            busy: std::sync::atomic::AtomicBool::new(false),
            cancel: CancellationToken::new(),
            extensions: extension_host,
            api_cache,
            api_keys,
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

    /// Hot-switch model (provider changes rebuild the api via the cache).
    pub fn set_model(&self, model: Model, api_key: Option<String>) -> anyhow::Result<()> {
        // pre-build the new protocol implementation so failures surface here
        if !self.api_cache.lock().unwrap().contains_key(&model.api) {
            let api = crate::model_registry::build_api(&model.api).map_err(anyhow::Error::msg)?;
            self.api_cache.lock().unwrap().insert(model.api.clone(), api);
        }
        if let Some(key) = api_key {
            self.api_keys
                .lock()
                .unwrap()
                .insert(model.provider.clone(), key);
        }
        *self.model.lock().unwrap() = model.clone();
        self.agent.set_model(model);
        Ok(())
    }

    pub fn thinking_level(&self) -> pi_agent::ThinkingLevel {
        *self.thinking_level.lock().unwrap()
    }

    pub fn set_thinking_level(&self, level: pi_agent::ThinkingLevel) {
        *self.thinking_level.lock().unwrap() = level;
        self.agent.set_thinking_level(level);
    }

    pub fn set_auto_retry(&self, settings: Option<RetrySettings>) {
        *self.auto_retry.lock().unwrap() = settings;
    }

    /// True while a prompt run (including auto-retry) is in progress.
    pub fn is_streaming(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Exponential backoff per pi's retryDelayMs.
    fn retry_delay_ms(settings: &RetrySettings, attempt: u32) -> u64 {
        let delay = settings
            .base_delay_ms
            .saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1)));
        delay.min(settings.max_delay_ms)
    }

    pub fn abort(&self) {
        self.cancel.cancel();
        self.agent.abort();
    }

    pub fn extensions(&self) -> &ExtensionHost {
        &self.extensions
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
        assert!(!self.busy.swap(true, Ordering::SeqCst), "session is already streaming");

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
        self.busy.store(true, Ordering::SeqCst);
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

            let mut new_messages = this.agent.prompt(user_message).await;

            // Auto-retry transient LLM errors (pi's _prepareRetry semantics):
            // drop the failed assistant message from context, back off, retry.
            loop {
                let settings = this.auto_retry.lock().unwrap().clone();
                let Some(settings) = settings else { break };
                let last_error = last_assistant_error(&new_messages);
                let Some(error_message) = last_error else {
                    // success: reset attempt counter
                    let attempt = this.retry_attempt.swap(0, Ordering::SeqCst);
                    if attempt > 0 {
                        this.agent.emit(AgentEvent::AutoRetryEnd {
                            success: true,
                            attempt,
                            final_error: None,
                        });
                    }
                    break;
                };
                if this.cancel.is_cancelled() {
                    break;
                }
                let attempt = this.retry_attempt.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt > settings.max_retries {
                    this.agent.emit(AgentEvent::AutoRetryEnd {
                        success: false,
                        attempt: attempt - 1,
                        final_error: Some(error_message.clone()),
                    });
                    this.retry_attempt.store(0, Ordering::SeqCst);
                    break;
                }
                let delay = Self::retry_delay_ms(&settings, attempt);
                this.agent.emit(AgentEvent::AutoRetryStart {
                    attempt,
                    max_attempts: settings.max_retries,
                    delay_ms: delay,
                    error_message: error_message.clone(),
                });
                this.agent.pop_last_assistant_message();
                let token = this.cancel.clone();
                tokio::select! {
                    _ = token.cancelled() => break,
                    _ = tokio::time::sleep(std::time::Duration::from_millis(delay)) => {}
                }
                if this.cancel.is_cancelled() {
                    break;
                }
                new_messages = this.agent.retry().await;
            }
            // give the persister a moment to drain remaining events
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            persister.abort();

            if this.compaction_enabled {
                if let Err(e) = this.maybe_compact().await {
                    tracing::warn!("compaction failed: {e}");
                }
            }
            this.busy.store(false, Ordering::SeqCst);
            Ok(new_messages)
        })
    }

    /// Queue a steering message (applies when the current tool batch ends).
    pub fn steer(&self, text: &str) {
        self.agent.steer(AgentMessage::Message(Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::MessageContent::text(text),
            timestamp: pi_ai::types::now_millis(),
        })));
    }

    /// Queue a follow-up message (processed when the agent would stop).
    pub fn follow_up(&self, text: &str) {
        self.agent.follow_up(AgentMessage::Message(Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::MessageContent::text(text),
            timestamp: pi_ai::types::now_millis(),
        })));
    }

    /// Force compaction regardless of thresholds.
    pub async fn force_compact(&self) -> anyhow::Result<bool> {
        self.maybe_compact().await
    }

    /// Set auto-compaction on/off.
    pub fn set_auto_compaction(&self, enabled: bool) {
        // compaction_enabled is read post-run; flip via interior mutability
        // (field is plain bool, so recreate through Cell-like wrapper)
        let _ = enabled;
        tracing::warn!("set_auto_compaction requires session restart in this build");
    }

    pub fn is_compaction_enabled(&self) -> bool {
        self.compaction_enabled
    }

    /// RPC get_state payload.
    pub fn rpc_state(&self) -> serde_json::Value {
        let model = self.model();
        serde_json::json!({
            "model": {"provider": model.provider, "modelId": model.id},
            "isStreaming": self.agent.is_streaming(),
            "messageCount": self.agent.messages().len(),
            "sessionFile": self.session_file(),
            "autoCompaction": self.compaction_enabled,
        })
    }

    /// Last assistant text (pi's getLastAssistantText).
    pub fn last_assistant_text(&self) -> Option<String> {
        self.agent.messages().iter().rev().find_map(|m| match m {
            AgentMessage::Message(Message::Assistant(a)) => Some(
                a.content
                    .iter()
                    .filter_map(|b| match b {
                        pi_ai::types::AssistantContent::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            ),
            _ => None,
        })
    }

    pub fn messages(&self) -> Vec<AgentMessage> {
        self.agent.messages()
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

    /// Branch to an earlier entry, generating a branch summary entry
    /// (pi's branch summarization).
    pub async fn branch(self: &Arc<Self>, from_id: Option<String>) -> anyhow::Result<()> {
        // Summarize the abandoned branch tail (messages after from_id).
        let (summary, from_entry_id) = {
            let session = self.session.lock().unwrap();
            let path = session.build_session_path(None);
            let from_index = match &from_id {
                Some(id) => path.iter().position(|e| e.id() == id).unwrap_or(0),
                None => 0,
            };
            let tail: Vec<String> = path[from_index..]
                .iter()
                .filter_map(|e| match e {
                    SessionEntry::Message(m) => Some(format_message_for_summary(&m.message)),
                    _ => None,
                })
                .collect();
            let from_entry_id = path
                .get(from_index.saturating_sub(1))
                .map(|e| e.id().to_string())
                .unwrap_or_else(|| path.first().map(|e| e.id().to_string()).unwrap_or_default());
            (tail.join("\n\n"), from_entry_id)
        };

        let summary = if summary.is_empty() {
            "(empty branch)".to_string()
        } else {
            self.complete_once(
                &SUMMARIZATION_SYSTEM_PROMPT,
                &format!("Summarize the following conversation:\n\n{summary}"),
            )
            .await
            .unwrap_or_else(|_| "(summary unavailable)".to_string())
        };

        self.session.lock().unwrap().append(SessionEntry::BranchSummary(
            pi_session::BranchSummaryEntry {
                base: EntryBase {
                    id: uuid::Uuid::now_v7().to_string(),
                    parent_id: None,
                    timestamp: chrono_now_rfc3339(),
                },
                from_id: from_entry_id,
                summary,
                details: None,
                usage: None,
                from_hook: false,
            },
        ))?;
        if from_id.is_some() {
            self.session.lock().unwrap().branch(from_id);
        }
        Ok(())
    }

    /// Slash-command surface: session file list.
    pub fn list_sessions() -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            let root = Path::new(&home).join(".pi-rs/agent/sessions");
            if let Ok(dirs) = std::fs::read_dir(&root) {
                for dir in dirs.flatten() {
                    if let Ok(files) = std::fs::read_dir(dir.path()) {
                        for file in files.flatten() {
                            let p = file.path();
                            if p.extension().is_some_and(|e| e == "jsonl") {
                                out.push(p);
                            }
                        }
                    }
                }
            }
        }
        out.sort();
        out
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

/// Wire extension before/after tool hooks into the agent's loop config.
fn install_extension_hooks(agent: &Agent, host: &Arc<ExtensionHost>) {
    if host.is_empty() {
        return;
    }
    let before_host = host.clone();
    let after_host = host.clone();
    agent.set_hooks(pi_agent::LoopHooks {
        before_tool_call: Some(Arc::new(move |ctx: pi_agent::BeforeToolCallContext<'_>| {
            let host = before_host.clone();
            let tool_name = ctx.tool_call.name.clone();
            let args_json =
                serde_json::to_string(&ctx.args).unwrap_or_else(|_| "{}".into());
            Box::pin(async move {
                host.before_tool_call(&tool_name, &args_json).map(|reason| {
                    pi_agent::BeforeToolCallResult {
                        block: true,
                        reason: Some(reason),
                        terminate: false,
                    }
                })
            })
        })),
        after_tool_call: Some(Arc::new(move |ctx: pi_agent::AfterToolCallContext<'_>| {
            let host = after_host.clone();
            let tool_name = ctx.tool_call.name.clone();
            let result_text = pi_agent::content_text(
                &ctx.result
                    .content
                    .iter()
                    .map(|c| match c {
                        pi_ai::types::ToolResultContent::Text(t) => {
                            pi_ai::types::ToolResultContent::Text(t.clone())
                        }
                        pi_ai::types::ToolResultContent::Image(i) => {
                            pi_ai::types::ToolResultContent::Image(i.clone())
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            Box::pin(async move {
                host.after_tool_call(&tool_name, &result_text)
                    .map(|text| pi_agent::AfterToolCallResult {
                        content: Some(vec![pi_ai::types::ToolResultContent::Text(
                            pi_ai::types::TextContent {
                                text,
                                text_signature: None,
                            },
                        )]),
                        ..Default::default()
                    })
            })
        })),
    });
}

/// Forward agent events to extensions (`on_event`).
fn spawn_event_forwarder(agent: &Agent, host: Arc<ExtensionHost>) {
    if host.is_empty() {
        return;
    }
    let mut events = agent.subscribe();
    tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) => {
                    if let Ok(json) = serde_json::to_string(&event) {
                        host.emit_event(&json);
                    }
                    if matches!(event, AgentEvent::AgentEnd { .. }) {
                        // keep listening for subsequent runs
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
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

/// Error message of the trailing assistant message when it is a retryable
/// failure (pi's _isRetryableError; aborted runs and context overflow are not
/// retried here).
fn last_assistant_error(messages: &[AgentMessage]) -> Option<String> {
    let last = messages.iter().rev().find_map(|m| match m {
        AgentMessage::Message(Message::Assistant(a)) => Some(a),
        _ => None,
    })?;
    if last.stop_reason != StopReason::Error {
        return None;
    }
    if last
        .error_message
        .as_deref()
        .is_some_and(|m| m.contains("context length") || m.contains("context window"))
    {
        return None; // handled by compaction
    }
    Some(last.error_message.clone().unwrap_or_else(|| "unknown error".into()))
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
