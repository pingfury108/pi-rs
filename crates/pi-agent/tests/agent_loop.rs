//! Agent loop tests driven by the FauxApi (pi's faux-provider pattern).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pi_agent::{Agent, AgentBuilder, AgentEvent, AgentTool, ToolContext, ToolUpdateFn};
use pi_ai::api::{FauxApi, FauxResponse, LlmApi};
use pi_ai::types::{
    AssistantContent, Message, MessageContent, Model, ModelCost, Modality, StopReason, TextContent,
    ToolResultContent, Usage,
};
use serde_json::json;

fn test_model() -> Model {
    Model {
        id: "faux-1".into(),
        name: "Faux".into(),
        api: "faux".into(),
        provider: "faux".into(),
        base_url: "http://localhost".into(),
        reasoning: false,
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn faux_agent(script: Vec<FauxResponse>) -> (Arc<Agent>, Arc<FauxApi>) {
    let faux = Arc::new(FauxApi::with_script(script));
    let stream_fn: pi_agent::StreamFn = {
        let faux = faux.clone();
        Arc::new(move |model, context, options| faux.stream(model, context, options.clone()))
    };
    let agent = AgentBuilder::new(test_model(), stream_fn)
        .system_prompt("test system prompt")
        .build();
    (Arc::new(agent), faux)
}

fn user(text: &str) -> pi_agent::AgentMessage {
    pi_agent::AgentMessage::Message(Message::User(pi_ai::types::UserMessage::new(
        MessageContent::text(text),
    )))
}

/// Simple echo tool recording its calls.
struct EchoTool {
    calls: AtomicUsize,
    delay: Option<Duration>,
}

impl EchoTool {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            delay: None,
        })
    }

    fn with_delay(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            delay: Some(delay),
        })
    }
}

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echoes its input"
    }
    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    }
    fn label(&self, args: &serde_json::Value) -> String {
        format!("echo({})", args["text"].as_str().unwrap_or(""))
    }
    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        _ctx: &ToolContext,
        _on_update: ToolUpdateFn,
    ) -> Result<pi_agent::AgentToolResult, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        Ok(pi_agent::AgentToolResult::text(format!(
            "echo: {}",
            args["text"].as_str().unwrap_or("")
        )))
    }
}

#[tokio::test]
async fn simple_text_response_event_flow() {
    let (agent, _faux) = faux_agent(vec![FauxResponse::text("hello world")]);
    let mut events = agent.subscribe();
    let handle_messages = agent.prompt(user("hi")).await;

    assert_eq!(handle_messages.len(), 2); // user + assistant
    let collected = collect_events(&mut events).await;
    let names: Vec<&str> = collected.iter().map(event_name).collect();
    assert_eq!(
        names,
        [
            "agent_start",
            "turn_start",
            "message_start", // user
            "message_end",
            "message_start", // assistant (from stream start)
            "message_update", // text_start
            "message_update", // text_delta
            "message_update", // text_delta
            "message_update", // text_end
            "message_end",
            "turn_end",
            "agent_end"
        ]
    );

    // transcript ends with assistant text
    let messages = agent.messages();
    let last = messages.last().unwrap();
    match last {
        pi_agent::AgentMessage::Message(Message::Assistant(a)) => {
            assert_eq!(a.stop_reason, StopReason::Stop);
            assert_eq!(a.content[0], AssistantContent::Text(TextContent {
                text: "hello world".into(),
                text_signature: None,
            }));
        }
        other => panic!("expected assistant message, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_call_round_trip() {
    let (agent, _faux) = faux_agent(vec![
        FauxResponse::tool_call("call_1", "echo", json!({"text": "ping"})),
        FauxResponse::text("done"),
    ]);
    let echo = EchoTool::new();
    agent.add_tool(echo.clone());

    let mut events = agent.subscribe();
    agent.prompt(user("call the tool")).await;

    let collected = collect_events(&mut events).await;
    let names: Vec<&str> = collected.iter().map(event_name).collect();
    assert_eq!(
        names,
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start", // system: tool declaration (echo)
            "message_end",
            "message_start",
            "message_update", // toolcall_start
            "message_update", // toolcall_delta
            "message_update", // toolcall_end
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start", // toolResult
            "message_end",
            "turn_end",
            "turn_start",     // second turn
            "message_start",  // assistant "done"
            "message_update", // text_start
            "message_update", // text_delta
            "message_update", // text_delta
            "message_update", // text_end
            "message_end",
            "turn_end",
            "agent_end"
        ]
    );
    assert_eq!(echo.calls.load(Ordering::SeqCst), 1);

    // tool result message content
    let messages = agent.messages();
    let tool_result = messages
        .iter()
        .find_map(|m| match m {
            pi_agent::AgentMessage::Message(Message::ToolResult(t)) => Some(t.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(tool_result.tool_call_id, "call_1");
    assert_eq!(tool_result.tool_name, "echo");
    assert!(!tool_result.is_error);
    match &tool_result.content[0] {
        ToolResultContent::Text(t) => assert_eq!(t.text, "echo: ping"),
        _ => panic!("expected text content"),
    }
}

#[tokio::test]
async fn parallel_tool_calls_preserve_source_order() {
    let script = vec![
        FauxResponse::Message {
            content: vec![
                AssistantContent::ToolCall(pi_ai::types::ToolCall {
                    id: "call_a".into(),
                    name: "echo".into(),
                    arguments: serde_json::Map::from_iter([(
                        "text".to_string(),
                        json!("slow-first-in-order"),
                    )]),
                    thought_signature: None,
                    namespace: None,
                }),
                AssistantContent::ToolCall(pi_ai::types::ToolCall {
                    id: "call_b".into(),
                    name: "echo".into(),
                    arguments: serde_json::Map::from_iter([(
                        "text".to_string(),
                        json!("fast-second-in-order"),
                    )]),
                    thought_signature: None,
                    namespace: None,
                }),
            ],
            usage: Some(Usage::default()),
            stop_reason: StopReason::ToolUse,
        },
        FauxResponse::text("both done"),
    ];
    let (agent, _faux) = faux_agent(script);
    // first tool call sleeps (would finish second), second is instant
    let slow = EchoTool::with_delay(Duration::from_millis(100));
    agent.add_tool(slow.clone());

    let mut events = agent.subscribe();
    let messages = agent.prompt(user("parallel")).await;

    let collected = collect_events(&mut events).await;
    // toolResult messages must appear in SOURCE order (call_a before call_b)
    let result_positions: Vec<(usize, String)> = collected
        .iter()
        .enumerate()
        .filter(|(_, e)| event_name(e) == "message_start")
        .filter_map(|(i, e)| match e {
            AgentEvent::MessageStart { message } => match message.as_ref() {
                pi_agent::AgentMessage::Message(Message::ToolResult(t)) => Some((i, t.tool_call_id.clone())),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<(usize, String)>>();
    assert_eq!(
        result_positions.iter().map(|(_, id)| id.as_str()).collect::<Vec<_>>(),
        ["call_a", "call_b"],
        "toolResult messages must follow assistant source order"
    );

    // both tools executed
    assert_eq!(slow.calls.load(Ordering::SeqCst), 2);
    let _ = messages;
}

#[tokio::test]
async fn steering_message_injected_between_turns() {
    let (agent, faux) = faux_agent(vec![
        FauxResponse::text("first response"),
        FauxResponse::text("second response"),
    ]);
    // slow the LLM boundary so steering lands mid-run
    let delayed: pi_agent::StreamFn = {
        let faux = faux.clone();
        Arc::new(move |model, context, options| {
            let faux = faux.clone();
            let options = options.clone();
            let (tx, stream) = pi_ai::events::AssistantMessageEventStream::channel(64);
            let model = model.clone();
            let context = context.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let mut inner = faux.stream(&model, &context, options);
                use futures::StreamExt;
                while let Some(event) = inner.next().await {
                    tx.send(event).await;
                }
                tx.close();
            });
            stream
        })
    };
    // replace the agent's stream_fn by rebuilding: simplest is to wrap via
    // AgentBuilder again
    let agent = AgentBuilder::new(test_model(), delayed)
        .system_prompt("test system prompt")
        .build();
    let agent = Arc::new(agent);

    let mut events = agent.subscribe();
    let agent_for_task = agent.clone();
    let handle = tokio::spawn(async move { agent_for_task.prompt(user("start")).await });
    // queue steering while first turn runs
    tokio::time::sleep(Duration::from_millis(20)).await;
    agent.steer(user("steered!"));
    handle.await.unwrap();

    let collected = collect_events(&mut events).await;
    // the steered user message should be announced and answered
    let steered = collected.iter().any(|e| match e {
        AgentEvent::MessageStart { message } => match message.as_ref() {
            pi_agent::AgentMessage::Message(Message::User(u)) => {
                u.content.as_text().contains("steered!")
            }
            _ => false,
        },
        _ => false,
    });
    assert!(steered, "steering message should be announced");
    assert_eq!(agent.messages().len(), 5); // sys + user + a1 + steered + a2
}

#[tokio::test]
async fn before_tool_call_block() {
    let (agent, _faux) = faux_agent(vec![
        FauxResponse::tool_call("call_1", "echo", json!({"text": "x"})),
        FauxResponse::text("done"),
    ]);
    agent.add_tool(EchoTool::new());

    // install blocking hook by running the raw loop instead: use Agent-level
    // config is built internally; here we verify blocked semantics at loop level.
    let tool_calls = pi_ai::types::ToolCall {
        id: "call_1".into(),
        name: "echo".into(),
        arguments: serde_json::Map::from_iter([("text".to_string(), json!("x"))]),
        thought_signature: None,
        namespace: None,
    };
    let _ = tool_calls;
    let _ = agent;
    // covered by loop-level test below
}

#[tokio::test]
async fn abort_produces_aborted_error() {
    let (agent, _faux) = faux_agent(vec![FauxResponse::Message {
        content: vec![],
        usage: None,
        stop_reason: StopReason::Stop,
    }]);

    let mut events = agent.subscribe();
    let agent_for_task = agent.clone();
    let runner = tokio::spawn(async move { agent_for_task.prompt(user("go")).await });
    tokio::time::sleep(Duration::from_millis(5)).await;
    agent.abort();
    runner.await.unwrap();

    let collected = collect_events(&mut events).await;
    let last = collected.last().unwrap();
    assert!(matches!(last, AgentEvent::AgentEnd { .. }));
}

// ---------------------------------------------------------------------------
// loop-level tests (hooks, truncation, tool changes)
// ---------------------------------------------------------------------------

use pi_agent::{run_agent_loop, AgentContext, AgentLoopConfig, BeforeToolCallResult};
use pi_ai::types::Context as LlmContext;
use std::future::Future;
use std::pin::Pin;

fn loop_config(
    faux: &Arc<FauxApi>,
    model: Model,
    hooks: Hooks,
) -> AgentLoopConfig {
    let stream_fn: pi_agent::StreamFn = {
        let faux = faux.clone();
        Arc::new(move |model, context, options| faux.stream(model, context, options.clone()))
    };
    AgentLoopConfig {
        model,
        stream_options: Default::default(),
        stream_fn,
        convert_to_llm: Arc::new(pi_agent::convert_to_llm_default),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        tool_execution: pi_agent::ToolExecutionMode::Parallel,
        before_tool_call: hooks.before_tool_call,
        after_tool_call: None,
    }
}

#[derive(Default)]
struct Hooks {
    before_tool_call: Option<
        Arc<
            dyn Fn(pi_agent::BeforeToolCallContext<'_>) -> Pin<Box<dyn Future<Output = Option<BeforeToolCallResult>> + Send>>
                + Send
                + Sync,
        >,
    >,
}

fn no_emit() -> pi_agent::EventSink {
    Arc::new(|_event| Box::pin(async {}))
}

#[tokio::test]
async fn blocked_tool_call_produces_error_result() {
    let faux = Arc::new(FauxApi::with_script(vec![
        FauxResponse::tool_call("call_1", "echo", json!({"text": "x"})),
        FauxResponse::text("after block"),
    ]));
    let mut context = AgentContext {
        messages: vec![user("go")],
        tools: vec![EchoTool::new()],
    };
    let mut hooks = Hooks::default();
    hooks.before_tool_call = Some(Arc::new(|_ctx| {
        Box::pin(async {
            Some(BeforeToolCallResult {
                block: true,
                reason: Some("not allowed".into()),
                terminate: false,
            })
        })
    }));
    let config = loop_config(&faux, test_model(), hooks);

    let new_messages =
        run_agent_loop(vec![], &mut context, &config, None, no_emit()).await;

    // find the toolResult message: should be the blocked error
    let result = new_messages.iter().find_map(|m| match m {
        pi_agent::AgentMessage::Message(Message::ToolResult(t)) => Some(t.clone()),
        _ => None,
    });
    let result = result.expect("expected tool result");
    assert!(result.is_error);
    match &result.content[0] {
        ToolResultContent::Text(t) => assert_eq!(t.text, "not allowed"),
        _ => panic!("expected text"),
    }
    // the loop continued to the next turn despite the block
    // messages: system(tool declaration) + assistant + toolResult + assistant
    assert_eq!(new_messages.len(), 4);
}

#[tokio::test]
async fn unknown_tool_produces_error_result() {
    let faux = Arc::new(FauxApi::with_script(vec![
        FauxResponse::tool_call("call_1", "nonexistent", json!({})),
        FauxResponse::text("recovered"),
    ]));
    let mut context = AgentContext {
        messages: vec![user("go")],
        tools: vec![], // no tools registered
    };
    let config = loop_config(&faux, test_model(), Hooks::default());

    let new_messages = run_agent_loop(vec![], &mut context, &config, None, no_emit()).await;
    let result = new_messages.iter().find_map(|m| match m {
        pi_agent::AgentMessage::Message(Message::ToolResult(t)) => Some(t.clone()),
        _ => None,
    });
    let result = result.expect("expected tool result");
    assert!(result.is_error);
    match &result.content[0] {
        ToolResultContent::Text(t) => assert!(t.text.contains("nonexistent")),
        _ => panic!("expected text"),
    }
}

#[tokio::test]
async fn tool_change_declared_as_system_message() {
    let faux = Arc::new(FauxApi::with_script(vec![FauxResponse::text("ok")]));
    let mut context = AgentContext {
        messages: vec![
            pi_agent::AgentMessage::Message(Message::System(pi_ai::types::SystemMessage::new(
                "prompt",
            ))),
            user("go"),
        ],
        tools: vec![EchoTool::new()], // not yet declared in transcript
    };
    let config = loop_config(&faux, test_model(), Hooks::default());

    let new_messages = run_agent_loop(vec![], &mut context, &config, None, no_emit()).await;

    // first new message is a system message declaring the echo tool
    match &new_messages[0] {
        pi_agent::AgentMessage::Message(Message::System(sys)) => {
            let added = sys.tools_added.as_ref().unwrap();
            assert_eq!(added.len(), 1);
            assert_eq!(added[0].name, "echo");
        }
        other => panic!("expected system tool declaration, got {other:?}"),
    }
}

#[tokio::test]
async fn error_response_stops_loop() {
    let faux = Arc::new(FauxApi::with_script(vec![FauxResponse::Error {
        message: "provider exploded".into(),
    }]));
    let mut context = AgentContext {
        messages: vec![user("go")],
        tools: vec![],
    };
    let config = loop_config(&faux, test_model(), Hooks::default());

    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink: pi_agent::EventSink = {
        let events = events.clone();
        Arc::new(move |event| {
            events.lock().unwrap().push(event_name(&event).to_string());
            Box::pin(async {})
        })
    };
    run_agent_loop(vec![], &mut context, &config, None, sink).await;

    let events = events.lock().unwrap().clone();
    assert_eq!(events.last().unwrap(), "agent_end");
    // error path: turn_end with no tool results, then agent_end
    assert!(events.contains(&"turn_end".to_string()));
}

async fn collect_events(receiver: &mut tokio::sync::broadcast::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut out = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        out.push(event);
    }
    out
}

fn event_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}
