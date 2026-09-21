//! Minimal embedding example: build a coding-agent-style assistant with one
//! custom business tool, a scripted (faux) LLM, event streaming and hooks.
//!
//! Run: cargo run -p pi-agent --example custom_agent
//!
//! Replace the faux provider with a real one (see pi-core::model_registry)
//! and your tool with real logic to ship.

use std::sync::Arc;

use async_trait::async_trait;
use pi_agent::{Agent, AgentBuilder, AgentEvent, AgentTool, ToolContext, ToolUpdateFn};
use pi_ai::api::{FauxApi, FauxResponse, LlmApi};
use pi_ai::types::{
    Model, ModelCost, Modality, StopReason,
};

// ---------------------------------------------------------------------------
// 1. Your business tool: implement one trait, that's it.
// ---------------------------------------------------------------------------

struct GetOrderTool;

#[async_trait]
impl AgentTool for GetOrderTool {
    fn name(&self) -> &str {
        "get_order"
    }

    fn description(&self) -> &str {
        "Look up an order by id"
    }

    /// JSON Schema for the arguments the LLM must provide.
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. A-1001"}
            },
            "required": ["order_id"]
        })
    }

    fn label(&self, args: &serde_json::Value) -> String {
        format!("get_order({})", args["order_id"].as_str().unwrap_or("?"))
    }

    /// Return Err(...) to surface an error result to the model.
    async fn execute(
        &self,
        _tool_call_id: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
        _on_update: ToolUpdateFn,
    ) -> Result<pi_agent::AgentToolResult, String> {
        let order_id = args["order_id"].as_str().unwrap_or_default();
        // ctx.cancel: cooperative abort for long operations
        if ctx.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err("Operation aborted".into());
        }
        Ok(pi_agent::AgentToolResult::text(format!(
            "order {order_id}: shipped, ETA tomorrow"
        ))
        .with_details(serde_json::json!({"orderId": order_id, "status": "shipped"})))
    }
}

// ---------------------------------------------------------------------------
// 2. An LLM backend. Here: the scripted faux provider for a self-contained
//    demo. Swap with pi_ai::api::AnthropicApi / OpenAICompletionsApi / ...
// ---------------------------------------------------------------------------

fn faux_stream_fn(faux: Arc<FauxApi>) -> pi_agent::StreamFn {
    Arc::new(move |model, context, options| faux.stream(model, context, options.clone()))
}

fn demo_model() -> Model {
    Model {
        id: "demo-1".into(),
        name: "Demo".into(),
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

#[tokio::main]
async fn main() {
    // Script the LLM: first call -> tool call, second call -> final answer.
    let faux = Arc::new(FauxApi::with_script(vec![
        FauxResponse::tool_call("call_1", "get_order", serde_json::json!({"order_id": "A-1001"})),
        FauxResponse::text("Your order A-1001 has shipped and arrives tomorrow."),
    ]));

    let agent = Arc::new(
        AgentBuilder::new(demo_model(), faux_stream_fn(faux))
            .system_prompt("You are a helpful order assistant.")
            .build(),
    );

    // Register tools the model may call.
    agent.add_tool(Arc::new(GetOrderTool));

    // 3. Optional policy hooks: block or rewrite tool calls.
    agent.set_hooks(pi_agent::LoopHooks {
        before_tool_call: Some(Arc::new(|ctx| {
            Box::pin(async move {
                // e.g. deny orders for other users; block returns an error
                // tool result to the model.
                if ctx.args["order_id"] == "A-0000" {
                    return Some(pi_agent::BeforeToolCallResult {
                        block: true,
                        reason: Some("order A-0000 is not yours".into()),
                        terminate: false,
                    });
                }
                None
            })
        })),
        after_tool_call: None,
    });

    // 4. Subscribe to the event stream (drives any UI), then run the prompt.
    let mut events = agent.subscribe();
    let runner = {
        let agent = agent.clone();
        tokio::spawn(async move {
            agent
                .prompt(pi_agent::AgentMessage::Message(pi_ai::types::Message::User(
                    pi_ai::types::UserMessage::new(pi_ai::types::MessageContent::text(
                        "Where is my order A-1001?",
                    )),
                )))
                .await
        })
    };

    while let Ok(event) = events.recv().await {
        match &event {
            AgentEvent::MessageUpdate {
                assistant_message_event,
                ..
            } => {
                use pi_ai::events::AssistantMessageEvent as E;
                if let E::TextDelta { delta, .. } = assistant_message_event.as_ref() {
                    print!("{delta}");
                    use std::io::Write as _;
                    std::io::stdout().flush().ok();
                }
            }
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                println!("\n[tool: {tool_name}]");
            }
            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }
    println!();

    // 5. The final transcript (system + user + assistant + toolResult + ...).
    let messages = runner.await.expect("prompt task failed");
    println!("--- transcript: {} messages ---", messages.len());
}
