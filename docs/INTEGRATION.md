# 集成指南：在其他 Rust 项目中使用 pi-rs

pi-rs 的 crate 分层是单向依赖，按需取用：

```
pi-ai ──── LLM 抽象（协议/流式/模型目录）
pi-agent ── agent loop（工具执行/事件流/队列/hooks）
pi-tools ── 内置编码工具（read/bash/edit/...）
pi-session  JSONL 会话持久化
pi-core ─── AgentSession（全家桶：工具+会话+压缩+扩展+skills）
pi-cli ──── 参考实现（print/json/repl/rpc）
```

## 姿势一：最小嵌入（只带 loop，工具自己写）

依赖 `pi-agent`（可不含 pi-tools）。完整示例：`crates/pi-agent/examples/custom_agent.rs`。

```rust
// Cargo.toml
// pi-agent = { git = "..." }
// pi-ai = { git = "..." }

// 1. 实现业务工具
#[async_trait]
impl AgentTool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "..." }
    fn parameters(&self) -> Value { /* JSON Schema */ }
    fn label(&self, args: &Value) -> String { /* UI 标签 */ }
    async fn execute(&self, id: &str, args: Value, ctx: &ToolContext,
                     on_update: ToolUpdateFn) -> Result<AgentToolResult, String> {
        // ctx.cancel 协作取消；on_update 流式部分结果
    }
}

// 2. 组装 agent（stream_fn 注入 LLM 后端）
let agent = AgentBuilder::new(model, stream_fn).system_prompt("...").build();
agent.add_tool(Arc::new(MyTool));

// 3. 策略 hooks（审批/脱敏）
agent.set_hooks(LoopHooks { before_tool_call: Some(...), after_tool_call: Some(...) });

// 4. 事件驱动 UI + 运行
let mut events = agent.subscribe();
let runner = tokio::spawn({ let a = agent.clone(); async move { a.prompt(msg).await } });
while let Ok(event) = events.recv().await { /* 渲染 */ }
let transcript = runner.await?;
```

## 姿势二：全家桶（AgentSession）

依赖 `pi-core`（自动带上工具/会话/压缩/skills/扩展）。示例：`crates/pi-core/examples/full_session.rs`。

```rust
let session = AgentSession::new(SessionOptions {
    cwd: project_dir,               // AGENTS.md/skills/.pi/extensions 从这里加载
    model,                          // pi_core::build_model(...) 或自定义
    api,                            // pi_core::build_api("anthropic-messages") 等
    api_key: None,                  // 自动从 env / ~/.pi-rs/agent/auth.json 解析
    force_system_prompt: None,      // 整体替换系统提示词
    append_system_prompt: None,     // 追加段落
    resume_file: None,              // Some(path) 恢复 pi 兼容会话
    compaction_enabled: true,       // 自动上下文压缩
    auto_retry: None,               // 默认 3 次指数退避
})?;

let mut events = session.subscribe();
let runner = session.prompt("...");
// ... 事件循环 ...
runner.await?;
```

热能力：`session.set_model(model, key)`（跨 provider 切换）、
`session.set_thinking_level(level)`、`session.force_compact()`、
`session.steer(text)` / `session.follow_up(text)`（运行中插话）、`session.abort()`。

## 姿势三：进程隔离（非 Rust 集成）

`pi-rs --rpc`：stdin JSONL 命令 / stdout 响应+事件流。命令集见 `crates/pi-cli/src/rpc.rs`
（prompt/steer/abort/get_state/set_model/compact/get_messages/...）。
任何能读写 stdio 的语言/进程都能驱动一个完整 agent。

## 自定义 LLM 后端

实现 `pi_ai::api::LlmApi`（10 行协议适配 + SSE 映射），或者如果后端能接受
`{model, context, options}` JSON 并回 SSE 事件流，直接用内置的 `pi-messages`
网关协议，零代码。

## 测试

`pi_ai::api::FauxApi` 提供脚本化 LLM 响应（`FauxResponse::text / tool_call / Error`），
不需要真实 key 即可测试完整 agent 行为。参考 `crates/pi-agent/tests/agent_loop.rs`。

## 版本策略

crate 处于 0.1.0，API 仍在演进。建议 git tag 锁定依赖；破坏性变更会记录在
CHANGELOG（待建）。
