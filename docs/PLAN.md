# pi-rs：用 Rust 复刻 Pi Agent Harness 核心方案

> 目标：复刻 [badlogic/pi](../pi) 的核心能力（LLM 抽象、Agent Loop、内置工具、Session 持久化、无头 CLI），**不实现 TUI**。
> 参考：`packages/ai`、`packages/agent`、`packages/coding-agent/src/core`。

## 一、原项目架构理解

| 包 | 职责 | 规模 |
|---|---|---|
| `packages/ai` | 统一多 Provider LLM API：消息类型、流式事件、50+ provider、模型目录、auth | ~19k 行 |
| `packages/agent` | Agent 运行时：agent-loop、Agent 类、事件流、steering/follow-up 队列 | ~6k 行 |
| `packages/coding-agent` | 应用层：AgentSession、内置工具、session 持久化、compaction、系统提示词 | ~20k+ 行 |
| `packages/tui` | 终端 UI（不复刻） | — |

核心数据流：

```
AgentSession (状态/生命周期/compaction/重试)
    └─> Agent.prompt(msg)
          └─> agent_loop:
              while (有 tool call || steering/follow-up 队列非空):
                  transformContext → convertToLlm → streamFn(model, context)
                  ← AssistantMessageEvent 流 (text_delta / toolcall_delta ...)
                  ← tool 执行 (sequential/parallel, before/after hooks)
                  ← toolResult 消息回填 → 下一轮
          ← AgentEvent 流 (agent_start/turn_*/message_*/tool_execution_*/agent_end)
    └─> 每个事件同步追加到 JSONL session 文件（entry 树，支持分支）
```

必须保留的精髓：
- **AgentMessage ≠ LLM Message**：`convertToLlm` 在 LLM 调用边界才转换，UI-only 消息被过滤
- **StreamFn 注入**：loop 不直接依赖 provider，方便测试（faux provider）
- **Session 是 JSONL entry 树**：`parentId` 链，支持 branch/compact/replay
- **工具用 Operations trait 抽象**，便于沙箱替换
- **Hooks**：`beforeToolCall`（可拦截）、`afterToolCall`（可改写）、`shouldStopAfterTurn`、`prepareNextTurn`

## 二、复刻范围

### ✅ 复刻（core）
1. pi-ai：类型系统 + 流式协议 + **2 个 provider**（Anthropic Messages + OpenAI Chat Completions，后者覆盖所有兼容 API）
2. pi-agent：agent-loop 完整语义（steering、follow-up、并行工具、hooks、abort）
3. 内置工具：read / bash / edit / write / grep / find / ls（截断规则、file mutation queue）
4. Session JSONL 持久化（格式对齐 v3，可直接读 pi 原生 session 文件）
5. AgentSession：系统提示词构建、AGENTS.md/skills 加载、auto compaction、模型切换、重试
6. CLI 无头模式：`-p "prompt"` 单发 + `--mode json` 事件流 + RPC（stdio JSON lines）

### ❌ 不做 / 缓做
- TUI 全部（chord/tui、interactive mode、themes）
- OAuth 类 provider（Copilot/Codex/Qwen token plan）→ 二期
- Bedrock / Vertex / Azure / Mistral conversations → 二期
- TS 动态 extensions（jiti）→ 一期只留 hook 接口，后期考虑 rhai/WASM
- HTML 导出、bug report、package-manager、telemetry

## 三、技术选型

| TS 概念 | Rust 方案 |
|---|---|
| async 循环/并发 | `tokio` |
| EventStream | `tokio::sync::mpsc` + `async-stream` 生成器，`Pin<Box<dyn Stream>>` |
| AbortSignal | `tokio_util::sync::CancellationToken` |
| TypeBox schema | `schemars` derive JSON Schema + `serde_json::Value` 校验 |
| SSE 解析 | `eventsource-stream` |
| 部分工具参数增量 JSON | 补尾括号策略（手写，对齐 pi 实现） |
| HTTP | `reqwest`（`stream` feature） |
| 错误 | `thiserror`（库）+ `anyhow`（CLI） |
| UUID v7 | `uuid` crate v7 feature |
| CLI | `clap` (derive) |

## 四、Workspace 结构

```
pi-rs/
├── Cargo.toml                  # [workspace]
└── crates/
    ├── pi-ai/                  # LLM 抽象层（对标 packages/ai）
    │   └── src/
    │       ├── types.rs        # Content/Message/Context/Tool/Usage/StopReason
    │       ├── events.rs       # AssistantMessageEvent + EventStream
    │       ├── model.rs        # Model 目录、registry
    │       ├── api/            # anthropic.rs / openai_completions.rs / faux.rs
    │       ├── auth.rs         # env key + auth.json
    │       ├── retry.rs        # 重试、overflow 检测
    │       └── partial_json.rs # 工具参数增量解析
    ├── pi-agent/               # Agent 运行时（对标 packages/agent）
    │   └── src/
    │       ├── types.rs        # AgentMessage/AgentEvent/AgentTool/LoopConfig
    │       ├── loop_.rs        # agent_loop 核心
    │       ├── agent.rs        # Agent: prompt/steer/follow_up/abort/subscribe
    │       └── stream_fn.rs
    ├── pi-tools/               # 内置工具（对标 core/tools）
    │   └── src/                # bash/read/edit/write/grep/find/ls + truncate + mutation_queue
    ├── pi-session/             # JSONL 持久化（对标 session-manager）
    │   └── src/
    │       ├── format.rs       # SessionHeader + Entry 枚举
    │       ├── manager.rs      # append/branch/tree/read/resume
    │       └── compaction.rs   # 摘要压缩 + 文件操作追踪
    ├── pi-core/                # AgentSession（对标 core/agent-session）
    │   └── src/                # session.rs / system_prompt.rs / convert.rs / settings.rs
    └── pi-cli/                 # bin: -p / --mode json / rpc
```

依赖方向单向：`pi-cli → pi-core → {pi-agent, pi-session, pi-tools} → pi-ai`。

## 五、关键类型设计

```rust
// pi-ai: 消息与内容块
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    Text { text: String },
    Thinking { thinking: String, #[serde(default)] signature: String },
    ToolCall { id: String, name: String, arguments: serde_json::Value },
    Image { mime_type: String, data: String }, // base64
}

pub struct Context {
    pub system_prompt: Option<String>,
    pub tools: Vec<ToolDef>,
    pub messages: Vec<Message>,
}

// 流式事件 —— 与 pi 事件名对齐
pub enum AssistantMessageEvent {
    Start { partial: AssistantMessage },
    TextStart, TextDelta { delta: String }, TextEnd,
    ThinkingStart, ThinkingDelta { delta: String }, ThinkingEnd,
    ToolcallStart { index: usize }, ToolcallDelta { index: usize, delta: String },
    ToolcallEnd { tool_call: ToolCall },
    Done { reason: StopReason, usage: Usage },
    Error { error: LlmError },
}

pub type EventStream = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send>>;

// pi-agent: 工具 trait
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn schema(&self) -> serde_json::Value;
    fn label(&self, args: &Value) -> String;
    async fn execute(
        &self, tool_call_id: &str, args: Value,
        ctx: &ToolContext, on_update: Updater<'_>,
    ) -> ToolResult;
}
```

## 六、实施阶段

| 阶段 | 内容 | 预估 | 验收标准 | 状态 |
|---|---|---|---|---|
| Phase 0 | workspace 骨架 + CI | 0.5 天 | `cargo build && cargo test` 通过 | ✅ 完成 |
| Phase 1 | pi-ai 类型 + Anthropic 流式 + faux provider | 2~3 天 | 带工具调用的对话完整流式回放 | ✅ 完成（含 OpenAI Completions，mock server 测试全绿） |
| Phase 2 | 模型目录 + auth 存储 | 2 天 | anthropic/deepseek/openrouter 行为一致 | ⬜ 模型目录待做 |
| Phase 3 | pi-agent loop | 3 天 | faux provider 驱动的 loop 状态机测试全绿 | ✅ 完成（steering/parallel/hooks/abort/tool-declaration） |
| Phase 4 | pi-tools 七件套 | 3 天 | 与 pi 工具语义对拍 | ✅ 完成（18 lib + 8 integration 测试） |
| Phase 5 | pi-session + compaction | 3 天 | 能读 pi 原生 session 并 resume | ✅ 完成（真实 pi session 验收通过，LLM 摘要生成待 Phase 6） |
| Phase 6 | pi-core AgentSession + CLI | 2~3 天 | `pi-rs -p "..."` 端到端可用 | ✅ 完成（真实 API 冒烟通过） |
| Phase 7(后置) | OAuth provider、Bedrock/Vertex、rhai 插件、RPC server | — | — | ⬜ |

### 已完成的关键决策（与计划的差异）

- `pi-ai` 的 `AssistantMessageEventStream`：用 `mpsc` + `async-stream` 实现，`done`/`error` 事件携带最终消息，`result()` 由 stream 自身追踪终端事件
- OpenAI Completions 在 Phase 1 一并完成（`api/openai_completions.rs`），reasoning_content 映射为 thinking 块
- `partial_json.rs`：单遍扫描 + 补尾修复（关闭未闭合字符串/容器、悬挂 `:`/`,`、部分字面量），替代 pi 的 partial-json npm 依赖
- Agent 事件时序验收：`tests/agent_loop.rs` 精确断言事件序列（含 parallel 模式的 completion-order vs source-order 语义）
- 工具参数校验：Phase 3 只做 object 类型检查，完整 JSON Schema 校验待接 schemars
- pi-tools 搜索实现：grep/find 不再 shell 出 rg/fd 二进制，改用 ripgrep 官方库 `ignore` + `globset`（内置 .gitignore/隐藏文件语义）
- edit 工具完整移植 fuzzy 匹配管线（exact → NFKC/行尾空白/智能引号归一化 → 唯一性/重叠校验 → 分组行级覆写保留原字节），diff 用 `similar` 生成 display diff + unified patch
- bash 输出截断后 spill 到临时文件（`pi-rs-bash-*.log`），abort/timeout 杀进程；进程组级 kill 留待后续
- session 格式验证：本机真实 pi v3 session 文件（940 entries）可被 `SessionManager::open` 读取，`build_context_messages` 正确产出 230 条上下文消息，thinking/model 沿路径正确解析（`tests/pi_compat.rs`，无文件环境自动跳过）
- compaction 的 entry 应用逻辑已实现（`build_context_entries`）；LLM 摘要生成已挂载到 AgentSession（token 估算切点 + 同模型摘要 + CompactionEntry 落盘）

### 端到端冒烟验收（2026-09-20，kimi-coding 真实 API）

```
pi-rs --provider kimi-coding -p "Read note.txt..."   →  LLM 调用 read → 回复文件内容 ✓
pi-rs --mode json -p "..."                            →  完整 AgentEvent JSONL 流 ✓
pi-rs --resume <session.jsonl> -p "What did I ask?"   →  上下文恢复，准确回忆 ✓
```

Session 文件为 pi v3 兼容 JSONL（header + message entries, camelCase）。

## 八、CLI 用法

```bash
pi-rs --provider anthropic -p "分析这个 repo"          # 单发（text 流式输出）
pi-rs --provider kimi-coding --model kimi-latest -p "..."
pi-rs --provider deepseek --mode json -p "..."          # AgentEvent JSONL
pi-rs --resume session.jsonl -p "..."                   # 续接会话
pi-rs --no-compact -p "..."                             # 禁用自动 compaction
```

API key 解析顺序：`--api-key` → 环境变量（`KIMI_CODING_API_KEY` / provider 默认名）→ `~/.pi-rs/agent/auth.json`（pi 格式兼容）。

## 九、已知边界（后续迭代）

- 模型目录为静态内置表（8 个 provider），未接 pi 的全量生成目录
- read 工具暂不支持图片输出（返回文本 note）
- bash 进程组级 kill 未接 libc
- 工具参数完整 JSON Schema 校验未接 schemars
- steering/compaction hooks、steering 队列的 Agent 级 API 已就绪，CLI 未暴露
- 无 extensions/rhai 插件、无 RPC server 模式（Phase 7）

核心约 6000~8000 行 Rust。

## 七、风险点

1. **流式事件时序对齐**是最大工作量，照 pi README 的 event sequence 图写状态机测试
2. **session 格式兼容**：serde 用 camelCase，AgentMessage custom role 用 tagged enum
3. **部分 JSON 解析**边界多，先写纯函数 + 大量用例
4. **不要照抄 AgentSession**（3699 行混入太多 UI 关切），按层拆解
