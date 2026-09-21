# pi-rs

Rust port of the [pi coding agent](https://github.com/badlogic/pi) core (no TUI).

See [docs/PLAN.md](docs/PLAN.md) for the full plan.

## Crates

| Crate | Port of | Status |
|---|---|---|
| `pi-ai` | `@earendil-works/pi-ai` | ✅ anthropic-messages + openai-completions + faux |
| `pi-agent` | `@earendil-works/pi-agent-core` | ✅ 完整 loop 语义 |
| `pi-tools` | `coding-agent/src/core/tools` | ✅ read/bash/edit/write/grep/find/ls |
| `pi-session` | `coding-agent/src/core/session-manager` | ✅ pi v3 格式兼容 |
| `pi-core` | `coding-agent/src/core/agent-session` | ✅ 系统提示词 + compaction |
| `pi-cli` | `coding-agent/src/modes` (print/json) | ✅ 端到端可用 |

## Build

```bash
cargo build --release
export KIMI_CODING_API_KEY=...   # or use ~/.pi-rs/agent/auth.json
./target/release/pi-rs --provider kimi-coding -p "What is 2+2?"
```

See [docs/PLAN.md](docs/PLAN.md) for architecture and roadmap,
[docs/INTEGRATION.md](docs/INTEGRATION.md) for embedding pi-rs into your own
project (three integration patterns with runnable examples):

```bash
cargo run -p pi-agent --example custom_agent   # minimal: custom tool + faux LLM
cargo run -p pi-core  --example full_session   # full stack: session+compaction
```
