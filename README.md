<div align="center">

# <img src="hexagon-multiple-green.svg" width="36" height="36" alt="" valign="middle" /> openheim

[openheim.io](https://openheim.io)

[![CI](https://github.com/weirdstuff-dev/openheim/actions/workflows/ci.yml/badge.svg)](https://github.com/weirdstuff-dev/openheim/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/weirdstuff-dev/openheim)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange)](https://www.rust-lang.org)
</div>

**A fast, multi-provider LLM agent runtime built in Rust.**

Openheim runs an iterative agent loop — it calls your LLM, executes tools on its behalf, feeds results back, and repeats until the task is done. It works as an interactive REPL, a headless CLI, an ACP stdio agent (for Zed, Claude Code, and other ACP clients), or a self-hosted ACP-over-WebSocket server.

---

## Why Rust?

Openheim is built in Rust from the ground up:

- **Low memory** — runs in a fraction of the RAM a Python agent would need
- **Fast startup** — no interpreter warmup
- **True concurrency** — async Tokio runtime, multiple agents without threading headaches
- **Safe by default** — Rust's ownership model means fewer footguns in long-running agent processes

---

## Features

- **Multi-provider** — OpenAI, Anthropic Claude, Google Gemini, and any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, etc.)
- **Tool execution** — built-in shell, file read, and file write tools. Trait-based, so you can add your own.
- **MCP (Model Context Protocol)** — connect external MCP servers (stdio or Streamable HTTP) and their tools are automatically exposed to the LLM as `{server_name}__{tool_name}`.
- **Conversation memory** — conversations (including full tool call history) persist to disk and resume across sessions
- **Skills** — drop a markdown file into `~/.openheim/skills/` and it's prepended to the system prompt. ACP clients can also pass skills per-session via `_meta`.
- **ACP transport** — implements the [Agent Client Protocol](https://github.com/block/agent-client-protocol) over stdio (for editor integrations) and WebSocket (for remote clients), with real-time streaming of message chunks and tool calls
- **Unified WebSocket** — single multiplexed `WS /ws` connection carries both ACP agent traffic (sessions, streaming, tool calls) and filesystem operations (file CRUD, live watching) via channel envelopes
- **Retry with backoff** — transient failures (429s, 5xx, network errors) are retried automatically with exponential backoff
- **Docker ready** — multi-stage Dockerfile and docker-compose included

---

## Quickstart

### Prerequisites

- Rust 1.85+
- An API key for at least one supported provider

### Install

```bash
git clone https://github.com/weirdstuff-dev/openheim.git
cd openheim
cargo build --release
```

### Configure

```bash
# Generate the default config
cargo run -- init

# Edit it
vim ~/.openheim/config.toml
```

Example config:

```toml
default_provider = "openai"
max_iterations = 10

[providers.openai]
api_base = "https://api.openai.com/v1"
default_model = "gpt-4"
models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
env_var = "OPENAI_API_KEY"
# timeout_secs = 120  # Request timeout in seconds (default: 120)
# max_tokens = 4096   # Maximum output tokens for LLM responses

[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-5-20250929"
models = ["claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
env_var = "ANTHROPIC_API_KEY"

[providers.gemini]
api_base = "https://generativelanguage.googleapis.com/v1beta"
default_model = "gemini-2.5-flash"
models = ["gemini-2.5-flash", "gemini-2.5-pro"]
env_var = "GEMINI_API_KEY"

# Local Ollama (no API key needed)
[providers.ollama]
api_base = "http://localhost:11434/v1"
default_model = "llama2"
models = ["llama2", "mistral", "codellama"]

# MCP servers — tools are exposed as "{server_name}__{tool_name}"
# [mcp_servers.filesystem]
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
#
# [mcp_servers.remote-tools]
# url = "http://localhost:8080/mcp"
```

### Run

```bash
# Interactive REPL (default — no subcommand)
cargo run

# Load skills in the REPL
cargo run -- --skills rust,debugging

# Single headless prompt, streams to stdout
cargo run -- run "List the files in the current directory"

# Single headless prompt with a model override
cargo run -- run "Hello" --model gpt-4-turbo

# ACP stdio agent (for Zed, Claude Code, and other ACP clients)
cargo run -- acp

# ACP-over-WebSocket server
cargo run -- serve
cargo run -- serve --host 0.0.0.0 --port 1217

# Initialize config
cargo run -- init
```

---

## How the agent loop works

```
User prompt
    │
    ▼
Send conversation + tools → LLM
    │
    ├─ Tool call requested? → Execute tool → feed result back → repeat
    │
    └─ Final response → done
```

Conversations are saved to `~/.openheim/history/` as JSON after every run.

---

## Skills

Skills are markdown files in `~/.openheim/skills/`. When loaded, their content is injected into the system prompt before the conversation starts.

Use them to give the agent a persona, a set of coding standards, domain knowledge, or anything you'd otherwise paste into the system prompt every time.

```bash
# Run the REPL with specific skills loaded
cargo run -- --skills rust,debugging
```

ACP clients (Zed, Claude Code, etc.) can pass skills per-session by including a `skills` array in the `_meta` field of the `NewSession` request — no flag needed on the server side.

---

## Server mode

Start with `cargo run -- serve` (defaults to `0.0.0.0:1217`).

The server speaks the [Agent Client Protocol](https://github.com/block/agent-client-protocol) over WebSocket and exposes a multiplexed WS endpoint plus REST API routes:

### WebSocket

| Endpoint | Description |
|---|---|
| `WS /ws` | Single multiplexed connection carrying two channels via JSON envelopes: **agent** (ACP sessions, streaming, tool calls) and **fs** (file CRUD, live watching) |

Every message is wrapped in `{ "channel": "<agent|fs>", "data": <payload> }`.

### REST API

| Endpoint | Description |
|---|---|
| `GET /api/config` | Public config (providers, models — API keys stripped) |
| `GET /api/models` | Available models per provider |
| `GET /api/skills` | List of installed skills |
| `GET /api/tools` | All registered tool definitions (built-in + MCP) |
| `GET /api/mcp-servers` | MCP server connection statuses |
| `GET /api/sessions` | All persisted sessions (metadata only, newest first) |
| `GET /api/sessions/{id}` | Full conversation — messages, tool calls, and metadata |

> **Frontend / WebSocket implementors:** see [OPENHEIM_SPEC.md](./OPENHEIM_SPEC.md) for the complete protocol reference, TypeScript interfaces, and sequence diagrams.

---

## Use as a library

Openheim can be embedded directly in your Rust application via the `openheim` crate. The library exposes the full agent runtime — sessions, streaming, conversation history, skills, and MCP servers — through a single `OpenheimClient` facade.

```toml
# Cargo.toml
[dependencies]
openheim = { path = "../openheim-core" }
tokio = { version = "1", features = ["full"] }
```

See **[docs/library.md](./docs/library.md)** for the full API reference, session management, multi-turn conversations, and MCP integration.

---

## Docker

```bash
# Build and start with docker-compose
docker-compose up --build

# Or run manually
docker build -t openheim .
docker run -p 1217:1217 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/workspace:/workspace \
  openheim serve
```

---

## Project structure

```
src/
  main.rs           Entry point and subcommand dispatch
  lib.rs            Public API surface
  error.rs          Error types (with retryable classification for backoff)
  config/           Config loading, provider/model resolution, defaults
  core/
    agent.rs        Agent loop (streaming variant)
    models.rs       Message, Tool, Choice, and WebSocket envelope types
    llm/            LLM client trait and provider implementations
      anthropic.rs    Anthropic Messages API client
      gemini.rs       Google Gemini API client
      openai.rs       OpenAI API client
      openai_compatible.rs  Generic OpenAI-compatible client (Ollama, etc.)
      retry.rs        Automatic retry with exponential backoff
  tools/            Tool trait, registry, and built-in tools
    execute_command.rs / read_file.rs / write_file.rs
  mcp/              MCP (Model Context Protocol) client integration
    client.rs       MCP server connection (stdio + Streamable HTTP)
    tool_handler.rs  Adapts MCP tools to the ToolHandler trait
  rag/              Conversation history, prompt builder, and skills manager
  acp/              ACP agent core — session state and protocol handling
  transport/
    stdio.rs        ACP-over-stdio transport (for editor integrations)
    ws.rs           Multiplexed WebSocket server (axum) + REST API + filesystem channel
    run.rs          Headless single-prompt transport
    WS_SPEC.md      Full WebSocket protocol reference for frontend implementors
  tui/              Interactive rustyline REPL
```

---

## Development

```bash
RUST_LOG=debug cargo run -- run "test"
cargo test
cargo fmt --check
cargo clippy
```

---

## Contributing

Contributions are welcome.

---

## License

See [LICENSE](./LICENSE) for details.