<div align="center">

# <img src="openheim-logo.png" width="36" height="36" alt="" valign="middle" /> openheim

[openheim.io](https://openheim.io)

[![CI](https://github.com/weirdstuff-dev/openheim/actions/workflows/ci.yml/badge.svg)](https://github.com/weirdstuff-dev/openheim/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/openheim)](https://crates.io/crates/openheim)
[![docs.rs](https://img.shields.io/docsrs/openheim)](https://docs.rs/openheim)
[![License](https://img.shields.io/github/license/weirdstuff-dev/openheim)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
</div>

**A fast, multi-provider LLM agent runtime built in Rust.**

<div align="center">
<table>
<tr>
  <td><img src="docs/screenshots/green.png" alt="Green theme" /></td>
  <td><img src="docs/screenshots/blue.png" alt="Blue theme" /></td>
</tr>
<tr>
  <td><img src="docs/screenshots/red.png" alt="Red theme" /></td>
  <td><img src="docs/screenshots/white.png" alt="White theme" /></td>
</tr>
</table>
</div>

Openheim runs an iterative agent loop — it calls your LLM, executes tools on its behalf, feeds results back, and repeats until the task is done. It works as an interactive terminal UI, a headless CLI, an ACP stdio agent (for Zed, Claude Code, and other ACP clients), or a self-hosted ACP-over-WebSocket server.

---

## Why Rust?

Openheim is built in Rust from the ground up:

- **Low memory**
- **Fast startup**
- **True concurrency**
- **Memory-safe by default**

---

## Features

- **Multi-provider** — OpenAI, Anthropic Claude, Google Gemini, and any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, etc.)
- **Tool execution** — built-in shell, file read/write/edit, directory listing, ripgrep-style search, web fetch, long-term memory (`remember`/`search_memory`/`forget`), and subagent-delegation tools. Trait-based, so you can add your own.
- **Subagents** — delegate a self-contained task to another agent (its own persona, model, and tools) via the always-on `delegate_task` tool. Named profiles live in `~/.openheim/agents/`; the orchestrator can also define inline, one-off subagents. See [docs/subagents.md](./docs/subagents.md).
- **Agent sandboxing** — configurable work-directory boundary restricts file access to a directory tree. Shell execution is disabled by default and can be enabled via `allow_shell = true` in config or `.allow_shell(true)` in the builder.
- **MCP (Model Context Protocol)** — connect external MCP servers (stdio or Streamable HTTP) and their tools are automatically exposed to the LLM as `{server_name}__{tool_name}`.
- **Conversation memory** — conversations (including full tool call history) persist to disk and resume across sessions
- **Long-term memory** — ask the agent to remember something and it saves a note via the `remember` tool; later sessions recall it with `search_memory`, and `forget` drops a note by id. Nothing is stored or injected automatically. Works out of the box with keyword search (SQLite FTS5, no network); add an embedding provider under `[memory]` (any OpenAI-compatible endpoint or Gemini) and search becomes semantic via [sqlite-vec](https://github.com/asg017/sqlite-vec).
- **Context-size tracking** — each session's current context size (the last LLM call's token usage) is tracked, persisted, and shown live in the TUI footer; embedders can read it via `SessionHandle::context_usage()` or `GET /api/sessions/{id}`
- **System identity** — edit `~/.openheim/system.md` to define how the agent presents itself. Required when preparing a session (created by `openheim init`).
- **Skills** — drop a markdown file into `~/.openheim/skills/` and it's injected into the system prompt. Set `default_skills` in config to auto-load skills every session; pass `--skills` for per-session additions. ACP clients can also pass skills per-session via `_meta`.
- **ACP transport** — implements the [Agent Client Protocol](https://github.com/block/agent-client-protocol) over stdio (for editor integrations) and WebSocket (for remote clients), with real-time streaming of message chunks and tool calls
- **Unified WebSocket** — single multiplexed `WS /ws` connection carries both ACP agent traffic (sessions, streaming, tool calls) and filesystem operations (file CRUD, live watching) via channel envelopes; filesystem operations are sandboxed to the configured `work_dir`
- **Retry with backoff** — transient failures (429s, 5xx, network errors) are retried automatically with exponential backoff
- **Docker ready** — multi-stage Dockerfile and docker-compose included

---

## Quickstart

### Prerequisites

- Rust 1.91+
- An API key for at least one supported provider

### Install

```bash
cargo install openheim
```

Or build from source:

```bash
git clone https://github.com/weirdstuff-dev/openheim.git
cd openheim-core
cargo build --release
```

### Configure

```bash
# Generate the default config and system.md
openheim init

# Edit them
vim ~/.openheim/config.toml
vim ~/.openheim/system.md
```

Example config:

```toml
default_provider = "anthropic"
max_iterations = 10

# Skills loaded in every new session automatically (no --skills flag needed)
# default_skills = ["rules"]

# Restrict the agent to a specific directory tree (defaults to invocation directory)
# work_dir = "/home/user/projects/myproject"

# Shell execution is disabled by default; set to true to expose the
# execute_command tool to the LLM
# allow_shell = true

[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-6"
models = ["claude-sonnet-4-6", "claude-opus-4-7"]
env_var = "ANTHROPIC_API_KEY"

[providers.openai]
api_base = "https://api.openai.com/v1"
default_model = "gpt-4o"
models = ["gpt-4o", "gpt-4-turbo"]
env_var = "OPENAI_API_KEY"

[providers.gemini]
api_base = "https://generativelanguage.googleapis.com/v1beta"
default_model = "gemini-2.0-flash"
models = ["gemini-2.0-flash", "gemini-2.5-pro"]
env_var = "GEMINI_API_KEY"

# Local Ollama (no API key needed)
[providers.ollama]
api_base = "http://localhost:11434/v1"
default_model = "llama3"
models = ["llama3", "mistral", "codellama"]

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
# Interactive TUI (default — no subcommand)
openheim

# Load skills in the TUI
openheim --skills rust,debugging

# Single headless prompt, streams to stdout
openheim run "List the files in the current directory"

# Single headless prompt with a model override
openheim run "Hello" --model gpt-4o

# ACP stdio agent (for Zed, Claude Code, and other ACP clients)
openheim acp

# ACP-over-WebSocket server
openheim serve
openheim serve --host 0.0.0.0 --port 1217

# Initialize config
openheim init
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

## Agent identity and skills

### `~/.openheim/system.md`

This file defines the agent's base identity. It is loaded when preparing each session (via `prepare()` / session setup) and is required — run `openheim init` to create it, then edit it freely.

```markdown
You are a senior software engineer who writes clean, idiomatic code.
You prefer simple solutions and ask clarifying questions before making large changes.
```

### Skills

Skills are markdown files in `~/.openheim/skills/`. They are injected into the system prompt after the identity block.

```bash
# Run with specific skills for this session
openheim --skills rust,debugging

# Always load certain skills (set in config.toml)
# default_skills = ["rules", "concise"]
```

The system message the LLM receives is assembled in this order:

```
You are a general purpose multiprovider LLM agent.

---

The user has given you the following identity:

<system.md content>

---

These are the skills you have mastered:

### rust
<rust.md content>
```

ACP clients (Zed, Claude Code, etc.) can pass skills per-session by including a `skills` array in the `_meta` field of the `NewSession` request — no flag needed on the server side.

---

## Server mode

Start with `openheim serve` (defaults to `0.0.0.0:1217`).

The server speaks the [Agent Client Protocol](https://github.com/block/agent-client-protocol) over WebSocket and exposes a multiplexed WS endpoint plus REST API routes:

### WebSocket

| Endpoint | Description |
|---|---|
| `WS /ws` | Single multiplexed connection carrying two channels via JSON envelopes: **agent** (ACP sessions, streaming, tool calls) and **fs** (file CRUD, live watching) |
| `WS /acp` | Bare ACP-only endpoint — raw JSON-RPC over the socket, no envelope, no `fs` channel |

Every message is wrapped in `{ "channel": "<agent|fs>", "data": <payload> }`. The **fs** channel is sandboxed to the agent's configured `work_dir` — the same boundary enforced for the agent's own file tools.

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

> **Frontend / WebSocket implementors:** see [docs/api.md](./docs/api.md) for the complete protocol reference, TypeScript interfaces, and sequence diagrams.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/architecture.md](./docs/architecture.md) | Module map and prompt flow |
| [docs/configuration.md](./docs/configuration.md) | Full `config.toml` reference |
| [docs/library.md](./docs/library.md) | Embedding openheim as a Rust library |
| [docs/skills.md](./docs/skills.md) | Writing and enabling skill files |
| [docs/subagents.md](./docs/subagents.md) | Delegating to named and inline subagent profiles |
| [docs/deployment.md](./docs/deployment.md) | Docker, systemd, reverse proxy, enterprise |
| [docs/custom-tools.md](./docs/custom-tools.md) | Implementing a custom `ToolHandler` |
| [docs/custom-llm-provider.md](./docs/custom-llm-provider.md) | Implementing a custom `LlmClient` |
| [docs/api.md](./docs/api.md) | REST + WebSocket API spec |
| [docs.rs/openheim](https://docs.rs/openheim) | Rust API reference (auto-generated) |

---

## Use as a library

Openheim can be embedded directly in your Rust application via the `openheim` crate. The library exposes the full agent runtime — sessions, streaming, conversation history, skills, and MCP servers — through a single `OpenheimClient` facade.

```toml
# Cargo.toml
[dependencies]
openheim = "0.6"
tokio = { version = "1", features = ["full"] }
```

Embedders that don't need the terminal UI or the built-in server can depend with `default-features = false` (optionally adding the `tui` or `server` feature back) to skip those dependency trees — see [docs/library.md](./docs/library.md) §"Feature flags".

See **[docs/library.md](./docs/library.md)** for the full API reference, session management, multi-turn conversations, and MCP integration.

---

## Docker

```bash
# Build and start with docker compose
docker compose up --build

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
  client.rs         OpenheimClient / SessionHandle — library facade
  error.rs          Error types (with retryable classification for backoff)
  config/           Config loading, provider/model resolution, defaults
  core/
    agent.rs        Agent loop (streaming variant)
    models.rs       Message, Tool, Choice, and WebSocket envelope types
    permission.rs   PermissionGate trait — embedder hook for tool-call approval
    turn.rs         Cross-cutting turn controls (cancellation, etc.)
    client_io.rs    Optional delegation of file I/O to the ACP client
    llm/            LLM client trait and provider implementations
      anthropic.rs    Anthropic Messages API client
      gemini.rs       Google Gemini API client
      openai.rs       OpenAI API client
      openai_compatible.rs  Generic OpenAI-compatible client (Ollama, etc.)
      sse.rs          Shared Server-Sent Events decoder for streaming
      retry.rs        Automatic retry with exponential backoff
  tools/            Tool trait, registry, and built-in tools
    execute_command.rs / read_file.rs / write_file.rs / edit_file.rs / list_dir.rs / search.rs / web_fetch.rs
    delegate.rs       DelegateTool — delegate_task tool (subagents)
    args.rs           Shared tool-argument decoding
    sandbox.rs        Work-directory path validation (used by every file tool)
    scoped_executor.rs     Tool-name allowlist wrapper around any ToolExecutor
  mcp/              MCP (Model Context Protocol) client integration
    client.rs       MCP server connection (stdio + Streamable HTTP)
    tool_handler.rs  Adapts MCP tools to the ToolHandler trait
  memory/           Agent memory — conversation history, prompt builder, skills manager, system identity
    history.rs      HistoryManager — conversation persistence
    lease.rs        Advisory per-session write lease (guards ~/.openheim/history/ across processes)
    skills.rs / system.rs / prompt.rs
  rag/              Long-term memory via tool calls (feature `rag`)
    embedding/      EmbeddingClient trait + OpenAI-compatible and Gemini clients
    store.rs        VectorStore — SQLite schema, FTS5 keyword search, sqlite-vec KNN search
    tool.rs         remember / search_memory / forget tools
  subagents/        Subagent profiles — delegated, isolated agent personas (~/.openheim/agents/)
  acp/              Agent Client Protocol server implementation
    session.rs      Live session map — state and eviction policy
    state.rs        AgentState — shared handle, request handlers
    serve.rs        Connection loop wiring transports to the agent-client-protocol crate
    permission.rs   Adapts ACP session/request_permission to PermissionGate
    client_io.rs    Adapts ACP fs/* requests to ClientIo
    convert.rs      Maps ACP content blocks to core::models::ContentBlock
    util.rs         Shared ACP vocabulary (session modes, stop reasons, history replay)
  transport/
    stdio.rs        ACP-over-stdio transport (for editor integrations)
    ws.rs           Multiplexed WebSocket server (axum) + REST API + filesystem channel
    run.rs          Headless single-prompt transport
  tui/              Interactive ratatui terminal UI
    app.rs          App state, command dispatch (:models, :theme, …)
    permission.rs   TUI permission prompt (Allow once/always, Reject)
    render.rs       Frame rendering, theme colors, chat item layout
    types.rs        ChatItem, AgentUpdate, Status, Screen enums
```

---

## Development

```bash
RUST_LOG=debug openheim run "test"
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