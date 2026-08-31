# Architecture

Openheim is a multi-provider LLM agent runtime built around the [Agent Client Protocol (ACP)](https://github.com/block/agent-client-protocol). This document explains how the modules fit together and traces the path of a single prompt from entry point to response.

---

## Usage modes

Openheim can be used in two ways:

| Mode | Entry point | Use case |
|------|-------------|----------|
| **Library** | `OpenheimClient` in `src/client.rs` | Embedded in a Rust application |
| **Server** | `src/main.rs` subcommands | Standalone process driven by a client over a transport |

Both modes share the same agent logic. Transports speak the ACP wire protocol to `acp::serve`; the library facade calls the same `AgentState` request handlers directly, and the headless `run` mode connects an ACP client to `acp::serve` over an in-memory duplex pipe.

---

## Module map

```
src/
├── main.rs             CLI entry point (clap subcommands)
├── lib.rs              Public re-exports for library users
│
├── client.rs           OpenheimClient / SessionHandle — library facade
├── acp/                Agent Client Protocol server implementation
│   ├── session.rs      Live session map — state and eviction policy
│   ├── state.rs        AgentState — shared handle, request handlers
│   ├── serve.rs        Connection loop wiring transports to agent-client-protocol
│   ├── permission.rs   Adapts session/request_permission to core::PermissionGate
│   ├── client_io.rs    Adapts fs/* requests to core::ClientIo
│   ├── convert.rs      Maps ACP content blocks to core::models::ContentBlock
│   └── util.rs         Shared ACP vocabulary (session modes, stop reasons, history replay)
│
├── core/
│   ├── agent.rs        Agent loop — LLM ↔ tool call iteration
│   ├── permission.rs   PermissionGate trait — embedder hook for tool-call approval
│   ├── turn.rs         Cross-cutting turn controls (cancellation, …)
│   ├── client_io.rs    Optional delegation of file I/O to the ACP client
│   ├── llm/            LLM provider abstraction + implementations
│   │   ├── mod.rs      LlmClient trait
│   │   ├── anthropic.rs
│   │   ├── openai.rs
│   │   ├── gemini.rs
│   │   ├── openai_compatible.rs
│   │   ├── sse.rs      Shared Server-Sent Events decoder for streaming
│   │   └── retry.rs    Exponential-backoff wrapper
│   └── models.rs       Shared data types (Message, Tool, Choice, …)
│
├── config/             Config loading, provider resolution, HTTP client
├── error.rs            Unified Error / Result types
│
├── rag/                Retrieval-Augmented Generation utilities
│   ├── mod.rs          RagContext — history + skills + system identity
│   ├── history.rs      HistoryManager — conversation persistence
│   ├── skills.rs       SkillsManager — Markdown skill files
│   ├── system.rs       SystemLoader — reads ~/.openheim/system.md
│   └── prompt.rs       PromptBuilder — assembles structured system message
│
├── subagents/          Subagent profiles — delegated, isolated agent personas
│   └── mod.rs          AgentProfile, SubagentLoader — Markdown files in ~/.openheim/agents/
│
├── tools/              Tool abstraction and built-in implementations
│   ├── mod.rs          ToolHandler / ToolExecutor traits, SystemToolExecutor
│   ├── execute_command.rs
│   ├── read_file.rs
│   ├── write_file.rs
│   ├── edit_file.rs    Targeted string replacement, no whole-file rewrite
│   ├── list_dir.rs     Immediate directory contents
│   ├── search.rs       Regex search across files, ripgrep-style (ripgrep's own crates)
│   ├── web_fetch.rs    Fetch a public http(s) URL as text (SSRF-guarded)
│   ├── sandboxed_executor.rs  SandboxedExecutor — work_dir / allow_shell boundary
│   ├── scoped_executor.rs     ScopedExecutor — tool-name allowlist wrapper
│   └── delegate.rs            DelegateTool, with_delegation — delegate_task tool
│
├── mcp/                Model Context Protocol client
│   ├── mod.rs          load_mcp_tools(), McpServerStatus
│   ├── client.rs       McpClient — rmcp service wrapper
│   └── tool_handler.rs McpToolHandler — bridges MCP tools to ToolHandler
│
├── transport/          Transport-specific entry points
│   ├── stdio.rs        ACP over stdin/stdout
│   ├── ws.rs           ACP + REST over WebSocket (axum)
│   └── run.rs          Headless single-prompt mode
│
└── tui/                Interactive terminal UI (ratatui)
    ├── mod.rs          Entry point — event loop, AgentUpdate channel
    ├── app.rs          App state, command dispatch (:models, :theme, …)
    ├── permission.rs   TUI permission prompt (Allow once/always, Reject)
    ├── render.rs       Frame rendering, theme colors, chat item layout
    └── types.rs        ChatItem, AgentUpdate, Status, Screen enums
```

---

## Prompt flow

The following traces what happens when a user sends a prompt — from the transport layer all the way to the saved response.

```
User / Client
      │
      ▼
┌─────────────────────────────────────────────┐
│               Transport Layer               │
│                                             │
│  stdio ── ws (HTTP + WebSocket) ── headless │
│                                             │
│  All transports call acp::serve(transport,  │
│  state) and speak the ACP wire protocol.    │
└────────────────────┬────────────────────────┘
                     │ ACP PromptRequest
                     ▼
┌─────────────────────────────────────────────┐
│            acp::serve / AgentState          │
│                                             │
│  AgentState holds:                          │
│  • llm       Arc<dyn LlmClient>             │
│  • executor  Arc<dyn ToolExecutor>          │
│  • rag       RagContext                     │
│  • config    AgentConfig (model, iters, …)  │
│  • sessions  HashMap<id, SessionState>      │
└────────────────────┬────────────────────────┘
                     │ acp_prompt()
                     ▼
┌─────────────────────────────────────────────┐
│            rag::RagContext::prepare         │
│                                             │
│  1. Load conversation from disk (history)   │
│  2. Load ~/.openheim/system.md (identity)   │
│  3. Merge default_skills + session skills   │
│  4. Load skill files by name                │
│  5. Build PromptBuilder (system message)    │
│                                             │
│  Returns: (Conversation, PromptBuilder)     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│         core::agent  (agent loop)           │
│                                             │
│  for iteration in 0..max_iterations:        │
│    ┌─────────────────────────────────────┐  │
│    │ 1. PromptBuilder.build(history)     │  │
│    │    → prepend system message         │  │
│    │ 2. llm.send(messages, tools)        │  │
│    │    → Choice { message, finish }     │  │
│    │ 3. if tool_calls:                   │  │
│    │      executor.execute(name, args)   │  │
│    │      append tool result message     │  │
│    │      emit StreamEvent::ToolCall/    │  │
│    │        ToolResult/MessageAppended   │  │
│    │    else if finish_reason == Stop:   │  │
│    │      emit StreamEvent::Finished     │  │
│    │      break                          │  │
│    └─────────────────────────────────────┘  │
│                                             │
│  acp_prompt appends each MessageAppended    │
│  event to history.jsonl as it's emitted,    │
│  then rewrites the full log once more       │
│  after the loop (history.save_conversation) │
└──────────┬──────────────────┬───────────────┘
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────────────┐
│   LLM Backend    │  │    Tool Executor     │
│                  │  │                      │
│  AnthropicClient │  │  Built-in tools:     │
│  OpenAiClient    │  │    execute_command   │
│  GeminiClient    │  │    read_file         │
│  OpenAiCompatible│  │    write_file        │
│  (+ RetryClient) │  │    edit_file         │
└──────────────────┘  │    list_dir          │
                      │    search            │
                      │    web_fetch         │
                      │    delegate_task     │
                      │  MCP tools:          │
                      │    {server}__{tool}  │
                      │    (via rmcp)        │
                      └──────────────────────┘
```

---

## Data persistence

All persistence lives under `~/.openheim/` by default.

```
~/.openheim/
├── config.toml              Agent configuration (providers, MCP servers, default_skills, …)
├── system.md                Agent identity — loaded on every session (required)
├── history/
│   ├── {uuid}.json          Conversation metadata (rewritten wholesale)
│   ├── {uuid}.jsonl         Conversation messages (appended one per line)
│   └── …
└── skills/
    ├── rust.md              Named skill files (Markdown)
    └── …
```

`SystemLoader` reads `system.md` on every `prepare()` call — missing file is a hard error (run `openheim init` to create it). `HistoryManager` reads and writes each conversation's metadata (`.json`) and message log (`.jsonl`) — see `src/rag/history.rs`'s doc comment for why they're split. `SkillsManager` reads `.md` files from the skills directory. Both history and skills paths are configurable at construction time, which is how the test suite uses temporary directories.

---

## ACP and the library facade

The `OpenheimClient` facade (`src/client.rs`) wraps the same `AgentState` the transports use, behind a simple Rust API. Internally it:

1. Builds an `AgentState` (LLM client, tool executor, RAG context).
2. Calls its request handlers (`acp_prompt`, `acp_load_session`, `acp_cancel`, …) directly — no wire protocol, no background task. Streaming updates reach your callback through the same `SessionUpdate` events a transport would forward.

The duplex-pipe + ACP-client wiring does exist — in `src/transport/run.rs`, where the headless `openheim run` mode drives `acp::serve` over `tokio::io::duplex`. Either way there is no separate "library mode" agent logic: the facade and every transport share the exact same session and agent-loop code path.

---

## Key extension points

| What to extend | Where | How |
|----------------|-------|-----|
| Add a new LLM provider | `src/core/llm/` | Implement `LlmClient` |
| Add a built-in tool | `src/tools/` | Implement `ToolHandler`, register in `register_builtins` |
| Add a tool without touching source | any embedder | Implement `ToolHandler`, register via `OpenheimBuilder::tool()` |
| Add an external tool source | `src/mcp/` | MCP servers via configuration or `McpClient` |
| Add a new transport | `src/transport/` | Call `acp::serve(your_transport, state)` |
| Gate tool calls on user approval | any embedder | Implement `PermissionGate`, set via `SessionHandle::permission_gate()` |

See [custom-tools.md](./custom-tools.md) and [custom-llm-provider.md](./custom-llm-provider.md) for step-by-step guides.
