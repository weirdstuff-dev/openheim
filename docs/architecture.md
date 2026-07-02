# Architecture

Openheim is a multi-provider LLM agent runtime built around the [Agent Client Protocol (ACP)](https://github.com/block/agent-client-protocol). This document explains how the modules fit together and traces the path of a single prompt from entry point to response.

---

## Usage modes

Openheim can be used in two ways:

| Mode | Entry point | Use case |
|------|-------------|----------|
| **Library** | `OpenheimClient` in `src/client.rs` | Embedded in a Rust application |
| **Server** | `src/main.rs` subcommands | Standalone process driven by a client over a transport |

Both modes converge on the same `acp::serve` loop — the library mode runs it in-process over an in-memory pipe; server mode exposes it over a real transport.

---

## Module map

```
src/
├── main.rs             CLI entry point (clap subcommands)
├── lib.rs              Public re-exports for library users
│
├── client.rs           OpenheimClient / SessionHandle — library facade
├── acp/                Agent Client Protocol server implementation
│   ├── mod.rs          AgentState, serve(), request handlers
│   └── session.rs      Per-session state (model, skills, cwd)
│
├── core/
│   ├── agent.rs        Agent loop — LLM ↔ tool call iteration
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
│  (+ RetryClient) │  │                      │
│                  │  │  MCP tools:          │
└──────────────────┘  │    {server}__{tool}  │
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

The `OpenheimClient` facade (`src/client.rs`) wraps the ACP protocol behind a simple Rust API. Internally it:

1. Builds an `AgentState` (LLM client, tool executor, RAG context).
2. Creates an in-process duplex pipe (`tokio::io::duplex`).
3. Runs `acp::serve` on one end as a background task.
4. Drives an ACP `Client` on the other end in response to `session.prompt(…)` calls.

This means the library API and the server transports share the exact same agent logic — there is no separate "library mode" code path.

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
