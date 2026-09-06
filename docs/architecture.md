# Architecture

Openheim is a multi-provider LLM agent runtime built around the [Agent Client Protocol (ACP)](https://github.com/block/agent-client-protocol). This document explains how the modules fit together and traces the path of a single prompt from entry point to response.

---

## Usage modes

Openheim can be used in two ways:

| Mode | Entry point | Use case |
|------|-------------|----------|
| **Library** | `OpenheimClient` in `src/client.rs` | Embedded in a Rust application |
| **Server** | `src/main.rs` subcommands | Standalone process driven by a client over a transport |

Both modes share the same agent logic. Transports speak the ACP wire protocol to `acp::serve`; the library facade calls the same `core::runtime::AgentState` request handlers directly, and the headless `run` mode connects an ACP client to `acp::serve` over an in-memory duplex pipe.

---

## Module map

```
src/
├── main.rs             CLI entry point (clap subcommands)
├── lib.rs              Public re-exports for library users
│
├── client.rs           OpenheimClient / SessionHandle — library facade
├── acp/                Agent Client Protocol wire adapter
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
│   ├── runtime/        The runtime core — used by acp, the other transports, and the library facade
│   │   ├── state.rs    AgentState — shared handle, request handlers
│   │   ├── session.rs  Live session map — state and eviction policy
│   │   └── mod.rs      AgentMode (session tool policy)
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
├── memory/             Agent memory — sessions, skills, identity, prompt assembly
│   ├── mod.rs          MemoryContext — history + skills + system identity
│   ├── history.rs      HistoryManager — conversation persistence
│   ├── lease.rs        Advisory cross-process write lease per conversation
│   ├── skills.rs       SkillsManager — Markdown skill files
│   ├── system.rs       SystemLoader — reads ~/.openheim/system.md
│   └── prompt.rs       PromptBuilder — assembles structured system message
│
├── rag/                Long-term memory via tool calls (feature `rag`)
│   ├── mod.rs          LongTermMemory — remember / search / forget facade
│   ├── embedding/      EmbeddingClient trait, OpenAI-compatible + Gemini clients
│   ├── store.rs        VectorStore — SQLite: FTS5 keyword index + sqlite-vec (vec0, cosine KNN)
│   └── tool.rs         remember, search_memory, and forget tools
│
├── subagents/          Subagent profiles — delegated, isolated agent personas
│   └── mod.rs          AgentProfile, SubagentLoader — Markdown files in ~/.openheim/agents/
│
├── tools/              Tool abstraction and built-in implementations
│   ├── mod.rs          ToolHandler / ToolExecutor traits, SystemToolExecutor
│   ├── args.rs         parse_args / require_str — shared argument decoding
│   ├── sandbox.rs      validate_path — work_dir boundary used by every file tool
│   ├── execute_command.rs
│   ├── read_file.rs
│   ├── write_file.rs
│   ├── edit_file.rs    Targeted string replacement, no whole-file rewrite
│   ├── list_dir.rs     Immediate directory contents
│   ├── search.rs       Regex search across files, ripgrep-style (ripgrep's own crates)
│   ├── web_fetch.rs    Fetch a public http(s) URL as text (SSRF-guarded)
│   ├── scoped_executor.rs     ScopedExecutor — tool-name allowlist wrapper
│   └── delegate.rs            DelegateTool — delegate_task tool (subagents)
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
│      acp::serve / core::runtime::AgentState │
│                                             │
│  AgentState holds:                          │
│  • llm       Arc<dyn LlmClient>             │
│  • executor  Arc<dyn ToolExecutor>          │
│  • memory    MemoryContext                  │
│  • long_term_memory Arc<LongTermMemory>     │
│  • config    AgentConfig (model, iters, …)  │
│  • sessions  HashMap<id, SessionState>      │
└────────────────────┬────────────────────────┘
                     │ AgentState::prompt()
                     ▼
┌─────────────────────────────────────────────┐
│         memory::MemoryContext::prepare      │
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
│  AgentState::prompt appends each            │
│  MessageAppended event to history.jsonl,    │
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
                      │    remember          │
                      │    search_memory     │
                      │    forget            │
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
├── skills/
│   ├── rust.md              Named skill files (Markdown)
│   └── …
└── memory.db                Long-term memory notes, FTS5 index, and (optionally) sqlite-vec vectors
```

`SystemLoader` reads `system.md` on every `prepare()` call — missing file is a hard error (run `openheim init` to create it). `HistoryManager` reads and writes each conversation's metadata (`.json`) and message log (`.jsonl`) — see `src/memory/history.rs`'s doc comment for why they're split. `SkillsManager` reads `.md` files from the skills directory. Both history and skills paths are configurable at construction time, which is how the test suite uses temporary directories.

### Long-term memory

`core::runtime::AgentState::new` always opens `memory.db` and registers three tools. `remember(content)` inserts a note into a `memories` table; an FTS5 external-content index (`memories_fts`) tracks it via triggers. `search_memory(query)` runs an FTS5 BM25 query (each word quoted and OR-ed, so user input can't inject query syntax) and returns the best notes with their date. `forget(id)` deletes a note. Nothing is stored or injected automatically: the model calls the tools when the user asks it to remember, recall, or drop something, or when it judges a stored preference relevant.

When `[memory]` names an `embedding_provider` and `embedding_model`, the same store gains a `vec0` virtual table with cosine distance: `remember` embeds the note (OpenAI-compatible `/embeddings` or Gemini `batchEmbedContents`) and `search_memory` becomes a KNN `MATCH` over embeddings instead of keyword search. Notes written before embeddings were enabled are back-filled on the next call. The store records the embedding model and dimension; if either changes, the vectors are dropped and every note is re-embedded from its stored text, since vectors from different models aren't comparable.

---

## ACP and the library facade

The `OpenheimClient` facade (`src/client.rs`) wraps the same `core::runtime::AgentState` the transports use, behind a simple Rust API. Internally it:

1. Builds an `AgentState` (LLM client, tool executor, memory context, long-term memory).
2. Calls its request handlers (`prompt`, `load_session`, `cancel_session`, …) directly — no wire protocol, no background task. Streaming updates reach your callback through the same `SessionUpdate` events a transport would forward.

The duplex-pipe + ACP-client wiring does exist — in `src/transport/run.rs`, where the headless `openheim run` mode drives `acp::serve` over `tokio::io::duplex`. Either way there is no separate "library mode" agent logic: the facade and every transport share the exact same session and agent-loop code path.

Every transport (`stdio`, `ws`, `run`) builds its `AgentState` the same way: `OpenheimClient::builder().build().await?` followed by `OpenheimClient::state()` (`pub(crate)`, so only reachable from inside this crate) to get the `Arc<AgentState>` `acp::serve` wants. That's the same load-config → resolve → `MemoryContext::new` → `AgentState::new` sequence the builder already does for library users, so there's exactly one place that canonicalizes `work_dir`, merges builder-registered MCP servers, and registers custom tools — a hand-rolled sequence in a transport would silently skip all of that.

---

## Key extension points

| What to extend | Where | How |
|----------------|-------|-----|
| Add a new LLM provider | `src/core/llm/` | Implement `LlmClient` |
| Add an embeddings backend | `src/rag/embedding/` | Implement `EmbeddingClient`, pass to `LongTermMemory::new` |
| Add a built-in tool | `src/tools/` | Implement `ToolHandler`, register in `register_builtins` |
| Add a tool without touching source | any embedder | Implement `ToolHandler`, register via `OpenheimBuilder::tool()` |
| Add an external tool source | `src/mcp/` | MCP servers via configuration or `McpClient` |
| Add a new transport | `src/transport/` | Build via `OpenheimClient::builder().build()`, then call `acp::serve(your_transport, client.state().clone())` |
| Gate tool calls on user approval | any embedder | Implement `PermissionGate`, set via `SessionHandle::permission_gate()` |

See [custom-tools.md](./custom-tools.md) and [custom-llm-provider.md](./custom-llm-provider.md) for step-by-step guides.
