# openheim as a Rust library

Openheim can be embedded directly in your Rust application. The library exposes the full agent runtime — sessions, streaming, conversation history, RAG, skills, MCP servers, and tools — through a single `OpenheimClient` facade built on top of the [Agent Client Protocol (ACP)](https://github.com/block/agent-client-protocol).

---

## Add to your project

```toml
# Cargo.toml
[dependencies]
openheim = "0.8"
tokio = { version = "1", features = ["full"] }
```

### Feature flags

By default the `openheim` dependency also builds the CLI/TUI binary stack
(`clap`, `ratatui`, `crossterm`, `tracing-subscriber`) and the WebSocket
server stack (`axum`, `tower-http`, `notify`, `walkdir`, `futures`).
Embedders that drive the agent through `OpenheimClient` (or their own ACP
wiring) usually don't need those:

```toml
openheim = { version = "0.10", default-features = false }
# optionally: features = ["server"]  # axum WS/REST server (openheim::transport::ws)
# optionally: features = ["tui"]     # ratatui terminal UI (openheim::tui)
# optionally: features = ["rag"]     # remember/search_memory/forget long-term memory (rusqlite FTS5 + sqlite-vec)
```

Everything else — the client facade, agent loop, providers, tools, MCP, ACP,
and config — is always available.

---

## Quick start

```rust
use openheim::{OpenheimClient, SessionUpdate};

#[tokio::main]
async fn main() -> openheim::Result<()> {
    // Loads ~/.openheim/config.toml
    let client = OpenheimClient::builder().build().await?;

    let session = client
        .new_session()
        .cwd("/my/project")
        .start()
        .await?;

    session
        .prompt("What files are in the current directory?", |update| {
            if let SessionUpdate::AgentMessageChunk(chunk) = update {
                for block in &chunk.content {
                    if let openheim::ContentBlock::Text(t) = block {
                        print!("{}", t.text);
                    }
                }
            }
        })
        .await?;

    Ok(())
}
```

---

## Client initialisation

### From `~/.openheim/config.toml` (default)

```rust
let client = OpenheimClient::builder().build().await?;
```

### From a custom config file

```rust
let client = OpenheimClient::from_config("/etc/myapp/openheim.toml")
    .build()
    .await?;
```

### Programmatic config (no file needed)

```rust
let client = OpenheimClient::builder()
    .provider("anthropic")
    .api_key("sk-ant-...")
    .model("claude-opus-4-7")
    .max_iterations(15)
    .build()
    .await?;
```

Supported `provider` values: `"openai"`, `"anthropic"`, `"gemini"`, or any string for OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, etc.).

Default models when `.model()` is omitted:
- `"anthropic"` → `claude-sonnet-4-6`
- `"gemini"` → `gemini-2.0-flash`
- everything else → `gpt-4o`

### Security controls

Two builder methods control the agent's access boundary. Both override the corresponding `config.toml` fields when set.

```rust
let client = OpenheimClient::builder()
    .provider("openai")
    .api_key("sk-...")
    // Restrict file access to this directory tree
    .work_dir("/home/user/projects/myproject")
    // Remove the execute_command tool from the LLM's tool list entirely
    .allow_shell(false)
    .build()
    .await?;
```

**`.work_dir(path)`** — sets the root directory the agent may read and write. The agent cannot access files outside this tree. Relative paths in tool arguments are resolved against this directory. Defaults to the directory from which the process was invoked when not set in the builder or config file.

**`.allow_shell(bool)`** — controls whether the `execute_command` tool is exposed to the LLM. When `false` the tool is removed from the tool list entirely; the LLM never sees it and cannot request it. Defaults to `false`.

### With MCP servers

MCP servers can be added in either mode. Their tools become available to the agent automatically as `{server_name}__{tool_name}`.

```rust
use openheim::{McpServerConfig, OpenheimClient};
use std::collections::HashMap;

let client = OpenheimClient::builder()
    .provider("openai")
    .api_key(std::env::var("OPENAI_API_KEY").unwrap())
    // stdio MCP server
    .mcp_server("filesystem", McpServerConfig {
        command: Some("npx".into()),
        args: vec![
            "-y".into(),
            "@modelcontextprotocol/server-filesystem".into(),
            "/workspace".into(),
        ],
        env: HashMap::new(),
        url: None,
    })
    // Streamable HTTP MCP server
    .mcp_server("my-tools", McpServerConfig {
        command: None,
        args: vec![],
        env: HashMap::new(),
        url: Some("http://localhost:8080/mcp".into()),
    })
    .build()
    .await?;
```

MCP servers defined in a config file are always loaded; builder `.mcp_server()` calls are merged in on top.

### With custom tools

`.tool()` registers an in-process `ToolHandler` alongside the built-ins and any MCP-sourced tools, subject to the same `work_dir`/`allow_shell` sandbox boundary. See [custom-tools.md](./custom-tools.md) for how to implement `ToolHandler`.

```rust
let client = OpenheimClient::builder()
    .provider("openai")
    .api_key(std::env::var("OPENAI_API_KEY").unwrap())
    .tool(Box::new(FetchUrlTool::new()))
    .build()
    .await?;
```

---

## Sessions

Sessions are the unit of conversation. Each session has its own message history, model, skills, and working directory.

### Create a session

```rust
let session = client
    .new_session()
    .model("gpt-4o")                          // optional — overrides the config default
    .skills(vec!["rust".into(), "tdd".into()]) // optional — names of ~/.openheim/skills/*.md
    .cwd("/my/workspace")                      // optional — used for history filtering
    .start()
    .await?;

println!("session id: {}", session.id);
```

### Send a prompt (streaming)

`prompt` calls your callback once per ACP `SessionUpdate` event as the agent runs.

```rust
use openheim::{AcpToolCall, ContentBlock, SessionUpdate};

session
    .prompt("Refactor the auth module to use JWTs", |update| {
        match update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                for block in &chunk.content {
                    if let ContentBlock::Text(t) = block {
                        print!("{}", t.text);
                    }
                }
            }
            SessionUpdate::ToolCall(tc) => {
                println!("\n[tool] {} — running…", tc.name);
            }
            SessionUpdate::ToolCallUpdate(tcu) => {
                println!("[tool] {} — done", tcu.id);
            }
            _ => {}
        }
    })
    .await?;
```

### Send a prompt with images

`prompt_with_images` sends a turn that mixes text with one or more images — useful with any vision-capable provider (Anthropic, OpenAI, Gemini). Each image is a `(base64_data, mime_type)` pair; the text block (when non-empty) leads, followed by the images. It streams the same `SessionUpdate` events as `prompt`, which itself delegates to `prompt_with_images` with no images.

```rust
let png = std::fs::read("screenshot.png")?;
let data = base64::engine::general_purpose::STANDARD.encode(&png);

session
    .prompt_with_images(
        "What's in this screenshot?",
        vec![(data, "image/png".to_string())],
        |update| { /* same SessionUpdate events as `prompt` */ },
    )
    .await?;
```

### Multi-turn conversation

Call `prompt` multiple times on the same handle. The agent accumulates history on disk automatically.

```rust
session.prompt("My name is Alice", |_| {}).await?;
session.prompt("What's my name?", |update| { /* prints "Alice" */ }).await?;
```

### Context usage

`session.context_usage().await?` returns the token usage of the most recent LLM call — how full the context window is *right now*, not a running total across the session. `None` until a provider has reported usage.

```rust
if let Some(usage) = session.context_usage().await? {
    println!(
        "context: {} in / {} out ({} cache read / {} cache write)",
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_creation_tokens
    );
}
```

It's also persisted on the conversation, so it's readable via `client.get_session(id)` (`conversation.meta.context_usage`) without an active `SessionHandle` — see [Get full conversation](#get-full-conversation-messages--metadata) below.

### Permission gate, cancellation, and client I/O

By default a `SessionHandle` allows every tool call unconditionally (`AllowAll`) — the embedder is trusted to have already consented to the run. For an interactive embedder, supply your own [`PermissionGate`](../src/core/permission.rs) so the agent asks before running a tool call:

```rust
use openheim::core::permission::{PermissionDecision, PermissionGate};
use std::sync::Arc;

struct CliConfirmGate;

#[async_trait::async_trait]
impl PermissionGate for CliConfirmGate {
    async fn check(&self, _id: &str, tool_name: &str, arguments: &str) -> PermissionDecision {
        eprintln!("allow {tool_name}({arguments})? [y/N]");
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        if line.trim().eq_ignore_ascii_case("y") {
            PermissionDecision::AllowOnce
        } else {
            PermissionDecision::RejectOnce
        }
    }
}

let session = client
    .new_session()
    .start()
    .await?
    .permission_gate(Arc::new(CliConfirmGate));
```

`PermissionGate::check` is called once per tool call, before it executes — including tool calls made by a `delegate_task` subagent, which inherits the parent turn's gate rather than always-allowing.

`.client_io(Arc<dyn ClientIo>)` similarly lets `read_file`/`write_file`/`edit_file` be delegated to the embedder's own I/O (e.g. an editor's unsaved buffers) instead of local disk — see [`ClientIo`](../src/core/client_io.rs). `edit_file` uses it for both the read and the write, since an edit is a read followed by a write. Both `.permission_gate()` and `.client_io()` carry over automatically when a handle is reused via `.restore()`.

`session.cancel().await` cancels the turn currently in flight for that session (no-op if none is running) — call it from another task while `prompt()` is awaiting.

---

## History & session management

### List sessions

```rust
use std::path::Path;

// All sessions, newest first
let all = client.list_sessions(None).await?;

// Only sessions from a specific working directory
let workspace = client.list_sessions(Some(Path::new("/my/workspace"))).await?;

for info in &workspace {
    println!("{} — {}", info.id, info.title.as_deref().unwrap_or("untitled"));
}
```

### Get full conversation (messages + metadata)

```rust
let conv = client.get_session("550e8400-e29b-41d4-a716-446655440000").await?;

println!("model: {:?}", conv.meta.model);
println!("messages: {}", conv.messages.len());

for msg in &conv.messages {
    println!("[{:?}] {}", msg.role, msg.text().unwrap_or_default());
}
```

`msg.content` is a `Vec<core::models::ContentBlock>` (`Text`/`Thinking`/`Image`/`ToolUse`/`ToolResult`) rather than a plain string — `msg.text()` concatenates the `Text` blocks. Use `msg.tool_calls()` / `msg.tool_result_block()` for the other block types; see `docs/custom-llm-provider.md` for the full `ContentBlock` shape.

`conv.meta.context_usage` is an `Option<core::models::Usage>` — see [Context usage](#context-usage) above.

### Resume a session (load + continue prompting)

`load_session` registers the conversation in the live sessions map and replays the message history through your callback so you can populate a UI.

```rust
let session = client
    .load_session(
        "550e8400-e29b-41d4-a716-446655440000",
        "/my/workspace".into(),
        |update| {
            // replay previous messages into your UI
            match update {
                SessionUpdate::UserMessageChunk(chunk) => { /* render user bubble */ }
                SessionUpdate::AgentMessageChunk(chunk) => { /* render agent bubble */ }
                _ => {}
            }
        },
    )
    .await?;

// Continue where the conversation left off
session.prompt("Continue from where you left off", |update| { /* … */ }).await?;
```

### Delete a session

```rust
client.delete_session("550e8400-e29b-41d4-a716-446655440000").await?;
```

---

## Memory — direct history and skills access

`client.memory()` returns a `&MemoryContext` with direct access to the underlying `HistoryManager` and `SkillsManager`. This is useful for advanced use cases like building custom UIs, searching conversations, or managing skills programmatically.

```rust
let memory = client.memory();

// List all conversation metadata
let metas = memory.history.list_conversations()?;

// Load a full conversation
let conv = memory.history.load_conversation(&uuid)?;

// Save a conversation (e.g. after external edits)
memory.history.save_conversation(&conv)?;

// List available skills
let skills = memory.skills.list_skills()?;
// → ["debugging", "rust", "tdd"]

// Load skill content
let content = memory.skills.load_skill("rust")?;
println!("{content}");
```

---

## Long-term memory

With the `rag` feature, `client.long_term_memory()` returns the `LongTermMemory` behind the `remember` / `search_memory` / `forget` tools. It is keyword search (FTS5) unless the config's `[memory]` section names an embedding provider, in which case search is semantic. You can drive it directly, for example to seed memories or build a memory browser:

```rust
let memory = client.long_term_memory();
let note = memory.remember("The user's staging cluster is eu-west-1.").await?;

// Best match first; `hit.method` says whether the score is cosine similarity or a BM25 rank.
for hit in memory.search("where is staging?", Some(3)).await? {
    println!("#{} {} ({:?} {:.2})\n{}", hit.record.id, hit.record.created_at, hit.method, hit.score, hit.record.content);
}

memory.forget(note.id).await?;
```

To use a custom embeddings backend, implement `openheim::rag::EmbeddingClient` and build `LongTermMemory::new(VectorStore::open(path)?, Some(Arc::new(my_embedder)), top_k)` yourself; wrap it in `RememberTool` / `SearchMemoryTool` / `ForgetTool` and register them via `OpenheimBuilder::tool` if the agent should be able to call them.

---

## Introspection

### Available tools

```rust
for tool in client.tools() {
    println!("{}: {}", tool.function.name, tool.function.description.as_deref().unwrap_or(""));
}
```

### MCP server statuses

```rust
for status in client.mcp_servers() {
    println!(
        "{} [{}] connected={} tools={}{}",
        status.name,
        status.transport,
        status.connected,
        status.tool_count,
        status.error.as_deref().map(|e| format!(" error={e}")).unwrap_or_default(),
    );
}
```

### Available models

```rust
let models = client.models();
println!("default provider: {}", models.default_provider);
for (provider, info) in &models.providers {
    println!("  {provider}: {} (default)", info.default_model);
    for model in &info.models {
        println!("    - {model}");
    }
}
```

---

## Full example — multi-provider app with MCP and history

```rust
use openheim::{ContentBlock, McpServerConfig, OpenheimClient, SessionUpdate};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> openheim::Result<()> {
    let client = OpenheimClient::builder()
        .provider("anthropic")
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap())
        .model("claude-opus-4-7")
        .max_iterations(20)
        .mcp_server("fs", McpServerConfig {
            command: Some("npx".into()),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-filesystem".into(),
                "/workspace".into(),
            ],
            env: HashMap::new(),
            url: None,
        })
        .build()
        .await?;

    // Print MCP connection status
    for s in client.mcp_servers() {
        println!("[mcp] {} — connected={} tools={}", s.name, s.connected, s.tool_count);
    }

    // Check for an existing session or start fresh
    let all_sessions = client.list_sessions(Some(std::path::Path::new("/workspace"))).await?;
    let session = if let Some(last) = all_sessions.first() {
        println!("Resuming session: {}", last.id);
        client
            .load_session(&last.id.to_string(), "/workspace".into(), |_| {})
            .await?
    } else {
        client
            .new_session()
            .skills(vec!["rust".into()])
            .cwd("/workspace")
            .start()
            .await?
    };

    session
        .prompt("Summarise the project structure", |update| {
            if let SessionUpdate::AgentMessageChunk(chunk) = update {
                for block in &chunk.content {
                    if let ContentBlock::Text(t) = block {
                        print!("{}", t.text);
                    }
                }
            }
        })
        .await?;

    println!("\nDone. Session id: {}", session.id);
    Ok(())
}
```

---

## ACP event reference

All events received by the `prompt` callback are `agent_client_protocol::schema::SessionUpdate` variants, re-exported from `openheim`:

| Variant | When |
|---|---|
| `AgentMessageChunk(ContentChunk)` | Streaming text from the LLM |
| `UserMessageChunk(ContentChunk)` | Echoed user message (during `load_session` history replay) |
| `ToolCall(AcpToolCall)` | Agent is about to invoke a tool |
| `ToolCallUpdate(ToolCallUpdate)` | Tool finished; contains status and raw output |

`ContentChunk.content` is a `Vec<ContentBlock>`. Match on `ContentBlock::Text(t)` to get the text string.

---

## Error handling

All fallible operations return `openheim::Result<T>` (`std::result::Result<T, openheim::Error>`).

```rust
use openheim::{Error, OpenheimClient};

match client.get_session("bad-id").await {
    Ok(conv) => { /* … */ }
    Err(Error::ConfigError(msg)) => eprintln!("config: {msg}"),
    Err(Error::Other(msg)) => eprintln!("error: {msg}"),
    Err(e) => eprintln!("unexpected: {e}"),
}
```

Transient LLM errors (rate limits, 5xx, network timeouts) are retried automatically with exponential backoff before surfacing as `Error::HttpError` or `Error::ApiError`.
