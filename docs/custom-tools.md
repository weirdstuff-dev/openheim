# Custom Tools

Openheim's tool system is trait-based. Implementing `ToolHandler` is all you need to expose a new capability to the agent.

For an external tool source (databases, APIs, third-party services), prefer an MCP server — openheim will load its tools automatically. This guide covers writing a tool that runs in the same process as the agent.

---

## The `ToolHandler` trait

```rust
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Returns the tool's schema — what it's called and what arguments it accepts.
    fn definition(&self) -> Tool;

    /// Executes the tool with JSON-encoded arguments and returns the result as a string.
    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String>;
}
```

The `definition` method runs once at startup to populate the list sent to the LLM. The `execute` method is called each time the LLM decides to use the tool.

`turn` is the calling turn's `openheim::core::turn::TurnContext`. It carries everything the built-in tools use to behave well inside an agent session, and custom tools get exactly the same:

| Field | Type | Use it for |
|-------|------|-----------|
| `turn.cancel` | `&CancellationToken` | Race long-running work against it (`tokio::select!`) so a `session/cancel` can interrupt your tool. |
| `turn.work_dir` | `&Path` | The sandbox boundary. Validate any user- or LLM-supplied path with `openheim::tools::sandbox::validate_path(path, turn.work_dir)` before touching the filesystem; it resolves relative paths against `work_dir`, follows symlinks, and rejects anything outside. |
| `turn.client_io` | `&dyn ClientIo` | Ask the client (e.g. an editor's unsaved buffers) to read/write a file before falling back to local I/O. Returns `None` when there is no client to ask. |
| `turn.permission_gate` | `&Arc<dyn PermissionGate>` | Already consulted by the agent loop before your tool runs; only relevant if your tool spawns nested agent turns. |

Ignore the fields you don't need — a tool that calls an HTTP API only cares about `turn.cancel`, if that.

`openheim::tools::args` provides `parse_args(args)` and `require_str(&value, "key")`, which produce the same "failed to parse arguments" / "missing 'key' argument" errors the built-ins use, so the LLM sees consistent feedback.

---

## Step-by-step example

The following implements a `lookup` tool that queries a fixed third-party API and returns the response body.

### 1. Define the struct

```rust
use async_trait::async_trait;
use openheim::error::{Error, Result};
use openheim::core::models::Tool;
use openheim::core::turn::TurnContext;
use openheim::tools::ToolHandler;
use openheim::tools::args::{parse_args, require_str};
use serde_json::json;

pub struct LookupTool {
    client: reqwest::Client,
}

impl LookupTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}
```

### 2. Implement `definition`

Return a `Tool` with a JSON Schema describing the arguments. The LLM uses the `description` fields to decide when and how to call the tool.

```rust
fn definition(&self) -> Tool {
    Tool::function(
        "lookup",
        "Query the example API for a term and return the result as text. \
         Use for looking up documentation or reference entries.",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term to look up"
                }
            },
            "required": ["query"]
        }),
    )
}
```

### 3. Implement `execute`

Parse the JSON arguments, run the operation, and return a `String`. Return `Err` only for infrastructure failures — for user-visible failures (e.g. HTTP 404), prefer returning a descriptive string so the LLM can react to the failure. Racing the request against `turn.cancel` lets the user interrupt a slow fetch.

**Don't hand an LLM-supplied URL straight to `client.get(url).send()`.** The model can be steered (directly or via prompt injection in fetched content) into requesting internal addresses — cloud metadata endpoints, loopback, RFC1918 ranges — turning the tool into an SSRF primitive. Openheim already ships a hardened `web_fetch` built-in (scheme allowlist, DNS-resolve-then-check with the checked address pinned for the actual connection so a second, rebound lookup can't bypass it, no automatic redirects, timeout, and a response size cap); prefer registering that instead of writing your own general-purpose fetcher. If your tool genuinely needs its own HTTP call, restrict it to a fixed, non-LLM-controlled endpoint rather than an arbitrary model-supplied URL:

```rust
const API_ENDPOINT: &str = "https://api.example.com/v1/lookup";

async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
    let v = parse_args(args)?;
    let query = require_str(&v, "query")?; // used as a query param, never as the URL itself

    let request = async {
        let response = self.client
            .get(API_ENDPOINT)
            .query(&[("q", query)])
            .send()
            .await
            .map_err(|e| Error::ToolExecutionError(format!("request failed: {e}")))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::ToolExecutionError(format!("failed to read body: {e}")))?;

        if !status.is_success() {
            return Ok(format!("HTTP {status}: {body}"));
        }

        Ok(body)
    };

    tokio::select! {
        _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError("lookup cancelled".into())),
        result = request => result,
    }
}
```

A tool that touches the filesystem should validate its path first, and prefer the client's view of the file when there is one:

```rust
use openheim::tools::sandbox::validate_path;

async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
    let v = parse_args(args)?;
    let path = validate_path(require_str(&v, "path")?, turn.work_dir)?;
    let content = match turn.client_io.read_file(&path).await {
        Some(result) => result?,               // the client answered
        None => tokio::fs::read_to_string(&path).await?, // no client: local disk
    };
    Ok(content.lines().count().to_string())
}
```

### 4. Register the tool

The simplest path is the client builder, which wires the tool into a fully-configured runtime (sandbox, MCP servers, subagents, history):

```rust
let client = OpenheimClient::builder()
    .tool(Box::new(FetchUrlTool::new()))
    .build()
    .await?;
```

If you're driving the agent loop yourself, use `SystemToolExecutor::register` and build the `TurnContext` by hand:

```rust
use openheim::tools::SystemToolExecutor;
use openheim::core::agent::run_agent_with_history;
use openheim::core::client_io::NoClientIo;
use openheim::core::models::Message;
use openheim::core::permission::{AllowAll, PermissionGate};
use openheim::core::turn::TurnContext;
use openheim::config::{AgentConfig, load_config};
use std::path::Path;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> openheim::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;

    let mut executor = SystemToolExecutor::new();
    executor.register_builtins();                        // built-ins
    executor.register(Box::new(FetchUrlTool::new()));    // your tool
    let executor = Arc::new(executor);

    // Build the LLM client from config
    let http = openheim::config::build_http_client(agent_config.timeout_secs)?;
    let llm = openheim::config::create_client(&agent_config, &http);

    let mut messages = vec![Message::user("Fetch https://example.com and summarise it.")];

    let work_dir = std::env::current_dir()?;
    let turn = TurnContext {
        cancel: &CancellationToken::new(),
        permission_gate: &(Arc::new(AllowAll) as Arc<dyn PermissionGate>),
        work_dir: &work_dir,   // sandbox boundary for the file tools
        client_io: &NoClientIo, // no editor to delegate file I/O to
    };

    let result = run_agent_with_history(
        llm,
        executor,
        &agent_config,
        &mut messages,
        None, // prompt_builder
        &turn,
    )
    .await?;

    println!("{}", result.final_response);
    Ok(())
}
```

---

## Tool design guidelines

### Descriptions matter

The LLM decides when to call a tool based entirely on its `description`. Write descriptions that include:
- **What it does** — one clear sentence
- **When to use it** — scenarios where it applies
- **What it returns** — especially if the format is non-obvious

### Argument schemas

Keep argument schemas flat and explicit. Avoid deeply nested objects. Mark required fields in the `"required"` array. Add a `"description"` to every property — the LLM reads these to construct correct calls.

### Error as content

When an operation fails in a recoverable way (file not found, HTTP error, permission denied), return the error as a string rather than propagating it as `Err`. The agent loop feeds tool output back to the LLM, which can then decide to try something else. Reserve `Err` for unrecoverable failures that should stop the loop.

```rust
// Good: LLM can react to this
Ok(format!("Error reading {path}: file not found"))

// Also fine when the failure is unrecoverable or likely a bug
Err(Error::ToolExecutionError("database connection pool exhausted".into()))
```

### Idempotency

Prefer idempotent tools. The agent may call the same tool multiple times with identical arguments across retries. Mutations (writes, POSTs, deletes) should document this in their description so the LLM is careful.

---

## Using MCP instead

For tools backed by an external process or service, use an MCP server instead of implementing `ToolHandler` directly. MCP servers are loaded automatically from configuration and their tools are available in every session without any code changes.

See [configuration.md](./configuration.md) for how to configure MCP servers.
