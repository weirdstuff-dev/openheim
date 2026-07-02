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
    async fn execute(&self, args: &str) -> Result<String>;
}
```

The `definition` method runs once at startup to populate the list sent to the LLM. The `execute` method is called each time the LLM decides to use the tool.

---

## Step-by-step example

The following implements a `fetch_url` tool that downloads a URL and returns its body.

### 1. Define the struct

```rust
use async_trait::async_trait;
use openheim::error::{Error, Result};
use openheim::core::models::{FunctionDefinition, Tool};
use openheim::tools::ToolHandler;
use serde_json::json;

pub struct FetchUrlTool {
    client: reqwest::Client,
}

impl FetchUrlTool {
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
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "fetch_url".to_string(),
            description: "Download the content of a URL and return the response body as text. \
                          Use for fetching documentation, APIs, or web pages.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        },
    }
}
```

### 3. Implement `execute`

Parse the JSON arguments, run the operation, and return a `String`. Return `Err` only for infrastructure failures — for user-visible failures (e.g. HTTP 404), prefer returning a descriptive string so the LLM can react to the failure.

```rust
async fn execute(&self, args: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(args)
        .map_err(|e| Error::ParseError(format!("invalid args: {e}")))?;

    let url = v["url"]
        .as_str()
        .ok_or_else(|| Error::ParseError("missing 'url' argument".into()))?;

    let response = self.client
        .get(url)
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
}
```

### 4. Register the tool

Use `SystemToolExecutor::register` before running the agent:

```rust
use openheim::tools::SystemToolExecutor;
use openheim::core::agent::run_agent_with_history;
use openheim::core::models::Message;
use openheim::core::permission::{AllowAll, PermissionGate};
use openheim::core::turn::TurnContext;
use openheim::config::{AgentConfig, load_config};
use openheim::rag::RagContext;
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

    let turn = TurnContext {
        cancel: &CancellationToken::new(),
        permission_gate: &(Arc::new(AllowAll) as Arc<dyn PermissionGate>),
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
