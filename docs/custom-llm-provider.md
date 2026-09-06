# Custom LLM Provider

Openheim's `LlmClient` trait abstracts over any chat-completion backend. Implement it to connect to a provider that isn't built in — a self-hosted model, a private API, a research endpoint, or a mock for testing.

---

## The `LlmClient` trait

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a chat request and return the first choice from the provider.
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice>;
}
```

`messages` is the full conversation history (user, assistant, tool-result turns). `tools` is the list of currently registered tools in JSON-schema format. Return the model's next `Choice` — either a text response or a set of tool calls.

`send_streaming` is a second trait method with a default implementation that calls `send` and forwards the whole response as one `LlmChunk::Text`. Override it if your provider supports token-by-token streaming; otherwise the default is fine.

---

## Key types

```rust
// Input
pub struct Message {
    pub role: Role,              // User | Assistant | System | Tool
    pub content: Vec<ContentBlock>,
}

pub enum ContentBlock {
    Text { text: String },
    Thinking { thinking: String, signature: Option<String> }, // extended-thinking output; signature must round-trip unmodified
    Image { data: String, mime_type: String },                // data is base64-encoded
    ToolUse { id: String, name: String, arguments: String },  // arguments is a JSON string
    ToolResult { tool_call_id: String, tool_name: String, content: String, is_error: bool },
}

pub struct Tool {
    pub tool_type: String,                 // always "function"
    pub function: FunctionDefinition,
}

pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,    // JSON Schema object
}

// Output
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,     // None if your provider doesn't report token usage
}

pub enum FinishReason {
    Stop,               // normal completion
    ToolCalls,          // model wants to invoke tools
    MaxTokens,          // truncated at the token limit
    Other(String),      // provider-specific reason with no equivalent above
}

pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64, // tokens written to a prompt cache
    pub cache_read_tokens: u64,     // tokens served from a prompt cache
}
```

`content` is an ordered list of blocks rather than a single string — an assistant turn is commonly `[Thinking?, Text?, ToolUse*]`; a `Role::Tool` message holds exactly one `ToolResult` block; a `Role::User` message holds `Text`/`Image` blocks. `Message` has convenience accessors so you rarely need to pattern-match the enum directly:

- `message.text() -> Option<String>` — concatenation of all `Text` blocks
- `message.tool_calls() -> Vec<ToolUseBlock>` — all `ToolUse` blocks, each `{ id, name, arguments }`
- `message.tool_result_block() -> Option<ToolResultBlock>` — the `ToolResult` block on a `Role::Tool` message, `{ tool_call_id, tool_name, content, is_error }`

and constructors for building your own:

- `Message::user(text)`, `Message::assistant(text)` — single-`Text`-block message
- `Message::tool_result(tool_call_id, tool_name, content, is_error)` — single-`ToolResult`-block message

The agent loop treats `finish_reason == Some(FinishReason::Stop)` as the signal to end the conversation. Any other finish reason with no tool calls also ends the loop (with a warning). If `message.tool_calls()` is non-empty, the loop executes them and continues.

`usage` is optional — set it if your API returns token counts, otherwise leave it `None`. When present, it's surfaced to embedders as the session's current context-size snapshot (`SessionHandle::context_usage()` / `ConversationMeta.context_usage`); leaving it `None` just means that feature has nothing to report for this provider.

---

## Step-by-step example

The following implements a provider that speaks a hypothetical OpenAI-compatible API with a custom base URL and auth scheme.

### 1. Define the client struct

```rust
use async_trait::async_trait;
use openheim::core::models::{Choice, ContentBlock, FinishReason, Message, Role, Tool};
use openheim::error::{Error, Result};
use openheim::llm::LlmClient;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct MyCustomProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl MyCustomProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}
```

### 2. Define request/response shapes

Map openheim's types to what the remote API expects. Most chat-completion APIs follow the OpenAI schema closely, so if that's the case, use `OpenAiCompatibleClient` instead of writing this by hand — see `core/llm/openai.rs` for the reference conversion (including how it flattens `ContentBlock`s into OpenAI's `content`/`tool_calls` wire shape).

```rust
#[derive(Serialize)]
struct ApiRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: Option<String>,
}

#[derive(Deserialize)]
struct ApiResponse {
    choices: Vec<ApiChoice>,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    content: Option<String>,
    // tool_calls: Option<Vec<…>>,  add if the API supports function calling
}
```

### 3. Implement `send`

```rust
#[async_trait]
impl LlmClient for MyCustomProvider {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let api_messages: Vec<ApiMessage> = messages
            .iter()
            .map(|m| ApiMessage {
                role: match m.role {
                    Role::User => "user".into(),
                    Role::Assistant => "assistant".into(),
                    Role::System => "system".into(),
                    Role::Tool => "tool".into(),
                },
                // This example only forwards text; a real provider that supports
                // images or tool calls would walk `m.content` for those block
                // types too (see `ContentBlock` above).
                content: m.text(),
            })
            .collect();

        let api_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| serde_json::json!({
                "type": t.tool_type,
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                }
            }))
            .collect();

        let body = ApiRequest {
            model: self.model.clone(),
            messages: api_messages,
            tools: api_tools,
        };

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(Error::ReqwestError)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HttpError { status, body });
        }

        let api_resp: ApiResponse = response
            .json()
            .await
            .map_err(|e| Error::ParseError(format!("failed to parse response: {e}")))?;

        let choice = api_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::ApiError("empty choices array".into()))?;

        let content = match choice.message.content {
            Some(text) => vec![ContentBlock::Text { text }],
            None => vec![],
        };

        // Map the raw wire value onto the provider-agnostic `FinishReason` —
        // do this once, at the response-parsing boundary, per the pattern
        // used by `core::llm::{anthropic,gemini,openai}`.
        let finish_reason = choice.finish_reason.as_deref().map(|r| match r {
            "stop" => FinishReason::Stop,
            "tool_calls" => FinishReason::ToolCalls,
            "length" => FinishReason::MaxTokens,
            other => FinishReason::Other(other.to_string()),
        });

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content,
            },
            finish_reason,
            // This example's `ApiResponse` doesn't model a `usage` field —
            // add one (following `core/llm/openai.rs`'s `OpenAiUsage`) and
            // map it here if your API reports token counts.
            usage: None,
        })
    }
}
```

### 4. Wrap with `RetryClient` (optional but recommended)

`RetryClient` wraps any `LlmClient` and retries on transient errors (rate limits, 5xx, network timeouts) with exponential backoff. Non-streaming `send` calls are retried up to three times; streaming `send_streaming` calls are retried only while it is still safe — i.e. before your provider has emitted the first chunk to the caller. Once the first token has been forwarded, a mid-stream failure is returned as-is rather than replayed (which would duplicate output):

```rust
use openheim::llm::RetryClient;
use std::sync::Arc;

let base_provider = MyCustomProvider::new(
    "https://api.myprovider.com",
    std::env::var("MY_PROVIDER_KEY").unwrap(),
    "my-model-v1",
);

let llm: Arc<dyn LlmClient> = Arc::new(RetryClient::new(Arc::new(base_provider)));
```

### 5. Use with the agent loop

Pass the custom client directly to `run_agent_with_history`. It also needs a `TurnContext` (cancellation token + permission gate) — use `permission::AllowAll` and a fresh `CancellationToken` for a one-shot, non-interactive run:

```rust
use openheim::core::agent::run_agent_with_history;
use openheim::core::models::Message;
use openheim::core::permission::{AllowAll, PermissionGate};
use openheim::core::turn::TurnContext;
use openheim::config::load_config;
use openheim::tools::SystemToolExecutor;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> openheim::Result<()> {
    let llm: Arc<dyn openheim::llm::LlmClient> = Arc::new(
        RetryClient::new(Arc::new(MyCustomProvider::new(
            "https://api.myprovider.com",
            std::env::var("MY_PROVIDER_KEY").unwrap(),
            "my-model-v1",
        )))
    );

    let mut executor = SystemToolExecutor::new();
    executor.register_builtins();
    let executor = Arc::new(executor);

    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;

    let mut messages = vec![Message::user("Hello!")];

    let work_dir = std::env::current_dir()?;
    let turn = TurnContext {
        cancel: &CancellationToken::new(),
        permission_gate: &(Arc::new(AllowAll) as Arc<dyn PermissionGate>),
        work_dir: &work_dir,
        client_io: &openheim::core::client_io::NoClientIo,
    };

    let result = run_agent_with_history(
        llm,
        executor,
        &agent_config,
        &mut messages,
        None, // prompt_builder — Some(&builder) to prepend a system.md/skills identity
        &turn,
    )
    .await?;

    println!("{}", result.final_response);
    Ok(())
}
```

---

## Testing a custom provider

Use a mock `LlmClient` to test prompt logic without making real API calls. The agent's test suite in `src/core/agent.rs` shows the pattern:

```rust
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use openheim::core::models::{Choice, Message, Role, Tool};
use openheim::error::{Error, Result};
use openheim::llm::LlmClient;

struct MockLlm {
    responses: Mutex<Vec<Choice>>,
}

impl MockLlm {
    fn with_responses(responses: Vec<Choice>) -> Arc<Self> {
        Arc::new(Self { responses: Mutex::new(responses) })
    }
}

#[async_trait]
impl LlmClient for MockLlm {
    async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
        self.responses
            .lock()
            .unwrap()
            .pop()
            .ok_or_else(|| Error::ApiError("no more mock responses".into()))
    }
}
```

Build a tool-call response for a mock with `Message { role: Role::Assistant, content: vec![ContentBlock::ToolUse { id: "call_1".into(), name: "read_file".into(), arguments: "{}".into() }] }` — see `core::models::ContentBlock` for the other block types.

---

## Already covered by `OpenAiCompatibleClient`

If the target API speaks the OpenAI chat-completions format, use the built-in `OpenAiCompatibleClient` with a custom `base_url` in `config.toml` rather than implementing `LlmClient` from scratch:

```toml
[providers.my-provider]
api_base = "https://api.myprovider.com/v1"
default_model = "my-model-v1"
models = ["my-model-v1"]
env_var = "MY_PROVIDER_KEY"
```

This covers Ollama, vLLM, LM Studio, Mistral, and any other OpenAI-compatible endpoint.
