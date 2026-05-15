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

---

## Key types

```rust
// Input
pub struct Message {
    pub role: Role,                        // User | Assistant | System | Tool
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>, // set by the model when it calls tools
    pub tool_call_id: Option<String>,      // set on Role::Tool messages
    pub tool_name: Option<String>,
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
    pub finish_reason: Option<String>,    // "stop" | "tool_calls" | "length" | …
}
```

The agent loop treats `finish_reason == "stop"` as the signal to end the conversation. Any other finish reason with no tool calls also ends the loop (with a warning). If `message.tool_calls` is set, the loop executes them and continues.

---

## Step-by-step example

The following implements a provider that speaks a hypothetical OpenAI-compatible API with a custom base URL and auth scheme.

### 1. Define the client struct

```rust
use async_trait::async_trait;
use openheim::core::models::{Choice, Message, Role, Tool, ToolCall, FunctionCall};
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

Map openheim's types to what the remote API expects. Most chat-completion APIs follow the OpenAI schema closely, so if that's the case, use `OpenAiCompatibleClient` instead of writing this by hand.

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
                content: m.content.clone(),
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
            .map_err(|e| Error::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError(format!("HTTP {status}: {text}")));
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

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content: choice.message.content,
                tool_calls: None,
                tool_call_id: None,
                tool_name: None,
            },
            finish_reason: choice.finish_reason,
        })
    }
}
```

### 4. Wrap with `RetryClient` (optional but recommended)

`RetryClient` wraps any `LlmClient` and retries on transient errors (rate limits, 5xx, network timeouts) with exponential backoff:

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

Pass the custom client directly to `run_agent_with_history`:

```rust
use openheim::core::agent::run_agent_with_history;
use openheim::core::models::Message;
use openheim::config::{AgentConfig, load_config};
use openheim::tools::SystemToolExecutor;
use std::sync::Arc;

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

    let mut messages = vec![Message::user("Hello!".into())];

    let result = run_agent_with_history(
        llm,
        executor,
        &agent_config,
        &mut messages,
        None,
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
