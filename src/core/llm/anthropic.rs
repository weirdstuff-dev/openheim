use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::models::{Choice, FunctionCall, Message, Role, Tool, ToolCall};
use crate::error::{Error, Result};

use super::LlmClient;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Clone)]
pub struct AnthropicClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
}

impl AnthropicClient {
    pub fn new(client: ReqwestClient, api_base: String, api_key: String, model: String) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
        }
    }
}

// --- Anthropic request types ---

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

// --- Anthropic response types ---

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseBlock>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::Tool => {
                // Tool results must be sent as user messages with tool_result content blocks
                let block = AnthropicContentBlock::ToolResult {
                    tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                    content: msg.content.clone().unwrap_or_default(),
                };
                // Merge into the last user message if it exists, otherwise create new
                if let Some(last) = result.last_mut() {
                    let last: &mut AnthropicMessage = last;
                    if last.role == "user" {
                        last.content.push(block);
                        continue;
                    }
                }
                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![block],
                });
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let Some(text) = &msg.content {
                    if !text.is_empty() {
                        blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Object(
                                serde_json::Map::new(),
                            ));
                        blocks.push(AnthropicContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        });
                    }
                }
                if !blocks.is_empty() {
                    result.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: blocks,
                    });
                }
            }
            _ => {
                // user and system roles
                let role_str = match msg.role {
                    Role::User => "user",
                    Role::System => "user",
                    _ => "user",
                };
                let text = msg.content.clone().unwrap_or_default();
                result.push(AnthropicMessage {
                    role: role_str.to_string(),
                    content: vec![AnthropicContentBlock::Text { text }],
                });
            }
        }
    }

    result
}

fn convert_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t.function.parameters.clone(),
        })
        .collect()
}

fn convert_response(resp: AnthropicResponse) -> Choice {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in resp.content {
        match block {
            AnthropicResponseBlock::Text { text } => {
                text_parts.push(text);
            }
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_default(),
                    },
                });
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let finish_reason = match resp.stop_reason.as_deref() {
        Some("tool_use") => Some("tool_calls".to_string()),
        Some("end_turn") => Some("stop".to_string()),
        other => other.map(|s| s.to_string()),
    };

    Choice {
        message: Message {
            role: Role::Assistant,
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            tool_name: None,
        },
        finish_reason,
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: DEFAULT_MAX_TOKENS,
            messages: convert_messages(messages),
            tools: convert_tools(tools),
        };

        let endpoint = format!("{}/messages", self.api_base.trim_end_matches('/'));

        let response = self
            .client
            .post(&endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(Error::ReqwestError)?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = match response.text().await {
                Ok(t) => t,
                Err(_) => "<failed to read error body>".to_string(),
            };
            return Err(Error::ApiError(format!(
                "API request failed with status {}: {}",
                status, error_text
            )));
        }

        let anthropic_response: AnthropicResponse =
            response.json().await.map_err(Error::ReqwestError)?;

        Ok(convert_response(anthropic_response))
    }
}
