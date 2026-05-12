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
    max_tokens: u32,
}

impl AnthropicClient {
    pub fn new(
        client: ReqwestClient,
        api_base: String,
        api_key: String,
        model: String,
        max_tokens: Option<u32>,
    ) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
            max_tokens: max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        }
    }
}

// --- Anthropic request types ---

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
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

fn convert_messages(messages: &[Message]) -> Result<Vec<AnthropicMessage>> {
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
                if let Some(text) = &msg.content
                    && !text.is_empty()
                {
                    blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: Value =
                            serde_json::from_str(&tc.function.arguments).map_err(|e| {
                                Error::ParseError(format!(
                                    "invalid JSON in tool call arguments for '{}': {}",
                                    tc.function.name, e
                                ))
                            })?;
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
            Role::User => {
                let text = msg.content.clone().unwrap_or_default();
                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![AnthropicContentBlock::Text { text }],
                });
            }
            Role::System => {
                // extracted into the top-level system field of AnthropicRequest
            }
        }
    }

    Ok(result)
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
        let system: Option<String> = {
            let parts: Vec<&str> = messages
                .iter()
                .filter(|m| m.role == Role::System)
                .filter_map(|m| m.content.as_deref())
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        };

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system,
            messages: convert_messages(messages)?,
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
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read error body>".into());
            return Err(Error::HttpError { status, body });
        }

        let anthropic_response: AnthropicResponse =
            response.json().await.map_err(Error::ReqwestError)?;

        Ok(convert_response(anthropic_response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_messages_user_message() {
        let messages = vec![Message::user("hello".into())];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert!(
            matches!(&result[0].content[0], AnthropicContentBlock::Text { text } if text == "hello")
        );
    }

    #[test]
    fn convert_messages_system_is_excluded() {
        let messages = vec![Message {
            role: Role::System,
            content: Some("system prompt".into()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: Some("thinking".into()),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            }]),
            tool_call_id: None,
            tool_name: None,
        }];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        assert_eq!(result[0].content.len(), 2); // text + tool_use
    }

    #[test]
    fn convert_messages_invalid_tool_arguments_returns_error() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: "not valid json".into(),
                },
            }]),
            tool_call_id: None,
            tool_name: None,
        }];
        assert!(convert_messages(&messages).is_err());
    }

    #[test]
    fn convert_messages_merges_consecutive_tool_results() {
        let messages = vec![
            Message::tool_result("call_1".into(), "read_file".into(), "content1".into()),
            Message::tool_result("call_2".into(), "write_file".into(), "content2".into()),
        ];
        let result = convert_messages(&messages).unwrap();
        // Both tool results should merge into a single user message
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content.len(), 2);
    }

    #[test]
    fn convert_tools_maps_definitions() {
        let tools = vec![Tool {
            tool_type: "function".into(),
            function: crate::core::models::FunctionDefinition {
                name: "read_file".into(),
                description: "Read a file".into(),
                parameters: json!({"type": "object"}),
            },
        }];
        let result = convert_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "read_file");
        assert_eq!(result[0].description, "Read a file");
    }

    #[test]
    fn convert_tools_empty_input() {
        let result = convert_tools(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn convert_response_text_only() {
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text {
                text: "Hello!".into(),
            }],
            stop_reason: Some("end_turn".into()),
        };
        let choice = convert_response(resp);
        assert_eq!(choice.message.content.as_deref(), Some("Hello!"));
        assert!(choice.message.tool_calls.is_none());
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn convert_response_tool_use() {
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                input: json!({"path": "a.txt"}),
            }],
            stop_reason: Some("tool_use".into()),
        };
        let choice = convert_response(resp);
        assert!(choice.message.content.is_none());
        let tool_calls = choice.message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "read_file");
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn convert_response_mixed_text_and_tool() {
        let resp = AnthropicResponse {
            content: vec![
                AnthropicResponseBlock::Text {
                    text: "Let me read that.".into(),
                },
                AnthropicResponseBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "test.txt"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
        };
        let choice = convert_response(resp);
        assert_eq!(choice.message.content.as_deref(), Some("Let me read that."));
        assert_eq!(choice.message.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn convert_response_stop_reason_mapping() {
        // end_turn -> stop
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text {
                text: "done".into(),
            }],
            stop_reason: Some("end_turn".into()),
        };
        assert_eq!(
            convert_response(resp).finish_reason.as_deref(),
            Some("stop")
        );

        // tool_use -> tool_calls
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text { text: "x".into() }],
            stop_reason: Some("tool_use".into()),
        };
        assert_eq!(
            convert_response(resp).finish_reason.as_deref(),
            Some("tool_calls")
        );

        // max_tokens passes through
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text { text: "x".into() }],
            stop_reason: Some("max_tokens".into()),
        };
        assert_eq!(
            convert_response(resp).finish_reason.as_deref(),
            Some("max_tokens")
        );

        // None stays None
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text { text: "x".into() }],
            stop_reason: None,
        };
        assert!(convert_response(resp).finish_reason.is_none());
    }
}
