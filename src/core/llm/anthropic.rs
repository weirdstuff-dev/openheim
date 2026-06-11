use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::core::models::{Choice, FunctionCall, Message, Role, Tool, ToolCall};
use crate::error::{Error, Result};

use super::sse::SseDecoder;
use super::{LlmChunk, LlmClient};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

fn is_false(b: &bool) -> bool {
    !b
}

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
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct AnthropicThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: &'static str,
    budget_tokens: u32,
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
        #[serde(skip_serializing_if = "is_false")]
        is_error: bool,
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
                    is_error: msg.is_error,
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

fn convert_response(resp: AnthropicResponse) -> Result<Choice> {
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
                        arguments: serde_json::to_string(&input)?,
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

    Ok(Choice {
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
            is_error: false,
        },
        finish_reason,
    })
}

fn extract_system(messages: &[Message]) -> Option<String> {
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
}

/// Returns a thinking config if the model supports extended thinking and
/// `max_tokens` is large enough to accommodate a reasonable budget.
fn thinking_config(model: &str, max_tokens: u32) -> Option<AnthropicThinkingConfig> {
    let supported =
        model.contains("claude-3-7") || (model.starts_with("claude-") && model.contains("-4-"));
    if !supported || max_tokens < 2048 {
        return None;
    }
    // Leave at least 1024 tokens for the actual response.
    let budget = 5000u32.min(max_tokens - 1024);
    Some(AnthropicThinkingConfig {
        thinking_type: "enabled",
        budget_tokens: budget,
    })
}

/// Returns the interleaved-thinking beta header value for models that need it.
fn thinking_beta(model: &str) -> Option<&'static str> {
    if model.contains("claude-3-7") {
        Some("interleaved-thinking-2025-05-14")
    } else {
        None
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            stream: false,
            system: extract_system(messages),
            messages: convert_messages(messages)?,
            tools: convert_tools(tools),
            thinking: None,
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

        convert_response(anthropic_response)
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let thinking = thinking_config(&self.model, self.max_tokens);

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            stream: true,
            system: extract_system(messages),
            messages: convert_messages(messages)?,
            tools: convert_tools(tools),
            thinking,
        };

        let endpoint = format!("{}/messages", self.api_base.trim_end_matches('/'));

        let mut req = self
            .client
            .post(&endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json");

        if let Some(beta) = thinking_beta(&self.model) {
            req = req.header("anthropic-beta", beta);
        }

        let response = req
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

        // Parse SSE stream.
        let mut decoder = SseDecoder::new();
        let mut current_block_type: Option<String> = None;
        let mut current_tool_id: Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_json = String::new();
        let mut text_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut stop_reason: Option<String> = None;

        let mut response = response;
        while let Some(chunk) = response.chunk().await.map_err(Error::ReqwestError)? {
            decoder.feed(&chunk);

            while let Some(data) = decoder.next_payload() {
                if data == "[DONE]" || data.is_empty() {
                    continue;
                }

                let event: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                match event["type"].as_str().unwrap_or("") {
                    "content_block_start" => {
                        let block_type = event["content_block"]["type"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        if block_type == "tool_use" {
                            current_tool_id =
                                event["content_block"]["id"].as_str().map(String::from);
                            current_tool_name =
                                event["content_block"]["name"].as_str().map(String::from);
                            current_tool_json.clear();
                        }
                        current_block_type = Some(block_type);
                    }
                    "content_block_delta" => {
                        let delta = &event["delta"];
                        match delta["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                if let Some(text) = delta["text"].as_str() {
                                    text_content.push_str(text);
                                    let _ = chunk_tx.send(LlmChunk::Text(text.to_string()));
                                }
                            }
                            "thinking_delta" => {
                                if let Some(thinking) = delta["thinking"].as_str() {
                                    let _ = chunk_tx.send(LlmChunk::Thinking(thinking.to_string()));
                                }
                            }
                            "input_json_delta" => {
                                if let Some(partial) = delta["partial_json"].as_str() {
                                    current_tool_json.push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        if current_block_type.as_deref() == Some("tool_use") {
                            if let (Some(id), Some(name)) =
                                (current_tool_id.take(), current_tool_name.take())
                            {
                                tool_calls.push(ToolCall {
                                    id,
                                    call_type: "function".to_string(),
                                    function: FunctionCall {
                                        name,
                                        arguments: current_tool_json.clone(),
                                    },
                                });
                            }
                            current_tool_json.clear();
                        }
                        current_block_type = None;
                    }
                    "message_delta" => {
                        if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                            stop_reason = Some(reason.to_string());
                        }
                    }
                    "error" => {
                        let msg = event["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown streaming error")
                            .to_string();
                        return Err(Error::Other(format!("Anthropic streaming error: {msg}")));
                    }
                    _ => {}
                }
            }
        }

        let finish_reason = match stop_reason.as_deref() {
            Some("tool_use") => Some("tool_calls".to_string()),
            Some("end_turn") => Some("stop".to_string()),
            other => other.map(|s| s.to_string()),
        };

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content: if text_content.is_empty() {
                    None
                } else {
                    Some(text_content)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
                tool_name: None,
                is_error: false,
            },
            finish_reason,
        })
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
            is_error: false,
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
            is_error: false,
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
            is_error: false,
        }];
        assert!(convert_messages(&messages).is_err());
    }

    #[test]
    fn convert_messages_merges_consecutive_tool_results() {
        let messages = vec![
            Message::tool_result(
                "call_1".into(),
                "read_file".into(),
                "content1".into(),
                false,
            ),
            Message::tool_result(
                "call_2".into(),
                "write_file".into(),
                "content2".into(),
                false,
            ),
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
        let choice = convert_response(resp).unwrap();
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
        let choice = convert_response(resp).unwrap();
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
        let choice = convert_response(resp).unwrap();
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
            convert_response(resp).unwrap().finish_reason.as_deref(),
            Some("stop")
        );

        // tool_use -> tool_calls
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text { text: "x".into() }],
            stop_reason: Some("tool_use".into()),
        };
        assert_eq!(
            convert_response(resp).unwrap().finish_reason.as_deref(),
            Some("tool_calls")
        );

        // max_tokens passes through
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text { text: "x".into() }],
            stop_reason: Some("max_tokens".into()),
        };
        assert_eq!(
            convert_response(resp).unwrap().finish_reason.as_deref(),
            Some("max_tokens")
        );

        // None stays None
        let resp = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text { text: "x".into() }],
            stop_reason: None,
        };
        assert!(convert_response(resp).unwrap().finish_reason.is_none());
    }
}
