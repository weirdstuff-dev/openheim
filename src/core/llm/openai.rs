use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::core::models::{Choice, ContentBlock, Message, Role, Tool};
use crate::error::{Error, Result};

use super::sse::SseDecoder;
use super::{LlmChunk, LlmClient};

#[derive(Clone)]
pub struct OpenAiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
}

impl OpenAiClient {
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
            max_tokens,
        }
    }
}

// --- OpenAI request types ---

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiRequestToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// OpenAI accepts `content` as either a plain string or an array of typed
/// parts. Text-only messages stay a plain string to remain maximally
/// compatible with picky OpenAI-compatible backends; the array form is only
/// used once an image is actually present (see `convert_messages`).
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Debug, Serialize)]
struct OpenAiImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct OpenAiRequestToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAiRequestFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAiRequestFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunctionDef,
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionDef {
    name: String,
    description: String,
    parameters: Value,
}

// --- OpenAI response types ---

#[derive(Debug, Deserialize)]
struct OpenAiResponseEnvelope {
    choices: Vec<OpenAiResponseChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseToolCall {
    id: String,
    function: OpenAiResponseFunctionCall,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseFunctionCall {
    name: String,
    arguments: String,
}

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                if let Some(text) = msg.text() {
                    result.push(OpenAiMessage {
                        role: "system".to_string(),
                        content: Some(OpenAiContent::Text(text)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            Role::User => {
                let mut parts = Vec::new();
                let mut has_image = false;
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(OpenAiContentPart::Text { text: text.clone() });
                        }
                        ContentBlock::Image { data, mime_type } => {
                            has_image = true;
                            parts.push(OpenAiContentPart::ImageUrl {
                                image_url: OpenAiImageUrl {
                                    url: format!("data:{mime_type};base64,{data}"),
                                },
                            });
                        }
                        _ => {}
                    }
                }
                let content = if has_image {
                    OpenAiContent::Parts(parts)
                } else {
                    let text: String = parts
                        .into_iter()
                        .map(|p| match p {
                            OpenAiContentPart::Text { text } => text,
                            OpenAiContentPart::ImageUrl { .. } => unreachable!(),
                        })
                        .collect();
                    OpenAiContent::Text(text)
                };
                result.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Role::Assistant => {
                let text = msg.text();
                let calls = msg.tool_calls();
                let tool_calls = if calls.is_empty() {
                    None
                } else {
                    Some(
                        calls
                            .into_iter()
                            .map(|tc| OpenAiRequestToolCall {
                                id: tc.id,
                                call_type: "function".to_string(),
                                function: OpenAiRequestFunctionCall {
                                    name: tc.name,
                                    arguments: tc.arguments,
                                },
                            })
                            .collect(),
                    )
                };
                result.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: text.map(OpenAiContent::Text),
                    tool_calls,
                    tool_call_id: None,
                });
            }
            Role::Tool => {
                if let Some(tr) = msg.tool_result_block() {
                    result.push(OpenAiMessage {
                        role: "tool".to_string(),
                        content: Some(OpenAiContent::Text(tr.content)),
                        tool_calls: None,
                        tool_call_id: Some(tr.tool_call_id),
                    });
                }
            }
        }
    }

    result
}

fn convert_tools(tools: &[Tool]) -> Vec<OpenAiTool> {
    tools
        .iter()
        .map(|t| OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionDef {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters: t.function.parameters.clone(),
            },
        })
        .collect()
}

pub(super) async fn send_openai_style(
    client: &ReqwestClient,
    api_base: &str,
    api_key: &str,
    model: &str,
    max_tokens: Option<u32>,
    messages: &[Message],
    tools: &[Tool],
) -> Result<Choice> {
    let request = OpenAiRequest {
        model: model.to_string(),
        messages: convert_messages(messages),
        tools: convert_tools(tools),
        max_tokens,
    };

    let endpoint = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
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

    let envelope: OpenAiResponseEnvelope = response.json().await.map_err(Error::ReqwestError)?;

    let choice = envelope
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::ApiError("No response from LLM".to_string()))?;

    let mut content = Vec::new();
    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }
    if let Some(tcs) = choice.message.tool_calls {
        for tc in tcs {
            content.push(ContentBlock::ToolUse {
                id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments,
            });
        }
    }

    Ok(Choice {
        message: Message {
            role: Role::Assistant,
            content,
        },
        finish_reason: choice.finish_reason,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn send_openai_style_streaming(
    client: &ReqwestClient,
    api_base: &str,
    api_key: &str,
    model: &str,
    max_tokens: Option<u32>,
    messages: &[Message],
    tools: &[Tool],
    chunk_tx: mpsc::UnboundedSender<LlmChunk>,
) -> Result<Choice> {
    let request = OpenAiRequest {
        model: model.to_string(),
        messages: convert_messages(messages),
        tools: convert_tools(tools),
        max_tokens,
    };

    let mut body = serde_json::to_value(&request).map_err(|e| Error::ParseError(e.to_string()))?;
    body["stream"] = serde_json::Value::Bool(true);

    let endpoint = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let mut response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
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

    struct ToolCallAcc {
        id: String,
        name: String,
        args: String,
    }

    let mut text_buf = String::new();
    let mut tool_acc: Vec<ToolCallAcc> = Vec::new();
    let mut finish_reason: Option<String> = None;
    let mut decoder = SseDecoder::new();
    let mut done = false;

    while !done {
        let Some(bytes) = response.chunk().await.map_err(Error::ReqwestError)? else {
            break;
        };
        decoder.feed(&bytes);

        while let Some(data) = decoder.next_payload() {
            if data == "[DONE]" {
                done = true;
                break;
            }

            let Ok(event) = serde_json::from_str::<serde_json::Value>(&data) else {
                continue;
            };

            let choice = &event["choices"][0];

            if let Some(fr) = choice["finish_reason"].as_str() {
                finish_reason = Some(fr.to_string());
            }

            let delta = &choice["delta"];

            if let Some(reasoning) = delta["reasoning_content"].as_str()
                && !reasoning.is_empty()
            {
                let _ = chunk_tx.send(LlmChunk::Thinking(reasoning.to_string()));
            }

            if let Some(content) = delta["content"].as_str()
                && !content.is_empty()
            {
                text_buf.push_str(content);
                let _ = chunk_tx.send(LlmChunk::Text(content.to_string()));
            }

            if let Some(tcs) = delta["tool_calls"].as_array() {
                for tc in tcs {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    while tool_acc.len() <= idx {
                        tool_acc.push(ToolCallAcc {
                            id: String::new(),
                            name: String::new(),
                            args: String::new(),
                        });
                    }
                    if let Some(id) = tc["id"].as_str() {
                        tool_acc[idx].id = id.to_string();
                    }
                    if let Some(name) = tc["function"]["name"].as_str() {
                        tool_acc[idx].name.push_str(name);
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        tool_acc[idx].args.push_str(args);
                    }
                }
            }
        }
    }

    let mut content = Vec::new();
    if !text_buf.is_empty() {
        content.push(ContentBlock::Text { text: text_buf });
    }
    for (i, tc) in tool_acc.into_iter().enumerate() {
        if tc.name.is_empty() {
            continue;
        }
        content.push(ContentBlock::ToolUse {
            id: if tc.id.is_empty() {
                format!("call_{i}")
            } else {
                tc.id
            },
            name: tc.name,
            arguments: tc.args,
        });
    }

    Ok(Choice {
        message: Message {
            role: Role::Assistant,
            content,
        },
        finish_reason,
    })
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        send_openai_style(
            &self.client,
            &self.api_base,
            &self.api_key,
            &self.model,
            self.max_tokens,
            messages,
            tools,
        )
        .await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        send_openai_style_streaming(
            &self.client,
            &self.api_base,
            &self.api_key,
            &self.model,
            self.max_tokens,
            messages,
            tools,
            chunk_tx,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_messages_text_only_user_stays_plain_string() {
        let messages = vec![Message::user("hello")];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert!(matches!(&result[0].content, Some(OpenAiContent::Text(t)) if t == "hello"));
    }

    #[test]
    fn convert_messages_user_with_image_becomes_parts() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "what is this?".into(),
                },
                ContentBlock::Image {
                    data: "base64data".into(),
                    mime_type: "image/png".into(),
                },
            ],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        let Some(OpenAiContent::Parts(parts)) = &result[0].content else {
            panic!("expected parts");
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], OpenAiContentPart::Text { text } if text == "what is this?"));
        assert!(matches!(
            &parts[1],
            OpenAiContentPart::ImageUrl { image_url }
                if image_url.url == "data:image/png;base64,base64data"
        ));
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.txt"}"#.into(),
            }],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        let tool_calls = result[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn convert_messages_tool_result_uses_tool_role() {
        let messages = vec![Message::tool_result(
            "call_1",
            "read_file",
            "file content",
            false,
        )];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "tool");
        assert_eq!(result[0].tool_call_id.as_deref(), Some("call_1"));
        assert!(matches!(&result[0].content, Some(OpenAiContent::Text(t)) if t == "file content"));
    }

    #[test]
    fn openai_request_skips_empty_tools_and_max_tokens() {
        let req = OpenAiRequest {
            model: "gpt-4".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: None,
        };
        let json: Value = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none());
        assert!(json.get("max_tokens").is_none());
    }
}
