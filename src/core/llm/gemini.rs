use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::core::models::{Choice, ContentBlock, FinishReason, Message, Role, Tool, Usage};
use crate::error::{Error, Result};

use super::sse::SseDecoder;
use super::{LlmChunk, LlmClient};

#[derive(Clone)]
pub struct GeminiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
}

impl GeminiClient {
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

// --- Gemini request types ---

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiToolDeclaration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    max_output_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolDeclaration {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: Value,
}

// --- Gemini response types ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    cached_content_token_count: u64,
}

impl From<GeminiUsageMetadata> for Usage {
    fn from(u: GeminiUsageMetadata) -> Self {
        Usage {
            // Gemini's `promptTokenCount` already includes cached tokens; see
            // the same normalization in the OpenAI client.
            input_tokens: u
                .prompt_token_count
                .saturating_sub(u.cached_content_token_count),
            output_tokens: u.candidates_token_count,
            cache_creation_tokens: 0,
            cache_read_tokens: u.cached_content_token_count,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiContent,
    finish_reason: Option<String>,
}

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Result<Vec<GeminiContent>> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::Assistant => {
                let mut parts = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } if !text.is_empty() => {
                            parts.push(GeminiPart {
                                text: Some(text.clone()),
                                ..Default::default()
                            });
                        }
                        ContentBlock::ToolUse {
                            name, arguments, ..
                        } => {
                            let args: Value = serde_json::from_str(arguments).map_err(|e| {
                                Error::ParseError(format!(
                                    "invalid JSON in tool call arguments for '{name}': {e}"
                                ))
                            })?;
                            parts.push(GeminiPart {
                                function_call: Some(GeminiFunctionCall {
                                    name: name.clone(),
                                    args,
                                }),
                                ..Default::default()
                            });
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    result.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            Role::Tool => {
                // Tool results become functionResponse parts in a user turn.
                // Gemini matches by function name, not by call id.
                let Some(tr) = msg.tool_result_block() else {
                    continue;
                };
                let part = GeminiPart {
                    function_response: Some(GeminiFunctionResponse {
                        name: tr.tool_name,
                        response: serde_json::json!({ "result": tr.content }),
                    }),
                    ..Default::default()
                };
                // Merge into last user/function-response content if possible
                if let Some(last) = result.last_mut() {
                    let last: &mut GeminiContent = last;
                    if last.role == "user" {
                        last.parts.push(part);
                        continue;
                    }
                }
                result.push(GeminiContent {
                    role: "user".to_string(),
                    parts: vec![part],
                });
            }
            Role::User => {
                let mut parts = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(GeminiPart {
                                text: Some(text.clone()),
                                ..Default::default()
                            });
                        }
                        ContentBlock::Image { data, mime_type } => {
                            parts.push(GeminiPart {
                                inline_data: Some(GeminiInlineData {
                                    mime_type: mime_type.clone(),
                                    data: data.clone(),
                                }),
                                ..Default::default()
                            });
                        }
                        _ => {}
                    }
                }
                if parts.is_empty() {
                    parts.push(GeminiPart {
                        text: Some(String::new()),
                        ..Default::default()
                    });
                }
                result.push(GeminiContent {
                    role: "user".to_string(),
                    parts,
                });
            }
            Role::System => {
                // extracted into system_instruction field of GeminiRequest
            }
        }
    }

    Ok(result)
}

fn convert_tools(tools: &[Tool]) -> Vec<GeminiToolDeclaration> {
    if tools.is_empty() {
        return Vec::new();
    }

    let declarations = tools
        .iter()
        .map(|t| GeminiFunctionDeclaration {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
        })
        .collect();

    vec![GeminiToolDeclaration {
        function_declarations: declarations,
    }]
}

fn convert_response(resp: GeminiResponse) -> Result<Choice> {
    let usage = resp.usage_metadata.map(Usage::from);
    let candidate = resp
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| Error::ApiError("No candidates in Gemini response".to_string()))?;

    let mut content = Vec::new();
    let mut tool_call_count = 0usize;

    for part in candidate.content.parts {
        if let Some(text) = part.text {
            content.push(ContentBlock::Text { text });
        }
        if let Some(fc) = part.function_call {
            content.push(ContentBlock::ToolUse {
                id: format!("call_{tool_call_count}"),
                name: fc.name,
                arguments: serde_json::to_string(&fc.args)?,
            });
            tool_call_count += 1;
        }
    }

    let finish_reason = candidate.finish_reason.as_deref().map(map_finish_reason);

    Ok(Choice {
        message: Message {
            role: Role::Assistant,
            content,
        },
        finish_reason,
        usage,
    })
}

/// Maps Gemini's `finishReason` vocabulary onto the provider-agnostic
/// [`FinishReason`]; anything without a known equivalent passes through as
/// [`FinishReason::Other`] (lowercased, matching this function's prior
/// string-based behavior).
fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::MaxTokens,
        other => FinishReason::Other(other.to_lowercase()),
    }
}

fn gemini_system_instruction(messages: &[Message]) -> Option<GeminiContent> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .filter_map(|m| m.text())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart {
                text: Some(parts.join("\n\n")),
                ..Default::default()
            }],
        })
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let system_instruction = gemini_system_instruction(messages);

        let request = GeminiRequest {
            contents: convert_messages(messages)?,
            tools: convert_tools(tools),
            generation_config: self.max_tokens.map(|t| GeminiGenerationConfig {
                max_output_tokens: t,
            }),
            system_instruction,
        };

        let endpoint = format!(
            "{}/models/{}:generateContent",
            self.api_base.trim_end_matches('/'),
            self.model
        );

        let response = self
            .client
            .post(&endpoint)
            // Key in header, not query: reqwest embeds the full URL (query
            // included) in transport error strings, which would leak the key
            // into logs on any timeout/connect failure.
            .header("x-goog-api-key", &self.api_key)
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

        let gemini_response: GeminiResponse = response.json().await.map_err(Error::ReqwestError)?;

        convert_response(gemini_response)
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let request = GeminiRequest {
            contents: convert_messages(messages)?,
            tools: convert_tools(tools),
            generation_config: self.max_tokens.map(|t| GeminiGenerationConfig {
                max_output_tokens: t,
            }),
            system_instruction: gemini_system_instruction(messages),
        };

        let endpoint = format!(
            "{}/models/{}:streamGenerateContent",
            self.api_base.trim_end_matches('/'),
            self.model
        );

        let mut response = self
            .client
            .post(&endpoint)
            .query(&[("alt", "sse")])
            .header("x-goog-api-key", &self.api_key)
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

        let mut text_buf = String::new();
        let mut tool_calls_accum: Vec<(String, String)> = Vec::new();
        let mut finish_reason: Option<FinishReason> = None;
        let mut usage: Option<Usage> = None;
        let mut decoder = SseDecoder::new();

        while let Some(bytes) = response.chunk().await.map_err(Error::ReqwestError)? {
            decoder.feed(&bytes);

            while let Some(data) = decoder.next_payload() {
                let Ok(event) = serde_json::from_str::<GeminiResponse>(&data) else {
                    continue;
                };

                // Each chunk's `usageMetadata` is cumulative, not a delta, so
                // the last one seen before the stream ends is the true total.
                if let Some(u) = event.usage_metadata {
                    usage = Some(Usage::from(u));
                }

                let Some(candidate) = event.candidates.into_iter().next() else {
                    continue;
                };

                if let Some(fr) = candidate.finish_reason {
                    finish_reason = Some(map_finish_reason(&fr));
                }

                for part in candidate.content.parts {
                    if let Some(text) = part.text
                        && !text.is_empty()
                    {
                        text_buf.push_str(&text);
                        let _ = chunk_tx.send(LlmChunk::Text(text));
                    }
                    if let Some(fc) = part.function_call {
                        tool_calls_accum.push((fc.name, serde_json::to_string(&fc.args)?));
                    }
                }
            }
        }

        let mut content = Vec::new();
        if !text_buf.is_empty() {
            content.push(ContentBlock::Text { text: text_buf });
        }
        for (i, (name, arguments)) in tool_calls_accum.into_iter().enumerate() {
            content.push(ContentBlock::ToolUse {
                id: format!("call_{i}"),
                name,
                arguments,
            });
        }

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content,
            },
            finish_reason,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_messages_user() {
        let messages = vec![Message::user("hello")];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn convert_messages_user_with_image() {
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
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].parts.len(), 2);
        let inline = result[0].parts[1].inline_data.as_ref().unwrap();
        assert_eq!(inline.mime_type, "image/png");
        assert_eq!(inline.data, "base64data");
    }

    #[test]
    fn convert_messages_assistant_becomes_model() {
        let messages = vec![Message::assistant("response")];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "model");
        assert_eq!(result[0].parts[0].text.as_deref(), Some("response"));
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
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "model");
        assert!(result[0].parts[0].function_call.is_some());
        assert_eq!(
            result[0].parts[0].function_call.as_ref().unwrap().name,
            "read_file"
        );
    }

    #[test]
    fn convert_messages_invalid_tool_arguments_returns_error() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "not valid json".into(),
            }],
        }];
        assert!(convert_messages(&messages).is_err());
    }

    #[test]
    fn convert_messages_tool_result_as_function_response() {
        let messages = vec![Message::tool_result(
            "call_1",
            "read_file",
            "file content",
            false,
        )];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        let fr = result[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "read_file");
    }

    #[test]
    fn convert_messages_merges_tool_results_into_user() {
        let messages = vec![
            Message::tool_result("call_1", "read_file", "a", false),
            Message::tool_result("call_2", "write_file", "b", false),
        ];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].parts.len(), 2);
    }

    #[test]
    fn convert_tools_wraps_in_declaration() {
        let tools = vec![Tool {
            tool_type: "function".into(),
            function: crate::core::models::FunctionDefinition {
                name: "test_tool".into(),
                description: "A test tool".into(),
                parameters: json!({"type": "object"}),
            },
        }];
        let result = convert_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function_declarations.len(), 1);
        assert_eq!(result[0].function_declarations[0].name, "test_tool");
    }

    #[test]
    fn convert_tools_empty_returns_empty() {
        let result = convert_tools(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn convert_response_text_only() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart {
                        text: Some("Hello!".into()),
                        ..Default::default()
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
        };
        let choice = convert_response(resp).unwrap();
        assert_eq!(choice.message.text().as_deref(), Some("Hello!"));
        assert!(choice.message.tool_calls().is_empty());
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn convert_response_function_call() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart {
                        function_call: Some(GeminiFunctionCall {
                            name: "read_file".into(),
                            args: json!({"path": "test.txt"}),
                        }),
                        ..Default::default()
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
        };
        let choice = convert_response(resp).unwrap();
        assert!(choice.message.text().is_none());
        let tool_calls = choice.message.tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].id, "call_0");
    }

    #[test]
    fn convert_response_finish_reason_mapping() {
        // STOP -> Stop
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart {
                        text: Some("x".into()),
                        ..Default::default()
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
        };
        assert_eq!(
            convert_response(resp).unwrap().finish_reason,
            Some(FinishReason::Stop)
        );

        // MAX_TOKENS -> MaxTokens
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart {
                        text: Some("x".into()),
                        ..Default::default()
                    }],
                },
                finish_reason: Some("MAX_TOKENS".into()),
            }],
            usage_metadata: None,
        };
        assert_eq!(
            convert_response(resp).unwrap().finish_reason,
            Some(FinishReason::MaxTokens)
        );
    }

    #[test]
    fn convert_response_errors_on_empty_candidates() {
        let resp = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
        };
        let result = convert_response(resp);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No candidates"));
    }
}
