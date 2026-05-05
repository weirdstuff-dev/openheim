use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::models::{Choice, FunctionCall, Message, Role, Tool, ToolCall};
use crate::error::{Error, Result};

use super::LlmClient;

#[derive(Clone)]
pub struct GeminiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
}

impl GeminiClient {
    pub fn new(client: ReqwestClient, api_base: String, api_key: String, model: String, max_tokens: Option<u32>) -> Self {
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
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
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
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
                if let Some(text) = &msg.content {
                    if !text.is_empty() {
                        parts.push(GeminiPart {
                            text: Some(text.clone()),
                            function_call: None,
                            function_response: None,
                        });
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let args: Value = serde_json::from_str(&tc.function.arguments)
                            .map_err(|e| Error::ParseError(format!(
                                "invalid JSON in tool call arguments for '{}': {}",
                                tc.function.name, e
                            )))?;
                        parts.push(GeminiPart {
                            text: None,
                            function_call: Some(GeminiFunctionCall {
                                name: tc.function.name.clone(),
                                args,
                            }),
                            function_response: None,
                        });
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
                let name = msg.tool_name.clone()
                    .or_else(|| msg.tool_call_id.clone())
                    .unwrap_or_default();
                let part = GeminiPart {
                    text: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse {
                        name,
                        response: serde_json::json!({
                            "result": msg.content.clone().unwrap_or_default()
                        }),
                    }),
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
                let text = msg.content.clone().unwrap_or_default();
                result.push(GeminiContent {
                    role: "user".to_string(),
                    parts: vec![GeminiPart {
                        text: Some(text),
                        function_call: None,
                        function_response: None,
                    }],
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
    let candidate = resp
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| Error::ApiError("No candidates in Gemini response".to_string()))?;

    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for part in candidate.content.parts {
        if let Some(text) = part.text {
            text_parts.push(text);
        }
        if let Some(fc) = part.function_call {
            tool_calls.push(ToolCall {
                id: format!("call_{}", tool_calls.len()),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: fc.name,
                    arguments: serde_json::to_string(&fc.args).unwrap_or_default(),
                },
            });
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let finish_reason = match candidate.finish_reason.as_deref() {
        Some("STOP") => Some("stop".to_string()),
        Some("MAX_TOKENS") => Some("length".to_string()),
        other => other.map(|s| s.to_lowercase()),
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
        },
        finish_reason,
    })
}

#[async_trait]
impl LlmClient for GeminiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let system_instruction: Option<GeminiContent> = {
            let parts: Vec<&str> = messages
                .iter()
                .filter(|m| m.role == Role::System)
                .filter_map(|m| m.content.as_deref())
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(GeminiContent {
                    role: "user".to_string(),
                    parts: vec![GeminiPart {
                        text: Some(parts.join("\n\n")),
                        function_call: None,
                        function_response: None,
                    }],
                })
            }
        };

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
            .query(&[("key", &self.api_key)])
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(Error::ReqwestError)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_else(|_| "<failed to read error body>".into());
            return Err(Error::HttpError { status, body });
        }

        let gemini_response: GeminiResponse =
            response.json().await.map_err(Error::ReqwestError)?;

        convert_response(gemini_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_messages_user() {
        let messages = vec![Message::user("hello".into())];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn convert_messages_assistant_becomes_model() {
        let messages = vec![Message::assistant("response".into())];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "model");
        assert_eq!(result[0].parts[0].text.as_deref(), Some("response"));
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: None,
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
        assert_eq!(result[0].role, "model");
        assert!(result[0].parts[0].function_call.is_some());
        assert_eq!(result[0].parts[0].function_call.as_ref().unwrap().name, "read_file");
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
    fn convert_messages_tool_result_as_function_response() {
        let messages = vec![Message::tool_result(
            "call_1".into(),
            "read_file".into(),
            "file content".into(),
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
            Message::tool_result("call_1".into(), "read_file".into(), "a".into()),
            Message::tool_result("call_2".into(), "write_file".into(), "b".into()),
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
                        function_call: None,
                        function_response: None,
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
        };
        let choice = convert_response(resp).unwrap();
        assert_eq!(choice.message.content.as_deref(), Some("Hello!"));
        assert!(choice.message.tool_calls.is_none());
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn convert_response_function_call() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart {
                        text: None,
                        function_call: Some(GeminiFunctionCall {
                            name: "read_file".into(),
                            args: json!({"path": "test.txt"}),
                        }),
                        function_response: None,
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
        };
        let choice = convert_response(resp).unwrap();
        assert!(choice.message.content.is_none());
        let tool_calls = choice.message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
        assert_eq!(tool_calls[0].id, "call_0");
    }

    #[test]
    fn convert_response_finish_reason_mapping() {
        // STOP -> stop
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent { role: "model".into(), parts: vec![GeminiPart { text: Some("x".into()), function_call: None, function_response: None }] },
                finish_reason: Some("STOP".into()),
            }],
        };
        assert_eq!(convert_response(resp).unwrap().finish_reason.as_deref(), Some("stop"));

        // MAX_TOKENS -> length
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent { role: "model".into(), parts: vec![GeminiPart { text: Some("x".into()), function_call: None, function_response: None }] },
                finish_reason: Some("MAX_TOKENS".into()),
            }],
        };
        assert_eq!(convert_response(resp).unwrap().finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn convert_response_errors_on_empty_candidates() {
        let resp = GeminiResponse { candidates: vec![] };
        let result = convert_response(resp);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No candidates"));
    }
}
