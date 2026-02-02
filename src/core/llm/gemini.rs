use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::models::{Choice, FunctionCall, Message, Tool, ToolCall};
use crate::error::{Error, Result};

use super::LlmClient;

#[derive(Clone)]
pub struct GeminiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
}

impl GeminiClient {
    pub fn new(client: ReqwestClient, api_base: String, api_key: String, model: String) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
        }
    }
}

// --- Gemini request types ---

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiToolDeclaration>,
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

fn convert_messages(messages: &[Message]) -> Vec<GeminiContent> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "assistant" => {
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
                            .unwrap_or(Value::Object(serde_json::Map::new()));
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
            "tool" => {
                // Tool results become functionResponse parts in a user turn.
                // We need the tool name — but our Message only has tool_call_id, not name.
                // We use the tool_call_id as a fallback for the name field; the agent
                // loop should ideally provide the name. Gemini matches by name, not id.
                let part = GeminiPart {
                    text: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse {
                        name: msg.tool_call_id.clone().unwrap_or_default(),
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
            _ => {
                // user and any other role
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
        }
    }

    result
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
            role: "assistant".to_string(),
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        },
        finish_reason,
    })
}

#[async_trait]
impl LlmClient for GeminiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let request = GeminiRequest {
            contents: convert_messages(messages),
            tools: convert_tools(tools),
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

        let gemini_response: GeminiResponse =
            response.json().await.map_err(Error::ReqwestError)?;

        convert_response(gemini_response)
    }
}
