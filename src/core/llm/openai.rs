use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use tokio::sync::mpsc;

use crate::core::models::{
    ChatRequest, ChatResponse, Choice, FunctionCall, Message, Role, Tool, ToolCall,
};
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

pub(super) async fn send_openai_style(
    client: &ReqwestClient,
    api_base: &str,
    api_key: &str,
    model: &str,
    max_tokens: Option<u32>,
    messages: &[Message],
    tools: &[Tool],
) -> Result<Choice> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        tools: tools.to_vec(),
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

    let chat_response: ChatResponse = response.json().await.map_err(Error::ReqwestError)?;

    chat_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::ApiError("No response from LLM".to_string()))
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
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        tools: tools.to_vec(),
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

    let content = if text_buf.is_empty() {
        None
    } else {
        Some(text_buf)
    };
    let tool_calls: Vec<ToolCall> = tool_acc
        .into_iter()
        .enumerate()
        .filter(|(_, tc)| !tc.name.is_empty())
        .map(|(i, tc)| ToolCall {
            id: if tc.id.is_empty() {
                format!("call_{i}")
            } else {
                tc.id
            },
            call_type: "function".to_string(),
            function: FunctionCall {
                name: tc.name,
                arguments: tc.args,
            },
        })
        .collect();

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
