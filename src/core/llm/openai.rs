use async_trait::async_trait;
use reqwest::Client as ReqwestClient;

use crate::core::models::{ChatRequest, ChatResponse, Choice, Message, Tool};
use crate::error::{Error, Result};

use super::LlmClient;

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
}
