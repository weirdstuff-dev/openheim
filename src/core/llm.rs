use async_trait::async_trait;
use reqwest::Client as ReqwestClient;

use crate::error::{Error, Result};
use crate::core::models::{ChatRequest, ChatResponse, Choice, Message, Tool};

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice>;
}

#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
}

impl OpenAiCompatibleClient {
    pub fn new(client: ReqwestClient, api_base: String, api_key: String, model: String) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatibleClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
        };

        let endpoint = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));

        let response = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let chat_response: ChatResponse = response.json().await.map_err(Error::ReqwestError)?;

        chat_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::ApiError("No response from LLM".to_string()))
    }
}