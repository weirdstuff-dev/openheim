use async_trait::async_trait;
use reqwest::Client as ReqwestClient;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

use super::LlmClient;
use super::openai::send_openai_style;

#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
}

impl OpenAiCompatibleClient {
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

#[async_trait]
impl LlmClient for OpenAiCompatibleClient {
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
