use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use tokio::sync::mpsc;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

use super::{LlmClient, LlmChunk};
use super::openai::{send_openai_style, send_openai_style_streaming};

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
