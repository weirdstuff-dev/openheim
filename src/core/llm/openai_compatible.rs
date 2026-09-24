use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use tokio::sync::mpsc;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

use super::openai::OpenAiClient;
use super::{LlmChunk, LlmClient};

/// Client for any endpoint speaking OpenAI's Chat Completions format
/// (Ollama, OpenRouter, Together, vLLM, …). The wire format is exactly
/// OpenAI's, so this delegates to an [`OpenAiClient`]; it's a separate type
/// so the two provider kinds stay distinguishable and can diverge later.
#[derive(Clone)]
pub struct OpenAiCompatibleClient(OpenAiClient);

impl OpenAiCompatibleClient {
    pub fn new(
        client: ReqwestClient,
        api_base: String,
        api_key: String,
        model: String,
        max_tokens: Option<u32>,
    ) -> Self {
        Self(OpenAiClient::new(
            client, api_base, api_key, model, max_tokens,
        ))
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatibleClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        self.0.send(messages, tools).await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        self.0.send_streaming(messages, tools, chunk_tx).await
    }
}
