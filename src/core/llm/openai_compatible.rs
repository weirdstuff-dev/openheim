use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use tokio::sync::mpsc;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

use super::openai::{OpenAiClient, TokenLimitField};
use super::{LlmChunk, LlmClient};

/// Client for any endpoint speaking OpenAI's Chat Completions format
/// (Ollama, OpenRouter, Together, vLLM, …). Delegates to an
/// [`OpenAiClient`], except that the output limit goes out as `max_tokens`:
/// OpenAI itself moved to `max_completion_tokens`, but most compatible
/// backends only know the old name.
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
        Self(
            OpenAiClient::new(client, api_base, api_key, model, max_tokens)
                .with_token_limit_field(TokenLimitField::MaxTokens),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_backends_get_max_tokens() {
        let client = OpenAiCompatibleClient::new(
            ReqwestClient::new(),
            "http://localhost:11434/v1".into(),
            String::new(),
            "llama3".into(),
            Some(1000),
        );
        assert_eq!(client.0.token_limit_field(), TokenLimitField::MaxTokens);
    }
}
