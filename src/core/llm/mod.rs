mod anthropic;
mod gemini;
mod openai;
mod openai_compatible;
mod retry;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

/// A single streaming chunk produced during an LLM call.
#[derive(Debug)]
pub enum LlmChunk {
    /// A token or partial text from the model's response.
    Text(String),
    /// A chunk from the model's extended thinking (reasoning).
    Thinking(String),
}

/// Abstraction over a chat-completion API.
///
/// Implement this trait to add a custom provider. The built-in implementations
/// are [`AnthropicClient`], [`GeminiClient`], [`OpenAiClient`], and
/// [`OpenAiCompatibleClient`] (for any OpenAI-compatible endpoint).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a chat request and return the first choice from the provider.
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice>;

    /// Streaming variant: sends [`LlmChunk`]s to `chunk_tx` as they arrive,
    /// then returns the complete [`Choice`] once the response is finished.
    ///
    /// The default implementation calls [`LlmClient::send`] and emits the full
    /// response content as a single [`LlmChunk::Text`]. Override this to enable
    /// real token-by-token streaming.
    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let choice = self.send(messages, tools).await?;
        if let Some(ref content) = choice.message.content {
            let _ = chunk_tx.send(LlmChunk::Text(content.clone()));
        }
        Ok(choice)
    }
}

pub use anthropic::AnthropicClient;
pub use gemini::GeminiClient;
pub use openai::OpenAiClient;
pub use openai_compatible::OpenAiCompatibleClient;
pub use retry::RetryClient;
