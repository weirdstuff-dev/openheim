mod anthropic;
mod gemini;
mod openai;
mod openai_compatible;
mod retry;

use async_trait::async_trait;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

/// Abstraction over a chat-completion API.
///
/// Implement this trait to add a custom provider. The built-in implementations
/// are [`AnthropicClient`], [`GeminiClient`], [`OpenAiClient`], and
/// [`OpenAiCompatibleClient`] (for any OpenAI-compatible endpoint).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a chat request and return the first choice from the provider.
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice>;
}

pub use anthropic::AnthropicClient;
pub use gemini::GeminiClient;
pub use openai::OpenAiClient;
pub use openai_compatible::OpenAiCompatibleClient;
pub use retry::RetryClient;
