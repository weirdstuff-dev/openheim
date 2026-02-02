mod anthropic;
mod gemini;
mod openai;
mod openai_compatible;

use async_trait::async_trait;

use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice>;
}

pub use anthropic::AnthropicClient;
pub use gemini::GeminiClient;
pub use openai::OpenAiClient;
pub use openai_compatible::OpenAiCompatibleClient;
