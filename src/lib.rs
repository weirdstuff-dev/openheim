pub mod acp;
pub mod config;
pub mod transport;
pub mod tui;
pub mod core;
pub mod error;
pub mod mcp;
pub mod rag;
pub mod tools;

pub use config::{AgentConfig, AppConfig};
pub use core::{agent, llm, models};
pub use error::{Error, Result};
pub use models::*;

pub use llm::{LlmClient, OpenAiClient, OpenAiCompatibleClient, AnthropicClient, GeminiClient};
pub use rag::{RagContext, PromptBuilder};
