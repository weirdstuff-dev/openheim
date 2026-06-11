//! # openheim
//!
//! A fast, multi-provider LLM agent runtime written in Rust.
//!
//! ## Quick start
//!
//! ```no_run
//! use openheim::{OpenheimClient, Result};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let client = OpenheimClient::builder()
//!         .provider("openai")
//!         .api_key("sk-...")
//!         .model("gpt-4o")
//!         .build()
//!         .await?;
//!
//!     let session = client.new_session().start().await?;
//!     session.prompt("List the files in the current directory.", |_update| {}).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Providers
//!
//! | Provider   | Value         | Default model        |
//! |------------|---------------|----------------------|
//! | OpenAI     | `"openai"`    | `gpt-4o`             |
//! | Anthropic  | `"anthropic"` | `claude-sonnet-4-6`  |
//! | Google     | `"gemini"`    | `gemini-2.0-flash`   |
//! | Compatible | any string    | set via `.model()`   |
//!
//! ## Configuration file
//!
//! By default openheim loads `~/.openheim/config.toml`. Use
//! [`OpenheimClient::from_config`] to load from a custom path, or set
//! individual fields via the builder for fully programmatic configuration.
//!
//! ## MCP servers
//!
//! External tools are registered as MCP servers and namespaced as
//! `{server_name}__{tool_name}`. They are automatically available in every
//! agent session.
//!
//! ## Key types
//!
//! - [`OpenheimClient`] / [`OpenheimBuilder`] — main entry point
//! - [`SessionHandle`] — send prompts and receive streaming [`SessionUpdate`] events
//! - [`LlmClient`] — implement to add a custom provider
//! - [`RagContext`] — conversation history and skill injection
//! - [`Error`] / [`Result`] — unified error type

pub mod acp;
pub mod client;
pub mod config;
pub mod core;
pub mod error;
pub mod mcp;
pub mod rag;
pub mod subagents;
pub mod tools;
pub mod transport;
pub mod tui;

// Core types
pub use config::{AgentConfig, AppConfig, McpServerConfig, ModelsInfo};
pub use core::{agent, llm, models};
pub use error::{Error, Result};
pub use llm::{AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient};
pub use models::*;
pub use rag::{Conversation, ConversationMeta, HistoryManager, PromptBuilder, RagContext};

// Library facade
pub use client::{OpenheimBuilder, OpenheimClient, SessionBuilder, SessionHandle};

// ACP types re-exported so library users don't need a direct agent-client-protocol dependency
pub use agent_client_protocol::schema::{
    ContentBlock, ContentChunk, SessionInfo, SessionUpdate, ToolCall as AcpToolCall,
    ToolCallStatus, ToolCallUpdate,
};
