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
//! ## Feature flags
//!
//! | Feature  | Default    | Enables                                              |
//! |----------|------------|------------------------------------------------------|
//! | `cli`    | ✓          | The `openheim` binary (CLI, TUI, `serve`). Implies `tui` + `server`. |
//! | `tui`    | via `cli`  | The `tui` module (ratatui/crossterm terminal UI).    |
//! | `server` | via `cli`  | The `transport::ws` WebSocket/REST server (axum).    |
//!
//! Everything else — the client facade, agent loop, providers, tools, MCP,
//! ACP, and config — is always available. Embedders that don't need the
//! terminal UI or the built-in server should depend on openheim with
//! `default-features = false` (optionally adding `"tui"` or `"server"` back)
//! to skip the `clap`, `ratatui`, `crossterm`, `axum`, `tower-http`, `notify`,
//! `futures`, `walkdir`, and `tracing-subscriber` dependency trees.
//!
//! ## Key types
//!
//! - [`OpenheimClient`] / [`OpenheimBuilder`] — main entry point
//! - [`SessionHandle`] — send prompts and receive streaming [`SessionUpdate`] events
//! - [`LlmClient`] — implement to add a custom provider
//! - [`MemoryContext`] — conversation history, skills, and system identity
//! - [`Error`] / [`Result`] — unified error type

pub mod acp;
pub mod client;
pub mod config;
pub mod core;
pub mod error;
pub mod mcp;
pub mod memory;
pub mod subagents;
pub mod tools;
pub mod transport;
#[cfg(feature = "tui")]
pub mod tui;

// Core types
pub use config::{AgentConfig, AppConfig, McpServerConfig, ModelsInfo};
pub use core::{agent, llm, models};
pub use error::{Error, Result};
pub use llm::{AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient};
// `ContentBlock` is deliberately omitted here: `agent_client_protocol::schema::ContentBlock`
// below is the one library users see at the crate root. Reach the core content-block
// type via `openheim::core::models::ContentBlock` if needed.
pub use memory::{Conversation, ConversationMeta, HistoryManager, MemoryContext, PromptBuilder};
pub use models::{
    AgentResult, AgentStep, Choice, FinishReason, FunctionDefinition, Message, Role, StopReason,
    StreamEvent, Tool, ToolExecutionResult, ToolResultBlock, ToolUseBlock,
};

// Library facade
pub use client::{OpenheimBuilder, OpenheimClient, SessionBuilder, SessionHandle};

// ACP types re-exported so library users don't need a direct agent-client-protocol dependency
pub use agent_client_protocol::schema::{
    ContentBlock, ContentChunk, SessionInfo, SessionUpdate, ToolCall as AcpToolCall,
    ToolCallStatus, ToolCallUpdate,
};
