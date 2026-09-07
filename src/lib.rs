//! # openheim
//!
//! A fast, multi-provider LLM agent runtime written in Rust.
//!
//! ## Quick start
//!
//! ```no_run
//! use openheim::{OpenheimClient, Result};
//!
//! // `current_thread`: the library itself only needs `tokio`'s `rt` feature,
//! // not `rt-multi-thread` (that's what the `cli` feature adds for the
//! // `openheim` binary's own `#[tokio::main]`).
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> Result<()> {
//!     let client = OpenheimClient::builder()
//!         .provider("openai")
//!         .api_key("sk-...")
//!         .model("gpt-4o")
//!         .build()
//!         .await?;
//!
//!     let session = client.new_session().start().await?;
//!     session
//!         .prompt_events("List the files in the current directory.", |_event| {})
//!         .await?;
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
//! | `cli`    | ✓          | The `openheim` binary (CLI, TUI, `serve`). Implies `tui` + `server` + `acp`. |
//! | `tui`    | via `cli`  | The `tui` module (ratatui/crossterm terminal UI). Doesn't need `acp`. |
//! | `acp`    | via `cli`  | The `acp` and `transport` modules (Agent Client Protocol: `serve`, `stdio`, `run`, `ws`). |
//! | `server` | via `cli`  | The `transport::ws` WebSocket/REST server (axum). Implies `acp`. |
//! | `rag`    | via `cli`  | The `rag` module and `remember`/`search_memory`/`forget` tools (rusqlite with FTS5 + sqlite-vec). |
//!
//! Everything else — the client facade, agent loop, providers, tools, MCP,
//! and config — is always available. Embedders that don't need ACP, the
//! terminal UI, or the built-in server should depend on openheim with
//! `default-features = false` (optionally adding back `"acp"`, `"tui"`, or
//! `"server"`) to skip the `clap`, `ratatui`, `crossterm`, `axum`,
//! `tower-http`, `notify`, `walkdir`, `tracing-subscriber`, and
//! `agent-client-protocol{,-tokio}` dependency trees. `futures` is not
//! behind any feature — the agent loop uses it directly.
//!
//! ## Key types
//!
//! - [`OpenheimClient`] / [`OpenheimBuilder`] — main entry point
//! - [`SessionHandle`] — send prompts and receive streaming events (`StreamEvent`, or ACP's `SessionUpdate` with feature `acp`)
//! - [`LlmClient`] — implement to add a custom provider
//! - [`MemoryContext`] — conversation history, skills, and system identity
//! - [`rag::LongTermMemory`] — tool-driven long-term memory: FTS5 keyword search, optionally sqlite-vec semantic search (feature `rag`)
//! - [`Error`] / [`Result`] — unified error type

#[cfg(feature = "acp")]
pub mod acp;
pub mod client;
pub mod config;
pub mod core;
pub mod error;
pub mod mcp;
pub mod memory;
#[cfg(feature = "rag")]
pub mod rag;
pub mod subagents;
pub mod tools;
#[cfg(feature = "acp")]
pub mod transport;
#[cfg(feature = "tui")]
pub mod tui;

// Core types
pub use config::{AgentConfig, AppConfig, McpServerConfig, ModelsInfo};
pub use core::{agent, llm, models};
pub use error::{Error, Result};
pub use llm::{AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient};
pub use memory::{Conversation, ConversationMeta, HistoryManager, MemoryContext, PromptBuilder};
pub use models::{
    AgentResult, Choice, ContentBlock, FinishReason, FunctionDefinition, Message, Role, StopReason,
    StreamEvent, Tool, ToolResultBlock, ToolUseBlock,
};
#[cfg(feature = "rag")]
pub use rag::LongTermMemory;

// Library facade
pub use client::{OpenheimBuilder, OpenheimClient, SessionBuilder, SessionHandle};

// ACP's own vocabulary (`SessionUpdate`, `ContentBlock`, …) is reached via
// `openheim::acp::schema` (feature `acp`), not re-exported at the crate
// root — the root `ContentBlock` above is `core::models::ContentBlock`.
