pub mod acp;
pub mod client;
pub mod config;
pub mod core;
pub mod error;
pub mod mcp;
pub mod rag;
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
