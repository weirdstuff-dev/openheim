//! ACP stdio transport: serves the agent over stdin/stdout.
//!
//! Used when `openheim` is launched as a subprocess by an ACP-compatible client
//! such as an IDE extension or CLI wrapper. Reads ACP messages from stdin and
//! writes responses to stdout using the line-delimited JSON framing defined by
//! the Agent Client Protocol.

use std::sync::Arc;

use agent_client_protocol_tokio::Stdio;

use crate::{
    acp::{self, AgentState},
    config::load_config,
    memory::MemoryContext,
};

/// Loads configuration, initialises the agent runtime, and serves ACP over stdin/stdout.
///
/// Blocks until the client closes the connection (EOF on stdin).
pub async fn run() -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;
    let memory = MemoryContext::new(app_config.default_skills.clone())?;
    let state = Arc::new(AgentState::new(agent_config, app_config, memory, vec![]).await?);

    acp::serve(Stdio::new(), state)
        .await
        .map_err(|e| crate::error::Error::Other(e.to_string()))
}
