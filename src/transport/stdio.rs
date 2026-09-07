//! ACP stdio transport: serves the agent over stdin/stdout.
//!
//! Used when `openheim` is launched as a subprocess by an ACP-compatible client
//! such as an IDE extension or CLI wrapper. Reads ACP messages from stdin and
//! writes responses to stdout using the line-delimited JSON framing defined by
//! the Agent Client Protocol.

use agent_client_protocol_tokio::Stdio;

use crate::{acp, client::OpenheimClient};

/// Serves ACP over stdin/stdout for `client` — caller-built, so an embedder
/// with custom tools or a custom `LlmClient` can use this transport too.
///
/// Blocks until the client closes the connection (EOF on stdin).
pub async fn run(client: OpenheimClient) -> crate::error::Result<()> {
    let state = client.state().clone();

    acp::serve(Stdio::new(), state)
        .await
        .map_err(|e| crate::error::Error::Other(e.to_string()))
}
