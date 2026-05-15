mod client;
mod tool_handler;

use std::collections::BTreeMap;
use std::sync::Arc;

use client::McpClient;
use serde::Serialize;
use tool_handler::McpToolHandler;

use crate::{config::McpServerConfig, error::Result, tools::ToolHandler};

/// Connection status and tool-count summary for a single MCP server.
///
/// Returned by [`OpenheimClient::mcp_servers`] so callers can inspect which
/// servers connected successfully and how many tools each one exposed.
#[derive(Debug, Clone, Serialize)]
pub struct McpServerStatus {
    /// Name of the server as defined in the configuration.
    pub name: String,
    /// Transport type: `"stdio"`, `"http"`, or `"unknown"`.
    pub transport: &'static str,
    /// Spawn command for stdio servers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Base URL for HTTP servers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub connected: bool,
    pub tool_count: usize,
    /// Error message if the server failed to connect.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Connects to all configured MCP servers and returns their tool handlers and statuses.
///
/// Connection failures are non-fatal: a server that fails to connect produces a
/// [`McpServerStatus`] with `connected: false` and an error message, and the
/// function continues with the remaining servers.
pub(crate) async fn load_mcp_tools(
    configs: &BTreeMap<String, McpServerConfig>,
) -> (Vec<Box<dyn ToolHandler>>, Vec<McpServerStatus>) {
    let mut handlers: Vec<Box<dyn ToolHandler>> = Vec::new();
    let mut statuses: Vec<McpServerStatus> = Vec::new();

    for (name, config) in configs {
        let (transport, command, url) = if config.command.is_some() {
            ("stdio", config.command.clone(), None)
        } else if config.url.is_some() {
            ("http", None, config.url.clone())
        } else {
            ("unknown", None, None)
        };

        match connect_server(name, config).await {
            Ok(server_handlers) => {
                tracing::info!(server = %name, count = server_handlers.len(), "MCP server connected");
                statuses.push(McpServerStatus {
                    name: name.clone(),
                    transport,
                    command,
                    url,
                    connected: true,
                    tool_count: server_handlers.len(),
                    error: None,
                });
                handlers.extend(server_handlers);
            }
            Err(e) => {
                tracing::warn!(server = %name, error = %e, "MCP server failed to connect");
                statuses.push(McpServerStatus {
                    name: name.clone(),
                    transport,
                    command,
                    url,
                    connected: false,
                    tool_count: 0,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    (handlers, statuses)
}

async fn connect_server(name: &str, config: &McpServerConfig) -> Result<Vec<Box<dyn ToolHandler>>> {
    let client = Arc::new(McpClient::connect(name, config).await?);
    let tools = client.list_tools().await?;

    // Sanitise the prefix: hyphens and spaces become underscores so the
    // combined name is a valid identifier for tool-call APIs.
    let prefix: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    let handlers = tools
        .iter()
        .map(|tool| -> Box<dyn ToolHandler> {
            Box::new(McpToolHandler::new(Arc::clone(&client), tool, &prefix))
        })
        .collect();

    Ok(handlers)
}
