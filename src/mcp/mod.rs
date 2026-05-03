mod client;
mod tool_handler;

use std::collections::BTreeMap;
use std::sync::Arc;

use client::McpClient;
use tool_handler::McpToolHandler;

use crate::{config::McpServerConfig, error::Result, tools::ToolHandler};

pub(crate) async fn load_mcp_tools(configs: &BTreeMap<String, McpServerConfig>) -> Vec<Box<dyn ToolHandler>> {
    let mut handlers: Vec<Box<dyn ToolHandler>> = Vec::new();

    for (name, config) in configs {
        match connect_server(name, config).await {
            Ok(server_handlers) => {
                tracing::info!(
                    server = %name,
                    count = server_handlers.len(),
                    "MCP server connected"
                );
                handlers.extend(server_handlers);
            }
            Err(e) => {
                tracing::warn!(server = %name, error = %e, "MCP server failed to connect");
            }
        }
    }

    handlers
}

async fn connect_server(name: &str, config: &McpServerConfig) -> Result<Vec<Box<dyn ToolHandler>>> {
    let client = Arc::new(McpClient::connect(name, config).await?);
    let tools = client.list_tools().await?;

    // Sanitise the prefix: hyphens and spaces become underscores so the
    // combined name is a valid identifier for tool-call APIs.
    let prefix: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();

    let handlers = tools
        .iter()
        .map(|tool| -> Box<dyn ToolHandler> {
            Box::new(McpToolHandler::new(Arc::clone(&client), tool, &prefix))
        })
        .collect();

    Ok(handlers)
}
