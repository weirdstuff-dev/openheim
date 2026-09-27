use std::sync::Arc;

use async_trait::async_trait;
use rmcp::model::Tool as McpTool;

use crate::{
    core::{models::Tool, turn::TurnContext},
    error::Result,
    tools::ToolHandler,
};

use super::client::McpClient;

/// A [`ToolHandler`] that proxies calls to an MCP server's tool, offered to
/// the LLM as `{server}__{tool}` so servers can't collide.
pub struct McpToolHandler {
    client: Arc<McpClient>,
    /// Original tool name as reported by the MCP server.
    tool_name: String,
    /// Name offered to the LLM (see `mcp::exposed_tool_name`).
    prefixed_name: String,
    description: String,
    schema: serde_json::Value,
}

impl McpToolHandler {
    /// Creates a new handler for a specific MCP tool, offered to the LLM as
    /// `prefixed_name` (see `mcp::exposed_tool_name`).
    pub fn new(client: Arc<McpClient>, tool: &McpTool, prefixed_name: String) -> Self {
        let tool_name = tool.name.to_string();
        let description = tool.description.as_deref().unwrap_or("").to_string();
        let schema = serde_json::to_value(&tool.input_schema)
            .unwrap_or_else(|_| serde_json::json!({"type": "object", "properties": {}}));

        Self {
            client,
            tool_name,
            prefixed_name,
            description,
            schema,
        }
    }
}

#[async_trait]
impl ToolHandler for McpToolHandler {
    fn definition(&self) -> Tool {
        Tool::function(
            self.prefixed_name.clone(),
            self.description.clone(),
            self.schema.clone(),
        )
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        self.client.call_tool(&self.tool_name, args).await
    }
}
