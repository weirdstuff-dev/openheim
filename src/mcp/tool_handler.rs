use std::sync::Arc;

use async_trait::async_trait;
use rmcp::model::Tool as McpTool;

use crate::{
    core::models::{FunctionDefinition, Tool},
    error::Result,
    tools::ToolHandler,
};

use super::client::McpClient;

pub struct McpToolHandler {
    client: Arc<McpClient>,
    /// Original tool name as reported by the MCP server.
    tool_name: String,
    /// Name exposed to the LLM: `{server_prefix}__{tool_name}`.
    prefixed_name: String,
    description: String,
    schema: serde_json::Value,
}

impl McpToolHandler {
    pub fn new(client: Arc<McpClient>, tool: &McpTool, server_prefix: &str) -> Self {
        let tool_name = tool.name.to_string();
        let prefixed_name = format!("{}__{}", server_prefix, tool_name);
        let description = tool.description.as_deref().unwrap_or("").to_string();
        let schema = serde_json::to_value(&tool.input_schema)
            .unwrap_or_else(|_| serde_json::json!({"type": "object", "properties": {}}));

        Self { client, tool_name, prefixed_name, description, schema }
    }
}

#[async_trait]
impl ToolHandler for McpToolHandler {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: self.prefixed_name.clone(),
                description: self.description.clone(),
                parameters: self.schema.clone(),
            },
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        self.client.call_tool(&self.tool_name, args).await
    }
}
