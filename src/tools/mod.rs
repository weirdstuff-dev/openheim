mod execute_command;
mod read_file;
mod write_file;

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;

use crate::config::McpServerConfig;
use crate::core::models::Tool;
use crate::error::{Error, Result};

#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Returns the tool definition (name, description, JSON-schema parameters).
    fn definition(&self) -> Tool;

    /// Executes the tool with the given JSON-encoded arguments.
    async fn execute(&self, args: &str) -> Result<String>;
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    fn list_tools(&self) -> Vec<Tool>;
    async fn execute(&self, name: &str, args_json: &str) -> Result<String>;
}

pub struct SystemToolExecutor {
    handlers: HashMap<String, Box<dyn ToolHandler>>,
}

impl SystemToolExecutor {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub async fn build(
        mcp_configs: &BTreeMap<String, McpServerConfig>,
    ) -> (Self, Vec<crate::mcp::McpServerStatus>) {
        let mut executor = Self::new();
        executor.register_builtins();
        let (handlers, statuses) = crate::mcp::load_mcp_tools(mcp_configs).await;
        for handler in handlers {
            executor.register(handler);
        }
        (executor, statuses)
    }

    pub fn register_builtins(&mut self) {
        self.register(Box::new(execute_command::ExecuteCommandTool));
        self.register(Box::new(read_file::ReadFileTool));
        self.register(Box::new(write_file::WriteFileTool));
    }

    pub fn register(&mut self, handler: Box<dyn ToolHandler>) {
        let name = handler.definition().function.name.clone();
        if self.handlers.contains_key(&name) {
            tracing::warn!(name = %name, "Tool name collision: overwriting existing handler");
        }
        self.handlers.insert(name, handler);
    }
}

impl Default for SystemToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolExecutor for SystemToolExecutor {
    fn list_tools(&self) -> Vec<Tool> {
        self.handlers.values().map(|h| h.definition()).collect()
    }

    async fn execute(&self, name: &str, args_json: &str) -> Result<String> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| Error::ToolExecutionError(format!("Unknown tool: {}", name)))?;
        handler.execute(args_json).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_executor_is_empty() {
        let executor = SystemToolExecutor::new();
        assert_eq!(executor.handlers.len(), 0);
    }

    #[test]
    fn register_builtins_adds_three_tools() {
        let mut executor = SystemToolExecutor::new();
        executor.register_builtins();
        assert!(executor.handlers.contains_key("execute_command"));
        assert!(executor.handlers.contains_key("read_file"));
        assert!(executor.handlers.contains_key("write_file"));
        assert_eq!(executor.handlers.len(), 3);
    }

    #[tokio::test]
    async fn executor_returns_error_for_unknown_tool() {
        let executor = SystemToolExecutor::new();
        let result = executor.execute("nonexistent_tool", "{}").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown tool"));
    }
}
