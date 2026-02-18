mod execute_command;
mod read_file;
mod write_file;

use std::collections::HashMap;
use std::sync::OnceLock;

use async_trait::async_trait;

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
    async fn execute(&self, name: &str, args_json: &str) -> Result<String>;
}

pub struct SystemToolExecutor {
    handlers: HashMap<String, Box<dyn ToolHandler>>,
}

impl SystemToolExecutor {
    pub fn new() -> Self {
        let mut executor = Self {
            handlers: HashMap::new(),
        };
        executor.register(Box::new(execute_command::ExecuteCommandTool));
        executor.register(Box::new(read_file::ReadFileTool));
        executor.register(Box::new(write_file::WriteFileTool));
        executor
    }

    fn register(&mut self, handler: Box<dyn ToolHandler>) {
        let name = handler.definition().function.name.clone();
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
    async fn execute(&self, name: &str, args_json: &str) -> Result<String> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| Error::ToolExecutionError(format!("Unknown tool: {}", name)))?;
        handler.execute(args_json).await
    }
}

/// Global executor used by the free-standing helper functions.
static GLOBAL_EXECUTOR: OnceLock<SystemToolExecutor> = OnceLock::new();

fn global_executor() -> &'static SystemToolExecutor {
    GLOBAL_EXECUTOR.get_or_init(SystemToolExecutor::new)
}

pub fn get_available_tools() -> Vec<Tool> {
    global_executor()
        .handlers
        .values()
        .map(|h| h.definition())
        .collect()
}

pub async fn execute_tool(name: &str, arguments: &str) -> Result<String> {
    global_executor().execute(name, arguments).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_registers_all_tools() {
        let executor = SystemToolExecutor::new();
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

    #[test]
    fn get_available_tools_returns_three_definitions() {
        let tools = get_available_tools();
        assert_eq!(tools.len(), 3);
        let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"execute_command"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
    }
}
