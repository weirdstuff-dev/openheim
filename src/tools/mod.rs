pub mod executor;

pub use executor::{SystemToolExecutor, ToolExecutor};

use crate::error::{Error, Result};
use crate::core::models::{FunctionDefinition, Tool};
use once_cell::sync::Lazy;
use serde_json::json;
use tokio::runtime::Runtime;

static AVAILABLE_TOOLS: Lazy<Vec<Tool>> = Lazy::new(|| {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "execute_command".to_string(),
                description: "Execute a shell command (e.g., ls, pwd, echo). Use this for listing directories and running system commands.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        }
                    },
                    "required": ["command"]
                }),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "read_file".to_string(),
                description: "Read the contents of a file at the specified path.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "write_file".to_string(),
                description: "Write content to a file at the specified path. Creates the file if it doesn't exist.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
    ]
});

pub fn get_available_tools() -> &'static [Tool] {
    &AVAILABLE_TOOLS
}

pub async fn execute_tool(name: &str, arguments: &str) -> Result<String> {
    let exec = SystemToolExecutor::new();
    exec.execute(name, arguments).await
}

pub fn execute_tool_blocking(name: &str, arguments: &str) -> Result<String> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(execute_tool(name, arguments)),
        Err(_) => {
            // No current runtime — create a temporary one.
            let rt = Runtime::new()
                .map_err(|e| Error::Other(format!("Failed to create runtime: {}", e)))?;
            rt.block_on(execute_tool(name, arguments))
        }
    }
}