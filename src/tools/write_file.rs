use async_trait::async_trait;
use serde_json::json;
use std::path::Path;
use tokio::fs;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;

pub struct WriteFileTool;

#[async_trait]
impl ToolHandler for WriteFileTool {
    fn definition(&self) -> Tool {
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
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let path = args["path"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'path' argument".to_string()))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'content' argument".to_string()))?;

        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .await
                .map_err(Error::IoError)?;
        }

        fs::write(path, content).await.map_err(Error::IoError)?;
        Ok(format!("Successfully wrote to {}", path))
    }
}
