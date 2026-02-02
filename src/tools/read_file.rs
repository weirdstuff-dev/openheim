use async_trait::async_trait;
use serde_json::json;
use tokio::fs;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;

pub struct ReadFileTool;

#[async_trait]
impl ToolHandler for ReadFileTool {
    fn definition(&self) -> Tool {
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
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let path = args["path"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'path' argument".to_string()))?;

        let content = fs::read_to_string(path)
            .await
            .map_err(Error::IoError)?;
        Ok(content)
    }
}
