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

        let content = fs::read_to_string(path).await.map_err(Error::IoError)?;
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn definition_has_correct_name() {
        let tool = ReadFileTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "read_file");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_reads_existing_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "hello world").unwrap();
        let path = tmp.path().to_str().unwrap();

        let tool = ReadFileTool;
        let args = serde_json::json!({"path": path}).to_string();
        let result = tool.execute(&args).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_file() {
        let tool = ReadFileTool;
        let args = r#"{"path": "/tmp/openheim_nonexistent_file_test_12345.txt"}"#;
        let result = tool.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let tool = ReadFileTool;
        let result = tool.execute("not json").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[tokio::test]
    async fn execute_errors_for_missing_path() {
        let tool = ReadFileTool;
        let result = tool.execute(r#"{"other": "value"}"#).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path"));
    }
}
