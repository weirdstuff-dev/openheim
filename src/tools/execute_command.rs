use async_trait::async_trait;
use serde_json::json;
use tokio::process::Command;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;

pub struct ExecuteCommandTool;

#[async_trait]
impl ToolHandler for ExecuteCommandTool {
    fn definition(&self) -> Tool {
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
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let command = args["command"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'command' argument".to_string()))?;

        #[cfg(target_family = "unix")]
        let mut cmd = {
            let mut c = Command::new("sh");
            c.arg("-c").arg(command);
            c
        };

        #[cfg(target_family = "windows")]
        let mut cmd = {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(command);
            c
        };

        let output = cmd
            .output()
            .await
            .map_err(|e| Error::ToolExecutionError(format!("Failed to execute command: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(stdout)
        } else {
            Ok(format!(
                "Command failed:\nStdout: {}\nStderr: {}",
                stdout, stderr
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = ExecuteCommandTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "execute_command");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_runs_simple_command() {
        let tool = ExecuteCommandTool;
        let args = r#"{"command": "echo hello"}"#;
        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[tokio::test]
    async fn execute_returns_output_for_failing_command() {
        let tool = ExecuteCommandTool;
        let args = r#"{"command": "ls /nonexistent_dir_12345"}"#;
        let result = tool.execute(args).await.unwrap();
        assert!(result.contains("Command failed:"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let tool = ExecuteCommandTool;
        let result = tool.execute("bad json").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_command() {
        let tool = ExecuteCommandTool;
        let result = tool.execute(r#"{"other": "value"}"#).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("command"));
    }
}
