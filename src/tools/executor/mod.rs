use async_trait::async_trait;
use serde_json::Value;
use std::path::Path;
use tokio::fs;
use tokio::process::Command;

use crate::error::{Error, Result};

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, name: &str, args_json: &str) -> Result<String>;
}

#[derive(Clone, Debug, Default)]
pub struct SystemToolExecutor {}

impl SystemToolExecutor {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl ToolExecutor for SystemToolExecutor {
    async fn execute(&self, name: &str, args_json: &str) -> Result<String> {
        let args: Value = serde_json::from_str(args_json)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        match name {
            "execute_command" => {
                let command = args["command"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("Missing 'command' argument".to_string()))?;

                // Platform-aware shell invocation.
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
                    Ok(format!("Command failed:\nStdout: {}\nStderr: {}", stdout, stderr))
                }
            }

            "read_file" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("Missing 'path' argument".to_string()))?;

                let content = fs::read_to_string(path)
                    .await
                    .map_err(Error::IoError)?;
                Ok(content)
            }

            "write_file" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("Missing 'path' argument".to_string()))?;
                let content = args["content"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("Missing 'content' argument".to_string()))?;

                // Ensure parent directories exist if provided.
                if let Some(parent) = Path::new(path).parent()
                    && !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)
                            .await
                            .map_err(Error::IoError)?;
                    }

                fs::write(path, content).await.map_err(Error::IoError)?;
                Ok(format!("Successfully wrote to {}", path))
            }

            other => Err(Error::ToolExecutionError(format!("Unknown tool: {}", other))),
        }
    }
}