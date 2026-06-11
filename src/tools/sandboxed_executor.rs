//! Work-directory sandboxing wrapper around any [`ToolExecutor`].

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;

use crate::{
    core::models::Tool,
    error::{Error, Result},
};

use super::execute_command::run_command;
use super::read_file::read_file;
use super::write_file::write_file;
use super::{ToolExecutor, sandbox::validate_path};

/// Wraps an inner [`ToolExecutor`] and enforces a work-directory boundary.
///
/// The three built-in tools are intercepted:
/// - `read_file` / `write_file`: the requested path is validated to be within
///   `work_dir` (following symlinks for existing paths); access outside the
///   boundary is rejected with an error the LLM can read and react to.
/// - `execute_command`: when `allow_shell` is `false` the call is rejected
///   immediately. When `true` the command runs with its working directory set
///   to `work_dir` so relative paths behave correctly. Note that absolute
///   paths inside the shell command are not blocked at the application layer
///   — OS-level sandboxing is required for that.
///
/// All other tools are forwarded to the inner executor unchanged.
pub struct SandboxedExecutor {
    inner: Arc<dyn ToolExecutor>,
    work_dir: Arc<PathBuf>,
    allow_shell: bool,
}

impl SandboxedExecutor {
    pub fn new(inner: Arc<dyn ToolExecutor>, work_dir: PathBuf, allow_shell: bool) -> Self {
        Self {
            inner,
            work_dir: Arc::new(work_dir),
            allow_shell,
        }
    }
}

#[async_trait]
impl ToolExecutor for SandboxedExecutor {
    fn list_tools(&self) -> Vec<Tool> {
        let tools = self.inner.list_tools();
        if self.allow_shell {
            tools
        } else {
            tools
                .into_iter()
                .filter(|t| t.function.name != "execute_command")
                .collect()
        }
    }

    async fn execute(&self, name: &str, args_json: &str) -> Result<String> {
        match name {
            "read_file" => {
                let args: serde_json::Value = serde_json::from_str(args_json)
                    .map_err(|e| Error::ParseError(format!("failed to parse arguments: {}", e)))?;
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("missing 'path' argument".to_string()))?;
                let validated = validate_path(path, &self.work_dir)?;
                read_file(&validated).await
            }

            "write_file" => {
                let args: serde_json::Value = serde_json::from_str(args_json)
                    .map_err(|e| Error::ParseError(format!("failed to parse arguments: {}", e)))?;
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("missing 'path' argument".to_string()))?;
                let content = args["content"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("missing 'content' argument".to_string()))?;
                let validated = validate_path(path, &self.work_dir)?;
                write_file(&validated, content).await
            }

            "execute_command" => {
                if !self.allow_shell {
                    return Err(Error::ToolExecutionError(
                        "shell command execution is disabled by configuration".to_string(),
                    ));
                }
                let args: serde_json::Value = serde_json::from_str(args_json)
                    .map_err(|e| Error::ParseError(format!("failed to parse arguments: {}", e)))?;
                let command = args["command"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("missing 'command' argument".to_string()))?;

                run_command(command, Some(&self.work_dir)).await
            }

            _ => self.inner.execute(name, args_json).await,
        }
    }
}
