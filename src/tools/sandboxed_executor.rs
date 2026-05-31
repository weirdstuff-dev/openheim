//! Work-directory sandboxing wrapper around any [`ToolExecutor`].

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use tokio::{fs, process::Command};

use crate::{
    core::models::Tool,
    error::{Error, Result},
};

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
        self.inner.list_tools()
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
                let content = fs::read_to_string(&validated)
                    .await
                    .map_err(Error::IoError)?;
                Ok(content)
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
                if let Some(parent) = validated.parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent).await.map_err(Error::IoError)?;
                }
                fs::write(&validated, content)
                    .await
                    .map_err(Error::IoError)?;
                Ok(format!("Successfully wrote to {}", validated.display()))
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

                cmd.current_dir(&*self.work_dir);

                let output = cmd.output().await.map_err(|e| {
                    Error::ToolExecutionError(format!("failed to execute command: {}", e))
                })?;

                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if output.status.success() {
                    Ok(stdout)
                } else {
                    Err(Error::ToolExecutionError(format!(
                        "Command failed:\nStdout: {}\nStderr: {}",
                        stdout, stderr
                    )))
                }
            }

            _ => self.inner.execute(name, args_json).await,
        }
    }
}
