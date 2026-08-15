//! Work-directory sandboxing wrapper around any [`ToolExecutor`].

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;

use crate::{
    core::{client_io::ClientIo, models::Tool, turn::TurnContext},
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
///   boundary is rejected with an error the LLM can read and react to. The I/O
///   itself is delegated to `client_io` first (e.g. an ACP client's editor
///   buffers), falling back to local `tokio::fs` when it defers. The
///   `client_io` await is raced against `turn.cancel` so a slow or
///   unresponsive client can't block `session/cancel` from interrupting the
///   turn.
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
    client_io: Arc<dyn ClientIo>,
}

impl SandboxedExecutor {
    pub fn new(
        inner: Arc<dyn ToolExecutor>,
        work_dir: PathBuf,
        allow_shell: bool,
        client_io: Arc<dyn ClientIo>,
    ) -> Self {
        Self {
            inner,
            work_dir: Arc::new(work_dir),
            allow_shell,
            client_io,
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

    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String> {
        match name {
            "read_file" => {
                let args: serde_json::Value = serde_json::from_str(args_json)
                    .map_err(|e| Error::ParseError(format!("failed to parse arguments: {}", e)))?;
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| Error::ParseError("missing 'path' argument".to_string()))?;
                let validated = validate_path(path, &self.work_dir)?;
                tokio::select! {
                    _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError(
                        "read_file cancelled".to_string(),
                    )),
                    result = self.client_io.read_file(&validated) => match result {
                        Some(result) => result,
                        None => read_file(&validated).await,
                    },
                }
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
                tokio::select! {
                    _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError(
                        "write_file cancelled".to_string(),
                    )),
                    result = self.client_io.write_file(&validated, content) => match result {
                        Some(Ok(())) => Ok(format!("Successfully wrote to {}", validated.display())),
                        Some(Err(e)) => Err(e),
                        None => write_file(&validated, content).await,
                    },
                }
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

            _ => self.inner.execute(name, args_json, turn).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::client_io::NoClientIo;
    use crate::core::models::FunctionDefinition;
    use crate::tools::test_support::TurnHarness;

    struct EmptyExecutor;

    #[async_trait]
    impl ToolExecutor for EmptyExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            vec![Tool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "execute_command".to_string(),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                },
            }]
        }

        async fn execute(
            &self,
            name: &str,
            _args_json: &str,
            _turn: &TurnContext<'_>,
        ) -> Result<String> {
            Err(Error::ToolExecutionError(format!(
                "unexpected call: {name}"
            )))
        }
    }

    /// Always answers from an in-memory string, never touching local disk.
    struct FixedClientIo(&'static str);

    #[async_trait]
    impl ClientIo for FixedClientIo {
        async fn read_file(&self, _path: &std::path::Path) -> Option<Result<String>> {
            Some(Ok(self.0.to_string()))
        }

        async fn write_file(&self, _path: &std::path::Path, _content: &str) -> Option<Result<()>> {
            Some(Ok(()))
        }
    }

    #[tokio::test]
    async fn read_file_prefers_client_io_over_local_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            dir.path().to_path_buf(),
            false,
            Arc::new(FixedClientIo("client content")),
        );
        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let harness = TurnHarness::new();
        let result = executor
            .execute("read_file", &args, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result, "client content");
    }

    #[tokio::test]
    async fn read_file_falls_back_to_local_disk_when_client_io_defers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            dir.path().to_path_buf(),
            false,
            Arc::new(NoClientIo),
        );
        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let harness = TurnHarness::new();
        let result = executor
            .execute("read_file", &args, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result, "local content");
    }

    #[tokio::test]
    async fn write_file_via_client_io_does_not_touch_local_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            dir.path().to_path_buf(),
            false,
            Arc::new(FixedClientIo("unused")),
        );
        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "hello"}).to_string();
        let harness = TurnHarness::new();
        let result = executor
            .execute("write_file", &args, &harness.turn())
            .await
            .unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(!path.exists());
    }

    /// Never resolves on its own; used to prove cancellation aborts the wait
    /// on an unresponsive client rather than blocking the turn.
    struct HangingClientIo;

    #[async_trait]
    impl ClientIo for HangingClientIo {
        async fn read_file(&self, _path: &std::path::Path) -> Option<Result<String>> {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this wait before the sleep elapses");
        }

        async fn write_file(&self, _path: &std::path::Path, _content: &str) -> Option<Result<()>> {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this wait before the sleep elapses");
        }
    }

    #[tokio::test]
    async fn read_file_cancel_aborts_hanging_client_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            dir.path().to_path_buf(),
            false,
            Arc::new(HangingClientIo),
        );
        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let harness = TurnHarness::new();
        let cancel = harness.cancel_handle();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            executor.execute("read_file", &args, &harness.turn()),
        )
        .await
        .expect("cancellation should abort the hanging client_io call");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn write_file_cancel_aborts_hanging_client_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            dir.path().to_path_buf(),
            false,
            Arc::new(HangingClientIo),
        );
        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "hello"}).to_string();
        let harness = TurnHarness::new();
        let cancel = harness.cancel_handle();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            executor.execute("write_file", &args, &harness.turn()),
        )
        .await
        .expect("cancellation should abort the hanging client_io call");
        assert!(result.is_err());
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn write_file_falls_back_to_local_disk_when_client_io_defers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            dir.path().to_path_buf(),
            false,
            Arc::new(NoClientIo),
        );
        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "hello"}).to_string();
        let harness = TurnHarness::new();
        executor
            .execute("write_file", &args, &harness.turn())
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    /// End-to-end C1 repro: `x` does not exist, so pre-normalization the
    /// ancestor walk only saw the work dir and handed the raw `..`-bearing
    /// path to `write_file`, whose `create_dir_all` then built the missing
    /// prefix and let the kernel resolve the `..`s outside the sandbox.
    #[tokio::test]
    async fn write_file_cannot_escape_through_nonexistent_prefix() {
        let work = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let escape_target = outside.path().join("pwned.txt");
        let requested = work
            .path()
            .join("x")
            .join("..")
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("pwned.txt");

        let executor = SandboxedExecutor::new(
            Arc::new(EmptyExecutor),
            work.path().to_path_buf(),
            false,
            Arc::new(NoClientIo),
        );
        let args = serde_json::json!({
            "path": requested.to_str().unwrap(),
            "content": "escaped",
        })
        .to_string();
        let harness = TurnHarness::new();
        let err = executor
            .execute("write_file", &args, &harness.turn())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
        assert!(!escape_target.exists());
    }
}
