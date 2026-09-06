//! Built-in tool: `write_file` — writes content to a file, creating parent directories as needed.

use std::path::Path;

use async_trait::async_trait;
use serde_json::json;
use tokio::fs;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::{parse_args, require_str};
use super::sandbox::validate_path;

/// Writes `content` to `path`, asking `turn.client_io` first and falling back
/// to local `tokio::fs` (creating any missing parent directories) when it
/// defers. The `client_io` await is raced against `turn.cancel` so an
/// unresponsive client can't block cancellation.
///
/// `path` must already be validated against the work directory; shared by
/// [`WriteFileTool`] and `edit_file`.
pub(crate) async fn write_text(path: &Path, content: &str, turn: &TurnContext<'_>) -> Result<()> {
    tokio::select! {
        _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError(
            "file write cancelled".to_string(),
        )),
        result = turn.client_io.write_file(path, content) => match result {
            Some(result) => result,
            None => write_local(path, content).await,
        },
    }
}

async fn write_local(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).await.map_err(Error::IoError)?;
    }
    fs::write(path, content).await.map_err(Error::IoError)
}

/// Writes content to a file at the given path, creating any missing parent directories.
///
/// Creates the file if it does not exist; overwrites it if it does. The path
/// must be inside the work directory.
pub struct WriteFileTool;

#[async_trait]
impl ToolHandler for WriteFileTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "write_file",
            "Write content to a file at the specified path. Creates the file if it doesn't exist.",
            json!({
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
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let path = require_str(&args, "path")?;
        let content = require_str(&args, "content")?;
        let validated = validate_path(path, turn.work_dir)?;
        write_text(&validated, content, turn).await?;
        Ok(format!("Successfully wrote to {}", validated.display()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::test_support::{FixedClientIo, HangingClientIo, TurnHarness};

    #[test]
    fn definition_has_correct_name() {
        let tool = WriteFileTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "write_file");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_writes_file_and_creates_parents() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("sub").join("test.txt");

        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "written"}).to_string();
        let result = WriteFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert!(result.contains("Successfully wrote"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "written");
    }

    #[tokio::test]
    async fn execute_errors_for_missing_content() {
        let harness = TurnHarness::new();
        let result = WriteFileTool
            .execute(r#"{"path": "test.txt"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("content"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = WriteFileTool.execute("bad json", &harness.turn()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn write_file_via_client_io_does_not_touch_local_disk() {
        let harness = TurnHarness::new().with_client_io(Arc::new(FixedClientIo("unused")));
        let path = harness.work_dir().join("a.txt");

        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "hello"}).to_string();
        let result = WriteFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn write_file_cancel_aborts_hanging_client_io() {
        let harness = TurnHarness::new().with_client_io(Arc::new(HangingClientIo));
        let path = harness.work_dir().join("a.txt");

        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "hello"}).to_string();
        let cancel = harness.cancel_handle();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            WriteFileTool.execute(&args, &harness.turn()),
        )
        .await
        .expect("cancellation should abort the hanging client_io call");
        assert!(result.is_err());
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn write_file_falls_back_to_local_disk_when_client_io_defers() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");

        let args =
            serde_json::json!({"path": path.to_str().unwrap(), "content": "hello"}).to_string();
        WriteFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    /// End-to-end escape repro: `x` does not exist, so without normalization
    /// the ancestor walk only saw the work dir and handed the raw `..`-bearing
    /// path to the writer, whose `create_dir_all` then built the missing
    /// prefix and let the kernel resolve the `..`s outside the sandbox.
    #[tokio::test]
    async fn write_file_cannot_escape_through_nonexistent_prefix() {
        let harness = TurnHarness::new();
        let outside = tempfile::tempdir().unwrap();
        let escape_target = outside.path().join("pwned.txt");
        let requested = harness
            .work_dir()
            .join("x")
            .join("..")
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("pwned.txt");

        let args = serde_json::json!({
            "path": requested.to_str().unwrap(),
            "content": "escaped",
        })
        .to_string();
        let err = WriteFileTool
            .execute(&args, &harness.turn())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
        assert!(!escape_target.exists());
    }
}
