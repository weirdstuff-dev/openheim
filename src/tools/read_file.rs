//! Built-in tool: `read_file` — reads a file and returns its contents.

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

/// Reads `path` as UTF-8 text, asking `turn.client_io` first (e.g. an ACP
/// client's editor buffers) and falling back to local `tokio::fs` when it
/// defers. The `client_io` await is raced against `turn.cancel` so a slow or
/// unresponsive client can't block `session/cancel` from interrupting the
/// turn.
///
/// `path` must already be validated against the work directory; shared by
/// [`ReadFileTool`] and `edit_file`.
pub(crate) async fn read_text(path: &Path, turn: &TurnContext<'_>) -> Result<String> {
    tokio::select! {
        _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError(
            "file read cancelled".to_string(),
        )),
        result = turn.client_io.read_file(path) => match result {
            Some(result) => result,
            None => fs::read_to_string(path).await.map_err(Error::IoError),
        },
    }
}

/// Reads a file at the given path and returns its UTF-8 contents.
///
/// Returns an error if the path is outside the work directory, the file does
/// not exist, cannot be read, or is not valid UTF-8.
pub struct ReadFileTool;

#[async_trait]
impl ToolHandler for ReadFileTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "read_file",
            "Read the contents of a file at the specified path.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let path = require_str(&args, "path")?;
        let validated = validate_path(path, turn.work_dir)?;
        read_text(&validated, turn).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::test_support::{FixedClientIo, HangingClientIo, TurnHarness};

    #[test]
    fn definition_has_correct_name() {
        let tool = ReadFileTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "read_file");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_reads_existing_file() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "hello world").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let result = ReadFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn execute_resolves_relative_paths_against_work_dir() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("rel.txt"), "relative").unwrap();

        let result = ReadFileTool
            .execute(r#"{"path": "rel.txt"}"#, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result, "relative");
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_file() {
        let harness = TurnHarness::new();
        let result = ReadFileTool
            .execute(r#"{"path": "missing.txt"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_rejects_path_outside_work_dir() {
        let harness = TurnHarness::new();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "x").unwrap();

        let args = serde_json::json!({"path": secret.to_str().unwrap()}).to_string();
        let err = ReadFileTool
            .execute(&args, &harness.turn())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = ReadFileTool.execute("not json", &harness.turn()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[tokio::test]
    async fn execute_errors_for_missing_path() {
        let harness = TurnHarness::new();
        let result = ReadFileTool
            .execute(r#"{"other": "value"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path"));
    }

    #[tokio::test]
    async fn read_file_prefers_client_io_over_local_disk() {
        let harness = TurnHarness::new().with_client_io(Arc::new(FixedClientIo("client content")));
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let result = ReadFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "client content");
    }

    #[tokio::test]
    async fn read_file_falls_back_to_local_disk_when_client_io_defers() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let result = ReadFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "local content");
    }

    #[tokio::test]
    async fn read_file_cancel_aborts_hanging_client_io() {
        let harness = TurnHarness::new().with_client_io(Arc::new(HangingClientIo));
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let cancel = harness.cancel_handle();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ReadFileTool.execute(&args, &harness.turn()),
        )
        .await
        .expect("cancellation should abort the hanging client_io call");
        assert!(result.is_err());
    }
}
