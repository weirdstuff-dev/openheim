//! Built-in tool: `write_file` — writes content to a file, creating parent directories as needed.

use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::parse;
use super::capabilities::{ToolCapabilities, ToolKindHint};

/// Writes `content` to `path` (already resolved with
/// `TurnContext::resolve_path`), asking `turn.client_io` first and falling
/// back to local disk, creating missing parent directories. The client is
/// raced against `turn.cancel`. Shared by [`WriteFileTool`] and `edit_file`.
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
    let (path, content) = (path.to_path_buf(), content.to_string());
    tokio::task::spawn_blocking(move || write_atomic(&path, content.as_bytes()))
        .await
        .map_err(Error::from)?
        .map_err(Error::IoError)
}

/// Replaces `path` with `content` via a temp file in the same directory,
/// synced and then renamed over it, so a crash or a full disk leaves the old
/// file or the new one, never a truncated mix. Missing parent directories
/// are created.
///
/// An existing file keeps its permissions (an executable script stays
/// executable) and, on Unix, its owner and group; its content is never
/// readable by anyone else meanwhile, as the temp file starts out private.
/// One this process may not write is refused, as a plain write would refuse
/// it. Where a rename can't keep what the file is (its owner can't be kept,
/// or it has other hard links), it's overwritten in place instead, without
/// the atomicity.
fn write_atomic(path: &Path, content: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir)?;
    let existing = match std::fs::metadata(path) {
        Ok(meta) => Some(meta),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if let Some(meta) = &existing {
        // Fails, as a plain write would, if this process may not write the
        // file; the rename alone needs only the directory to be writable.
        // Opened without truncating, so the file is untouched.
        std::fs::OpenOptions::new().write(true).open(path)?;
        #[cfg(unix)]
        if std::os::unix::fs::MetadataExt::nlink(meta) > 1 {
            return std::fs::write(path, content);
        }
    }

    // A fixed-length name, so a target name near the filesystem's limit
    // still leaves room for it.
    let tmp = dir.join(format!(".openheim-{}.tmp", uuid::Uuid::new_v4()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if existing.is_some() {
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    }
    let mut file = options.open(&tmp)?;
    let written = (|| {
        if let Some(meta) = &existing {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if std::os::unix::fs::fchown(&file, Some(meta.uid()), Some(meta.gid())).is_err() {
                    return Ok(false);
                }
            }
            file.set_permissions(meta.permissions())?;
        }
        file.write_all(content)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path).map(|()| true)
    })();
    drop(file);
    match written {
        Ok(true) => Ok(()),
        // The owner or group can't be kept: a replacement would change who
        // may read the file.
        Ok(false) => {
            let _ = std::fs::remove_file(&tmp);
            std::fs::write(path, content)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Writes content to a file at the given path, creating any missing parent directories.
///
/// Creates the file if it does not exist; overwrites it if it does. The path
/// must be inside the work directory.
pub struct WriteFileTool;

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

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
        let args: WriteFileArgs = parse(args)?;
        let validated = turn.resolve_path(&args.path)?;
        write_text(&validated, &args.content, turn).await?;
        Ok(format!("Successfully wrote to {}", validated.display()))
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            kind: ToolKindHint::Edit,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::test_support::{FixedClientIo, HangingClientIo, TurnHarness};

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

    /// Names in `dir`, sorted.
    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    // A failed write (here, the target is a directory) leaves no temp file.
    #[test]
    fn a_failed_write_cleans_up_its_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("sub");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("keep.txt"), "kept").unwrap();

        assert!(write_atomic(&target, b"new").is_err());

        assert_eq!(entries(dir.path()), ["sub"]);
        assert_eq!(entries(&target), ["keep.txt"]);
    }
}
