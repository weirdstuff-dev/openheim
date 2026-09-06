//! Built-in tool: `list_dir` — lists the immediate contents of a directory.

use std::collections::BinaryHeap;
use std::path::Path;

use async_trait::async_trait;
use serde_json::json;
use tokio::fs;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::parse_args;
use super::sandbox::validate_path;

/// Entries beyond this count are omitted, with a marker noting how many were
/// left out, so a directory with tens of thousands of entries can't blow out
/// the LLM's context.
const MAX_ENTRIES: usize = 500;

/// Lists `path`'s immediate contents (not recursive) and returns them as
/// newline-separated text, one entry per line, sorted by name. Directories
/// are suffixed with `/` and symlinks are shown as `name -> target`, so the
/// model can tell entry kinds apart without a second call.
async fn list_dir(path: &Path) -> Result<String> {
    let mut read_dir = fs::read_dir(path).await.map_err(Error::IoError)?;

    // Keep only the `MAX_ENTRIES` alphabetically-first entries as we go, via a
    // bounded max-heap: pushing past the cap and popping the greatest keeps
    // the heap holding the smallest names seen so far. This avoids buffering
    // every name and rendered label for directories with huge entry counts,
    // while `total` still tracks the true count for the omitted-entries
    // marker. Ord on the tuple compares the raw name first, so a `/` or
    // ` -> target` suffix on the label never perturbs the ordering.
    let mut entries: BinaryHeap<(String, String)> = BinaryHeap::with_capacity(MAX_ENTRIES + 1);
    let mut total = 0usize;
    while let Some(entry) = read_dir.next_entry().await.map_err(Error::IoError)? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let file_type = entry.file_type().await.map_err(Error::IoError)?;
        let label = if file_type.is_symlink() {
            match fs::read_link(entry.path()).await {
                Ok(target) => format!("{name} -> {}", target.display()),
                Err(_) => name.clone(),
            }
        } else if file_type.is_dir() {
            format!("{name}/")
        } else {
            name.clone()
        };
        total += 1;
        entries.push((name, label));
        if entries.len() > MAX_ENTRIES {
            entries.pop();
        }
    }

    if total == 0 {
        return Ok("(empty directory)".to_string());
    }

    let mut labels: Vec<String> = entries
        .into_sorted_vec()
        .into_iter()
        .map(|(_, label)| label)
        .collect();
    if total > MAX_ENTRIES {
        labels.push(format!(
            "... [{} more entries omitted]",
            total - MAX_ENTRIES
        ));
    }
    Ok(labels.join("\n"))
}

/// Lists the immediate contents of a directory at the given path.
///
/// Not recursive. Directories are suffixed with `/` and symlinks show their
/// target. Defaults to the work directory if no path is given; the path must
/// be inside the work directory.
pub struct ListDirTool;

#[async_trait]
impl ToolHandler for ListDirTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "list_dir",
            "List the immediate contents of a directory (not recursive). Directories are suffixed with '/' and symlinks are shown as 'name -> target'. Defaults to the work directory if no path is given.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory to list. Defaults to the work directory if omitted."
                    }
                }
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let path = args["path"].as_str().unwrap_or(".");
        let validated = validate_path(path, turn.work_dir)?;
        list_dir(&validated).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_support::TurnHarness;

    #[test]
    fn definition_has_correct_name() {
        let tool = ListDirTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "list_dir");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_lists_files_and_dirs_sorted() {
        let harness = TurnHarness::new();
        let dir = harness.work_dir();
        std::fs::write(dir.join("b.txt"), "x").unwrap();
        std::fs::write(dir.join("a.txt"), "x").unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();

        let args = serde_json::json!({"path": dir.to_str().unwrap()}).to_string();
        let result = ListDirTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "a.txt\nb.txt\nsub/");
    }

    #[tokio::test]
    async fn execute_reports_empty_directory() {
        let harness = TurnHarness::new();
        let result = ListDirTool.execute("{}", &harness.turn()).await.unwrap();
        assert_eq!(result, "(empty directory)");
    }

    #[tokio::test]
    async fn execute_defaults_to_work_dir_when_path_omitted() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("marker.txt"), "x").unwrap();

        let result = ListDirTool.execute("{}", &harness.turn()).await.unwrap();
        assert_eq!(result, "marker.txt");
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_dir() {
        let harness = TurnHarness::new();
        let result = ListDirTool
            .execute(r#"{"path": "missing"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_rejects_path_outside_work_dir() {
        let harness = TurnHarness::new();
        let outside = tempfile::tempdir().unwrap();
        let args = serde_json::json!({"path": outside.path().to_str().unwrap()}).to_string();
        let err = ListDirTool
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
        let result = ListDirTool.execute("bad json", &harness.turn()).await;
        assert!(result.is_err());
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn execute_shows_symlink_targets() {
        let harness = TurnHarness::new();
        let dir = harness.work_dir();
        let target = dir.join("target.txt");
        std::fs::write(&target, "x").unwrap();
        std::os::unix::fs::symlink(&target, dir.join("link")).unwrap();

        let result = ListDirTool.execute("{}", &harness.turn()).await.unwrap();
        assert!(
            result.contains(&format!("link -> {}", target.display())),
            "unexpected output: {result}"
        );
    }

    #[tokio::test]
    async fn execute_truncates_past_max_entries() {
        let harness = TurnHarness::new();
        for i in 0..(MAX_ENTRIES + 5) {
            std::fs::write(harness.work_dir().join(format!("f{i:04}.txt")), "x").unwrap();
        }

        let result = ListDirTool.execute("{}", &harness.turn()).await.unwrap();
        assert_eq!(result.lines().count(), MAX_ENTRIES + 1);
        assert!(
            result.contains("5 more entries omitted"),
            "unexpected output: {result}"
        );
    }
}
