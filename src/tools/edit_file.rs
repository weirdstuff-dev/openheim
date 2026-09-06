//! Built-in tool: `edit_file` — replaces an exact string in a file, without
//! rewriting (or resending) the whole thing.

use std::path::Path;

use async_trait::async_trait;
use serde_json::json;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::{parse_args, require_str};
use super::read_file::read_text;
use super::sandbox::validate_path;
use super::write_file::write_text;

/// Applies a find-and-replace edit to `content`, returning the edited text
/// and how many occurrences were replaced.
///
/// `old_string` must occur in `content` at least once; unless `replace_all`
/// is set it must occur *exactly* once, so a call can't silently touch more
/// of the file than the caller intended to.
///
/// Pure and I/O-free; [`EditFileTool`] supplies the current content via
/// `client_io`/local disk and writes the result back the same way.
pub(crate) fn apply_edit(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<(String, usize)> {
    if old_string.is_empty() {
        return Err(Error::ToolExecutionError(
            "old_string must not be empty; use write_file to create a new file".to_string(),
        ));
    }
    if old_string == new_string {
        return Err(Error::ToolExecutionError(
            "old_string and new_string are identical; nothing to edit".to_string(),
        ));
    }

    let count = content.matches(old_string).count();
    if count == 0 {
        return Err(Error::ToolExecutionError(
            "old_string not found in file".to_string(),
        ));
    }
    if count > 1 && !replace_all {
        return Err(Error::ToolExecutionError(format!(
            "old_string appears {count} times in the file; provide more surrounding context to \
             make it unique, or set replace_all to true"
        )));
    }

    let edited = if replace_all {
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };
    Ok((edited, count))
}

fn success_message(path: &Path, count: usize) -> String {
    format!(
        "Successfully edited {} ({count} replacement{})",
        path.display(),
        if count == 1 { "" } else { "s" }
    )
}

/// Edits a file at the given path by replacing an exact occurrence of one
/// string with another, without rewriting the rest of the file.
///
/// `old_string` must match the file's existing content exactly (including
/// whitespace/indentation) and must be unique in the file unless
/// `replace_all` is set. It's a read followed by a write, so both go through
/// `client_io` when the client provides one, and the path must be inside the
/// work directory.
pub struct EditFileTool;

#[async_trait]
impl ToolHandler for EditFileTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "edit_file",
            "Edit a file by replacing an exact occurrence of old_string with new_string, without rewriting the whole file. old_string must match the file's existing content exactly (including whitespace/indentation) and must be unique in the file unless replace_all is set. Use write_file instead to create a new file or replace a file's entire contents.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to replace. Must be unique in the file unless replace_all is set."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace old_string with"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string instead of requiring exactly one. Defaults to false."
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let path = require_str(&args, "path")?;
        let old_string = require_str(&args, "old_string")?;
        let new_string = require_str(&args, "new_string")?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);
        let validated = validate_path(path, turn.work_dir)?;

        let content = read_text(&validated, turn).await?;
        let (edited, count) = apply_edit(&content, old_string, new_string, replace_all)?;
        write_text(&validated, &edited, turn).await?;
        Ok(success_message(&validated, count))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::test_support::{FixedClientIo, TurnHarness};

    #[test]
    fn definition_has_correct_name() {
        let tool = EditFileTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "edit_file");
        assert_eq!(def.tool_type, "function");
    }

    #[test]
    fn apply_edit_replaces_unique_occurrence() {
        let (edited, count) = apply_edit("hello world", "world", "there", false).unwrap();
        assert_eq!(edited, "hello there");
        assert_eq!(count, 1);
    }

    #[test]
    fn apply_edit_errors_when_old_string_not_found() {
        let err = apply_edit("hello world", "missing", "x", false).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn apply_edit_errors_on_ambiguous_match_without_replace_all() {
        let err = apply_edit("foo foo foo", "foo", "bar", false).unwrap_err();
        assert!(err.to_string().contains("appears 3 times"));
    }

    #[test]
    fn apply_edit_replaces_all_when_requested() {
        let (edited, count) = apply_edit("foo foo foo", "foo", "bar", true).unwrap();
        assert_eq!(edited, "bar bar bar");
        assert_eq!(count, 3);
    }

    #[test]
    fn apply_edit_errors_on_empty_old_string() {
        let err = apply_edit("hello", "", "x", false).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn apply_edit_errors_when_strings_are_identical() {
        let err = apply_edit("hello", "hello", "hello", false).unwrap_err();
        assert!(err.to_string().contains("identical"));
    }

    #[tokio::test]
    async fn execute_edits_file_on_disk() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("main.rs");
        std::fs::write(&path, "fn main() { old_code() }").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "old_code()",
            "new_string": "new_code()",
        })
        .to_string();
        let result = EditFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert!(result.contains("Successfully edited"), "{result}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn main() { new_code() }"
        );
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_file() {
        let harness = TurnHarness::new();
        let args = serde_json::json!({
            "path": "missing.txt",
            "old_string": "a",
            "new_string": "b",
        })
        .to_string();
        let result = EditFileTool.execute(&args, &harness.turn()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_old_string() {
        let harness = TurnHarness::new();
        let result = EditFileTool
            .execute(r#"{"path": "x", "new_string": "b"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("old_string"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = EditFileTool.execute("bad json", &harness.turn()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn edit_file_prefers_client_io_over_local_disk() {
        let harness = TurnHarness::new().with_client_io(Arc::new(FixedClientIo("client content")));
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "client",
            "new_string": "CLIENT",
        })
        .to_string();
        let result = EditFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert!(result.contains("Successfully edited"), "{result}");
        // FixedClientIo's write is a no-op that never touches local disk, so
        // the file on disk still holds its original content unmodified —
        // proof the edit went through client_io, not local fs.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "local content");
    }

    #[tokio::test]
    async fn edit_file_falls_back_to_local_disk_when_client_io_defers() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "fn main() { old() }").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "old()",
            "new_string": "new()",
        })
        .to_string();
        EditFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn main() { new() }"
        );
    }

    #[tokio::test]
    async fn edit_file_rejects_path_outside_work_dir() {
        let harness = TurnHarness::new();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("secret.txt");
        std::fs::write(&target, "old").unwrap();

        let args = serde_json::json!({
            "path": target.to_str().unwrap(),
            "old_string": "old",
            "new_string": "new",
        })
        .to_string();
        let err = EditFileTool
            .execute(&args, &harness.turn())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "old");
    }
}
