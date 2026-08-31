//! Built-in tool: `edit_file` — replaces an exact string in a file, without
//! rewriting (or resending) the whole thing.

use std::path::Path;

use async_trait::async_trait;
use serde_json::json;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;
use super::read_file::read_file;
use super::write_file::write_file;

/// Applies a find-and-replace edit to `content`, returning the edited text
/// and how many occurrences were replaced.
///
/// `old_string` must occur in `content` at least once; unless `replace_all`
/// is set it must occur *exactly* once, so a call can't silently touch more
/// of the file than the caller intended to.
///
/// Pure and I/O-free — the single source of truth for the `edit_file`
/// behaviour, shared by [`EditFileTool`] and [`crate::tools::SandboxedExecutor`]
/// (which supplies the current content via `client_io`/local disk and writes
/// the result back the same way).
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

/// Formats the success message shared by both the plain and sandboxed paths.
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
/// `replace_all` is set.
pub struct EditFileTool;

#[async_trait]
impl ToolHandler for EditFileTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "edit_file".to_string(),
                description: "Edit a file by replacing an exact occurrence of old_string with new_string, without rewriting the whole file. old_string must match the file's existing content exactly (including whitespace/indentation) and must be unique in the file unless replace_all is set. Use write_file instead to create a new file or replace a file's entire contents.".to_string(),
                parameters: json!({
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
            },
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let path = args["path"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'path' argument".to_string()))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'old_string' argument".to_string()))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'new_string' argument".to_string()))?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        let path = Path::new(path);
        let content = read_file(path).await?;
        let (edited, count) = apply_edit(&content, old_string, new_string, replace_all)?;
        write_file(path, &edited).await?;
        Ok(success_message(path, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        write!(tmp, "fn main() {{ old_code() }}").unwrap();
        let path = tmp.path().to_str().unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": path,
            "old_string": "old_code()",
            "new_string": "new_code()",
        })
        .to_string();
        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("Successfully edited"), "{result}");

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content, "fn main() { new_code() }");
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_file() {
        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": "/tmp/openheim_nonexistent_edit_target_12345.txt",
            "old_string": "a",
            "new_string": "b",
        })
        .to_string();
        let result = tool.execute(&args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_old_string() {
        let tool = EditFileTool;
        let result = tool
            .execute(r#"{"path": "/tmp/x", "new_string": "b"}"#)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("old_string"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let tool = EditFileTool;
        let result = tool.execute("bad json").await;
        assert!(result.is_err());
    }
}
