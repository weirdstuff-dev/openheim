//! Built-in tool: `list_dir` — lists the immediate contents of a directory.

use std::path::Path;

use async_trait::async_trait;
use serde_json::json;
use tokio::fs;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;

/// Entries beyond this count are omitted, with a marker noting how many were
/// left out, so a directory with tens of thousands of entries can't blow out
/// the LLM's context.
const MAX_ENTRIES: usize = 500;

/// Lists `path`'s immediate contents (not recursive) and returns them as
/// newline-separated text, one entry per line, sorted by name. Directories
/// are suffixed with `/` and symlinks are shown as `name -> target`, so the
/// model can tell entry kinds apart without a second call.
///
/// Single source of truth for the `list_dir` behaviour, shared by
/// [`ListDirTool`] and [`crate::tools::SandboxedExecutor`] (which calls this
/// after validating the path against the work-directory boundary).
pub(crate) async fn list_dir(path: &Path) -> Result<String> {
    let mut read_dir = fs::read_dir(path).await.map_err(Error::IoError)?;

    let mut entries: Vec<(String, String)> = Vec::new();
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
        entries.push((name, label));
    }

    if entries.is_empty() {
        return Ok("(empty directory)".to_string());
    }

    // Sort by the raw name, not the formatted label, so a `/` or ` -> target`
    // suffix never perturbs the ordering the model would expect from `ls`.
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let total = entries.len();
    let mut labels: Vec<String> = entries.into_iter().map(|(_, label)| label).collect();
    if total > MAX_ENTRIES {
        labels.truncate(MAX_ENTRIES);
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
/// target. Defaults to the current directory if no path is given.
pub struct ListDirTool;

#[async_trait]
impl ToolHandler for ListDirTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "list_dir".to_string(),
                description: "List the immediate contents of a directory (not recursive). Directories are suffixed with '/' and symlinks are shown as 'name -> target'. Defaults to the current directory if no path is given.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory to list. Defaults to the current directory if omitted."
                        }
                    }
                }),
            },
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let path = args["path"].as_str().unwrap_or(".");
        list_dir(Path::new(path)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = ListDirTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "list_dir");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_lists_files_and_dirs_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.txt"), "x").unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let tool = ListDirTool;
        let args = serde_json::json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = tool.execute(&args).await.unwrap();
        assert_eq!(result, "a.txt\nb.txt\nsub/");
    }

    #[tokio::test]
    async fn execute_reports_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirTool;
        let args = serde_json::json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = tool.execute(&args).await.unwrap();
        assert_eq!(result, "(empty directory)");
    }

    #[tokio::test]
    async fn execute_defaults_to_current_directory_when_path_omitted() {
        let tool = ListDirTool;
        let result = tool.execute("{}").await.unwrap();
        // Cargo.toml lives at the crate root, which is the test process's cwd.
        assert!(result.contains("Cargo.toml"), "unexpected output: {result}");
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_dir() {
        let tool = ListDirTool;
        let args = r#"{"path": "/tmp/openheim_nonexistent_dir_test_12345"}"#;
        let result = tool.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let tool = ListDirTool;
        let result = tool.execute("bad json").await;
        assert!(result.is_err());
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn execute_shows_symlink_targets() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        std::fs::write(&target, "x").unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("link")).unwrap();

        let tool = ListDirTool;
        let args = serde_json::json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = tool.execute(&args).await.unwrap();
        assert!(
            result.contains(&format!("link -> {}", target.display())),
            "unexpected output: {result}"
        );
    }

    #[tokio::test]
    async fn execute_truncates_past_max_entries() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..(MAX_ENTRIES + 5) {
            std::fs::write(dir.path().join(format!("f{i:04}.txt")), "x").unwrap();
        }

        let tool = ListDirTool;
        let args = serde_json::json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = tool.execute(&args).await.unwrap();
        assert_eq!(result.lines().count(), MAX_ENTRIES + 1);
        assert!(
            result.contains("5 more entries omitted"),
            "unexpected output: {result}"
        );
    }
}
