//! Built-in tool: `search` — regex search across files, ripgrep-style.
//!
//! Built on ripgrep's own crates (`grep-searcher`/`grep-regex` for matching,
//! `ignore` for the directory walk) rather than shelling out to an `rg`
//! binary, so it works regardless of `allow_shell` and doesn't depend on
//! anything being installed on the host.

use std::path::Path;

use async_trait::async_trait;
use grep::regex::RegexMatcherBuilder;
use grep::searcher::sinks::UTF8;
use grep::searcher::{BinaryDetection, SearcherBuilder};
use ignore::WalkBuilder;
use serde_json::json;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;

/// Matches beyond this count are omitted, with a marker noting the cut-off,
/// so a broad pattern over a large tree can't blow out the LLM's context.
const MAX_RESULTS: usize = 200;

/// Files scanned beyond this count abort the walk early (with the same
/// truncation marker as the match cap) — a backstop for a pattern that
/// matches nothing over a huge, mostly-non-matching tree, where the match
/// cap alone would never kick in.
const MAX_FILES_SCANNED: usize = 50_000;

/// Searches every file under `root` for `pattern` (a regex) and returns
/// matches as `path:line: content`, one per line, in walk order.
///
/// Single source of truth for the `search` behaviour, shared by
/// [`SearchTool`] and [`crate::tools::SandboxedExecutor`] (which calls this
/// after validating `root` against the work-directory boundary).
///
/// Directory walking follows `.gitignore`/`.ignore` rules and skips hidden
/// entries, matching ripgrep's own defaults. Binary files are detected and
/// skipped rather than searched. The walk and regex matching are both
/// blocking/CPU-bound, so this runs on a blocking thread pool via
/// [`tokio::task::spawn_blocking`] rather than the async runtime.
pub(crate) async fn search(pattern: &str, root: &Path, case_insensitive: bool) -> Result<String> {
    let pattern = pattern.to_string();
    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || search_blocking(&pattern, &root, case_insensitive))
        .await
        .map_err(|e| Error::ToolExecutionError(format!("search task panicked: {e}")))?
}

fn search_blocking(pattern: &str, root: &Path, case_insensitive: bool) -> Result<String> {
    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(case_insensitive)
        .build(pattern)
        .map_err(|e| Error::ToolExecutionError(format!("Invalid pattern '{pattern}': {e}")))?;

    let mut searcher = SearcherBuilder::new()
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .line_number(true)
        .build();

    let mut results: Vec<String> = Vec::new();
    let mut files_scanned = 0usize;
    let mut truncated = false;

    for entry in WalkBuilder::new(root).build() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }

        files_scanned += 1;
        if files_scanned > MAX_FILES_SCANNED {
            truncated = true;
            break;
        }

        let path = entry.path();
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(path);
        let display_path = if relative.as_os_str().is_empty() {
            path.display().to_string()
        } else {
            relative.display().to_string()
        };

        // Errors here are almost always "not readable as text" (permissions,
        // a binary file that slipped past detection, ...) — skip the file
        // rather than failing the whole search over it.
        let _ = searcher.search_path(
            &matcher,
            path,
            UTF8(|line_num, line| {
                results.push(format!("{display_path}:{line_num}: {}", line.trim_end()));
                Ok(results.len() < MAX_RESULTS)
            }),
        );

        if results.len() >= MAX_RESULTS {
            truncated = true;
            break;
        }
    }

    if results.is_empty() {
        return Ok(if truncated {
            format!("(no matches found; search stopped after {MAX_FILES_SCANNED} files)")
        } else {
            "(no matches found)".to_string()
        });
    }
    let mut out = results.join("\n");
    if truncated {
        out.push_str(&format!(
            "\n... [results truncated at {MAX_RESULTS} matches]"
        ));
    }
    Ok(out)
}

/// Searches files under a path for lines matching a regex pattern, ripgrep-style.
///
/// Respects `.gitignore`/`.ignore` and skips hidden files and binary files,
/// matching ripgrep's own defaults. Returns matches as `path:line: content`.
/// Defaults to searching the current directory if no path is given.
pub struct SearchTool;

#[async_trait]
impl ToolHandler for SearchTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "search".to_string(),
                description: "Search files under a path for lines matching a regex pattern (ripgrep-style). Respects .gitignore and skips hidden and binary files. Returns matches as 'path:line: content', capped at 200 results.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The regex pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "The file or directory to search. Defaults to the current directory if omitted."
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Match case-insensitively. Defaults to false."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'pattern' argument".to_string()))?;
        let path = args["path"].as_str().unwrap_or(".");
        let case_insensitive = args["case_insensitive"].as_bool().unwrap_or(false);

        search(pattern, Path::new(path), case_insensitive).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = SearchTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "search");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_finds_matches_across_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world\nfoo\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "another hello\n").unwrap();

        let tool = SearchTool;
        let args = serde_json::json!({"pattern": "hello", "path": dir.path().to_str().unwrap()})
            .to_string();
        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("a.txt:1: hello world"), "{result}");
        assert!(result.contains("b.txt:1: another hello"), "{result}");
        assert!(!result.contains("foo"), "{result}");
    }

    #[tokio::test]
    async fn execute_reports_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "nothing relevant\n").unwrap();

        let tool = SearchTool;
        let args = serde_json::json!({"pattern": "needle", "path": dir.path().to_str().unwrap()})
            .to_string();
        let result = tool.execute(&args).await.unwrap();
        assert_eq!(result, "(no matches found)");
    }

    #[tokio::test]
    async fn execute_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        // `.gitignore` is only honored inside an actual git repo (matching
        // ripgrep's own `require_git` default), so the tree needs a `.git`
        // dir for this to exercise anything.
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "secret needle\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "visible needle\n").unwrap();

        let tool = SearchTool;
        let args = serde_json::json!({"pattern": "needle", "path": dir.path().to_str().unwrap()})
            .to_string();
        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("kept.txt"), "{result}");
        assert!(!result.contains("ignored.txt"), "{result}");
    }

    #[tokio::test]
    async fn execute_is_case_insensitive_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "HELLO\n").unwrap();

        let tool = SearchTool;
        let args = serde_json::json!({
            "pattern": "hello",
            "path": dir.path().to_str().unwrap(),
            "case_insensitive": true,
        })
        .to_string();
        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("HELLO"), "{result}");
    }

    #[tokio::test]
    async fn execute_errors_for_invalid_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchTool;
        let args =
            serde_json::json!({"pattern": "(unclosed", "path": dir.path().to_str().unwrap()})
                .to_string();
        let result = tool.execute(&args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_pattern() {
        let tool = SearchTool;
        let result = tool.execute(r#"{"path": "."}"#).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pattern"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let tool = SearchTool;
        let result = tool.execute("bad json").await;
        assert!(result.is_err());
    }
}
