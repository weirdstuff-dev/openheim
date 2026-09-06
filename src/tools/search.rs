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
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::{parse_args, require_str};
use super::sandbox::validate_path;

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
/// Directory walking follows `.gitignore`/`.ignore` rules and skips hidden
/// entries, matching ripgrep's own defaults. Binary files are detected and
/// skipped rather than searched. The walk and regex matching are both
/// blocking/CPU-bound, so this runs on a blocking thread pool via
/// [`tokio::task::spawn_blocking`] rather than the async runtime.
async fn search(pattern: &str, root: &Path, case_insensitive: bool) -> Result<String> {
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
/// Defaults to searching the work directory if no path is given; the path
/// must be inside the work directory.
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
                            "description": "The file or directory to search. Defaults to the work directory if omitted."
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

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let pattern = require_str(&args, "pattern")?;
        let path = args["path"].as_str().unwrap_or(".");
        let case_insensitive = args["case_insensitive"].as_bool().unwrap_or(false);
        let validated = validate_path(path, turn.work_dir)?;
        search(pattern, &validated, case_insensitive).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_support::TurnHarness;

    #[test]
    fn definition_has_correct_name() {
        let tool = SearchTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "search");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_finds_matches_across_files() {
        let harness = TurnHarness::new();
        let dir = harness.work_dir();
        std::fs::write(dir.join("a.txt"), "hello world\nfoo\n").unwrap();
        std::fs::write(dir.join("b.txt"), "another hello\n").unwrap();

        let args =
            serde_json::json!({"pattern": "hello", "path": dir.to_str().unwrap()}).to_string();
        let result = SearchTool.execute(&args, &harness.turn()).await.unwrap();
        assert!(result.contains("a.txt:1: hello world"), "{result}");
        assert!(result.contains("b.txt:1: another hello"), "{result}");
        assert!(!result.contains("foo"), "{result}");
    }

    #[tokio::test]
    async fn execute_defaults_to_work_dir_when_path_omitted() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), "needle here\n").unwrap();

        let result = SearchTool
            .execute(r#"{"pattern": "needle"}"#, &harness.turn())
            .await
            .unwrap();
        assert!(
            result.contains("needle here"),
            "unexpected output: {result}"
        );
    }

    #[tokio::test]
    async fn execute_reports_no_matches() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), "nothing relevant\n").unwrap();

        let result = SearchTool
            .execute(r#"{"pattern": "needle"}"#, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result, "(no matches found)");
    }

    #[tokio::test]
    async fn execute_respects_gitignore() {
        let harness = TurnHarness::new();
        let dir = harness.work_dir();
        // `.gitignore` is only honored inside an actual git repo (matching
        // ripgrep's own `require_git` default), so the tree needs a `.git`
        // dir for this to exercise anything.
        std::fs::create_dir(dir.join(".git")).unwrap();
        std::fs::write(dir.join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.join("ignored.txt"), "secret needle\n").unwrap();
        std::fs::write(dir.join("kept.txt"), "visible needle\n").unwrap();

        let result = SearchTool
            .execute(r#"{"pattern": "needle"}"#, &harness.turn())
            .await
            .unwrap();
        assert!(result.contains("kept.txt"), "{result}");
        assert!(!result.contains("ignored.txt"), "{result}");
    }

    #[tokio::test]
    async fn execute_is_case_insensitive_when_requested() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), "HELLO\n").unwrap();

        let result = SearchTool
            .execute(
                r#"{"pattern": "hello", "case_insensitive": true}"#,
                &harness.turn(),
            )
            .await
            .unwrap();
        assert!(result.contains("HELLO"), "{result}");
    }

    #[tokio::test]
    async fn execute_errors_for_invalid_pattern() {
        let harness = TurnHarness::new();
        let result = SearchTool
            .execute(r#"{"pattern": "(unclosed"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_pattern() {
        let harness = TurnHarness::new();
        let result = SearchTool
            .execute(r#"{"path": "."}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pattern"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = SearchTool.execute("bad json", &harness.turn()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_rejects_path_outside_work_dir() {
        let harness = TurnHarness::new();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "needle\n").unwrap();

        let args = serde_json::json!({
            "pattern": "needle",
            "path": outside.path().to_str().unwrap(),
        })
        .to_string();
        let err = SearchTool
            .execute(&args, &harness.turn())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
    }
}
