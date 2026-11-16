//! Built-in tool: `search` — regex search across files, ripgrep-style.
//!
//! Built on ripgrep's crates rather than an `rg` binary, so it works without
//! `allow_shell` or anything installed on the host.

use std::path::Path;

use async_trait::async_trait;
use grep::regex::RegexMatcherBuilder;
use grep::searcher::sinks::UTF8;
use grep::searcher::{BinaryDetection, SearcherBuilder};
use ignore::WalkBuilder;
use serde::Deserialize;
use serde_json::json;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::parse;
use super::capabilities::{ToolCapabilities, ToolKindHint};

/// Matches beyond this count are omitted, with a marker noting the cut-off.
const MAX_RESULTS: usize = 200;

/// The walk stops after this many files, for a pattern that matches little
/// in a huge tree.
const MAX_FILES_SCANNED: usize = 50_000;

/// Searches every file under `root` for `pattern` (a regex) and returns
/// matches as `path:line: content`, in walk order. Follows ripgrep's
/// defaults (`.gitignore`-aware, hidden and binary files skipped). Runs on
/// the blocking pool, since the walk and matching are blocking.
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

        // A file that can't be read as text is skipped, not fatal.
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
/// Defaults to searching the turn's working directory (`TurnContext::cwd`)
/// if no path is given; the path must be inside the work directory.
pub struct SearchTool;

#[derive(Deserialize)]
struct SearchArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    /// `Option` rather than a defaulted `bool` so an explicit `null` (which
    /// models do send for optional flags) means "unset", not a parse error.
    #[serde(default)]
    case_insensitive: Option<bool>,
}

#[async_trait]
impl ToolHandler for SearchTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "search",
            "Search files under a path for lines matching a regex pattern (ripgrep-style). Respects .gitignore and skips hidden and binary files. Returns matches as 'path:line: content', capped at 200 results.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "The file or directory to search. Defaults to the current working directory if omitted."
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Match case-insensitively. Defaults to false."
                    }
                },
                "required": ["pattern"]
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args: SearchArgs = parse(args)?;
        let validated = turn.resolve_path(args.path.as_deref().unwrap_or("."))?;
        search(
            &args.pattern,
            &validated,
            args.case_insensitive.unwrap_or(false),
        )
        .await
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            kind: ToolKindHint::Search,
            ..Default::default()
        }
    }
}
