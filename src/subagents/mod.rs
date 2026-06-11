//! Subagent profiles: named, user-defined agent personas that the orchestrating
//! agent can delegate self-contained tasks to via the `delegate_task` tool
//! (see [`crate::tools::delegate`]).
//!
//! A profile is a Markdown file in `~/.openheim/agents/{name}.md`, discovered the
//! same way [`crate::rag::SkillsManager`] discovers skill files. The file may start
//! with a `+++`-delimited TOML frontmatter block describing the profile; the rest
//! of the file is used verbatim as the subagent's system prompt.
//!
//! ```markdown
//! +++
//! description = "Reviews code changes for correctness, security, and style."
//! model = "claude-haiku-4-5"
//! tools = ["read_file"]
//! max_iterations = 6
//! +++
//! You are a meticulous code reviewer focused on correctness and idiomatic style.
//! ```
//!
//! All frontmatter fields are optional. A file with no frontmatter is treated as a
//! profile with an empty description whose entire contents form the system prompt.

use std::path::PathBuf;

use serde::Deserialize;

use crate::config::config_dir;
use crate::error::{Error, Result};

/// A user-defined subagent persona loaded from `~/.openheim/agents/{name}.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentProfile {
    /// Derived from the filename (without the `.md` extension).
    pub name: String,
    /// One-line summary shown to the orchestrating LLM so it knows when to delegate here.
    pub description: String,
    /// Overrides the parent's model when set (paired with `provider`).
    pub model: Option<String>,
    /// Overrides the parent's provider when set (paired with `model`).
    pub provider: Option<String>,
    /// Restricts the subagent to this set of tool names. `None` inherits the parent's tools.
    pub tools: Option<Vec<String>>,
    /// Caps the subagent's agent-loop iterations. `None` inherits the parent's setting.
    pub max_iterations: Option<usize>,
    /// The subagent's system prompt — the body of the Markdown file.
    pub system_prompt: String,
}

/// Deserialized shape of a profile's `+++` frontmatter block. All fields optional.
#[derive(Debug, Default, Deserialize)]
struct AgentProfileMeta {
    #[serde(default)]
    description: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    max_iterations: Option<usize>,
}

/// Discovers and loads [`AgentProfile`]s from `~/.openheim/agents/`.
///
/// Mirrors [`crate::rag::SkillsManager`]: a profile is named after its file
/// (`{name}.md`), and the directory is created on first use if missing.
#[derive(Clone)]
pub struct SubagentLoader {
    agents_dir: PathBuf,
}

impl SubagentLoader {
    /// Creates a `SubagentLoader` backed by `~/.openheim/agents/`, creating the
    /// directory if it doesn't exist.
    pub fn new() -> Result<Self> {
        let dir = config_dir()?.join("agents");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { agents_dir: dir })
    }

    /// Loads every valid `.md` profile in the agents directory, sorted by name.
    ///
    /// A file whose frontmatter fails to parse is skipped with a warning rather
    /// than failing the whole load — one malformed profile shouldn't prevent the
    /// agent from starting.
    pub fn load(&self) -> Result<Vec<AgentProfile>> {
        let mut profiles = Vec::new();
        for entry in std::fs::read_dir(&self.agents_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        file = %path.display(),
                        error = %e,
                        "Skipping unreadable subagent profile"
                    );
                    continue;
                }
            };
            match parse_profile(name, &content) {
                Ok(profile) => profiles.push(profile),
                Err(e) => tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "Skipping invalid subagent profile"
                ),
            }
        }
        profiles.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(profiles)
    }
}

/// Parses a profile file's content into an [`AgentProfile`] named `name`.
fn parse_profile(name: &str, content: &str) -> Result<AgentProfile> {
    let (meta, body) = split_frontmatter(content)?;
    Ok(AgentProfile {
        name: name.to_string(),
        description: meta.description,
        model: meta.model,
        provider: meta.provider,
        tools: meta.tools,
        max_iterations: meta.max_iterations,
        system_prompt: body.trim().to_string(),
    })
}

/// Splits an optional `+++`-delimited TOML frontmatter block from the Markdown body.
///
/// Files that don't start with `+++` have no frontmatter: the whole content becomes
/// the system prompt and the metadata defaults to empty. An opening `+++` with no
/// matching closing delimiter is a parse error.
fn split_frontmatter(content: &str) -> Result<(AgentProfileMeta, &str)> {
    let trimmed = content.trim_start();
    let Some(rest) = trimmed.strip_prefix("+++") else {
        return Ok((AgentProfileMeta::default(), content));
    };
    // Skip to the end of the opening delimiter's line.
    let rest = rest.strip_prefix('\n').unwrap_or(rest);
    let end = rest
        .find("\n+++")
        .ok_or_else(|| Error::ParseError("unterminated '+++' frontmatter block".into()))?;
    let frontmatter = &rest[..end];
    // `rest[end..]` starts with the `\n` before the closing delimiter, e.g.
    // "\n+++\nBody...". Skip past that line entirely to reach the body.
    let after_delim = &rest[end + 1..];
    let body = match after_delim.find('\n') {
        Some(i) => &after_delim[i + 1..],
        None => "",
    };
    let meta: AgentProfileMeta = toml::from_str(frontmatter)
        .map_err(|e| Error::ParseError(format!("invalid frontmatter: {e}")))?;
    Ok((meta, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_profile_with_frontmatter() {
        let content = "+++\n\
description = \"Reviews diffs for correctness and style.\"\n\
model = \"claude-haiku-4-5\"\n\
tools = [\"read_file\"]\n\
max_iterations = 6\n\
+++\n\
You are a meticulous code reviewer.\n";

        let profile = parse_profile("code-reviewer", content).unwrap();
        assert_eq!(profile.name, "code-reviewer");
        assert_eq!(
            profile.description,
            "Reviews diffs for correctness and style."
        );
        assert_eq!(profile.model.as_deref(), Some("claude-haiku-4-5"));
        assert_eq!(profile.provider, None);
        assert_eq!(profile.tools, Some(vec!["read_file".to_string()]));
        assert_eq!(profile.max_iterations, Some(6));
        assert_eq!(profile.system_prompt, "You are a meticulous code reviewer.");
    }

    #[test]
    fn treats_file_without_frontmatter_as_plain_prompt() {
        let content = "You are a helpful research assistant.\nAlways cite sources.";
        let profile = parse_profile("researcher", content).unwrap();
        assert_eq!(profile.name, "researcher");
        assert_eq!(profile.description, "");
        assert_eq!(profile.model, None);
        assert_eq!(profile.tools, None);
        assert_eq!(
            profile.system_prompt,
            "You are a helpful research assistant.\nAlways cite sources."
        );
    }

    #[test]
    fn rejects_unterminated_frontmatter() {
        let content = "+++\ndescription = \"oops\"\nNo closing delimiter here.";
        let err = parse_profile("broken", content).unwrap_err();
        assert!(err.to_string().contains("unterminated"));
    }

    #[test]
    fn rejects_malformed_frontmatter_toml() {
        let content = "+++\nthis is not valid toml :::\n+++\nBody.";
        let err = parse_profile("broken", content).unwrap_err();
        assert!(err.to_string().contains("invalid frontmatter"));
    }

    #[test]
    fn defaults_are_applied_for_partial_frontmatter() {
        let content = "+++\ndescription = \"Just a description.\"\n+++\nPrompt body.";
        let profile = parse_profile("partial", content).unwrap();
        assert_eq!(profile.description, "Just a description.");
        assert_eq!(profile.model, None);
        assert_eq!(profile.provider, None);
        assert_eq!(profile.tools, None);
        assert_eq!(profile.max_iterations, None);
        assert_eq!(profile.system_prompt, "Prompt body.");
    }
}
