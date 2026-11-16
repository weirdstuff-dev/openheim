//! Built-in tools for long-term memory: `remember`, `search_memory`,
//! `edit_memory`, and `forget`.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};
use crate::tools::ToolHandler;
use crate::tools::args::{NonEmptyString, parse};
use crate::tools::{ToolCapabilities, ToolKindHint};

use super::LongTermMemory;
use super::store::SearchMethod;

pub const REMEMBER_TOOL_NAME: &str = "remember";
pub const SEARCH_MEMORY_TOOL_NAME: &str = "search_memory";
pub const EDIT_MEMORY_TOOL_NAME: &str = "edit_memory";
pub const FORGET_TOOL_NAME: &str = "forget";

/// Hard ceiling on `top_k` a model may request, so one call can't dump the
/// whole store into the context window.
const MAX_TOP_K: usize = 20;

/// Longest note accepted, in characters. Memory is for facts and
/// preferences, not documents.
const MAX_NOTE_CHARS: usize = 4000;

/// Rejects a note over [`MAX_NOTE_CHARS`]; shared by `remember` and `edit_memory`.
fn check_note_len(content: &str) -> Result<()> {
    if content.chars().count() > MAX_NOTE_CHARS {
        return Err(Error::ToolExecutionError(format!(
            "note is longer than {MAX_NOTE_CHARS} characters; split it into smaller notes"
        )));
    }
    Ok(())
}

#[derive(Deserialize)]
struct RememberArgs {
    content: NonEmptyString,
}

#[derive(Deserialize)]
struct SearchMemoryArgs {
    query: NonEmptyString,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Deserialize)]
struct EditMemoryArgs {
    id: i64,
    content: NonEmptyString,
}

#[derive(Deserialize)]
struct ForgetArgs {
    id: i64,
}

/// Stores a note in long-term memory.
pub struct RememberTool {
    memory: Arc<LongTermMemory>,
}

impl RememberTool {
    pub fn new(memory: Arc<LongTermMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl ToolHandler for RememberTool {
    fn definition(&self) -> Tool {
        Tool::function(
            REMEMBER_TOOL_NAME,
            "Save a fact, preference, decision, or piece of context to long-term \
             memory so it can be recalled in future sessions with `search_memory`. \
             Use it when the user asks you to remember something, or states \
             something clearly worth keeping. Write one self-contained note per \
             call, in plain prose, including enough context to make sense on its own.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": format!("The note to remember (max {MAX_NOTE_CHARS} characters)")
                    }
                },
                "required": ["content"]
            }),
        )
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let args: RememberArgs = parse(args)?;
        check_note_len(&args.content)?;
        let record = self.memory.remember(&args.content).await?;
        Ok(format!("Remembered as memory #{}.", record.id))
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            kind: ToolKindHint::Think,
            ..Default::default()
        }
    }
}

/// Retrieves notes from long-term memory.
pub struct SearchMemoryTool {
    memory: Arc<LongTermMemory>,
}

impl SearchMemoryTool {
    pub fn new(memory: Arc<LongTermMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl ToolHandler for SearchMemoryTool {
    fn definition(&self) -> Tool {
        let how = if self.memory.is_semantic() {
            "Search is semantic: describe what you're looking for in natural language."
        } else {
            "Search is keyword-based: use the distinctive words the note is likely to contain."
        };
        Tool::function(
            SEARCH_MEMORY_TOOL_NAME,
            format!(
                "Search notes previously saved with `remember`. Use it when the user refers \
                 to something from an earlier session, asks what you remember, or when a \
                 stored preference or decision would change your answer. Returns the most \
                 relevant notes with their id and date. {how}"
            ),
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to recall"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": format!("Number of notes to return (default from config, max {MAX_TOP_K})"),
                        "minimum": 1,
                        "maximum": MAX_TOP_K
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let args: SearchMemoryArgs = parse(args)?;
        let top_k = args.top_k.map(|k| k.clamp(1, MAX_TOP_K));

        let found = self.memory.search_detailed(&args.query, top_k).await?;
        // How the search fell short of a full semantic search, if it did.
        let caveat = if found.semantic_failed {
            Some(
                "Semantic search is unavailable right now, so this was a keyword search; \
                 a note worded differently may not show up."
                    .to_string(),
            )
        } else if found.unindexed > 0 {
            Some(format!(
                "{} note(s) aren't in the semantic index yet and were searched by keyword only.",
                found.unindexed
            ))
        } else {
            None
        };
        let hits = found.hits;
        if hits.is_empty() {
            let stats = self.memory.stats().await?;
            if stats.memories == 0 {
                return Ok("Long-term memory is empty; nothing has been remembered yet.".into());
            }
            return Ok(match caveat {
                Some(caveat) => format!("No relevant memories found. ({caveat})"),
                None => "No relevant memories found.".to_string(),
            });
        }

        let mut out = format!("Found {} memory/memories:\n", hits.len());
        if let Some(caveat) = caveat {
            out.push_str(&format!("({caveat})\n"));
        }
        for hit in &hits {
            let score = match hit.method {
                SearchMethod::Semantic => format!("similarity {:.3}", hit.score),
                SearchMethod::Keyword => format!("rank {:.2}", hit.score),
            };
            out.push_str(&format!(
                "\n[#{}] {} ({score})\n{}\n",
                hit.record.id,
                hit.record.created_at.format("%Y-%m-%d"),
                hit.record.content.trim()
            ));
        }
        Ok(out)
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            kind: ToolKindHint::Search,
            ..Default::default()
        }
    }
}

/// Replaces the content of an existing note in long-term memory.
pub struct EditMemoryTool {
    memory: Arc<LongTermMemory>,
}

impl EditMemoryTool {
    pub fn new(memory: Arc<LongTermMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl ToolHandler for EditMemoryTool {
    fn definition(&self) -> Tool {
        Tool::function(
            EDIT_MEMORY_TOOL_NAME,
            "Replace the content of an existing note in long-term memory by its \
             id (the `#N` shown by `search_memory` or returned by `remember`). \
             Use it when a stored fact or preference has changed rather than \
             `forget`-ing it and `remember`-ing a new one. Search first if you \
             don't know the id.",
            json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Id of the memory to edit"
                    },
                    "content": {
                        "type": "string",
                        "description": format!("The note's new content (max {MAX_NOTE_CHARS} characters)")
                    }
                },
                "required": ["id", "content"]
            }),
        )
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let EditMemoryArgs { id, content } = parse(args)?;
        check_note_len(&content)?;
        match self.memory.edit(id, &content).await? {
            Some(_) => Ok(format!("Updated memory #{id}.")),
            None => Ok(format!("No memory #{id} exists; nothing to edit.")),
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            kind: ToolKindHint::Edit,
            ..Default::default()
        }
    }
}

/// Deletes a note from long-term memory by id.
pub struct ForgetTool {
    memory: Arc<LongTermMemory>,
}

impl ForgetTool {
    pub fn new(memory: Arc<LongTermMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl ToolHandler for ForgetTool {
    fn definition(&self) -> Tool {
        Tool::function(
            FORGET_TOOL_NAME,
            "Permanently delete a note from long-term memory by its id (the \
             `#N` shown by `search_memory` or returned by `remember`). Use it \
             when the user asks you to forget something or when a note is \
             outdated and being replaced. Search first if you don't know the id.",
            json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Id of the memory to delete"
                    }
                },
                "required": ["id"]
            }),
        )
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let ForgetArgs { id } = parse(args)?;
        if self.memory.forget(id).await? {
            Ok(format!("Forgot memory #{id}."))
        } else {
            Ok(format!("No memory #{id} exists; nothing to forget."))
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            kind: ToolKindHint::Delete,
            ..Default::default()
        }
    }
}
