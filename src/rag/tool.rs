//! Built-in tools: `remember`, `search_memory`, and `forget` — the agent's
//! long-term memory, used only when it decides to (typically because the
//! user asked it to remember, recall, or drop something).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::core::models::{FunctionDefinition, Tool};
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};
use crate::tools::ToolHandler;
use crate::tools::args::parse_args;

use super::LongTermMemory;
use super::store::SearchMethod;

pub const REMEMBER_TOOL_NAME: &str = "remember";
pub const SEARCH_MEMORY_TOOL_NAME: &str = "search_memory";
pub const FORGET_TOOL_NAME: &str = "forget";

/// Hard ceiling on `top_k` a model may request, so one call can't dump the
/// whole store into the context window.
const MAX_TOP_K: usize = 20;

/// Longest note accepted, in characters. Memory is for facts and
/// preferences, not documents.
const MAX_NOTE_CHARS: usize = 4000;

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
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: REMEMBER_TOOL_NAME.to_string(),
                description: "Save a fact, preference, decision, or piece of context to long-term \
                              memory so it can be recalled in future sessions with `search_memory`. \
                              Use it when the user asks you to remember something, or states \
                              something clearly worth keeping. Write one self-contained note per \
                              call, in plain prose, including enough context to make sense on its own."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": format!("The note to remember (max {MAX_NOTE_CHARS} characters)")
                        }
                    },
                    "required": ["content"]
                }),
            },
        }
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let content = args["content"]
            .as_str()
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .ok_or_else(|| Error::ParseError("Missing 'content' argument".to_string()))?;
        if content.chars().count() > MAX_NOTE_CHARS {
            return Err(Error::ToolExecutionError(format!(
                "note is longer than {MAX_NOTE_CHARS} characters; split it into smaller notes"
            )));
        }
        let record = self.memory.remember(content).await?;
        Ok(format!("Remembered as memory #{}.", record.id))
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
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: SEARCH_MEMORY_TOOL_NAME.to_string(),
                description: format!(
                    "Search notes previously saved with `remember`. Use it when the user refers \
                     to something from an earlier session, asks what you remember, or when a \
                     stored preference or decision would change your answer. Returns the most \
                     relevant notes with their id and date. {how}"
                ),
                parameters: json!({
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
            },
        }
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let query = args["query"]
            .as_str()
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .ok_or_else(|| Error::ParseError("Missing 'query' argument".to_string()))?;
        let top_k = args["top_k"]
            .as_u64()
            .map(|k| (k as usize).clamp(1, MAX_TOP_K));

        let hits = self.memory.search(query, top_k).await?;
        if hits.is_empty() {
            let stats = self.memory.stats().await?;
            if stats.memories == 0 {
                return Ok("Long-term memory is empty; nothing has been remembered yet.".into());
            }
            return Ok("No relevant memories found.".to_string());
        }

        let mut out = format!("Found {} memory/memories:\n", hits.len());
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
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: FORGET_TOOL_NAME.to_string(),
                description: "Permanently delete a note from long-term memory by its id (the \
                              `#N` shown by `search_memory` or returned by `remember`). Use it \
                              when the user asks you to forget something or when a note is \
                              outdated and being replaced. Search first if you don't know the id."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "Id of the memory to delete"
                        }
                    },
                    "required": ["id"]
                }),
            },
        }
    }

    async fn execute(&self, args: &str, _turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let id = args["id"]
            .as_i64()
            .ok_or_else(|| Error::ParseError("Missing or non-integer 'id' argument".to_string()))?;
        if self.memory.forget(id).await? {
            Ok(format!("Forgot memory #{id}."))
        } else {
            Ok(format!("No memory #{id} exists; nothing to forget."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::embedding::test_support::HashEmbedder;
    use crate::rag::store::VectorStore;
    use crate::tools::test_support::TurnHarness;

    fn semantic() -> Arc<LongTermMemory> {
        Arc::new(LongTermMemory::new(
            VectorStore::open_in_memory().unwrap(),
            Some(Arc::new(HashEmbedder::new(32))),
            3,
        ))
    }

    fn keyword() -> Arc<LongTermMemory> {
        Arc::new(LongTermMemory::new(
            VectorStore::open_in_memory().unwrap(),
            None,
            3,
        ))
    }

    #[test]
    fn definitions_name_tools_and_required_args() {
        let m = semantic();
        let remember = RememberTool::new(m.clone()).definition();
        assert_eq!(remember.function.name, "remember");
        assert_eq!(remember.function.parameters["required"], json!(["content"]));
        let search = SearchMemoryTool::new(m.clone()).definition();
        assert_eq!(search.function.name, "search_memory");
        assert_eq!(search.function.parameters["required"], json!(["query"]));
        assert!(search.function.description.contains("semantic"));
        let forget = ForgetTool::new(m).definition();
        assert_eq!(forget.function.name, "forget");
        assert_eq!(forget.function.parameters["required"], json!(["id"]));

        let keyword_search = SearchMemoryTool::new(keyword()).definition();
        assert!(keyword_search.function.description.contains("keyword"));
    }

    #[tokio::test]
    async fn empty_memory_says_so() {
        let harness = TurnHarness::new();
        let turn = harness.turn();
        let tool = SearchMemoryTool::new(keyword());
        let out = tool
            .execute(r#"{"query":"anything"}"#, &turn)
            .await
            .unwrap();
        assert!(out.contains("empty"), "{out}");
    }

    #[tokio::test]
    async fn remember_then_search_round_trip_semantic() {
        let harness = TurnHarness::new();
        let turn = harness.turn();
        let m = semantic();
        let remember = RememberTool::new(m.clone());
        let out = remember
            .execute(
                r#"{"content":"The user prefers tabs over spaces in Go code."}"#,
                &turn,
            )
            .await
            .unwrap();
        assert!(out.starts_with("Remembered as memory #"), "{out}");
        remember
            .execute(r#"{"content":"Deploys happen on Tuesdays."}"#, &turn)
            .await
            .unwrap();

        let search = SearchMemoryTool::new(m);
        let out = search
            .execute(r#"{"query":"tabs or spaces preference","top_k":1}"#, &turn)
            .await
            .unwrap();
        assert!(out.starts_with("Found 1 memory"), "{out}");
        assert!(out.contains("tabs over spaces"));
        assert!(!out.contains("Tuesdays"));
        assert!(out.contains("(similarity "));
    }

    #[tokio::test]
    async fn remember_then_search_round_trip_keyword() {
        let harness = TurnHarness::new();
        let turn = harness.turn();
        let m = keyword();
        RememberTool::new(m.clone())
            .execute(
                r#"{"content":"The staging cluster is in eu-west-1."}"#,
                &turn,
            )
            .await
            .unwrap();
        let out = SearchMemoryTool::new(m)
            .execute(r#"{"query":"staging cluster region"}"#, &turn)
            .await
            .unwrap();
        assert!(out.contains("eu-west-1"), "{out}");
        assert!(out.contains("(rank "));
    }

    #[tokio::test]
    async fn remember_rejects_blank_and_oversized_notes() {
        let harness = TurnHarness::new();
        let turn = harness.turn();
        let tool = RememberTool::new(keyword());
        assert!(tool.execute(r#"{}"#, &turn).await.is_err());
        assert!(tool.execute(r#"{"content":"   "}"#, &turn).await.is_err());
        let huge = json!({ "content": "x".repeat(MAX_NOTE_CHARS + 1) }).to_string();
        assert!(tool.execute(&huge, &turn).await.is_err());
        assert!(tool.execute("nope", &turn).await.is_err());
    }

    #[tokio::test]
    async fn search_rejects_missing_query() {
        let harness = TurnHarness::new();
        let turn = harness.turn();
        let tool = SearchMemoryTool::new(keyword());
        assert!(tool.execute(r#"{}"#, &turn).await.is_err());
        assert!(tool.execute(r#"{"query":""}"#, &turn).await.is_err());
    }

    #[tokio::test]
    async fn forget_deletes_by_id_and_reports_unknown_ids() {
        let harness = TurnHarness::new();
        let turn = harness.turn();
        let m = keyword();
        let out = RememberTool::new(m.clone())
            .execute(r#"{"content":"Temporary fact."}"#, &turn)
            .await
            .unwrap();
        let id: i64 = out
            .trim_start_matches("Remembered as memory #")
            .trim_end_matches('.')
            .parse()
            .unwrap();

        let forget = ForgetTool::new(m.clone());
        let args = json!({ "id": id }).to_string();
        assert_eq!(
            forget.execute(&args, &turn).await.unwrap(),
            format!("Forgot memory #{id}.")
        );
        assert!(
            forget
                .execute(&args, &turn)
                .await
                .unwrap()
                .contains("nothing to forget")
        );
        assert_eq!(m.stats().await.unwrap().memories, 0);

        assert!(forget.execute(r#"{}"#, &turn).await.is_err());
        assert!(forget.execute(r#"{"id":"seven"}"#, &turn).await.is_err());
    }
}
