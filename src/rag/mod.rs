//! Long-term memory driven by tool calls, with retrieval that is keyword
//! search by default and semantic search when an embeddings provider is
//! configured.
//!
//! Nothing is stored or retrieved automatically. The agent gets four tools —
//! `remember`, `search_memory`, `edit_memory`, and `forget` ([`RememberTool`],
//! [`SearchMemoryTool`], [`EditMemoryTool`], [`ForgetTool`]) — and uses them
//! when the user asks it to keep, recall, correct, or drop something. Notes
//! live in a SQLite file
//! (`~/.openheim/memory.db`) with an FTS5 index, so memory works with zero
//! configuration and no network. Adding `embedding_provider` /
//! `embedding_model` to the `[memory]` config section upgrades
//! `search_memory` to nearest-neighbour search over embeddings stored with
//! the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension; notes
//! saved before that are back-filled over the next calls. When the
//! embeddings provider fails, notes are saved without a vector and searches
//! fall back to keywords, so memory keeps working through an outage.
//!
//! | Submodule | Responsibility |
//! |-----------|----------------|
//! | [`embedding`] | `EmbeddingClient` trait + OpenAI-compatible and Gemini implementations |
//! | [`store`]     | `VectorStore` — SQLite schema, FTS5 index, sqlite-vec table, both queries |
//! | [`tool`]      | The `remember` / `search_memory` / `edit_memory` / `forget` [`crate::tools::ToolHandler`]s |
//!
//! ```text
//! remember(content)          ──▶ [embed] ──▶ memories (+ memories_fts, + vec_memories)
//!                                                            ▲
//! search_memory(query)       ──▶ embedder configured? ──yes──▶ KNN (cosine)
//!                                               └───no───▶ FTS5 BM25
//! edit_memory(id, content)   ──▶ [embed] ──▶ replace note text (+ vector)
//! forget(id)                 ──▶ delete note (+ vector)
//! ```
//!
//! Conversation transcripts, skills, and the system identity are a different
//! kind of memory and live in [`crate::memory`].

pub mod embedding;
pub mod store;
pub mod tool;

use std::path::Path;
use std::sync::Arc;

use crate::config::{AppConfig, build_http_client};
use crate::error::Result;

pub use embedding::{EmbeddingClient, GeminiEmbeddingClient, OpenAiEmbeddingClient};
pub use store::{MemoryHit, MemoryRecord, SearchMethod, StoreStats, VectorStore};
pub use tool::{
    EDIT_MEMORY_TOOL_NAME, EditMemoryTool, FORGET_TOOL_NAME, ForgetTool, REMEMBER_TOOL_NAME,
    RememberTool, SEARCH_MEMORY_TOOL_NAME, SearchMemoryTool,
};

/// Default result count when the `[memory]` section doesn't set `top_k`.
pub const DEFAULT_TOP_K: usize = 5;

/// Most notes one memory call back-fills, so the first call after enabling
/// embeddings on a large store doesn't stall the turn. The rest follow on
/// later calls.
const MAX_BACKFILL: usize = 4 * embedding::MAX_BATCH;

/// A store plus an optional embedder.
///
/// Cheap to share behind an `Arc`; all methods take `&self`. Blocking
/// SQLite work runs on `spawn_blocking`, embedding calls are async HTTP.
pub struct LongTermMemory {
    store: Arc<VectorStore>,
    embedder: Option<Arc<dyn EmbeddingClient>>,
    top_k: usize,
}

impl LongTermMemory {
    /// `embedder` is `None` for keyword-only memory. `top_k` is the default
    /// result count for [`Self::search`].
    pub fn new(
        store: VectorStore,
        embedder: Option<Arc<dyn EmbeddingClient>>,
        top_k: usize,
    ) -> Self {
        Self {
            store: Arc::new(store),
            embedder,
            top_k,
        }
    }

    /// Builds the memory described by `config.memory` (all of it optional):
    /// opens `db_path`, or `memory.db` under `data_dir` (the resolved
    /// [`RuntimePaths::data_dir`](crate::config::RuntimePaths::data_dir)),
    /// and attaches an embedder when `embedding_provider` / `embedding_model`
    /// are set. Does not touch the network.
    pub fn from_config(config: &AppConfig, data_dir: &Path) -> Result<Self> {
        let memory = config.memory.as_ref();
        let db_path = match memory.and_then(|m| m.db_path.clone()) {
            Some(p) => p,
            None => data_dir.join("memory.db"),
        };
        let embedder = match config.resolve_embedding()? {
            Some(embedding) => {
                let http = build_http_client(embedding.timeout_secs)?;
                Some(embedding::create_embedding_client(&embedding, &http))
            }
            None => None,
        };
        // `top_k = 0` would make `search_memory` useless; treat it as unset.
        let top_k = memory.map_or(DEFAULT_TOP_K, |m| {
            if m.top_k == 0 { DEFAULT_TOP_K } else { m.top_k }
        });
        let store = VectorStore::open(&db_path)?;
        Ok(Self::new(store, embedder, top_k))
    }

    /// Whether `search_memory` is semantic (embedder configured) or keyword.
    pub fn is_semantic(&self) -> bool {
        self.embedder.is_some()
    }

    /// Embeds one text with `embedder`, makes sure the store's vector space
    /// matches it, and back-fills some of the notes that have no vector yet
    /// (see [`Self::backfill`]). Fails only if `text` itself couldn't be
    /// embedded.
    async fn embed_ready(&self, embedder: &dyn EmbeddingClient, text: &str) -> Result<Vec<f32>> {
        let mut vectors = embedding::embed_batched(embedder, &[text.to_string()]).await?;
        let vector = vectors.pop().ok_or_else(|| {
            crate::error::Error::ApiError("embeddings provider returned no vector".into())
        })?;
        let store = Arc::clone(&self.store);
        let model = embedder.model().to_string();
        let dims = vector.len();
        tokio::task::spawn_blocking(move || store.ensure_embedding_space(&model, dims)).await??;
        self.backfill(embedder).await;
        Ok(vector)
    }

    /// Embeds up to [`MAX_BACKFILL`] notes that have no vector yet: written
    /// before embeddings were enabled, orphaned by a model change, or saved
    /// while the embeddings provider was failing. A failure is logged, not
    /// returned; the notes stay keyword-searchable and a later call retries.
    async fn backfill(&self, embedder: &dyn EmbeddingClient) {
        if let Err(e) = self.try_backfill(embedder).await {
            tracing::warn!(error = %e, "back-filling memory embeddings failed; a later call retries");
        }
    }

    async fn try_backfill(&self, embedder: &dyn EmbeddingClient) -> Result<()> {
        let store = Arc::clone(&self.store);
        let pending =
            tokio::task::spawn_blocking(move || store.unembedded_up_to(MAX_BACKFILL)).await??;
        if pending.is_empty() {
            return Ok(());
        }
        tracing::info!(count = pending.len(), "embedding memories without vectors");
        let texts: Vec<String> = pending.iter().map(|r| r.content.clone()).collect();
        let vectors = embedding::embed_batched(embedder, &texts).await?;
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || {
            for (record, vector) in pending.iter().zip(vectors) {
                store.set_embedding(record.id, &vector)?;
            }
            Ok::<_, crate::error::Error>(())
        })
        .await?
    }

    /// The vector for `text` to store with it, or `None` without an embedder
    /// or when embedding fails. A note stored without one is embedded by a
    /// later back-fill.
    async fn vector_to_store(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        match self.embed_ready(embedder.as_ref(), text).await {
            Ok(vector) => Some(vector),
            Err(e) => {
                tracing::warn!(error = %e, "embedding a memory failed; storing it without a vector for now");
                None
            }
        }
    }

    /// Stores `content` and returns the new record. With an embedder, the
    /// note is embedded first; if that fails, it's stored without a vector
    /// (keyword-searchable at once, embedded by a later back-fill).
    pub async fn remember(&self, content: &str) -> Result<MemoryRecord> {
        let vector = self.vector_to_store(content).await;
        let store = Arc::clone(&self.store);
        let content = content.to_string();
        tokio::task::spawn_blocking(move || store.insert(&content, vector.as_deref())).await?
    }

    /// Returns the notes best matching `query`, best first — by embedding
    /// similarity when an embedder is configured, otherwise by FTS5 keyword
    /// rank. If the query can't be embedded, falls back to keyword search;
    /// [`MemoryHit::method`] says which was used. `top_k` defaults to the
    /// configured value.
    pub async fn search(&self, query: &str, top_k: Option<usize>) -> Result<Vec<MemoryHit>> {
        let k = top_k.unwrap_or(self.top_k);
        if k == 0 || query.trim().is_empty() {
            return Ok(vec![]);
        }
        let store = Arc::clone(&self.store);
        if let Some(embedder) = &self.embedder {
            match self.embed_ready(embedder.as_ref(), query).await {
                Ok(vector) => {
                    return tokio::task::spawn_blocking(move || store.search_semantic(&vector, k))
                        .await?;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "embedding a memory query failed; searching by keyword");
                }
            }
        }
        let query = query.to_string();
        tokio::task::spawn_blocking(move || store.search_keyword(&query, k)).await?
    }

    /// Deletes a note by id. Returns whether it existed.
    pub async fn forget(&self, id: i64) -> Result<bool> {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || store.delete(id)).await?
    }

    /// Replaces a note's content, keeping its id and creation date. With an
    /// embedder, the new content is embedded first; if that fails, the old
    /// vector is dropped and a later back-fill embeds the new text. Returns
    /// `None` if `id` doesn't exist.
    pub async fn edit(&self, id: i64, content: &str) -> Result<Option<MemoryRecord>> {
        let vector = self.vector_to_store(content).await;
        let store = Arc::clone(&self.store);
        let content = content.to_string();
        tokio::task::spawn_blocking(move || store.update(id, &content, vector.as_deref())).await?
    }

    pub async fn stats(&self) -> Result<StoreStats> {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || store.stats()).await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedding::test_support::{FlakyEmbedder, HashEmbedder};

    fn semantic(embedder: HashEmbedder) -> LongTermMemory {
        LongTermMemory::new(
            VectorStore::open_in_memory().unwrap(),
            Some(Arc::new(embedder)),
            2,
        )
    }

    fn keyword() -> LongTermMemory {
        LongTermMemory::new(VectorStore::open_in_memory().unwrap(), None, 2)
    }

    #[tokio::test]
    async fn keyword_memory_needs_no_embedder() {
        let m = keyword();
        assert!(!m.is_semantic());
        let a = m
            .remember("The staging cluster is in eu-west-1.")
            .await
            .unwrap();
        m.remember("Deploys happen on Tuesdays.").await.unwrap();

        let hits = m.search("staging cluster", None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, a.id);
        assert_eq!(hits[0].method, SearchMethod::Keyword);
        assert_eq!(m.stats().await.unwrap().dimensions, None);

        assert!(m.forget(a.id).await.unwrap());
        assert!(m.search("staging", None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn semantic_remember_then_search_end_to_end() {
        let m = semantic(HashEmbedder::new(32));
        assert!(m.is_semantic());
        let a = m.remember("Cats purr when content.").await.unwrap();
        m.remember("Compilers optimise loops.").await.unwrap();
        assert_eq!(m.stats().await.unwrap().memories, 2);

        let hits = m.search("purring cats", None).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record.id, a.id);
        assert_eq!(hits[0].method, SearchMethod::Semantic);

        let one = m.search("compilers", Some(1)).await.unwrap();
        assert_eq!(one.len(), 1);
        assert!(one[0].record.content.contains("Compilers"));

        assert!(m.forget(a.id).await.unwrap());
        assert_eq!(m.stats().await.unwrap().memories, 1);
    }

    #[tokio::test]
    async fn blank_query_or_zero_k_returns_nothing_without_embedding() {
        let m = semantic(HashEmbedder::new(8));
        assert!(m.search("   ", None).await.unwrap().is_empty());
        assert!(m.search("x", Some(0)).await.unwrap().is_empty());
        assert_eq!(m.stats().await.unwrap().dimensions, None);
    }

    #[tokio::test]
    async fn enabling_embeddings_later_backfills_keyword_era_notes() {
        let store = VectorStore::open_in_memory().unwrap();
        let plain = LongTermMemory::new(store, None, 5);
        plain.remember("alpha beta").await.unwrap();
        plain.remember("gamma delta").await.unwrap();

        let upgraded = LongTermMemory {
            store: Arc::clone(&plain.store),
            embedder: Some(Arc::new(HashEmbedder::new(16))),
            top_k: 5,
        };
        let hits = upgraded.search("gamma", Some(1)).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.content, "gamma delta");
        assert_eq!(hits[0].method, SearchMethod::Semantic);
        assert_eq!(upgraded.stats().await.unwrap().dimensions, Some(16));
    }

    #[tokio::test]
    async fn model_switch_reembeds_existing_notes() {
        let store = VectorStore::open_in_memory().unwrap();
        let first = LongTermMemory::new(store, Some(Arc::new(HashEmbedder::new(16))), 5);
        first.remember("alpha beta").await.unwrap();
        first.remember("gamma delta").await.unwrap();

        let second = LongTermMemory {
            store: Arc::clone(&first.store),
            embedder: Some(Arc::new(HashEmbedder {
                dims: 24,
                model: "hash-v2".into(),
            })),
            top_k: 5,
        };
        let hits = second.search("gamma", Some(1)).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.content, "gamma delta");
        assert_eq!(second.stats().await.unwrap().dimensions, Some(24));
        assert_eq!(second.stats().await.unwrap().memories, 2);
    }

    fn flaky(embedder: &Arc<FlakyEmbedder>) -> LongTermMemory {
        LongTermMemory::new(
            VectorStore::open_in_memory().unwrap(),
            Some(Arc::clone(embedder) as Arc<dyn EmbeddingClient>),
            5,
        )
    }

    #[tokio::test]
    async fn remember_and_search_work_while_embeddings_are_down() {
        let embedder = Arc::new(FlakyEmbedder::down(401));
        let m = flaky(&embedder);

        let saved = m
            .remember("The staging cluster is in eu-west-1.")
            .await
            .unwrap();
        m.remember("Deploys happen on Tuesdays.").await.unwrap();
        assert_eq!(m.stats().await.unwrap().memories, 2);

        let hits = m.search("staging cluster", None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, saved.id);
        assert_eq!(hits[0].method, SearchMethod::Keyword);
    }

    #[tokio::test]
    async fn edit_works_while_embeddings_are_down_and_drops_the_stale_vector() {
        let embedder = Arc::new(FlakyEmbedder::failing(401, 0));
        let m = flaky(&embedder);
        let note = m.remember("Deploys happen on Tuesdays.").await.unwrap();
        assert!(m.store.unembedded().unwrap().is_empty());

        embedder
            .failures
            .store(usize::MAX, std::sync::atomic::Ordering::SeqCst);
        let edited = m
            .edit(note.id, "Deploys happen on Thursdays.")
            .await
            .unwrap();
        assert_eq!(edited.unwrap().content, "Deploys happen on Thursdays.");
        // The old vector described the old text, so it's gone until a
        // back-fill embeds the new one.
        assert_eq!(m.store.unembedded().unwrap()[0].id, note.id);
    }

    #[tokio::test]
    async fn notes_saved_during_an_outage_are_embedded_once_it_ends() {
        let embedder = Arc::new(FlakyEmbedder::down(401));
        let m = flaky(&embedder);
        m.remember("alpha beta").await.unwrap();
        m.remember("gamma delta").await.unwrap();

        embedder
            .failures
            .store(0, std::sync::atomic::Ordering::SeqCst);
        let hits = m.search("gamma", Some(1)).await.unwrap();
        assert_eq!(hits[0].method, SearchMethod::Semantic);
        assert_eq!(hits[0].record.content, "gamma delta");
        assert!(m.store.unembedded().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_failing_backfill_does_not_fail_the_call() {
        let store = VectorStore::open_in_memory().unwrap();
        store.insert("poison note", None).unwrap();
        let embedder = Arc::new(FlakyEmbedder::poisoned("poison"));
        let m = LongTermMemory::new(
            store,
            Some(Arc::clone(&embedder) as Arc<dyn EmbeddingClient>),
            5,
        );

        // The back-fill of "poison note" fails on every call; the calls
        // themselves still embed and succeed.
        let saved = m.remember("healthy note").await.unwrap();
        let hits = m.search("healthy", Some(1)).await.unwrap();
        assert_eq!(hits[0].record.id, saved.id);
        assert_eq!(hits[0].method, SearchMethod::Semantic);
        assert_eq!(m.store.unembedded().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn one_call_backfills_at_most_max_backfill_notes() {
        let store = VectorStore::open_in_memory().unwrap();
        let total = MAX_BACKFILL + 10;
        for i in 0..total {
            store.insert(&format!("note {i}"), None).unwrap();
        }
        let embedder = Arc::new(FlakyEmbedder::failing(503, 0));
        let m = LongTermMemory::new(
            store,
            Some(Arc::clone(&embedder) as Arc<dyn EmbeddingClient>),
            5,
        );

        m.search("note", None).await.unwrap();
        // The query itself, plus the capped back-fill.
        assert_eq!(embedder.embedded(), 1 + MAX_BACKFILL);
        assert_eq!(m.store.unembedded().unwrap().len(), 10);

        m.search("note", None).await.unwrap();
        assert!(m.store.unembedded().unwrap().is_empty());
    }

    #[test]
    fn config_without_memory_section_means_keyword_only() {
        let config: AppConfig = toml::from_str(
            r#"
            default_provider = "openai"
            [providers.openai]
            api_base = "https://api.openai.com/v1"
            default_model = "gpt-4o"
            models = ["gpt-4o"]
        "#,
        )
        .unwrap();
        assert!(config.resolve_embedding().unwrap().is_none());
        assert_eq!(config.memory.as_ref().map_or(DEFAULT_TOP_K, |m| m.top_k), 5);
    }

    #[test]
    fn from_config_clamps_explicit_zero_top_k_to_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("memory.db");
        let config: AppConfig = toml::from_str(&format!(
            r#"
            default_provider = "openai"
            [providers.openai]
            api_base = "https://api.openai.com/v1"
            default_model = "gpt-4o"
            models = ["gpt-4o"]
            [memory]
            db_path = "{}"
            top_k = 0
        "#,
            db_path.display()
        ))
        .unwrap();

        let memory = LongTermMemory::from_config(&config, dir.path()).unwrap();
        assert_eq!(memory.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn from_config_defaults_the_database_to_the_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config: AppConfig = toml::from_str(
            r#"
            default_provider = "openai"
            [providers.openai]
            api_base = "https://api.openai.com/v1"
            default_model = "gpt-4o"
            models = ["gpt-4o"]
        "#,
        )
        .unwrap();

        LongTermMemory::from_config(&config, dir.path()).unwrap();
        assert!(dir.path().join("memory.db").exists());
    }

    #[test]
    fn from_config_preserves_a_configured_positive_top_k() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("memory.db");
        let config: AppConfig = toml::from_str(&format!(
            r#"
            default_provider = "openai"
            [providers.openai]
            api_base = "https://api.openai.com/v1"
            default_model = "gpt-4o"
            models = ["gpt-4o"]
            [memory]
            db_path = "{}"
            top_k = 7
        "#,
            db_path.display()
        ))
        .unwrap();

        let memory = LongTermMemory::from_config(&config, dir.path()).unwrap();
        assert_eq!(memory.top_k, 7);
    }
}
