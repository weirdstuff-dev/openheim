//! SQLite persistence for long-term memory: FTS5 keyword search always, plus
//! sqlite-vec semantic search when vectors are available.
//!
//! Schema (one file, `~/.openheim/memory.db` by default):
//!
//! | Table | Purpose |
//! |-------|---------|
//! | `meta` | key/value: the embedding `model` and vector `dimensions` this store was built with |
//! | `memories` | one row per remembered note — text and creation time |
//! | `memories_fts` | FTS5 index over `memories.content`, kept in sync by triggers |
//! | `vec_memories` | `vec0` virtual table keyed by memory id, cosine distance (only once an embedder is configured) |
//!
//! The `vec0` table needs a fixed dimension at creation time, which is only
//! known once the first embedding comes back — so it's created lazily by
//! [`VectorStore::ensure_embedding_space`]. If the model or dimension later
//! differs from what's recorded, the vectors are dropped and re-embedded from
//! the stored text (see [`VectorStore::unembedded`]): vectors from two
//! models aren't comparable, and a silent mix would return nonsense. Notes
//! written while no embedder was configured show up in `unembedded` too, so
//! enabling embeddings later back-fills them.
//!
//! [`rusqlite::Connection`] is `!Sync`, so it sits behind a `Mutex`; every
//! method here is blocking and callers run them on `spawn_blocking`.

use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::sync::{Mutex, Once};

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};

use crate::error::{Error, Result};

static REGISTER_SQLITE_VEC: Once = Once::new();

/// Registers sqlite-vec as an auto-extension so every connection opened
/// afterwards has `vec0` and the `vec_*` functions. Idempotent.
fn register_sqlite_vec() {
    REGISTER_SQLITE_VEC.call_once(|| {
        type InitFn = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> c_int;
        // SAFETY: `sqlite3_vec_init` is the extension's C entry point with
        // exactly the `sqlite3_loadext_entry` signature `InitFn` spells out;
        // the `sqlite-vec` crate only declares it with an opaque type, hence
        // the transmute. `sqlite3_auto_extension` merely records the pointer
        // for future connections, and this runs once, before any connection
        // is opened — the pattern the sqlite-vec docs prescribe for Rust.
        unsafe {
            let init: InitFn = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            rusqlite::ffi::sqlite3_auto_extension(Some(init));
        }
    });
}

/// One remembered note.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecord {
    pub id: i64,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

/// How a hit was found; determines what [`MemoryHit::score`] means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMethod {
    /// Embedding nearest-neighbour; `score` is cosine similarity in `[-1, 1]`.
    Semantic,
    /// FTS5 BM25 keyword match; `score` is the negated BM25 rank (higher is better, unbounded).
    Keyword,
}

/// One retrieved note.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryHit {
    pub record: MemoryRecord,
    /// Higher is better; scale depends on `method`.
    pub score: f32,
    pub method: SearchMethod,
}

/// Store size summary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreStats {
    pub memories: usize,
    /// `None` until an embedder has fixed the vector space.
    pub dimensions: Option<usize>,
}

/// Blocking handle to the memory SQLite file.
pub struct VectorStore {
    conn: Mutex<Connection>,
}

impl VectorStore {
    /// Opens (creating if needed) the store at `path`, including parent
    /// directories.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        register_sqlite_vec();
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// A throwaway in-memory store; used by tests and handy for embedders
    /// that want ephemeral memory.
    pub fn open_in_memory() -> Result<Self> {
        register_sqlite_vec();
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS meta (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS memories (
                 id         INTEGER PRIMARY KEY,
                 content    TEXT NOT NULL,
                 created_at TEXT NOT NULL
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                 content,
                 content='memories',
                 content_rowid='id',
                 tokenize='unicode61 remove_diacritics 2'
             );
             CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                 INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
             END;
             CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                 VALUES ('delete', old.id, old.content);
             END;
             CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                 VALUES ('delete', old.id, old.content);
                 INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
             END;",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| Error::DatabaseError("memory store mutex poisoned".into()))
    }

    fn meta(conn: &Connection, key: &str) -> Result<Option<String>> {
        Ok(conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    fn has_vec_table(conn: &Connection) -> Result<bool> {
        let n: i64 = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'vec_memories'",
            [],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Makes sure the store's vector space is `(model, dimensions)`.
    ///
    /// Creates the `vec0` table on first use. If the store was built with a
    /// different model or dimension, the vectors are dropped (the note text
    /// is kept) and `Ok(true)` is returned; the caller should then re-embed
    /// everything [`Self::unembedded`] reports.
    pub fn ensure_embedding_space(&self, model: &str, dimensions: usize) -> Result<bool> {
        if dimensions == 0 {
            return Err(Error::ApiError(
                "embeddings provider returned an empty vector".into(),
            ));
        }
        let mut conn = self.lock()?;
        let stored_model = Self::meta(&conn, "model")?;
        let stored_dims = Self::meta(&conn, "dimensions")?.and_then(|d| d.parse::<usize>().ok());
        let matches = stored_model.as_deref() == Some(model) && stored_dims == Some(dimensions);
        if matches && Self::has_vec_table(&conn)? {
            return Ok(false);
        }
        let reset = stored_model.is_some() || stored_dims.is_some();
        if reset {
            tracing::warn!(
                old_model = ?stored_model,
                old_dimensions = ?stored_dims,
                new_model = model,
                new_dimensions = dimensions,
                "embedding space changed; memories will be re-embedded"
            );
        }
        let tx = conn.transaction()?;
        tx.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
        tx.execute_batch(&format!(
            "CREATE VIRTUAL TABLE vec_memories USING vec0(
                 memory_id INTEGER PRIMARY KEY,
                 embedding FLOAT[{dimensions}] distance_metric=cosine
             );"
        ))?;
        Self::set_meta(&tx, "model", model)?;
        Self::set_meta(&tx, "dimensions", &dimensions.to_string())?;
        tx.commit()?;
        Ok(reset)
    }

    /// Notes that have text but no vector — written before an embedder was
    /// configured, after a model switch, or if a crash landed between the
    /// two inserts. Empty when everything is embedded.
    pub fn unembedded(&self) -> Result<Vec<MemoryRecord>> {
        let conn = self.lock()?;
        if !Self::has_vec_table(&conn)? {
            return Self::all_records(&conn);
        }
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.created_at FROM memories m
             WHERE NOT EXISTS (SELECT 1 FROM vec_memories v WHERE v.memory_id = m.id)
             ORDER BY m.id",
        )?;
        let rows = stmt.query_map([], row_to_record)?;
        rows.map(|r| r.map_err(Error::from)).collect()
    }

    fn all_records(conn: &Connection) -> Result<Vec<MemoryRecord>> {
        let mut stmt = conn.prepare("SELECT id, content, created_at FROM memories ORDER BY id")?;
        let rows = stmt.query_map([], row_to_record)?;
        rows.map(|r| r.map_err(Error::from)).collect()
    }

    /// Stores a note, with its embedding when one is available, and returns
    /// the new record. Passing a vector requires
    /// [`Self::ensure_embedding_space`] to have run.
    pub fn insert(&self, content: &str, embedding: Option<&[f32]>) -> Result<MemoryRecord> {
        let mut conn = self.lock()?;
        if embedding.is_some() && !Self::has_vec_table(&conn)? {
            return Err(Error::DatabaseError(
                "vector table not initialised; call ensure_embedding_space first".into(),
            ));
        }
        let created_at = Utc::now();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO memories (content, created_at) VALUES (?1, ?2)",
            params![content, created_at.to_rfc3339()],
        )?;
        let id = tx.last_insert_rowid();
        if let Some(embedding) = embedding {
            tx.execute(
                "INSERT INTO vec_memories (memory_id, embedding) VALUES (?1, ?2)",
                params![id, vector_blob(embedding)],
            )?;
        }
        tx.commit()?;
        Ok(MemoryRecord {
            id,
            content: content.to_string(),
            created_at,
        })
    }

    /// (Re-)writes the vector for an existing note.
    pub fn set_embedding(&self, id: i64, embedding: &[f32]) -> Result<()> {
        let conn = self.lock()?;
        conn.execute("DELETE FROM vec_memories WHERE memory_id = ?1", [id])?;
        conn.execute(
            "INSERT INTO vec_memories (memory_id, embedding) VALUES (?1, ?2)",
            params![id, vector_blob(embedding)],
        )?;
        Ok(())
    }

    /// Removes a note and its vector. Returns whether it existed.
    pub fn delete(&self, id: i64) -> Result<bool> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        if Self::has_vec_table(&tx)? {
            tx.execute("DELETE FROM vec_memories WHERE memory_id = ?1", [id])?;
        }
        let removed = tx.execute("DELETE FROM memories WHERE id = ?1", [id])?;
        tx.commit()?;
        Ok(removed > 0)
    }

    /// The `k` notes nearest to `query` by cosine distance, closest first.
    /// Empty when no vectors exist yet.
    pub fn search_semantic(&self, query: &[f32], k: usize) -> Result<Vec<MemoryHit>> {
        let conn = self.lock()?;
        if k == 0 || !Self::has_vec_table(&conn)? {
            return Ok(vec![]);
        }
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.created_at, knn.distance
             FROM (
                 SELECT memory_id, distance
                 FROM vec_memories
                 WHERE embedding MATCH ?1 AND k = ?2
             ) AS knn
             JOIN memories m ON m.id = knn.memory_id
             ORDER BY knn.distance",
        )?;
        let rows = stmt.query_map(params![vector_blob(query), k as i64], |r| {
            Ok(MemoryHit {
                record: row_to_record(r)?,
                score: 1.0 - r.get::<_, f64>(3)? as f32,
                method: SearchMethod::Semantic,
            })
        })?;
        rows.map(|r| r.map_err(Error::from)).collect()
    }

    /// The `k` notes best matching `query`'s words by BM25, best first. Any
    /// word matching counts (OR semantics), so a partially matching note is
    /// still found; ranking prefers notes matching more, rarer words.
    pub fn search_keyword(&self, query: &str, k: usize) -> Result<Vec<MemoryHit>> {
        let Some(match_expr) = fts_query(query) else {
            return Ok(vec![]);
        };
        if k == 0 {
            return Ok(vec![]);
        }
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.created_at, bm25(memories_fts) AS rank
             FROM memories_fts
             JOIN memories m ON m.id = memories_fts.rowid
             WHERE memories_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![match_expr, k as i64], |r| {
            Ok(MemoryHit {
                record: row_to_record(r)?,
                score: -(r.get::<_, f64>(3)? as f32),
                method: SearchMethod::Keyword,
            })
        })?;
        rows.map(|r| r.map_err(Error::from)).collect()
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let conn = self.lock()?;
        let memories: i64 = conn.query_row("SELECT count(*) FROM memories", [], |r| r.get(0))?;
        let dimensions = Self::meta(&conn, "dimensions")?.and_then(|d| d.parse().ok());
        Ok(StoreStats {
            memories: memories as usize,
            dimensions,
        })
    }
}

/// Turns free text into a safe FTS5 expression: each alphanumeric word is
/// double-quoted (so `-`, `*`, `AND`, quotes, … in user input can't change
/// the query's meaning) and the words are OR-ed. `None` when there are no
/// words to search for.
fn fts_query(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{w}\""))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

fn row_to_record(r: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let created_raw: String = r.get(2)?;
    let created_at = DateTime::parse_from_rfc3339(&created_raw)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| DateTime::<Utc>::UNIX_EPOCH);
    Ok(MemoryRecord {
        id: r.get(0)?,
        content: r.get(1)?,
        created_at,
    })
}

/// sqlite-vec's `float[N]` columns take raw little-endian `f32` bytes.
fn vector_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dir: usize, dims: usize) -> Vec<f32> {
        let mut v = vec![0f32; dims];
        v[dir] = 1.0;
        v
    }

    #[test]
    fn open_in_memory_has_sqlite_vec_and_fts5() {
        let store = VectorStore::open_in_memory().unwrap();
        let conn = store.lock().unwrap();
        let version: String = conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .unwrap();
        assert!(version.starts_with('v'), "{version}");
        let fts: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'memories_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(fts, 1);
    }

    #[test]
    fn keyword_search_works_without_any_vectors() {
        let store = VectorStore::open_in_memory().unwrap();
        let a = store
            .insert("The staging cluster lives in eu-west-1.", None)
            .unwrap();
        store.insert("Deploys happen on Tuesdays.", None).unwrap();
        store
            .insert("Café menus use accented words like résumé.", None)
            .unwrap();

        let hits = store
            .search_keyword("where is the staging cluster?", 5)
            .unwrap();
        assert_eq!(hits[0].record.id, a.id);
        assert_eq!(hits[0].method, SearchMethod::Keyword);
        assert!(hits[0].score > 0.0);
        assert!(hits.iter().all(|h| !h.record.content.contains("Tuesdays")));

        // Diacritics are folded; hostile syntax is neutralised.
        assert_eq!(store.search_keyword("resume", 5).unwrap().len(), 1);
        assert!(
            store
                .search_keyword("\"unbalanced AND - *", 5)
                .unwrap()
                .is_empty()
        );
        assert!(store.search_keyword("   ", 5).unwrap().is_empty());
        assert!(store.search_keyword("staging", 0).unwrap().is_empty());
        assert_eq!(store.stats().unwrap().dimensions, None);
    }

    #[test]
    fn fts_query_quotes_and_ors_words() {
        assert_eq!(
            fts_query("a b-c").as_deref(),
            Some("\"a\" OR \"b\" OR \"c\"")
        );
        assert_eq!(fts_query("!!!"), None);
    }

    #[test]
    fn delete_keeps_fts_in_sync() {
        let store = VectorStore::open_in_memory().unwrap();
        let rec = store.insert("ephemeral note", None).unwrap();
        assert_eq!(store.search_keyword("ephemeral", 5).unwrap().len(), 1);
        assert!(store.delete(rec.id).unwrap());
        assert!(!store.delete(rec.id).unwrap());
        assert!(store.search_keyword("ephemeral", 5).unwrap().is_empty());
        assert_eq!(store.stats().unwrap().memories, 0);
    }

    #[test]
    fn ensure_embedding_space_creates_then_is_stable() {
        let store = VectorStore::open_in_memory().unwrap();
        assert!(!store.ensure_embedding_space("m", 3).unwrap());
        assert!(!store.ensure_embedding_space("m", 3).unwrap());
        assert_eq!(store.stats().unwrap().dimensions, Some(3));
    }

    #[test]
    fn notes_written_before_embeddings_are_reported_unembedded() {
        let store = VectorStore::open_in_memory().unwrap();
        let early = store.insert("pre-embedding note", None).unwrap();
        assert_eq!(store.unembedded().unwrap().len(), 1);
        store.ensure_embedding_space("m", 2).unwrap();
        assert_eq!(store.unembedded().unwrap()[0].id, early.id);
        store.set_embedding(early.id, &unit(0, 2)).unwrap();
        assert!(store.unembedded().unwrap().is_empty());
        assert_eq!(
            store.search_semantic(&unit(0, 2), 1).unwrap()[0].record.id,
            early.id
        );
    }

    #[test]
    fn changing_model_keeps_text_but_drops_vectors() {
        let store = VectorStore::open_in_memory().unwrap();
        store.ensure_embedding_space("m", 3).unwrap();
        let rec = store.insert("note", Some(&unit(0, 3))).unwrap();
        assert!(store.unembedded().unwrap().is_empty());

        assert!(store.ensure_embedding_space("other", 4).unwrap());
        assert_eq!(store.stats().unwrap().memories, 1);
        assert_eq!(store.stats().unwrap().dimensions, Some(4));
        assert!(store.search_semantic(&unit(0, 4), 5).unwrap().is_empty());
        assert_eq!(store.search_keyword("note", 5).unwrap().len(), 1);
        let pending = store.unembedded().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, rec.id);
    }

    #[test]
    fn insert_with_vector_before_ensure_is_an_error() {
        let store = VectorStore::open_in_memory().unwrap();
        let err = store.insert("t", Some(&unit(0, 2))).unwrap_err();
        assert!(matches!(err, Error::DatabaseError(_)));
    }

    #[test]
    fn semantic_search_returns_nearest_first() {
        let store = VectorStore::open_in_memory().unwrap();
        store.ensure_embedding_space("m", 3).unwrap();
        let a = store.insert("a", Some(&unit(0, 3))).unwrap();
        let b = store.insert("b", Some(&unit(1, 3))).unwrap();
        store.insert("c", Some(&unit(2, 3))).unwrap();

        let hits = store.search_semantic(&[0.1, 0.9, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record.id, b.id);
        assert_eq!(hits[1].record.id, a.id);
        assert!(hits[0].score > hits[1].score);
        assert!(hits[0].score > 0.9);
        assert_eq!(hits[0].method, SearchMethod::Semantic);
        assert!(hits[0].record.created_at <= Utc::now());
    }

    #[test]
    fn semantic_search_on_empty_store_is_empty() {
        let store = VectorStore::open_in_memory().unwrap();
        assert!(store.search_semantic(&[1.0], 5).unwrap().is_empty());
        store.ensure_embedding_space("m", 1).unwrap();
        assert!(store.search_semantic(&[1.0], 5).unwrap().is_empty());
        assert!(store.search_semantic(&[1.0], 0).unwrap().is_empty());
    }

    #[test]
    fn open_creates_parent_directories_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("memory.db");
        {
            let store = VectorStore::open(&path).unwrap();
            store.ensure_embedding_space("m", 2).unwrap();
            store.insert("kept", Some(&unit(0, 2))).unwrap();
        }
        let reopened = VectorStore::open(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().memories, 1);
        assert!(!reopened.ensure_embedding_space("m", 2).unwrap());
        assert_eq!(
            reopened.search_semantic(&unit(0, 2), 1).unwrap()[0]
                .record
                .content,
            "kept"
        );
        assert_eq!(reopened.search_keyword("kept", 1).unwrap().len(), 1);
    }
}
