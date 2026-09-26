use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::models::{Message, Role, Usage};
use crate::error::{Error, Result};
use crate::memory::lease::{self, SessionLease};
use std::path::PathBuf;

/// Persistent metadata for a conversation, stored in its `.json` file (see
/// [`HistoryManager`] for the on-disk layout).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// The first 80 characters of the first user message (see
    /// [`Self::fill_title_from`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Names of skills active in this conversation (correspond to `~/.openheim/skills/*.md`).
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cwd: Option<std::path::PathBuf>,
    /// The last LLM call's usage, i.e. how full the context window is now.
    /// `None` until the first turn completes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub context_usage: Option<Usage>,
}

impl ConversationMeta {
    /// Metadata for a brand-new conversation: created and updated now, no
    /// title, cwd or context usage yet.
    pub fn new(
        id: Uuid,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            created_at: now,
            updated_at: now,
            model,
            provider,
            title: None,
            skills,
            cwd: None,
            context_usage: None,
        }
    }

    /// Sets `title` from `message` (its first 80 characters) if there is no
    /// title yet and `message` is a user message with text.
    pub fn fill_title_from(&mut self, message: &Message) {
        if self.title.is_none()
            && message.role == Role::User
            && let Some(text) = message.text()
        {
            self.title = Some(text.chars().take(80).collect());
        }
    }
}

/// A complete conversation: metadata plus the full ordered message list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub meta: ConversationMeta,
    pub messages: Vec<Message>,
}

/// On-disk (de)serialization shape for a conversation's `{id}.json` meta file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConversationEnvelope {
    meta: ConversationMeta,
}

/// Manages persisted conversation history on disk.
///
/// Each conversation is stored as two files in `~/.openheim/history/` (or a
/// custom directory via [`HistoryManager::with_dir`]): `{uuid}.json` holds
/// [`ConversationMeta`], rewritten wholesale on every change; `{uuid}.jsonl`
/// holds the message log, one [`Message`] per line, appended rather than
/// rewritten, so a crash loses at most the message being written. Whole-file
/// writes go through a temp file and rename.
#[derive(Clone)]
pub struct HistoryManager {
    history_dir: PathBuf,
}

impl HistoryManager {
    fn meta_path(&self, id: &Uuid) -> PathBuf {
        self.history_dir.join(format!("{}.json", id))
    }

    fn log_path(&self, id: &Uuid) -> PathBuf {
        self.history_dir.join(format!("{}.jsonl", id))
    }

    /// Acquires the write lease for conversation `id`, an advisory lock
    /// between `openheim` processes sharing this history directory (see
    /// `memory::lease`). Fails with [`Error::SessionLocked`] if a live
    /// process holds it; a stale one is taken over. Hold the returned
    /// [`SessionLease`] only while writing (one turn); reads need no lease.
    pub fn acquire_lease(&self, id: &Uuid) -> Result<SessionLease> {
        lease::acquire(&self.history_dir, id)
    }

    /// Writes `contents` to `path` via a temp file and rename, so a crash
    /// leaves the old content or the new, never a truncated file.
    fn write_atomic(path: &std::path::Path, contents: &str) -> Result<()> {
        let mut tmp_path = path.as_os_str().to_owned();
        tmp_path.push(".tmp");
        let tmp_path = PathBuf::from(tmp_path);
        std::fs::write(&tmp_path, contents)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Reads a conversation's message log. A corrupt last line (what a crash
    /// mid-append leaves) is dropped instead of failing the load.
    fn read_message_log(&self, id: &Uuid) -> Result<Vec<Message>> {
        let path = self.log_path(id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let data = std::fs::read_to_string(&path)?;
        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        let mut messages = Vec::with_capacity(lines.len());
        for (i, line) in lines.iter().enumerate() {
            match serde_json::from_str(line) {
                Ok(message) => messages.push(message),
                Err(e) if i == lines.len() - 1 => {
                    tracing::warn!(
                        "dropping unparseable trailing line in {}: {e}",
                        path.display()
                    );
                }
                Err(e) => return Err(Error::JsonError(e)),
            }
        }
        Ok(messages)
    }

    /// Rewrites a conversation's whole message log, for
    /// [`Self::save_conversation`].
    fn write_message_log(&self, id: &Uuid, messages: &[Message]) -> Result<()> {
        let mut buf = String::new();
        for message in messages {
            buf.push_str(&serde_json::to_string(message)?);
            buf.push('\n');
        }
        Self::write_atomic(&self.log_path(id), &buf)
    }

    /// Creates and saves a new, empty conversation.
    pub fn create_conversation(
        &self,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Result<Conversation> {
        self.create_with_id(Uuid::new_v4(), model, provider, skills)
    }

    /// [`Self::create_conversation`] under a caller-chosen `id`.
    fn create_with_id(
        &self,
        id: Uuid,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Result<Conversation> {
        let conv = Conversation {
            meta: ConversationMeta::new(id, model, provider, skills),
            messages: Vec::new(),
        };
        self.save_conversation(&conv)?;
        Ok(conv)
    }

    /// Loads a conversation from disk by its UUID.
    ///
    /// Returns an error if the meta file does not exist or cannot be
    /// deserialised. Messages come from the `.jsonl` log (empty if the
    /// conversation has none yet).
    pub fn load_conversation(&self, id: &Uuid) -> Result<Conversation> {
        let path = self.meta_path(id);
        if !path.exists() {
            return Err(Error::NotFound(format!(
                "Conversation {} not found at {}",
                id,
                path.display()
            )));
        }
        let data = std::fs::read_to_string(&path)?;
        let envelope: ConversationEnvelope = serde_json::from_str(&data)?;
        let messages = self.read_message_log(id)?;
        Ok(Conversation {
            meta: envelope.meta,
            messages,
        })
    }

    /// Saves a whole conversation, rewriting its message log, and fills in
    /// the title if there is none. For routine writes use
    /// [`Self::append_message`] and [`Self::save_meta`] instead.
    ///
    /// Fails with [`Error::HistoryDiverged`] rather than rewrite if the log
    /// on disk isn't a prefix of `conv.messages`: another process has written
    /// to it since `conv` was loaded. Messages this process appended are
    /// expected in `conv.messages` too (as `AgentState::prompt` does).
    pub fn save_conversation(&self, conv: &Conversation) -> Result<()> {
        let on_disk = self.read_message_log(&conv.meta.id)?;
        let diverged = on_disk.len() > conv.messages.len()
            || on_disk
                .iter()
                .zip(&conv.messages)
                .any(|(disk, mem)| disk != mem);
        if diverged {
            return Err(Error::HistoryDiverged {
                session_id: conv.meta.id.to_string(),
            });
        }

        let mut meta = conv.meta.clone();
        if let Some(first_user) = conv.messages.iter().find(|m| m.role == Role::User) {
            meta.fill_title_from(first_user);
        }

        self.write_message_log(&conv.meta.id, &conv.messages)?;
        self.save_meta(&meta)
    }

    /// Writes only a conversation's meta file, bumping `updated_at`.
    pub fn save_meta(&self, meta: &ConversationMeta) -> Result<()> {
        let mut meta = meta.clone();
        meta.updated_at = Utc::now();
        let envelope = ConversationEnvelope { meta };
        Self::write_atomic(
            &self.meta_path(&envelope.meta.id),
            &serde_json::to_string_pretty(&envelope)?,
        )
    }

    /// Appends one message to a conversation's log and updates its meta file
    /// (`updated_at`, and the title if this is the first user message). Call
    /// it as each message is produced (`StreamEvent::MessageAppended`). A
    /// missing meta file is skipped; the message is still appended.
    pub fn append_message(&self, id: &Uuid, message: &Message) -> Result<()> {
        let line = format!("{}\n", serde_json::to_string(message)?);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path(id))?;
        use std::io::Write;
        file.write_all(line.as_bytes())?;

        if let Ok(data) = std::fs::read_to_string(self.meta_path(id))
            && let Ok(mut envelope) = serde_json::from_str::<ConversationEnvelope>(&data)
        {
            envelope.meta.fill_title_from(message);
            self.save_meta(&envelope.meta)?;
        }
        Ok(())
    }

    /// Deletes a conversation's meta file and message log by UUID.
    ///
    /// Fails if the meta file doesn't exist; the log and lease file, which
    /// may not exist, are removed if they do.
    pub fn delete_conversation(&self, id: &Uuid) -> Result<()> {
        let path = self.meta_path(id);
        if !path.exists() {
            return Err(Error::NotFound(format!("Conversation {id} not found")));
        }
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(self.log_path(id));
        let _ = std::fs::remove_file(self.history_dir.join(format!("{id}.lock")));
        Ok(())
    }

    /// Returns metadata for all persisted conversations, sorted newest-first by `updated_at`.
    ///
    /// Only the meta file is read; message logs are not loaded.
    pub fn list_conversations(&self) -> Result<Vec<ConversationMeta>> {
        let mut metas = Vec::new();
        for entry in std::fs::read_dir(&self.history_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let data = std::fs::read_to_string(&path)?;
                if let Ok(envelope) = serde_json::from_str::<ConversationEnvelope>(&data) {
                    metas.push(envelope.meta);
                }
            }
        }
        metas.sort_by_key(|m| std::cmp::Reverse(m.updated_at));
        Ok(metas)
    }

    /// Creates a `HistoryManager` backed by a caller-chosen directory, e.g.
    /// for an injected [`AppConfig::data_dir`](crate::config::AppConfig) or
    /// in tests. Does not create the directory; the caller is responsible
    /// for it existing.
    pub fn with_dir(dir: std::path::PathBuf) -> Self {
        Self { history_dir: dir }
    }

    /// Resolves or creates a conversation for a new agent session.
    ///
    /// - `chat_id` is `Some` and the file exists → loads and returns it.
    /// - `chat_id` is `Some` but the file doesn't exist → creates a new conversation
    ///   with that exact ID (useful for client-assigned IDs).
    /// - `chat_id` is `None` → creates a fresh conversation with a new UUID.
    ///
    /// `model`, `provider` and `skills` only seed a conversation created
    /// here; an existing one is returned as saved.
    pub fn resolve_conversation(
        &self,
        chat_id: Option<Uuid>,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Result<Conversation> {
        match chat_id {
            Some(id) => {
                let path = self.meta_path(&id);
                if path.exists() {
                    self.load_conversation(&id)
                } else {
                    self.create_with_id(id, model, provider, skills)
                }
            }
            None => self.create_conversation(model, provider, skills),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_manager() -> (HistoryManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let mgr = HistoryManager::with_dir(dir.path().to_path_buf());
        (mgr, dir)
    }

    #[test]
    fn create_and_load_conversation_roundtrip() {
        let (mgr, _dir) = make_manager();
        let conv = mgr
            .create_conversation(Some("gpt-4".into()), Some("openai".into()), vec![])
            .unwrap();
        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.meta.id, conv.meta.id);
        assert_eq!(loaded.meta.model.as_deref(), Some("gpt-4"));
        assert_eq!(loaded.meta.provider.as_deref(), Some("openai"));
        assert!(loaded.messages.is_empty());
    }

    #[test]
    fn load_nonexistent_conversation_errors() {
        let (mgr, _dir) = make_manager();
        let id = Uuid::new_v4();
        let err = mgr.load_conversation(&id).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
        assert!(err.to_string().contains(&id.to_string()));
    }

    #[test]
    fn save_sets_title_from_first_user_message() {
        let (mgr, _dir) = make_manager();
        let mut conv = mgr.create_conversation(None, None, vec![]).unwrap();
        conv.messages.push(Message::user("hello world"));
        mgr.save_conversation(&conv).unwrap();
        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.meta.title.as_deref(), Some("hello world"));
    }

    #[test]
    fn save_truncates_long_title() {
        let (mgr, _dir) = make_manager();
        let mut conv = mgr.create_conversation(None, None, vec![]).unwrap();
        let long_msg: String = "a".repeat(100);
        conv.messages.push(Message::user(long_msg));
        mgr.save_conversation(&conv).unwrap();
        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.meta.title.as_ref().map(|t| t.len()), Some(80));
    }

    #[test]
    fn list_conversations_returns_most_recent_first() {
        let (mgr, _dir) = make_manager();
        mgr.create_conversation(None, None, vec![]).unwrap();
        mgr.create_conversation(None, None, vec![]).unwrap();
        let list = mgr.list_conversations().unwrap();
        assert_eq!(list.len(), 2);
        assert!(list[0].updated_at >= list[1].updated_at);
    }

    #[test]
    fn list_conversations_empty_dir() {
        let (mgr, _dir) = make_manager();
        let list = mgr.list_conversations().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn resolve_conversation_loads_existing_by_id() {
        let (mgr, _dir) = make_manager();
        let existing = mgr
            .create_conversation(Some("gpt-4".into()), None, vec![])
            .unwrap();
        let resolved = mgr
            .resolve_conversation(Some(existing.meta.id), None, None, vec![])
            .unwrap();
        assert_eq!(resolved.meta.id, existing.meta.id);
        assert_eq!(resolved.meta.model.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn resolve_conversation_creates_new_for_unknown_id() {
        let (mgr, _dir) = make_manager();
        let new_id = Uuid::new_v4();
        let resolved = mgr
            .resolve_conversation(Some(new_id), Some("claude".into()), None, vec![])
            .unwrap();
        assert_eq!(resolved.meta.id, new_id);
        assert_eq!(resolved.meta.model.as_deref(), Some("claude"));
    }

    #[test]
    fn resolve_conversation_creates_fresh_when_no_id() {
        let (mgr, _dir) = make_manager();
        let resolved = mgr.resolve_conversation(None, None, None, vec![]).unwrap();
        assert!(resolved.messages.is_empty());
        // Verify it was persisted
        mgr.load_conversation(&resolved.meta.id).unwrap();
    }

    #[test]
    fn conversation_skills_are_persisted() {
        let (mgr, _dir) = make_manager();
        let conv = mgr
            .create_conversation(None, None, vec!["coding".into(), "rust".into()])
            .unwrap();
        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.meta.skills, vec!["coding", "rust"]);
    }

    /// Messages persisted only via `append_message` (never
    /// `save_conversation`) must still be there on load — this is the state
    /// a crash before the end-of-turn save would leave behind.
    #[test]
    fn append_message_persists_without_a_full_save() {
        let (mgr, _dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();

        mgr.append_message(&conv.meta.id, &Message::user("first"))
            .unwrap();
        mgr.append_message(&conv.meta.id, &Message::assistant("second"))
            .unwrap();

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].text().as_deref(), Some("first"));
        assert_eq!(loaded.messages[1].text().as_deref(), Some("second"));
    }

    #[test]
    fn append_message_updates_timestamp_and_derives_title() {
        let (mgr, _dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();
        let created_updated_at = conv.meta.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(5));
        mgr.append_message(&conv.meta.id, &Message::user("hello there"))
            .unwrap();

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert!(loaded.meta.updated_at > created_updated_at);
        assert_eq!(loaded.meta.title.as_deref(), Some("hello there"));
    }

    #[test]
    fn save_conversation_after_appends_reconciles_the_log() {
        // save_conversation always rewrites the log from its own `messages`
        // argument; it must not end up with both the appended messages and
        // a second copy from the save.
        let (mgr, _dir) = make_manager();
        let mut conv = mgr.create_conversation(None, None, vec![]).unwrap();

        mgr.append_message(&conv.meta.id, &Message::user("first"))
            .unwrap();
        conv.messages.push(Message::user("first"));
        mgr.save_conversation(&conv).unwrap();

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
    }

    #[test]
    fn save_conversation_refuses_when_another_writer_appended_more_than_we_know_of() {
        // Simulates two processes sharing a session: this process loads the
        // conversation with 0 messages, then a foreign process appends
        // directly to the log (bypassing this process's in-memory `conv`).
        // A stale full save must not clobber the foreign line.
        let (mgr, _dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();

        mgr.append_message(&conv.meta.id, &Message::user("from another process"))
            .unwrap();

        // `conv` is still the stale, pre-append in-memory view (0 messages).
        let err = mgr.save_conversation(&conv).unwrap_err();
        assert!(matches!(err, Error::HistoryDiverged { .. }));

        // The foreign message must still be there — the refusal didn't corrupt it.
        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(
            loaded.messages[0].text().as_deref(),
            Some("from another process")
        );
    }

    #[test]
    fn save_conversation_succeeds_when_conv_already_reflects_its_own_appends() {
        // The normal single-process pattern (see the `acp` turn loop):
        // `append_message` is called as each message is produced, and the
        // same message is also pushed onto `conv.messages`, so by the time
        // the end-of-turn full save runs, `conv.messages` is a superset of
        // (here, exactly equal to) what's already on disk. This must not be
        // mistaken for a foreign writer.
        let (mgr, _dir) = make_manager();
        let mut conv = mgr.create_conversation(None, None, vec![]).unwrap();

        let msg = Message::user("first");
        mgr.append_message(&conv.meta.id, &msg).unwrap();
        conv.messages.push(msg);

        mgr.save_conversation(&conv).unwrap();

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
    }

    #[test]
    fn save_conversation_refuses_when_a_shared_message_was_overwritten_in_place() {
        // Same scenario as the "more than we know of" case, but the foreign
        // writer's append lands at the same index this process's stale
        // `conv` already occupies with a *different* message — so a naive
        // length-only check would miss it and silently clobber the foreign
        // line with `conv`'s version.
        let (mgr, _dir) = make_manager();
        let mut conv = mgr.create_conversation(None, None, vec![]).unwrap();

        mgr.append_message(&conv.meta.id, &Message::user("from another process"))
            .unwrap();
        // Stale in-memory view: same length as disk, but disagrees on content.
        conv.messages.push(Message::user("from this process"));

        let err = mgr.save_conversation(&conv).unwrap_err();
        assert!(matches!(err, Error::HistoryDiverged { .. }));

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(
            loaded.messages[0].text().as_deref(),
            Some("from another process")
        );
    }

    #[test]
    fn a_truncated_trailing_log_line_does_not_lose_earlier_messages() {
        let (mgr, dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();
        mgr.append_message(&conv.meta.id, &Message::user("intact"))
            .unwrap();

        // Simulate a crash mid-write: append a partial, unparseable JSON
        // line directly, bypassing `append_message`.
        let log_path = dir.path().join(format!("{}.jsonl", conv.meta.id));
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .unwrap();
        write!(file, "{{\"role\":\"user\",\"conte").unwrap();

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text().as_deref(), Some("intact"));
    }

    #[test]
    fn delete_conversation_removes_the_message_log_too() {
        let (mgr, dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();
        mgr.append_message(&conv.meta.id, &Message::user("hi"))
            .unwrap();

        mgr.delete_conversation(&conv.meta.id).unwrap();

        assert!(!dir.path().join(format!("{}.json", conv.meta.id)).exists());
        assert!(!dir.path().join(format!("{}.jsonl", conv.meta.id)).exists());
    }

    #[test]
    fn delete_conversation_removes_the_lease_lockfile_too() {
        let (mgr, dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();
        let lease = mgr.acquire_lease(&conv.meta.id).unwrap();
        assert!(dir.path().join(format!("{}.lock", conv.meta.id)).exists());

        mgr.delete_conversation(&conv.meta.id).unwrap();

        assert!(!dir.path().join(format!("{}.lock", conv.meta.id)).exists());
        drop(lease); // held past the delete on purpose: must not resurrect the file
        assert!(!dir.path().join(format!("{}.lock", conv.meta.id)).exists());
    }
}
