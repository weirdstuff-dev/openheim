use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::models::{Effort, Message, Role, Usage};
use crate::core::runtime::AgentMode;
use crate::error::{Error, Result};
use crate::memory::lease::{self, SessionLease};
use std::path::PathBuf;

/// Persistent metadata for a conversation, stored in its `.json` file (see
/// `HistoryManager` for the on-disk layout).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// The session's mode as of its last turn; `None` (saved before modes
    /// were stored) loads as [`AgentMode::Execute`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub mode: Option<AgentMode>,
    /// The effort level chosen for the session, if one was; `None` runs
    /// at the provider's `effort`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub effort: Option<Effort>,
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
            mode: None,
            effort: None,
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
#[non_exhaustive]
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

    /// Makes the log at `path` end on a line boundary so an append starts a
    /// new line. A trailing line without a newline (what a crash mid-append
    /// leaves) is cut off if it doesn't parse, matching what
    /// [`Self::read_message_log`] drops; if it parses, its newline is added.
    /// A missing or empty log, or one already ending in a newline, is left
    /// alone.
    fn repair_log_tail(path: &std::path::Path) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let len = file.metadata()?.len();
        if len == 0 {
            return Ok(());
        }
        let mut last = [0u8; 1];
        file.seek(SeekFrom::Start(len - 1))?;
        file.read_exact(&mut last)?;
        if last[0] == b'\n' {
            return Ok(());
        }

        let mut data = Vec::new();
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut data)?;
        let tail_start = data.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        let tail = &data[tail_start..];
        let tail_parses = std::str::from_utf8(tail)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .is_some_and(|s| serde_json::from_str::<Message>(s).is_ok());
        if tail_parses {
            file.seek(SeekFrom::End(0))?;
            file.write_all(b"\n")?;
        } else {
            tracing::warn!(
                "truncating unparseable trailing line in {} before append",
                path.display()
            );
            file.set_len(tail_start as u64)?;
        }
        Ok(())
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
        let path = self.log_path(id);
        Self::repair_log_tail(&path)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
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
    fn appending_after_a_complete_line_missing_its_newline_keeps_that_line() {
        let (mgr, dir) = make_manager();
        let conv = mgr.create_conversation(None, None, vec![]).unwrap();

        let log_path = dir.path().join(format!("{}.jsonl", conv.meta.id));
        std::fs::write(
            &log_path,
            serde_json::to_string(&Message::user("no newline")).unwrap(),
        )
        .unwrap();

        mgr.append_message(&conv.meta.id, &Message::user("next"))
            .unwrap();

        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        let texts: Vec<_> = loaded.messages.iter().map(|m| m.text()).collect();
        assert_eq!(
            texts,
            vec![Some("no newline".to_string()), Some("next".to_string())]
        );
    }
}
