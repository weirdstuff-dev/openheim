use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::config_dir;
use crate::core::models::{Message, Role};
use crate::error::{Error, Result};
use crate::rag::lease::{self, SessionLease};
use std::path::PathBuf;

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
    fn get_last_conversation_returns_none_when_empty() {
        let (mgr, _dir) = make_manager();
        let result = mgr.get_last_conversation().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn get_last_conversation_returns_most_recent() {
        let (mgr, _dir) = make_manager();
        mgr.create_conversation(None, None, vec![]).unwrap();
        let second = mgr.create_conversation(None, None, vec![]).unwrap();
        // Save second with a message so its updated_at is newer
        let mut conv = second.clone();
        conv.messages.push(Message::user("latest"));
        mgr.save_conversation(&conv).unwrap();
        let last = mgr.get_last_conversation().unwrap().unwrap();
        assert_eq!(last.meta.id, conv.meta.id);
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

    /// The actual regression test for this durability item: messages
    /// persisted only via `append_message` (never `save_conversation`) must
    /// still be there on load — this is what a crash before the end-of-turn
    /// save would leave behind.
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
    fn pre_split_format_conversation_still_loads() {
        // A conversation written before message logs were split into a
        // `.jsonl` sidecar: a single `{id}.json` containing both `meta` and
        // the full `messages` array, no `.jsonl` sibling.
        let (mgr, dir) = make_manager();
        let id = Uuid::new_v4();
        let now = Utc::now();
        let legacy = Conversation {
            meta: ConversationMeta {
                id,
                created_at: now,
                updated_at: now,
                model: None,
                provider: None,
                title: None,
                skills: vec![],
                cwd: None,
            },
            messages: vec![Message::user("from the old format")],
        };
        std::fs::write(
            dir.path().join(format!("{id}.json")),
            serde_json::to_string_pretty(&legacy).unwrap(),
        )
        .unwrap();

        let loaded = mgr.load_conversation(&id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(
            loaded.messages[0].text().as_deref(),
            Some("from the old format")
        );
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

/// Persistent metadata for a conversation session.
///
/// Stored in the `meta` field of each conversation's `.json` file. Does not
/// include the full message history — see [`Conversation`] for that, and
/// [`HistoryManager`]'s doc comment for how the two are actually laid out on
/// disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Title derived from the first user message (up to 80 characters). Set on
    /// the first [`HistoryManager::save_conversation`] call that includes messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Names of skills active in this conversation (correspond to `~/.openheim/skills/*.md`).
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cwd: Option<std::path::PathBuf>,
}

/// A complete conversation: metadata plus the full ordered message list.
///
/// Not itself the on-disk format — see [`HistoryManager`]'s doc comment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub meta: ConversationMeta,
    pub messages: Vec<Message>,
}

/// On-disk (de)serialization shape for a conversation's `{id}.json` meta
/// file, and also — for backward compatibility — the *entire* shape of a
/// conversation file written before message logs were split out (see
/// [`HistoryManager`]'s doc comment). `messages` is only ever populated by
/// deserializing one of those old files; new code never sets it, since
/// current-format meta files never write it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConversationEnvelope {
    meta: ConversationMeta,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    messages: Vec<Message>,
}

/// Manages persisted conversation history on disk.
///
/// Each conversation is stored as two files in `~/.openheim/history/` (or a
/// custom directory when constructed with [`HistoryManager::with_dir`]):
/// `{uuid}.json` holds [`ConversationMeta`] (small, rewritten wholesale on
/// every change), and `{uuid}.jsonl` holds the message log, one JSON-encoded
/// [`Message`] per line, appended to as the conversation grows rather than
/// rewritten — see [`Self::append_message`]. This means a crash mid-turn
/// loses at most the one message that was mid-write, not every message
/// appended since the conversation was created, and avoids repeatedly
/// rewriting an ever-growing file for every message (the old format, a
/// single JSON blob containing the entire message array, was O(n) to save
/// per message written).
///
/// Both files are written atomically (temp file + rename) so a crash mid-write
/// can't corrupt the previous, already-saved content.
///
/// Conversation files written before this split (a single `{uuid}.json`
/// containing both `meta` and the full `messages` array, no `.jsonl`
/// sibling) still load correctly — see [`Self::load_conversation`] — and are
/// transparently upgraded to the split layout the next time they're saved.
#[derive(Clone)]
pub struct HistoryManager {
    history_dir: PathBuf,
}

impl HistoryManager {
    /// Creates a `HistoryManager` backed by `~/.openheim/history/`, creating the
    /// directory if it doesn't exist.
    pub fn new() -> Result<Self> {
        let dir = config_dir()?.join("history");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { history_dir: dir })
    }

    fn meta_path(&self, id: &Uuid) -> PathBuf {
        self.history_dir.join(format!("{}.json", id))
    }

    fn log_path(&self, id: &Uuid) -> PathBuf {
        self.history_dir.join(format!("{}.jsonl", id))
    }

    /// Acquires the write lease for conversation `id`: an advisory,
    /// cross-process lock so this session can't be written to by two
    /// `openheim` processes sharing the same history directory at once.
    ///
    /// Returns [`Error::SessionLocked`] if another still-live process
    /// already holds it. A stale lease (its process has since exited or the
    /// lock has aged past its TTL — see `rag::lease`) is taken over
    /// automatically. Hold the returned [`SessionLease`] for exactly the
    /// span that actually writes — a single `session/prompt` turn
    /// (`acp::AgentState::acp_prompt` acquires and releases one per turn) —
    /// not for as long as the session merely stays loaded/live; it releases
    /// on drop.
    ///
    /// Only needed around an actual write — [`Self::load_conversation`] and
    /// [`Self::list_conversations`] stay lease-free, since reading a
    /// conversation never risks clobbering another process's writes, and
    /// neither does merely activating/holding a session open without
    /// prompting it.
    pub fn acquire_lease(&self, id: &Uuid) -> Result<SessionLease> {
        lease::acquire(&self.history_dir, id)
    }

    /// Writes `contents` to `path` via a temp file + rename in the same
    /// directory, so a crash or kill mid-write leaves either the old content
    /// or the new content, never a truncated mix of both (`std::fs::write`
    /// truncates in place, which a crash mid-write can turn into an
    /// unparseable file).
    fn write_atomic(path: &std::path::Path, contents: &str) -> Result<()> {
        let mut tmp_path = path.as_os_str().to_owned();
        tmp_path.push(".tmp");
        let tmp_path = PathBuf::from(tmp_path);
        std::fs::write(&tmp_path, contents)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Reads and parses a conversation's message log, tolerating a corrupt or
    /// truncated trailing line (the one a crash mid-append would produce) by
    /// dropping it instead of failing the whole load — every complete line
    /// before it is still a message that was actually, durably appended.
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

    /// Rewrites a conversation's message log from scratch. Used to write the
    /// complete log for a [`Self::save_conversation`] call and to upgrade an
    /// old single-file conversation to the split layout; per-message
    /// persistence during a turn should use [`Self::append_message`] instead,
    /// which doesn't pay this method's O(n) cost per call.
    fn write_message_log(&self, id: &Uuid, messages: &[Message]) -> Result<()> {
        let mut buf = String::new();
        for message in messages {
            buf.push_str(&serde_json::to_string(message)?);
            buf.push('\n');
        }
        Self::write_atomic(&self.log_path(id), &buf)
    }

    /// Creates a new conversation, persists it immediately, and returns it.
    ///
    /// The conversation starts with no messages and no title. The title is derived
    /// from the first user message when [`save_conversation`] is later called.
    pub fn create_conversation(
        &self,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Result<Conversation> {
        let now = Utc::now();
        let conv = Conversation {
            meta: ConversationMeta {
                id: Uuid::new_v4(),
                created_at: now,
                updated_at: now,
                model,
                provider,
                title: None,
                skills,
                cwd: None,
            },
            messages: Vec::new(),
        };
        self.save_conversation(&conv)?;
        Ok(conv)
    }

    /// Loads a conversation from disk by its UUID.
    ///
    /// Returns an error if the meta file does not exist or cannot be
    /// deserialised. Messages come from the `.jsonl` log if one exists
    /// (current format), or from the meta file's own `messages` field
    /// otherwise (a conversation saved before message logs were split out).
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
        let messages = if self.log_path(id).exists() {
            self.read_message_log(id)?
        } else {
            envelope.messages
        };
        Ok(Conversation {
            meta: envelope.meta,
            messages,
        })
    }

    /// Saves a conversation to disk, updating `updated_at` to now.
    ///
    /// If the conversation has no title yet and contains at least one user message,
    /// the title is set to the first 80 characters of that message.
    ///
    /// Rewrites the *entire* message log (see [`Self::write_message_log`]);
    /// this is the right call for creating a conversation or for an
    /// end-of-turn consistency checkpoint, but a turn that wants to persist
    /// messages as they're produced should call [`Self::append_message`]
    /// instead of calling this once per message.
    ///
    /// Refuses (returning [`Error::HistoryDiverged`]) instead of rewriting if
    /// the on-disk log already has more messages than `conv` does: that can
    /// only mean another process appended to this conversation after `conv`
    /// was loaded, and a full rewrite from `conv.messages` would silently
    /// drop them. This process's own [`Self::append_message`] calls don't
    /// trigger it, since callers are expected to push the same message onto
    /// `conv.messages` when they append it (see the `acp` turn loop), so
    /// `conv.messages` and the on-disk log grow in lockstep.
    pub fn save_conversation(&self, conv: &Conversation) -> Result<()> {
        let on_disk_len = self.read_message_log(&conv.meta.id)?.len();
        if on_disk_len > conv.messages.len() {
            return Err(Error::HistoryDiverged {
                session_id: conv.meta.id.to_string(),
            });
        }

        let mut meta = conv.meta.clone();
        meta.updated_at = Utc::now();

        if meta.title.is_none()
            && let Some(msg) = conv.messages.iter().find(|m| m.role == Role::User)
            && let Some(content) = msg.text()
        {
            let title: String = content.chars().take(80).collect();
            meta.title = Some(title);
        }

        self.write_message_log(&conv.meta.id, &conv.messages)?;
        let envelope = ConversationEnvelope {
            meta,
            messages: Vec::new(),
        };
        Self::write_atomic(
            &self.meta_path(&conv.meta.id),
            &serde_json::to_string_pretty(&envelope)?,
        )
    }

    /// Appends one message to a conversation's on-disk log without rewriting
    /// the rest of it, and bumps `updated_at` (deriving `title` too, if this
    /// is the conversation's first user message) in the small meta file.
    ///
    /// Meant to be called as each message is produced during a turn — see
    /// `StreamEvent::MessageAppended` — so a crash mid-turn loses at most the
    /// message that was mid-write, rather than every message the turn had
    /// produced so far. Silently a no-op-on-meta if the conversation's meta
    /// file doesn't exist (shouldn't happen in practice — every conversation
    /// is created via [`Self::save_conversation`] first — but a missing meta
    /// file is not a reason to lose the message itself).
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
            envelope.meta.updated_at = Utc::now();
            if envelope.meta.title.is_none()
                && message.role == Role::User
                && let Some(text) = message.text()
            {
                envelope.meta.title = Some(text.chars().take(80).collect());
            }
            envelope.messages = Vec::new();
            Self::write_atomic(
                &self.meta_path(id),
                &serde_json::to_string_pretty(&envelope)?,
            )?;
        }
        Ok(())
    }

    /// Deletes a conversation's meta file and message log by UUID.
    ///
    /// Returns an error if the meta file does not exist; the `.jsonl` log
    /// and `.lock` lease file (neither of which necessarily exist — an
    /// empty or pre-split-format conversation has no log, and a session
    /// that was never activated for writing has no lease) are removed on a
    /// best-effort basis.
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

    /// Returns the most-recently-updated conversation, or `None` if none exist.
    ///
    /// Loads the full conversation (messages + metadata) for the most recent entry.
    pub fn get_last_conversation(&self) -> Result<Option<Conversation>> {
        let metas = self.list_conversations()?;
        match metas.first() {
            Some(meta) => Ok(Some(self.load_conversation(&meta.id)?)),
            None => Ok(None),
        }
    }

    #[cfg(test)]
    pub fn with_dir(dir: std::path::PathBuf) -> Self {
        Self { history_dir: dir }
    }

    /// Resolves or creates a conversation for a new agent session.
    ///
    /// - `chat_id` is `Some` and the file exists → loads and returns it.
    /// - `chat_id` is `Some` but the file doesn't exist → creates a new conversation
    ///   with that exact ID (useful for client-assigned IDs).
    /// - `chat_id` is `None` → creates a fresh conversation with a new UUID.
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
                    let now = Utc::now();
                    let conv = Conversation {
                        meta: ConversationMeta {
                            id,
                            created_at: now,
                            updated_at: now,
                            model,
                            provider,
                            title: None,
                            skills,
                            cwd: None,
                        },
                        messages: Vec::new(),
                    };
                    self.save_conversation(&conv)?;
                    Ok(conv)
                }
            }
            None => self.create_conversation(model, provider, skills),
        }
    }
}
