use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::config_dir;
use crate::core::models::{Message, Role};
use crate::error::{Error, Result};
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
        let conv = mgr.create_conversation(Some("gpt-4".into()), Some("openai".into()), vec![]).unwrap();
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
        assert!(err.to_string().contains(&id.to_string()));
    }

    #[test]
    fn save_sets_title_from_first_user_message() {
        let (mgr, _dir) = make_manager();
        let mut conv = mgr.create_conversation(None, None, vec![]).unwrap();
        conv.messages.push(Message::user("hello world".into()));
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
        conv.messages.push(Message::user("latest".into()));
        mgr.save_conversation(&conv).unwrap();
        let last = mgr.get_last_conversation().unwrap().unwrap();
        assert_eq!(last.meta.id, conv.meta.id);
    }

    #[test]
    fn resolve_conversation_loads_existing_by_id() {
        let (mgr, _dir) = make_manager();
        let existing = mgr.create_conversation(Some("gpt-4".into()), None, vec![]).unwrap();
        let resolved = mgr.resolve_conversation(Some(existing.meta.id), None, None, vec![]).unwrap();
        assert_eq!(resolved.meta.id, existing.meta.id);
        assert_eq!(resolved.meta.model.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn resolve_conversation_creates_new_for_unknown_id() {
        let (mgr, _dir) = make_manager();
        let new_id = Uuid::new_v4();
        let resolved = mgr.resolve_conversation(Some(new_id), Some("claude".into()), None, vec![]).unwrap();
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
        let conv = mgr.create_conversation(None, None, vec!["coding".into(), "rust".into()]).unwrap();
        let loaded = mgr.load_conversation(&conv.meta.id).unwrap();
        assert_eq!(loaded.meta.skills, vec!["coding", "rust"]);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cwd: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub meta: ConversationMeta,
    pub messages: Vec<Message>,
}

/// Lightweight struct for deserializing only the metadata portion of a conversation file.
#[derive(Debug, Deserialize)]
struct ConversationEnvelope {
    meta: ConversationMeta,
}

#[derive(Clone)]
pub struct HistoryManager {
    history_dir: PathBuf,
}

impl HistoryManager {
    pub fn new() -> Result<Self> {
        let dir = config_dir()?.join("history");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { history_dir: dir })
    }

    fn conversation_path(&self, id: &Uuid) -> PathBuf {
        self.history_dir.join(format!("{}.json", id))
    }

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

    pub fn load_conversation(&self, id: &Uuid) -> Result<Conversation> {
        let path = self.conversation_path(id);
        if !path.exists() {
            return Err(Error::Other(format!(
                "Conversation {} not found at {}",
                id,
                path.display()
            )));
        }
        let data = std::fs::read_to_string(&path)?;
        let conv: Conversation = serde_json::from_str(&data)?;
        Ok(conv)
    }

    pub fn save_conversation(&self, conv: &Conversation) -> Result<()> {
        let path = self.conversation_path(&conv.meta.id);
        let mut conv_to_save = conv.clone();
        conv_to_save.meta.updated_at = Utc::now();

        if conv_to_save.meta.title.is_none()
            && let Some(msg) = conv_to_save.messages.iter().find(|m| m.role == Role::User)
                && let Some(content) = &msg.content {
                    let title: String = content.chars().take(80).collect();
                    conv_to_save.meta.title = Some(title);
                }

        let json = serde_json::to_string_pretty(&conv_to_save)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn delete_conversation(&self, id: &Uuid) -> Result<()> {
        let path = self.conversation_path(id);
        if !path.exists() {
            return Err(Error::Other(format!("Conversation {id} not found")));
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

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
        metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(metas)
    }

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

    pub fn resolve_conversation(
        &self,
        chat_id: Option<Uuid>,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Result<Conversation> {
        match chat_id {
            Some(id) => {
                let path = self.conversation_path(&id);
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
