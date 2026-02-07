use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::config_dir;
use crate::core::models::{Message, Role};
use crate::error::{Error, Result};
use std::path::PathBuf;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub meta: ConversationMeta,
    pub messages: Vec<Message>,
}

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

        if conv_to_save.meta.title.is_none() {
            if let Some(msg) = conv_to_save.messages.iter().find(|m| m.role == Role::User) {
                if let Some(content) = &msg.content {
                    let title: String = content.chars().take(80).collect();
                    conv_to_save.meta.title = Some(title);
                }
            }
        }

        let json = serde_json::to_string_pretty(&conv_to_save)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn list_conversations(&self) -> Result<Vec<ConversationMeta>> {
        let mut metas = Vec::new();
        for entry in std::fs::read_dir(&self.history_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let data = std::fs::read_to_string(&path)?;
                if let Ok(conv) = serde_json::from_str::<Conversation>(&data) {
                    metas.push(conv.meta);
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

    pub fn resolve_conversation(
        &self,
        chat_id: Option<Uuid>,
        model: Option<String>,
        provider: Option<String>,
        skills: Vec<String>,
    ) -> Result<Conversation> {
        match chat_id {
            Some(id) => self.load_conversation(&id),
            None => self.create_conversation(model, provider, skills),
        }
    }
}
