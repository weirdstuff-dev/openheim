pub mod history;
pub mod prompt;
pub mod skills;

pub use history::{Conversation, ConversationMeta, HistoryManager};
pub use prompt::PromptBuilder;
pub use skills::SkillsManager;

use crate::error::Result;
use uuid::Uuid;

#[derive(Clone)]
pub struct RagContext {
    pub history: HistoryManager,
    pub skills: SkillsManager,
}

impl RagContext {
    pub fn new() -> Result<Self> {
        Ok(Self {
            history: HistoryManager::new()?,
            skills: SkillsManager::new()?,
        })
    }

    pub fn prepare(
        &self,
        chat_id: Option<Uuid>,
        skill_names: &[String],
        model: Option<String>,
        provider: Option<String>,
    ) -> Result<(Conversation, PromptBuilder)> {
        let conversation = self.history.resolve_conversation(
            chat_id,
            model,
            provider,
            skill_names.to_vec(),
        )?;

        let mut builder = PromptBuilder::new();

        // Load skills: use conversation's stored skills if continuing, otherwise use provided ones
        let skills_to_load = if chat_id.is_some() && !conversation.meta.skills.is_empty() {
            &conversation.meta.skills
        } else {
            skill_names
        };

        if !skills_to_load.is_empty() {
            let loaded = self.skills.load_skills(skills_to_load)?;
            for (name, content) in &loaded {
                builder.add_skill(name, content);
            }
        }

        Ok((conversation, builder))
    }
}
