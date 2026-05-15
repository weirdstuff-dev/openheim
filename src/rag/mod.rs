//! Conversation history and skill injection for agent prompts.

pub mod history;
pub mod prompt;
pub mod skills;

pub use history::{Conversation, ConversationMeta, HistoryManager};
pub use prompt::PromptBuilder;
pub use skills::SkillsManager;

use crate::error::Result;
use uuid::Uuid;

/// Holds the conversation history store and skill definitions used to build agent prompts.
#[derive(Clone)]
pub struct RagContext {
    /// Persisted conversation history.
    pub history: HistoryManager,
    /// Named skill files loaded from `~/.openheim/skills/`.
    pub skills: SkillsManager,
}

impl RagContext {
    /// Initialise history and skills from the default openheim data directory.
    pub fn new() -> Result<Self> {
        Ok(Self {
            history: HistoryManager::new()?,
            skills: SkillsManager::new()?,
        })
    }

    /// Load or create a conversation and build the prompt context for an agent turn.
    ///
    /// Returns the resolved [`Conversation`] and a [`PromptBuilder`] already populated
    /// with any requested skills. When `chat_id` refers to an existing conversation the
    /// skills stored on that conversation take precedence over `skill_names`.
    pub fn prepare(
        &self,
        chat_id: Option<Uuid>,
        skill_names: &[String],
        model: Option<String>,
        provider: Option<String>,
    ) -> Result<(Conversation, PromptBuilder)> {
        let conversation =
            self.history
                .resolve_conversation(chat_id, model, provider, skill_names.to_vec())?;

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
