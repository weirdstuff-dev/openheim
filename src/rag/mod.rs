//! Conversation history and skill injection for agent prompts.

pub mod history;
pub mod prompt;
pub mod skills;
pub mod system;

pub use history::{Conversation, ConversationMeta, HistoryManager};
pub use prompt::PromptBuilder;
pub use skills::SkillsManager;
pub use system::SystemLoader;

use crate::error::Result;
use uuid::Uuid;

/// Holds the conversation history store and skill definitions used to build agent prompts.
#[derive(Clone)]
pub struct RagContext {
    /// Persisted conversation history.
    pub history: HistoryManager,
    /// Named skill files loaded from `~/.openheim/skills/`.
    pub skills: SkillsManager,
    /// System identity loaded from `~/.openheim/system.md`.
    pub system: SystemLoader,
    /// Skills included in every new session (from `default_skills` in config).
    default_skills: Vec<String>,
}

impl RagContext {
    /// Initialise history, skills, and system identity from the default openheim data directory.
    ///
    /// `default_skills` are merged with any per-session skills on each new conversation.
    pub fn new(default_skills: Vec<String>) -> Result<Self> {
        Ok(Self {
            history: HistoryManager::new()?,
            skills: SkillsManager::new()?,
            system: SystemLoader::new()?,
            default_skills,
        })
    }

    /// Load or create a conversation and build the prompt context for an agent turn.
    ///
    /// Returns the resolved [`Conversation`] and a [`PromptBuilder`] already populated
    /// with the system identity and any requested skills.
    ///
    /// For **new** conversations, `default_skills` are merged with `skill_names` (defaults
    /// first, deduplicated) and persisted on the conversation. For **existing** conversations
    /// the stored skill list is used as-is, preserving the state from when the session began.
    pub fn prepare(
        &self,
        chat_id: Option<Uuid>,
        skill_names: &[String],
        model: Option<String>,
        provider: Option<String>,
    ) -> Result<(Conversation, PromptBuilder)> {
        // Always pass merged skills to resolve_conversation. For existing conversations
        // the parameter is ignored (stored list wins); for new ones it gets persisted.
        let merged_skills = merge_skills(&self.default_skills, skill_names);

        let conversation =
            self.history
                .resolve_conversation(chat_id, model, provider, merged_skills)?;

        let mut builder = PromptBuilder::new();

        // Always inject the system identity as the first layer.
        let system_content = self.system.load()?;
        tracing::debug!(chars = system_content.len(), "prepare: loaded system.md");
        builder.set_system(system_content);

        // Load skills from the conversation's stored list (already contains merged
        // defaults for new conversations, or the original set for existing ones).
        if !conversation.meta.skills.is_empty() {
            let loaded = self.skills.load_skills(&conversation.meta.skills)?;
            for (name, content) in &loaded {
                tracing::debug!(skill = %name, "prepare: loaded skill");
                builder.add_skill(name, content);
            }
        }

        Ok((conversation, builder))
    }
}

/// Merge `defaults` with `session` skills, preserving order and deduplicating.
/// Defaults come first; session skills are appended only if not already present.
fn merge_skills(defaults: &[String], session: &[String]) -> Vec<String> {
    let mut merged = defaults.to_vec();
    for s in session {
        if !merged.contains(s) {
            merged.push(s.clone());
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_skills_defaults_first_no_duplicates() {
        let defaults = vec!["rules".to_string(), "coding".to_string()];
        let session = vec!["coding".to_string(), "rust".to_string()];
        let merged = merge_skills(&defaults, &session);
        assert_eq!(merged, vec!["rules", "coding", "rust"]);
    }

    #[test]
    fn merge_skills_empty_defaults() {
        let merged = merge_skills(&[], &["rust".to_string()]);
        assert_eq!(merged, vec!["rust"]);
    }

    #[test]
    fn merge_skills_empty_session() {
        let merged = merge_skills(&["rules".to_string()], &[]);
        assert_eq!(merged, vec!["rules"]);
    }
}
