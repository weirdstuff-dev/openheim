use crate::core::models::{ContentBlock, Message, Role};

/// Builds an LLM message sequence by prepending a structured system message.
///
/// The system message combines an optional identity (from `system.md`) and
/// any number of named skills; see [`Self::build`] for the layout.
#[derive(Default)]
pub struct PromptBuilder {
    system_identity: Option<String>,
    skills: Vec<(String, String)>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the system identity loaded from `system.md`.
    pub fn set_system(&mut self, content: String) {
        self.system_identity = Some(content);
    }

    /// Registers a named skill to include in the system message.
    pub fn add_skill(&mut self, name: &str, content: &str) {
        self.skills.push((name.to_string(), content.to_string()));
    }

    /// Constructs the full message list for an LLM request.
    ///
    /// When there is substantive content (non-blank identity or at least one
    /// skill), a `Role::System` message is inserted at position 0 using the
    /// template below. Otherwise `history` is returned unchanged.
    ///
    /// ```text
    /// You are a general purpose multiprovider LLM agent.
    ///
    /// The user has given you the following identity:
    ///
    /// <system.md content>
    ///
    /// ---
    ///
    /// These are the skills you have mastered:
    ///
    /// ### <skill name>
    ///
    /// <skill content>
    /// ```
    pub fn build(&self, history: &[Message]) -> Vec<Message> {
        let orig = self.system_identity.as_deref();
        let identity = orig.filter(|s| !s.trim().is_empty());
        let has_content = identity.is_some() || !self.skills.is_empty();

        if !has_content {
            return history.to_vec();
        }

        let mut sections: Vec<String> = Vec::new();

        sections.push("You are a general purpose multiprovider LLM agent.".to_string());

        if let Some(id) = identity {
            sections.push(format!(
                "The user has given you the following identity:\n\n{id}"
            ));
        }

        if !self.skills.is_empty() {
            let skill_blocks: Vec<String> = self
                .skills
                .iter()
                .map(|(name, content)| format!("### {name}\n\n{content}"))
                .collect();
            sections.push(format!(
                "These are the skills you have mastered:\n\n{}",
                skill_blocks.join("\n\n---\n\n")
            ));
        }

        let system_content = sections.join("\n\n---\n\n");
        tracing::debug!(
            len = system_content.len(),
            "build: system message assembled"
        );

        let mut messages = vec![Message {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: system_content,
            }],
        }];

        messages.extend_from_slice(history);
        messages
    }
}
