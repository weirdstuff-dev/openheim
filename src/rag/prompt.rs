use crate::core::models::{Message, Role};

/// Builds an LLM message sequence by prepending a structured system message.
///
/// The system message is assembled from an optional identity (from `system.md`)
/// and any number of named skills, laid out in a consistent template so the
/// model always understands what it is, what identity it has been given, and
/// what skills it has been asked to master.
///
/// If neither identity nor skills have been registered, no system message is
/// prepended and `history` is returned unchanged.
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
            content: Some(system_content),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            is_error: false,
        }];

        messages.extend_from_slice(history);
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_with_nothing_returns_history_unchanged() {
        let builder = PromptBuilder::new();
        let history = vec![Message::user("hello".into())];
        let result = builder.build(&history);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn empty_system_identity_is_ignored() {
        let mut builder = PromptBuilder::new();
        builder.set_system("   ".into());
        let history = vec![Message::user("hi".into())];
        let result = builder.build(&history);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn set_system_alone_produces_structured_message() {
        let mut builder = PromptBuilder::new();
        builder.set_system("I am a helpful agent.".into());
        let result = builder.build(&[Message::user("hi".into())]);

        assert_eq!(result.len(), 2);
        let content = result[0].content.as_deref().unwrap();
        assert!(content.contains("You are a general purpose multiprovider LLM agent."));
        assert!(content.contains("The user has given you the following identity:"));
        assert!(content.contains("I am a helpful agent."));
        assert!(!content.contains("skills"));
    }

    #[test]
    fn skill_alone_produces_structured_message() {
        let mut builder = PromptBuilder::new();
        builder.add_skill("rust", "Write idiomatic Rust.");
        let result = builder.build(&[Message::user("hi".into())]);

        assert_eq!(result.len(), 2);
        let content = result[0].content.as_deref().unwrap();
        assert!(content.contains("You are a general purpose multiprovider LLM agent."));
        assert!(content.contains("These are the skills you have mastered:"));
        assert!(content.contains("### rust"));
        assert!(content.contains("Write idiomatic Rust."));
        assert!(!content.contains("identity"));
    }

    #[test]
    fn identity_and_skills_are_both_present_in_order() {
        let mut builder = PromptBuilder::new();
        builder.set_system("Custom identity.".into());
        builder.add_skill("rust", "Be idiomatic.");
        builder.add_skill("testing", "Write tests.");
        let result = builder.build(&[Message::user("go".into())]);

        assert_eq!(result.len(), 2);
        let content = result[0].content.as_deref().unwrap();

        let base_pos = content.find("general purpose").unwrap();
        let identity_pos = content.find("Custom identity.").unwrap();
        let skills_pos = content.find("skills you have mastered").unwrap();
        let rust_pos = content.find("### rust").unwrap();
        let testing_pos = content.find("### testing").unwrap();

        assert!(base_pos < identity_pos);
        assert!(identity_pos < skills_pos);
        assert!(skills_pos < rust_pos);
        assert!(rust_pos < testing_pos);
    }

    #[test]
    fn multiple_skills_are_separated() {
        let mut builder = PromptBuilder::new();
        builder.add_skill("a", "Content A");
        builder.add_skill("b", "Content B");
        let content = builder.build(&[]).remove(0).content.unwrap();
        assert!(content.contains("### a"));
        assert!(content.contains("### b"));
        assert!(content.contains("---"));
    }

    #[test]
    fn build_preserves_history_order() {
        let mut builder = PromptBuilder::new();
        builder.add_skill("test", "Test skill");
        let history = vec![
            Message::user("first".into()),
            Message::assistant("second".into()),
            Message::user("third".into()),
        ];
        let result = builder.build(&history);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].content.as_deref(), Some("first"));
        assert_eq!(result[2].content.as_deref(), Some("second"));
        assert_eq!(result[3].content.as_deref(), Some("third"));
    }
}
