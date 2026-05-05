use crate::core::models::{Message, Role};

#[derive(Default)]
pub struct PromptBuilder {
    system_parts: Vec<String>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_skill(&mut self, name: &str, content: &str) {
        self.system_parts
            .push(format!("## Skill: {}\n\n{}", name, content));
    }

    pub fn build(&self, history: &[Message]) -> Vec<Message> {
        let mut messages = Vec::new();

        if !self.system_parts.is_empty() {
            let system_content = self.system_parts.join("\n\n---\n\n");
            messages.push(Message {
                role: Role::System,
                content: Some(system_content),
                tool_calls: None,
                tool_call_id: None,
                tool_name: None,
            });
        }

        messages.extend_from_slice(history);
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_with_no_skills_returns_history_unchanged() {
        let builder = PromptBuilder::new();
        let history = vec![Message::user("hello".into())];
        let result = builder.build(&history);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, Role::User);
        assert_eq!(result[0].content.as_deref(), Some("hello"));
    }

    #[test]
    fn build_with_one_skill_prepends_system_message() {
        let mut builder = PromptBuilder::new();
        builder.add_skill("coding", "You are a coding assistant.");
        let history = vec![Message::user("help".into())];
        let result = builder.build(&history);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, Role::System);
        let content = result[0].content.as_deref().unwrap();
        assert!(content.contains("## Skill: coding"));
        assert!(content.contains("You are a coding assistant."));
        assert_eq!(result[1].role, Role::User);
    }

    #[test]
    fn build_with_multiple_skills_joins_with_separator() {
        let mut builder = PromptBuilder::new();
        builder.add_skill("skill_a", "Content A");
        builder.add_skill("skill_b", "Content B");
        let history = vec![Message::user("test".into())];
        let result = builder.build(&history);

        assert_eq!(result.len(), 2);
        let content = result[0].content.as_deref().unwrap();
        assert!(content.contains("## Skill: skill_a"));
        assert!(content.contains("## Skill: skill_b"));
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

        assert_eq!(result.len(), 4); // system + 3 history
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].content.as_deref(), Some("first"));
        assert_eq!(result[2].content.as_deref(), Some("second"));
        assert_eq!(result[3].content.as_deref(), Some("third"));
    }
}
