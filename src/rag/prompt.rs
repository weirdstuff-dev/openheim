use crate::core::models::{Message, Role};

pub struct PromptBuilder {
    system_parts: Vec<String>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self {
            system_parts: Vec::new(),
        }
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
