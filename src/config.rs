use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub max_iterations: usize,
}

impl AgentConfig {
    pub fn new(api_base: String, api_key: String, model: String, max_iterations: usize) -> Self {
        Self {
            api_base,
            api_key,
            model,
            max_iterations,
        }
    }

    pub fn with_max_iterations(&self, max_iterations: usize) -> Self {
        Self {
            api_base: self.api_base.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            max_iterations,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            api_base: "https://example.llm.api/v1".to_string(),
            api_key: String::new(),
            model: "glm-4.7".to_string(),
            max_iterations: 10,
        }
    }
}
