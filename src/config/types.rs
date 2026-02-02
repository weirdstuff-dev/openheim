use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Top-level configuration loaded from ~/.openheim/config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub default_provider: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

fn default_max_iterations() -> usize {
    10
}

/// Per-provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_base: String,
    pub default_model: String,
    pub models: Vec<String>,
    /// Name of the environment variable holding the API key (e.g. "OPENAI_API_KEY")
    pub env_var: Option<String>,
    /// Inline API key (not recommended - prefer env_var)
    pub api_key: Option<String>,
}

impl ProviderConfig {
    /// Resolve the API key: try env_var first, then inline api_key, then empty string (for keyless providers like Ollama)
    pub fn resolve_api_key(&self) -> String {
        if let Some(env_var) = &self.env_var {
            if let Ok(key) = std::env::var(env_var) {
                if !key.trim().is_empty() {
                    return key;
                }
            }
        }
        self.api_key.clone().unwrap_or_default()
    }
}

/// Runtime configuration passed to agent/LLM code
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

    pub fn arc_with_max_iterations(self: &Arc<Self>, max_iterations: usize) -> Arc<Self> {
        if self.max_iterations == max_iterations {
            Arc::clone(self)
        } else {
            Arc::new(Self {
                api_base: self.api_base.clone(),
                api_key: self.api_key.clone(),
                model: self.model.clone(),
                max_iterations,
            })
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            api_base: String::new(),
            api_key: String::new(),
            model: String::new(),
            max_iterations: 10,
        }
    }
}
