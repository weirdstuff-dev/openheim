use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Top-level configuration loaded from ~/.openheim/config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub default_provider: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
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
    /// Request timeout in seconds (default: 120)
    pub timeout_secs: Option<u64>,
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
    pub provider_name: String,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub max_iterations: usize,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_timeout_secs() -> u64 {
    120
}

impl AgentConfig {
    pub fn new(provider_name: String, api_base: String, api_key: String, model: String, max_iterations: usize) -> Self {
        Self {
            provider_name,
            api_base,
            api_key,
            model,
            max_iterations,
            timeout_secs: default_timeout_secs(),
        }
    }

    pub fn with_max_iterations(&self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self.clone()
        }
    }

    pub fn arc_with_max_iterations(self: &Arc<Self>, max_iterations: usize) -> Arc<Self> {
        if self.max_iterations == max_iterations {
            Arc::clone(self)
        } else {
            Arc::new(self.with_max_iterations(max_iterations))
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider_name: String::new(),
            api_base: String::new(),
            api_key: String::new(),
            model: String::new(),
            max_iterations: 10,
            timeout_secs: default_timeout_secs(),
        }
    }
}
