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
    /// Maximum output tokens for LLM responses
    pub max_tokens: Option<u32>,
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
    /// Maximum output tokens for LLM responses (provider-specific defaults if not set)
    pub max_tokens: Option<u32>,
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
            max_tokens: None,
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
            max_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_provider(env_var: Option<&str>, api_key: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            api_base: "https://api.example.com".into(),
            default_model: "model-1".into(),
            models: vec!["model-1".into()],
            env_var: env_var.map(String::from),
            api_key: api_key.map(String::from),
            timeout_secs: None,
            max_tokens: None,
        }
    }

    #[test]
    fn resolve_api_key_from_env_var() {
        let var_name = "OPENHEIM_TEST_KEY_ENV";
        unsafe { std::env::set_var(var_name, "secret-from-env"); }
        let provider = sample_provider(Some(var_name), Some("inline-key"));
        assert_eq!(provider.resolve_api_key(), "secret-from-env");
        unsafe { std::env::remove_var(var_name); }
    }

    #[test]
    fn resolve_api_key_falls_back_to_inline() {
        let var_name = "OPENHEIM_TEST_KEY_MISSING";
        unsafe { std::env::remove_var(var_name); }
        let provider = sample_provider(Some(var_name), Some("inline-key"));
        assert_eq!(provider.resolve_api_key(), "inline-key");
    }

    #[test]
    fn resolve_api_key_returns_empty_when_none() {
        let var_name = "OPENHEIM_TEST_KEY_NONE";
        unsafe { std::env::remove_var(var_name); }
        let provider = sample_provider(Some(var_name), None);
        assert_eq!(provider.resolve_api_key(), "");
    }

    #[test]
    fn resolve_api_key_no_env_var_configured() {
        let provider = sample_provider(None, Some("inline-only"));
        assert_eq!(provider.resolve_api_key(), "inline-only");
    }

    #[test]
    fn resolve_api_key_empty_env_var_falls_back() {
        let var_name = "OPENHEIM_TEST_KEY_EMPTY";
        unsafe { std::env::set_var(var_name, "  "); }
        let provider = sample_provider(Some(var_name), Some("fallback"));
        assert_eq!(provider.resolve_api_key(), "fallback");
        unsafe { std::env::remove_var(var_name); }
    }

    #[test]
    fn agent_config_new_sets_defaults() {
        let cfg = AgentConfig::new(
            "openai".into(),
            "https://api.openai.com".into(),
            "key".into(),
            "gpt-4".into(),
            5,
        );
        assert_eq!(cfg.provider_name, "openai");
        assert_eq!(cfg.max_iterations, 5);
        assert_eq!(cfg.timeout_secs, 120);
        assert!(cfg.max_tokens.is_none());
    }

    #[test]
    fn with_max_iterations_clones_with_new_value() {
        let cfg = AgentConfig::new("p".into(), "b".into(), "k".into(), "m".into(), 5);
        let updated = cfg.with_max_iterations(20);
        assert_eq!(updated.max_iterations, 20);
        assert_eq!(updated.provider_name, "p");
    }

    #[test]
    fn arc_with_max_iterations_reuses_arc_when_same() {
        let cfg = Arc::new(AgentConfig::new("p".into(), "b".into(), "k".into(), "m".into(), 10));
        let same = cfg.arc_with_max_iterations(10);
        assert!(Arc::ptr_eq(&cfg, &same));
    }

    #[test]
    fn arc_with_max_iterations_creates_new_when_different() {
        let cfg = Arc::new(AgentConfig::new("p".into(), "b".into(), "k".into(), "m".into(), 10));
        let different = cfg.arc_with_max_iterations(20);
        assert!(!Arc::ptr_eq(&cfg, &different));
        assert_eq!(different.max_iterations, 20);
    }

    #[test]
    fn agent_config_default_has_correct_values() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.max_iterations, 10);
        assert_eq!(cfg.timeout_secs, 120);
        assert!(cfg.provider_name.is_empty());
    }

    #[test]
    fn app_config_deserializes_with_default_max_iterations() {
        let toml_str = r#"
            default_provider = "openai"
            [providers]
        "#;
        let cfg: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_iterations, 10);
    }
}
