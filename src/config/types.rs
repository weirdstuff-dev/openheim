use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Public model info for a single provider (no credentials).
#[derive(Debug, Clone, Serialize)]
pub struct ProviderModels {
    pub default_model: String,
    pub models: Vec<String>,
}

/// JSON-safe summary of all configured providers and their models.
#[derive(Debug, Clone, Serialize)]
pub struct ModelsInfo {
    pub default_provider: String,
    pub providers: BTreeMap<String, ProviderModels>,
}

/// Public info for a single MCP server (no credentials or env vars).
#[derive(Debug, Clone, Serialize)]
pub struct McpServerInfo {
    pub transport: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Top-level configuration loaded from ~/.openheim/config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub default_provider: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
}

/// Configuration for a single MCP server connection.
/// The map key in `[mcp_servers.<name>]` is used as the server name and tool-name prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Binary to spawn for stdio transport (e.g. `"npx"`, `"uvx"`).
    pub command: Option<String>,
    /// Arguments passed to `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the spawned process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Base URL for Streamable HTTP transport (e.g. `"http://localhost:8080/mcp"`).
    pub url: Option<String>,
}

fn default_max_iterations() -> usize {
    10
}

impl AppConfig {
    pub fn models_info(&self) -> ModelsInfo {
        ModelsInfo {
            default_provider: self.default_provider.clone(),
            providers: self
                .providers
                .iter()
                .map(|(name, p)| {
                    (
                        name.clone(),
                        ProviderModels {
                            default_model: p.default_model.clone(),
                            models: p.models.clone(),
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn to_public_json(&self) -> serde_json::Value {
        let mut val = serde_json::to_value(self).unwrap_or_default();
        if let Some(providers) = val.get_mut("providers").and_then(|v| v.as_object_mut()) {
            for p in providers.values_mut() {
                if let Some(obj) = p.as_object_mut() {
                    obj.remove("api_key");
                }
            }
        }
        if let Some(servers) = val.get_mut("mcp_servers").and_then(|v| v.as_object_mut()) {
            for s in servers.values_mut() {
                if let Some(env) = s.get_mut("env").and_then(|v| v.as_object_mut()) {
                    for v in env.values_mut() {
                        *v = serde_json::Value::String("<redacted>".to_string());
                    }
                }
            }
        }
        val
    }

    pub fn mcp_servers_info(&self) -> BTreeMap<String, McpServerInfo> {
        self.mcp_servers
            .iter()
            .map(|(name, cfg)| {
                let info = if cfg.command.is_some() {
                    McpServerInfo { transport: "stdio", command: cfg.command.clone(), url: None }
                } else if cfg.url.is_some() {
                    McpServerInfo { transport: "http", command: None, url: cfg.url.clone() }
                } else {
                    McpServerInfo { transport: "unknown", command: None, url: None }
                };
                (name.clone(), info)
            })
            .collect()
    }
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
        if let Some(env_var) = &self.env_var
            && let Ok(key) = std::env::var(env_var)
            && !key.trim().is_empty()
        {
            return key;
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
