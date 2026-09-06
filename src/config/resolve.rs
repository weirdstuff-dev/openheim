use super::types::{AgentConfig, AppConfig, EmbeddingConfig, ProviderConfig};
use crate::error::{Error, Result};

fn validate_provider(name: &str, provider: &ProviderConfig) -> Result<()> {
    if provider.api_base.is_empty() {
        return Err(Error::config(format!(
            "Provider '{}' has an empty api_base",
            name
        )));
    }
    if !provider.api_base.starts_with("http://") && !provider.api_base.starts_with("https://") {
        return Err(Error::config(format!(
            "Provider '{}' api_base '{}' must start with http:// or https://",
            name, provider.api_base
        )));
    }
    if provider.api_base.starts_with("http://") && !provider.resolve_api_key().is_empty() {
        return Err(Error::config(format!(
            "Provider '{}' api_base '{}' uses http:// but has an API key configured; \
             credentials must not be sent over an unencrypted connection. Use https:// \
             or drop the API key for keyless local endpoints",
            name, provider.api_base
        )));
    }
    if !provider.models.is_empty() && !provider.models.contains(&provider.default_model) {
        return Err(Error::config(format!(
            "Provider '{}' default_model '{}' is not listed in models: [{}]",
            name,
            provider.default_model,
            provider.models.join(", ")
        )));
    }
    Ok(())
}

impl AppConfig {
    /// The single place an [`AgentConfig`] is assembled from a provider entry;
    /// every `resolve_*` path below funnels through here so field defaults
    /// (like the timeout) cannot drift between them.
    fn agent_config(
        &self,
        provider_name: &str,
        provider: &ProviderConfig,
        model: String,
    ) -> AgentConfig {
        AgentConfig {
            provider_name: provider_name.to_string(),
            api_base: provider.api_base.clone(),
            api_key: provider.resolve_api_key(),
            model,
            max_iterations: self.max_iterations,
            timeout_secs: provider.resolve_timeout_secs(),
            max_tokens: provider.max_tokens,
        }
    }

    fn get_provider(&self, provider_name: &str) -> Result<&ProviderConfig> {
        self.providers.get(provider_name).ok_or_else(|| {
            Error::config(format!(
                "Provider '{}' not found in config. Available providers: {}",
                provider_name,
                self.provider_names()
            ))
        })
    }

    pub fn resolve(&self, model_name: Option<&str>) -> Result<AgentConfig> {
        match model_name {
            Some(name) => self.resolve_model(name),
            None => self.resolve_provider_default(&self.default_provider),
        }
    }

    /// Resolves a provider by name using its own `default_model`.
    pub fn resolve_provider_default(&self, provider_name: &str) -> Result<AgentConfig> {
        let provider = self.get_provider(provider_name)?;
        validate_provider(provider_name, provider)?;
        Ok(self.agent_config(provider_name, provider, provider.default_model.clone()))
    }

    pub fn resolve_with_provider(&self, provider_name: &str, model: &str) -> Result<AgentConfig> {
        let provider = self.get_provider(provider_name)?;
        validate_provider(provider_name, provider)?;
        if !provider.models.is_empty() && !provider.models.contains(&model.to_string()) {
            return Err(Error::config(format!(
                "Model '{}' is not allowed for provider '{}'. Allowed models: [{}]",
                model,
                provider_name,
                provider.models.join(", ")
            )));
        }
        Ok(self.agent_config(provider_name, provider, model.to_string()))
    }

    fn resolve_model(&self, model_name: &str) -> Result<AgentConfig> {
        for (name, provider) in &self.providers {
            if provider.models.contains(&model_name.to_string()) {
                validate_provider(name, provider)?;
                return Ok(self.agent_config(name, provider, model_name.to_string()));
            }
        }
        Err(Error::config(format!(
            "Model '{}' not found in any provider. Check the [providers] section in your config file.",
            model_name
        )))
    }

    /// Resolves the embeddings endpoint named by `[memory]`'s
    /// `embedding_provider`. Returns `Ok(None)` when none is configured
    /// (keyword-only memory).
    pub fn resolve_embedding(&self) -> Result<Option<EmbeddingConfig>> {
        let Some(memory) = &self.memory else {
            return Ok(None);
        };
        let Some(provider_name) = memory.embedding_provider.as_deref() else {
            if memory.embedding_model.is_some() {
                return Err(Error::config(
                    "[memory] embedding_model is set but embedding_provider is not",
                ));
            }
            return Ok(None);
        };
        let model = memory
            .embedding_model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .ok_or_else(|| {
                Error::config("[memory] embedding_model is required when embedding_provider is set")
            })?;
        if provider_name == "anthropic" {
            return Err(Error::config(
                "[memory] embedding_provider 'anthropic' has no embeddings API; use an OpenAI-compatible provider or 'gemini'",
            ));
        }
        let provider = self.get_provider(provider_name)?;
        validate_provider(provider_name, provider)?;
        Ok(Some(EmbeddingConfig {
            provider_name: provider_name.to_string(),
            api_base: provider.api_base.clone(),
            api_key: provider.resolve_api_key(),
            model: model.to_string(),
            timeout_secs: provider.resolve_timeout_secs(),
        }))
    }

    fn provider_names(&self) -> String {
        self.providers
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use std::collections::BTreeMap;

    fn sample_config() -> AppConfig {
        let mut providers = BTreeMap::new();
        providers.insert(
            "openai".into(),
            ProviderConfig {
                api_base: "https://api.openai.com/v1".into(),
                default_model: "gpt-4".into(),
                models: vec!["gpt-4".into(), "gpt-3.5-turbo".into()],
                env_var: None,
                api_key: Some("test-key".into()),
                timeout_secs: Some(60),
                max_tokens: Some(4096),
            },
        );
        providers.insert(
            "anthropic".into(),
            ProviderConfig {
                api_base: "https://api.anthropic.com/v1".into(),
                default_model: "claude-3".into(),
                models: vec!["claude-3".into()],
                env_var: None,
                api_key: Some("anthropic-key".into()),
                timeout_secs: None,
                max_tokens: None,
            },
        );
        AppConfig {
            default_provider: "openai".into(),
            max_iterations: 5,
            theme_color: None,
            providers,
            mcp_servers: BTreeMap::new(),
            default_skills: vec![],
            work_dir: None,
            allow_shell: true,
            memory: None,
        }
    }

    #[test]
    fn resolve_none_returns_default_provider() {
        let config = sample_config();
        let agent = config.resolve(None).unwrap();
        assert_eq!(agent.provider_name, "openai");
        assert_eq!(agent.model, "gpt-4");
        assert_eq!(agent.api_key, "test-key");
        assert_eq!(agent.max_iterations, 5);
        assert_eq!(agent.timeout_secs, 60);
        assert_eq!(agent.max_tokens, Some(4096));
    }

    #[test]
    fn resolve_specific_model_finds_correct_provider() {
        let config = sample_config();
        let agent = config.resolve(Some("claude-3")).unwrap();
        assert_eq!(agent.provider_name, "anthropic");
        assert_eq!(agent.model, "claude-3");
        assert_eq!(agent.api_key, "anthropic-key");
        assert_eq!(agent.timeout_secs, 120); // default when None
    }

    #[test]
    fn resolve_unknown_model_returns_error() {
        let config = sample_config();
        let err = config.resolve(Some("unknown-model")).unwrap_err();
        assert!(err.to_string().contains("unknown-model"));
    }

    #[test]
    fn resolve_default_errors_when_provider_missing() {
        let config = AppConfig {
            default_provider: "nonexistent".into(),
            max_iterations: 10,
            theme_color: None,
            providers: BTreeMap::new(),
            mcp_servers: BTreeMap::new(),
            default_skills: vec![],
            work_dir: None,
            allow_shell: true,
            memory: None,
        };
        let err = config.resolve(None).unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
    }

    fn provider_with_base(api_base: &str) -> ProviderConfig {
        ProviderConfig {
            api_base: api_base.into(),
            default_model: "gpt-4".into(),
            models: vec!["gpt-4".into()],
            env_var: None,
            api_key: Some("key".into()),
            timeout_secs: None,
            max_tokens: None,
        }
    }

    #[test]
    fn validate_rejects_empty_api_base() {
        let p = provider_with_base("");
        let err = validate_provider("test", &p).unwrap_err();
        assert!(err.to_string().contains("empty api_base"));
    }

    #[test]
    fn validate_rejects_non_http_api_base() {
        let p = provider_with_base("ftp://example.com");
        let err = validate_provider("test", &p).unwrap_err();
        assert!(err.to_string().contains("http://") || err.to_string().contains("https://"));
    }

    #[test]
    fn validate_rejects_http_api_base_with_api_key() {
        let p = provider_with_base("http://example.com/v1");
        let err = validate_provider("test", &p).unwrap_err();
        assert!(err.to_string().contains("http://"));
        assert!(err.to_string().contains("API key"));
    }

    #[test]
    fn validate_allows_http_api_base_without_api_key() {
        // Keyless local endpoints (e.g. Ollama) have nothing to leak, so
        // plain http:// is fine as long as no credential is configured.
        let mut p = provider_with_base("http://localhost:11434/v1");
        p.api_key = None;
        assert!(validate_provider("test", &p).is_ok());
    }

    #[test]
    fn validate_rejects_default_model_not_in_models() {
        let mut p = provider_with_base("https://api.example.com");
        p.default_model = "gpt-5".into();
        let err = validate_provider("test", &p).unwrap_err();
        assert!(err.to_string().contains("gpt-5"));
    }

    #[test]
    fn validate_accepts_valid_provider() {
        let p = provider_with_base("https://api.example.com");
        assert!(validate_provider("test", &p).is_ok());
    }

    #[test]
    fn resolve_provider_default_uses_providers_default_model() {
        let config = sample_config();
        let agent = config.resolve_provider_default("anthropic").unwrap();
        assert_eq!(agent.provider_name, "anthropic");
        assert_eq!(agent.model, "claude-3");
        assert_eq!(agent.timeout_secs, 120); // default when None
    }

    #[test]
    fn resolve_provider_default_errors_on_unknown_provider() {
        let config = sample_config();
        let err = config.resolve_provider_default("nope").unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn resolve_with_provider_rejects_unlisted_model() {
        let config = sample_config();
        let err = config
            .resolve_with_provider("openai", "gpt-99")
            .unwrap_err();
        assert!(err.to_string().contains("gpt-99"));
        assert!(err.to_string().contains("openai"));
    }

    #[test]
    fn resolve_with_provider_accepts_listed_model() {
        let config = sample_config();
        let agent = config
            .resolve_with_provider("openai", "gpt-3.5-turbo")
            .unwrap();
        assert_eq!(agent.model, "gpt-3.5-turbo");
    }

    #[test]
    fn resolve_embedding_none_without_provider() {
        let mut config = sample_config();
        assert!(config.resolve_embedding().unwrap().is_none());
        config.memory = Some(crate::config::MemoryConfig::default());
        assert!(config.resolve_embedding().unwrap().is_none());
    }

    #[test]
    fn resolve_embedding_uses_named_provider_credentials() {
        let mut config = sample_config();
        config.memory = Some(crate::config::MemoryConfig {
            embedding_provider: Some("openai".into()),
            embedding_model: Some("text-embedding-3-small".into()),
            ..Default::default()
        });
        let emb = config.resolve_embedding().unwrap().unwrap();
        assert_eq!(emb.provider_name, "openai");
        assert_eq!(emb.api_key, "test-key");
        assert_eq!(emb.api_base, "https://api.openai.com/v1");
        assert_eq!(emb.model, "text-embedding-3-small");
        assert_eq!(emb.timeout_secs, 60);
    }

    #[test]
    fn resolve_embedding_rejects_bad_combinations() {
        let mut config = sample_config();
        let mut memory = crate::config::MemoryConfig {
            embedding_provider: Some("nope".into()),
            embedding_model: Some("m".into()),
            ..Default::default()
        };
        config.memory = Some(memory.clone());
        assert!(config.resolve_embedding().is_err());

        memory.embedding_provider = Some("anthropic".into());
        config.memory = Some(memory.clone());
        assert!(
            config
                .resolve_embedding()
                .unwrap_err()
                .to_string()
                .contains("embeddings API")
        );

        memory.embedding_provider = Some("openai".into());
        memory.embedding_model = Some("  ".into());
        config.memory = Some(memory.clone());
        assert!(config.resolve_embedding().is_err());

        memory.embedding_provider = None;
        memory.embedding_model = Some("m".into());
        config.memory = Some(memory);
        assert!(config.resolve_embedding().is_err());
    }

    #[test]
    fn resolve_with_provider_allows_any_model_when_list_empty() {
        let mut config = sample_config();
        config.providers.get_mut("openai").unwrap().models.clear();
        let agent = config
            .resolve_with_provider("openai", "any-future-model")
            .unwrap();
        assert_eq!(agent.model, "any-future-model");
    }
}
