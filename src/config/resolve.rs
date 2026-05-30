use super::types::{AgentConfig, AppConfig, ProviderConfig};
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
    pub fn resolve(&self, model_name: Option<&str>) -> Result<AgentConfig> {
        match model_name {
            Some(name) => self.resolve_model(name),
            None => self.resolve_default(),
        }
    }

    fn resolve_default(&self) -> Result<AgentConfig> {
        let provider = self.providers.get(&self.default_provider).ok_or_else(|| {
            Error::config(format!(
                "Default provider '{}' not found in config. Available providers: {}",
                self.default_provider,
                self.provider_names()
            ))
        })?;
        validate_provider(&self.default_provider, provider)?;
        Ok(AgentConfig {
            provider_name: self.default_provider.clone(),
            api_base: provider.api_base.clone(),
            api_key: provider.resolve_api_key(),
            model: provider.default_model.clone(),
            max_iterations: self.max_iterations,
            timeout_secs: provider.timeout_secs.unwrap_or(120),
            max_tokens: provider.max_tokens,
        })
    }

    pub fn resolve_with_provider(&self, provider_name: &str, model: &str) -> Result<AgentConfig> {
        let provider = self.providers.get(provider_name).ok_or_else(|| {
            Error::config(format!(
                "Provider '{}' not found in config. Available providers: {}",
                provider_name,
                self.provider_names()
            ))
        })?;
        validate_provider(provider_name, provider)?;
        if !provider.models.is_empty() && !provider.models.contains(&model.to_string()) {
            return Err(Error::config(format!(
                "Model '{}' is not allowed for provider '{}'. Allowed models: [{}]",
                model,
                provider_name,
                provider.models.join(", ")
            )));
        }
        Ok(AgentConfig {
            provider_name: provider_name.to_string(),
            api_base: provider.api_base.clone(),
            api_key: provider.resolve_api_key(),
            model: model.to_string(),
            max_iterations: self.max_iterations,
            timeout_secs: provider.timeout_secs.unwrap_or(120),
            max_tokens: provider.max_tokens,
        })
    }

    fn resolve_model(&self, model_name: &str) -> Result<AgentConfig> {
        for (name, provider) in &self.providers {
            if provider.models.contains(&model_name.to_string()) {
                validate_provider(name, provider)?;
                return Ok(AgentConfig {
                    provider_name: name.clone(),
                    api_base: provider.api_base.clone(),
                    api_key: provider.resolve_api_key(),
                    model: model_name.to_string(),
                    max_iterations: self.max_iterations,
                    timeout_secs: provider.timeout_secs.unwrap_or(120),
                    max_tokens: provider.max_tokens,
                });
            }
        }
        Err(Error::config(format!(
            "Model '{}' not found in any provider. Check the [providers] section in your config file.",
            model_name
        )))
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
    fn resolve_with_provider_allows_any_model_when_list_empty() {
        let mut config = sample_config();
        config.providers.get_mut("openai").unwrap().models.clear();
        let agent = config
            .resolve_with_provider("openai", "any-future-model")
            .unwrap();
        assert_eq!(agent.model, "any-future-model");
    }
}
