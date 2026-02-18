use super::types::{AgentConfig, AppConfig};
use crate::error::{Error, Result};

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

    fn resolve_model(&self, model_name: &str) -> Result<AgentConfig> {
        for (name, provider) in &self.providers {
            if provider.models.contains(&model_name.to_string()) {
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
            "Model '{}' not found in any provider. Run `openheim --list` to see available models.",
            model_name
        )))
    }

    pub fn list_models(&self) -> Result<String> {
        if self.providers.is_empty() {
            return Err(Error::config(
                "No providers configured. Edit your config file to add at least one provider.",
            ));
        }

        let mut out = String::from("Configured providers:\n");
        for (name, provider) in &self.providers {
            let is_default = name == &self.default_provider;
            let suffix = if is_default { " (default)" } else { "" };
            out.push_str(&format!("\n  {}{}\n", name, suffix));
            out.push_str(&format!("    api_base: {}\n", provider.api_base));

            let models: Vec<String> = provider
                .models
                .iter()
                .map(|m| {
                    if m == &provider.default_model {
                        format!("{} (default)", m)
                    } else {
                        m.clone()
                    }
                })
                .collect();
            out.push_str(&format!("    models:   {}\n", models.join(", ")));
        }
        Ok(out)
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
            providers,
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
            providers: BTreeMap::new(),
        };
        let err = config.resolve(None).unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn list_models_shows_providers_and_models() {
        let config = sample_config();
        let output = config.list_models().unwrap();
        assert!(output.contains("openai"));
        assert!(output.contains("gpt-4 (default)"));
        assert!(output.contains("gpt-3.5-turbo"));
        assert!(output.contains("anthropic"));
        assert!(output.contains("(default)")); // default provider marker
    }

    #[test]
    fn list_models_errors_when_empty() {
        let config = AppConfig {
            default_provider: "openai".into(),
            max_iterations: 10,
            providers: BTreeMap::new(),
        };
        let err = config.list_models().unwrap_err();
        assert!(err.to_string().contains("No providers configured"));
    }
}
