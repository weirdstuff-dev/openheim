use super::types::{AgentConfig, AppConfig, EmbeddingConfig, ProviderConfig, ProviderKind};
use crate::error::{Error, Result};

/// Whether `url`'s host is this machine: `localhost`, `127.0.0.0/8` or
/// `::1`, judged as written, without a DNS lookup. Requests there are sent
/// directly, never through a proxy (see `config::client`), so they don't
/// leave the machine.
pub(crate) fn is_loopback_url(url: &str) -> bool {
    let Some(host) = reqwest::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_string))
    else {
        return false;
    };
    // IPv6 literals come bracketed: `[::1]`.
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Whether `url` is plain `http://` to a host off this machine, where
/// credentials would cross the network unencrypted. Loopback
/// ([`is_loopback_url`]) never leaves the machine, so a key may go there
/// over `http://`, as local model servers and proxies expect.
pub(crate) fn is_insecure_remote_http(url: &str) -> bool {
    match reqwest::Url::parse(url) {
        Ok(parsed) => parsed.scheme() == "http" && !is_loopback_url(url),
        Err(_) => url.starts_with("http://"),
    }
}

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
    if is_insecure_remote_http(&provider.api_base) && !provider.resolve_api_key().is_empty() {
        return Err(Error::config(format!(
            "Provider '{}' api_base '{}' uses http:// to a non-loopback host but has an API \
             key configured; credentials must not be sent over an unencrypted connection. \
             Use https://, or drop the API key for keyless endpoints",
            name, provider.api_base
        )));
    }
    if provider.context_window == Some(0) {
        return Err(Error::config(format!(
            "Provider '{}' context_window must be greater than 0",
            name
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
        let kind = provider.resolve_kind(provider_name);
        AgentConfig {
            provider_name: provider_name.to_string(),
            kind,
            api_base: provider.api_base.clone(),
            api_key: provider.resolve_api_key(),
            model,
            max_iterations: self.max_iterations,
            timeout_secs: provider.resolve_timeout_secs(),
            max_tokens: provider.max_tokens,
            context_window: provider.context_window,
            effort: provider.effort,
            replay_thinking: provider.replay_thinking.unwrap_or(false),
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
        let provider = self.get_provider(provider_name)?;
        let kind = provider.resolve_kind(provider_name);
        if kind == ProviderKind::Anthropic {
            return Err(Error::config(format!(
                "[memory] embedding_provider '{provider_name}' is an Anthropic provider, which has no embeddings API; use an OpenAI-compatible or Gemini provider"
            )));
        }
        validate_provider(provider_name, provider)?;
        Ok(Some(EmbeddingConfig {
            provider_name: provider_name.to_string(),
            kind,
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

    fn sample_config() -> AppConfig {
        let mut config = AppConfig::new("openai");
        config.max_iterations = 5;
        config.providers.insert(
            "openai".into(),
            ProviderConfig {
                api_key: Some("test-key".into()),
                timeout_secs: Some(60),
                max_tokens: Some(4096),
                ..ProviderConfig::for_tests(
                    "https://api.openai.com/v1",
                    &["gpt-4", "gpt-3.5-turbo"],
                )
            },
        );
        config.providers.insert(
            "anthropic".into(),
            ProviderConfig {
                api_key: Some("anthropic-key".into()),
                ..ProviderConfig::for_tests("https://api.anthropic.com/v1", &["claude-3"])
            },
        );
        config
    }

    fn provider_with_base(api_base: &str) -> ProviderConfig {
        ProviderConfig::for_tests(api_base, &["gpt-4"])
    }

    // A key sent to this machine over http:// never crosses the network,
    // as with a local proxy or model server that wants one.
    #[test]
    fn validate_allows_http_api_base_with_api_key_on_loopback() {
        for base in [
            "http://localhost:8080/v1",
            "http://LOCALHOST/v1",
            "http://127.0.0.1:4000",
            "http://127.1.2.3/v1",
            "http://[::1]:8080/v1",
        ] {
            let p = provider_with_base(base);
            assert!(validate_provider("test", &p).is_ok(), "{base}");
        }
    }

    #[test]
    fn only_plain_http_off_this_machine_is_insecure() {
        for url in [
            "http://example.com",
            "http://192.168.1.10:8080",
            "http://localhost.example.com",
            "http://[::2]",
            "http://0.0.0.0:8080",
        ] {
            assert!(is_insecure_remote_http(url), "{url}");
        }
        for url in [
            "https://example.com",
            "http://localhost",
            "http://127.0.0.1",
            "http://[::1]",
        ] {
            assert!(!is_insecure_remote_http(url), "{url}");
        }
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

    fn config_from_toml(providers: &str) -> AppConfig {
        toml::from_str(&format!("default_provider = \"x\"\n{providers}")).unwrap()
    }

    #[test]
    fn embeddings_reject_anthropic_by_kind_not_name() {
        let mut config = config_from_toml(
            r#"
            [providers.claude-work]
            kind = "anthropic"
            api_base = "https://api.anthropic.com/v1"
            default_model = "claude-sonnet-4-6"
            models = ["claude-sonnet-4-6"]
            api_key = "k"
            "#,
        );
        config.memory = Some(crate::config::MemoryConfig {
            embedding_provider: Some("claude-work".into()),
            embedding_model: Some("m".into()),
            ..Default::default()
        });
        let err = config.resolve_embedding().unwrap_err().to_string();
        assert!(err.contains("embeddings API"), "{err}");
    }
}
