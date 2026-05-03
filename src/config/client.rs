use reqwest::Client as ReqwestClient;
use std::sync::Arc;
use std::time::Duration;

use super::types::{AgentConfig, AppConfig};
use crate::core::llm::{
    AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient, RetryClient,
};
use crate::error::Result;

/// Build a reqwest client with the configured timeout.
pub fn build_http_client(timeout_secs: u64) -> ReqwestClient {
    ReqwestClient::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .expect("failed to build HTTP client")
}

/// Create the appropriate LLM client based on the provider name, wrapped with retry logic.
pub fn create_client(config: &AgentConfig, http_client: &ReqwestClient) -> Arc<dyn LlmClient> {
    let inner: Arc<dyn LlmClient> = match config.provider_name.as_str() {
        "openai" => Arc::new(OpenAiClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
        "anthropic" => Arc::new(AnthropicClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
        "gemini" => Arc::new(GeminiClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
        _ => Arc::new(OpenAiCompatibleClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
    };
    Arc::new(RetryClient::new(inner))
}

/// Resolves the LLM client and agent configuration based on an optional model name and max iterations.
///
/// If `model_name` is provided, resolves against the `AppConfig` to determine the correct provider.
/// Otherwise, uses the provided `default_llm` and `default_config`.
pub fn resolve_client_and_config(
    model_name: Option<&str>,
    max_iterations: Option<usize>,
    app_config: &AppConfig,
    default_llm: Arc<dyn LlmClient>,
    default_config: &AgentConfig,
) -> Result<(Arc<dyn LlmClient>, AgentConfig)> {
    if let Some(model) = model_name {
        let mut resolved = app_config.resolve(Some(model))?;
        if let Some(max_iter) = max_iterations {
            resolved.max_iterations = max_iter;
        }
        let resolved_http = build_http_client(resolved.timeout_secs);
        let client = create_client(&resolved, &resolved_http);
        Ok((client, resolved))
    } else {
        let mut cfg = default_config.clone();
        if let Some(max_iter) = max_iterations {
            cfg.max_iterations = max_iter;
        }
        Ok((default_llm, cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use crate::core::llm::LlmClient;
    use crate::core::models::{Choice, Message, Tool};
    use async_trait::async_trait;
    use std::collections::BTreeMap;

    struct DummyClient;

    #[async_trait]
    impl LlmClient for DummyClient {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> crate::error::Result<Choice> {
            unimplemented!()
        }
    }

    fn sample_app_config() -> AppConfig {
        let mut providers = BTreeMap::new();
        providers.insert(
            "openai".into(),
            ProviderConfig {
                api_base: "https://api.openai.com/v1".into(),
                default_model: "gpt-4".into(),
                models: vec!["gpt-4".into()],
                env_var: None,
                api_key: Some("key".into()),
                timeout_secs: Some(30),
                max_tokens: None,
            },
        );
        AppConfig {
            default_provider: "openai".into(),
            max_iterations: 10,
            providers,
            mcp_servers: BTreeMap::new(),
        }
    }

    #[test]
    fn resolve_client_uses_default_when_no_model() {
        let app_config = sample_app_config();
        let default_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);
        let default_config = AgentConfig::new(
            "openai".into(),
            "https://api.openai.com/v1".into(),
            "key".into(),
            "gpt-4".into(),
            10,
        );

        let (client, cfg) = resolve_client_and_config(
            None,
            None,
            &app_config,
            default_llm.clone(),
            &default_config,
        )
        .unwrap();

        assert!(Arc::ptr_eq(&client, &default_llm));
        assert_eq!(cfg.max_iterations, 10);
    }

    #[test]
    fn resolve_client_overrides_max_iterations() {
        let app_config = sample_app_config();
        let default_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);
        let default_config = AgentConfig::new(
            "openai".into(),
            "https://api.openai.com/v1".into(),
            "key".into(),
            "gpt-4".into(),
            10,
        );

        let (_client, cfg) = resolve_client_and_config(
            None,
            Some(25),
            &app_config,
            default_llm,
            &default_config,
        )
        .unwrap();

        assert_eq!(cfg.max_iterations, 25);
    }

    #[test]
    fn resolve_client_creates_new_for_specific_model() {
        let app_config = sample_app_config();
        let default_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);
        let default_config = AgentConfig::new(
            "openai".into(),
            "https://api.openai.com/v1".into(),
            "key".into(),
            "gpt-4".into(),
            10,
        );

        let (client, cfg) = resolve_client_and_config(
            Some("gpt-4"),
            Some(3),
            &app_config,
            default_llm.clone(),
            &default_config,
        )
        .unwrap();

        // New client is created, not the default one
        assert!(!Arc::ptr_eq(&client, &default_llm));
        assert_eq!(cfg.model, "gpt-4");
        assert_eq!(cfg.max_iterations, 3);
    }

    #[test]
    fn resolve_client_errors_for_unknown_model() {
        let app_config = sample_app_config();
        let default_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);
        let default_config = AgentConfig::default();

        let result = resolve_client_and_config(
            Some("nonexistent-model"),
            None,
            &app_config,
            default_llm,
            &default_config,
        );

        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("nonexistent-model"));
    }
}
