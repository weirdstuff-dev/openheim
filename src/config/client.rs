use reqwest::Client as ReqwestClient;
use std::sync::Arc;
use std::time::Duration;

use super::types::AgentConfig;
use crate::core::llm::{
    AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient, RetryClient,
};
use crate::error::Result;

/// Build a reqwest client with the configured timeout.
pub fn build_http_client(timeout_secs: u64) -> Result<ReqwestClient> {
    ReqwestClient::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| crate::error::Error::Other(format!("failed to build HTTP client: {}", e)))
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

/// Returns the LLM client to use for `target`, reusing `baseline_llm` when it
/// already points at the same provider and model, and otherwise building a fresh
/// client for `target`.
///
/// This is the single source of truth for the "reuse the parent client unless
/// the model/provider changed" pattern needed when a session switches models
/// (`acp_prompt`) or a subagent runs under a different model
/// (`DelegateTool::resolve_runtime`).
pub fn client_for_config(
    target: &AgentConfig,
    baseline: &AgentConfig,
    baseline_llm: &Arc<dyn LlmClient>,
) -> Result<Arc<dyn LlmClient>> {
    if target.provider_name == baseline.provider_name && target.model == baseline.model {
        Ok(baseline_llm.clone())
    } else {
        let http_client = build_http_client(target.timeout_secs)?;
        Ok(create_client(target, &http_client))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::LlmClient;
    use crate::core::models::{Choice, Message, Tool};
    use async_trait::async_trait;

    struct DummyClient;

    #[async_trait]
    impl LlmClient for DummyClient {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> crate::error::Result<Choice> {
            unimplemented!()
        }
    }

    fn baseline_config() -> AgentConfig {
        AgentConfig::new(
            "openai".into(),
            "https://api.openai.com/v1".into(),
            "key".into(),
            "gpt-4".into(),
            10,
        )
    }

    #[test]
    fn client_for_config_reuses_baseline_when_provider_and_model_match() {
        let baseline = baseline_config();
        let baseline_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);

        // Same provider + model, even with a different api_base/timeout.
        let mut target = baseline.clone();
        target.timeout_secs = 999;

        let client = client_for_config(&target, &baseline, &baseline_llm).unwrap();
        assert!(Arc::ptr_eq(&client, &baseline_llm));
    }

    #[test]
    fn client_for_config_builds_new_when_model_differs() {
        let baseline = baseline_config();
        let baseline_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);

        let mut target = baseline.clone();
        target.model = "gpt-4o".into();

        let client = client_for_config(&target, &baseline, &baseline_llm).unwrap();
        assert!(!Arc::ptr_eq(&client, &baseline_llm));
    }

    #[test]
    fn client_for_config_builds_new_when_provider_differs() {
        let baseline = baseline_config();
        let baseline_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);

        let target = AgentConfig::new(
            "anthropic".into(),
            "https://api.anthropic.com/v1".into(),
            "key".into(),
            "gpt-4".into(),
            10,
        );

        let client = client_for_config(&target, &baseline, &baseline_llm).unwrap();
        assert!(!Arc::ptr_eq(&client, &baseline_llm));
    }
}
