use reqwest::{Client as ReqwestClient, redirect::Policy};
use std::sync::Arc;
use std::time::Duration;

use super::types::{AgentConfig, ProviderKind};
use crate::core::llm::{
    AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient, RetryClient,
};
use crate::error::Result;

/// Builds a reqwest client whose timeout bounds the connect and each read,
/// not the whole request: a long streamed reply keeps going as long as bytes
/// arrive, while a dead connection is still caught.
///
/// Redirects are refused: reqwest keeps the `Authorization` header on a
/// same-host `https://` → `http://` redirect, which would send the API key
/// in the clear, and providers don't redirect anyway.
pub fn build_http_client(timeout_secs: u64) -> Result<ReqwestClient> {
    let timeout = Duration::from_secs(timeout_secs);
    ReqwestClient::builder()
        .connect_timeout(timeout)
        .read_timeout(timeout)
        .redirect(Policy::none())
        .build()
        .map_err(|e| crate::error::Error::Other(format!("failed to build HTTP client: {}", e)))
}

/// Create the LLM client for `config.kind`, wrapped with retry logic.
pub fn create_client(config: &AgentConfig, http_client: &ReqwestClient) -> Arc<dyn LlmClient> {
    let inner: Arc<dyn LlmClient> = match config.kind {
        ProviderKind::OpenAi => Arc::new(OpenAiClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
        ProviderKind::Anthropic => Arc::new(AnthropicClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
            config.thinking,
        )),
        ProviderKind::Gemini => Arc::new(GeminiClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
        ProviderKind::OpenAiCompatible => Arc::new(OpenAiCompatibleClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
            config.max_tokens,
        )),
    };
    Arc::new(RetryClient::new(inner))
}

/// The LLM client for `target`: `baseline_llm` when it has the same provider
/// and model, otherwise a new one. Used for sessions that switched models
/// and for subagents.
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
