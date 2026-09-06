use reqwest::{Client as ReqwestClient, redirect::Policy};
use std::sync::Arc;
use std::time::Duration;

use super::types::AgentConfig;
use crate::core::llm::{
    AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient, RetryClient,
};
use crate::error::Result;

/// Build a reqwest client with the configured timeout applied as a connect
/// timeout and a per-read idle timeout — deliberately *not* a total request
/// deadline.
///
/// A total timeout (`ClientBuilder::timeout`) spans the entire body read, so
/// any streaming generation longer than `timeout_secs` (trivial for thinking
/// models) dies mid-stream, unretryably: chunks already emitted to the caller
/// can't be replayed. Bounding only the connect phase and the gap between
/// reads still detects dead connections while letting healthy slow streams
/// run to completion as long as bytes keep arriving.
///
/// Redirects are disabled: reqwest's default policy strips the
/// `Authorization` header when a redirect changes host, but not when it only
/// downgrades `https://` to `http://` on the same host, which would send the
/// provider's API key over an unencrypted connection. None of the supported
/// providers redirect under normal operation, so refusing to follow one is
/// safe and turns a silent credential leak into a visible error instead.
pub fn build_http_client(timeout_secs: u64) -> Result<ReqwestClient> {
    let timeout = Duration::from_secs(timeout_secs);
    ReqwestClient::builder()
        .connect_timeout(timeout)
        .read_timeout(timeout)
        .redirect(Policy::none())
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
/// (`AgentState::prompt`) or a subagent runs under a different model
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
