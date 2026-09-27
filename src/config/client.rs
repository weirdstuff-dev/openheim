use reqwest::{Client as ReqwestClient, redirect::Policy};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
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

/// One `reqwest::Client` per timeout, built on first use and shared by
/// every LLM client made from it, so they share connection pools and TLS
/// sessions instead of each opening their own.
#[derive(Debug, Default)]
pub(crate) struct HttpClients(Mutex<HashMap<u64, ReqwestClient>>);

impl HttpClients {
    /// The shared client for `timeout_secs` (see [`build_http_client`]).
    pub(crate) fn get(&self, timeout_secs: u64) -> Result<ReqwestClient> {
        // A poisoned lock still holds a usable map: entries are only ever
        // inserted whole.
        let mut clients = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(&timeout_secs) {
            return Ok(client.clone());
        }
        let client = build_http_client(timeout_secs)?;
        clients.insert(timeout_secs, client.clone());
        Ok(client)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// Whether `a` and `b` name the same model, so one LLM client serves both.
pub(crate) fn same_model(a: &AgentConfig, b: &AgentConfig) -> bool {
    a.provider_name == b.provider_name && a.model == b.model
}

/// The LLM client for `target`: `baseline_llm` when it has the same provider
/// and model, otherwise a new one on `http`'s shared `reqwest::Client`. Used
/// for sessions that switched models and for subagents.
pub(crate) fn client_for_config(
    target: &AgentConfig,
    baseline: &AgentConfig,
    baseline_llm: &Arc<dyn LlmClient>,
    http: &HttpClients,
) -> Result<Arc<dyn LlmClient>> {
    if same_model(target, baseline) {
        Ok(baseline_llm.clone())
    } else {
        Ok(create_client(target, &http.get(target.timeout_secs)?))
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

        let client =
            client_for_config(&target, &baseline, &baseline_llm, &HttpClients::default()).unwrap();
        assert!(Arc::ptr_eq(&client, &baseline_llm));
    }

    #[test]
    fn client_for_config_builds_new_when_model_differs() {
        let baseline = baseline_config();
        let baseline_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);

        let mut target = baseline.clone();
        target.model = "gpt-4o".into();

        let client =
            client_for_config(&target, &baseline, &baseline_llm, &HttpClients::default()).unwrap();
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

        let client =
            client_for_config(&target, &baseline, &baseline_llm, &HttpClients::default()).unwrap();
        assert!(!Arc::ptr_eq(&client, &baseline_llm));
    }

    /// Clients for different models share one `reqwest::Client` per
    /// timeout rather than building their own.
    #[test]
    fn client_for_config_shares_http_clients_by_timeout() {
        let baseline = baseline_config();
        let baseline_llm: Arc<dyn LlmClient> = Arc::new(DummyClient);
        let http = HttpClients::default();

        for model in ["gpt-4o", "gpt-4o-mini"] {
            let mut target = baseline.clone();
            target.model = model.into();
            client_for_config(&target, &baseline, &baseline_llm, &http).unwrap();
        }
        assert_eq!(http.len(), 1);

        let mut slow = baseline.clone();
        slow.model = "o3".into();
        slow.timeout_secs = 600;
        client_for_config(&slow, &baseline, &baseline_llm, &http).unwrap();
        assert_eq!(http.len(), 2);
    }
}
