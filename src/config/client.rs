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
    build(timeout_secs, false)
}

/// [`build_http_client`] for requests to `api_base`. One on this machine
/// ([`super::is_loopback_url`]) is reached directly, ignoring any proxy
/// the environment sets (`HTTP_PROXY` and the like): a proxy would carry
/// its API key off the machine, over plain `http://` at that.
#[cfg(feature = "rag")]
pub(crate) fn build_http_client_for(timeout_secs: u64, api_base: &str) -> Result<ReqwestClient> {
    build(timeout_secs, super::is_loopback_url(api_base))
}

fn build(timeout_secs: u64, direct: bool) -> Result<ReqwestClient> {
    let timeout = Duration::from_secs(timeout_secs);
    let mut builder = ReqwestClient::builder()
        .connect_timeout(timeout)
        .read_timeout(timeout)
        .redirect(Policy::none());
    if direct {
        builder = builder.no_proxy();
    }
    builder
        .build()
        .map_err(|e| crate::error::Error::Other(format!("failed to build HTTP client: {}", e)))
}

/// Create the LLM client for `config.kind`, wrapped with retry logic.
pub fn create_client(config: &AgentConfig, http_client: &ReqwestClient) -> Arc<dyn LlmClient> {
    let inner: Arc<dyn LlmClient> = match config.kind {
        ProviderKind::OpenAi => Arc::new(
            OpenAiClient::new(
                http_client.clone(),
                config.api_base.clone(),
                config.api_key.clone(),
                config.model.clone(),
                config.max_tokens,
            )
            .with_replay_thinking(config.replay_thinking)
            .with_effort(config.effort),
        ),
        ProviderKind::Anthropic => Arc::new(
            AnthropicClient::new(
                http_client.clone(),
                config.api_base.clone(),
                config.api_key.clone(),
                config.model.clone(),
                config.max_tokens,
            )
            .with_effort(config.effort),
        ),
        ProviderKind::Gemini => Arc::new(
            GeminiClient::new(
                http_client.clone(),
                config.api_base.clone(),
                config.api_key.clone(),
                config.model.clone(),
                config.max_tokens,
            )
            .with_effort(config.effort),
        ),
        ProviderKind::OpenAiCompatible => Arc::new(
            OpenAiCompatibleClient::new(
                http_client.clone(),
                config.api_base.clone(),
                config.api_key.clone(),
                config.model.clone(),
                config.max_tokens,
            )
            .with_replay_thinking(config.replay_thinking)
            .with_effort(config.effort),
        ),
    };
    Arc::new(RetryClient::new(inner))
}

/// One `reqwest::Client` per timeout (and per whether it bypasses proxies),
/// built on first use and shared by every LLM client made from it, so they
/// share connection pools and TLS sessions instead of each opening their
/// own.
#[derive(Debug, Default)]
pub(crate) struct HttpClients(Mutex<HashMap<(u64, bool), ReqwestClient>>);

impl HttpClients {
    /// The shared client for requests to `api_base` with `timeout_secs`
    /// (see [`build_http_client_for`]).
    pub(crate) fn get(&self, timeout_secs: u64, api_base: &str) -> Result<ReqwestClient> {
        let key = (timeout_secs, super::is_loopback_url(api_base));
        // A poisoned lock still holds a usable map: entries are only ever
        // inserted whole.
        let mut clients = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(&key) {
            return Ok(client.clone());
        }
        let client = build(key.0, key.1)?;
        clients.insert(key, client.clone());
        Ok(client)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// Whether `a` and `b` name the same model at the same effort level, so
/// one LLM client serves both.
pub(crate) fn same_client(a: &AgentConfig, b: &AgentConfig) -> bool {
    a.provider_name == b.provider_name && a.model == b.model && a.effort == b.effort
}

/// The LLM client for `target`: `baseline_llm` when it has the same provider,
/// model and effort level, otherwise a new one on `http`'s shared
/// `reqwest::Client`. Used for sessions that switched models or effort
/// levels, and for subagents.
pub(crate) fn client_for_config(
    target: &AgentConfig,
    baseline: &AgentConfig,
    baseline_llm: &Arc<dyn LlmClient>,
    http: &HttpClients,
) -> Result<Arc<dyn LlmClient>> {
    if same_client(target, baseline) {
        Ok(baseline_llm.clone())
    } else {
        Ok(create_client(
            target,
            &http.get(target.timeout_secs, &target.api_base)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A loopback endpoint gets its own, proxy-free client; any loopback
    // spelling shares it.
    #[test]
    fn loopback_endpoints_get_a_client_of_their_own() {
        let http = HttpClients::default();
        http.get(30, "https://api.openai.com/v1").unwrap();
        http.get(30, "http://localhost:11434/v1").unwrap();
        assert_eq!(http.len(), 2);
        http.get(30, "http://127.0.0.1:8080").unwrap();
        http.get(30, "https://api.anthropic.com/v1").unwrap();
        assert_eq!(http.len(), 2);
    }
}
