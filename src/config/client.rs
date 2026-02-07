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
