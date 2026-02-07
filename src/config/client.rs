use reqwest::Client as ReqwestClient;
use std::sync::Arc;

use super::types::{AgentConfig, AppConfig};
use crate::core::llm::{
    AnthropicClient, GeminiClient, LlmClient, OpenAiClient, OpenAiCompatibleClient,
};
use crate::error::Result;

/// Create the appropriate LLM client based on the provider name.
pub fn create_client(config: &AgentConfig, http_client: &ReqwestClient) -> Arc<dyn LlmClient> {
    match config.provider_name.as_str() {
        "openai" => Arc::new(OpenAiClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
        )),
        "anthropic" => Arc::new(AnthropicClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
        )),
        "gemini" => Arc::new(GeminiClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
        )),
        _ => Arc::new(OpenAiCompatibleClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
        )),
    }
}

/// Resolves the LLM client and agent configuration based on an optional model name and max iterations.
///
/// If `model_name` is provided, resolves against the `AppConfig` to determine the correct provider.
/// Otherwise, uses the provided `default_llm` and `default_config`.
pub fn resolve_client_and_config(
    model_name: Option<&str>,
    max_iterations: Option<usize>,
    app_config: &AppConfig,
    http_client: &ReqwestClient,
    default_llm: Arc<dyn LlmClient>,
    default_config: &AgentConfig,
) -> Result<(Arc<dyn LlmClient>, AgentConfig)> {
    if let Some(model) = model_name {
        let mut resolved = app_config.resolve(Some(model))?;
        if let Some(max_iter) = max_iterations {
            resolved.max_iterations = max_iter;
        }
        let client = create_client(&resolved, http_client);
        Ok((client, resolved))
    } else {
        let mut cfg = default_config.clone();
        if let Some(max_iter) = max_iterations {
            cfg.max_iterations = max_iter;
        }
        Ok((default_llm, cfg))
    }
}
