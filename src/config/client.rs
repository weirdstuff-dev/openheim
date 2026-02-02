use reqwest::Client as ReqwestClient;
use std::sync::Arc;

use super::types::{AgentConfig, AppConfig};
use crate::core::llm::{LlmClient, OpenAiCompatibleClient};

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
) -> Result<(Arc<dyn LlmClient>, AgentConfig), String> {
    if let Some(model) = model_name {
        match app_config.resolve(Some(model)) {
            Ok(mut resolved) => {
                if let Some(max_iter) = max_iterations {
                    resolved.max_iterations = max_iter;
                }
                let client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatibleClient::new(
                    http_client.clone(),
                    resolved.api_base.clone(),
                    resolved.api_key.clone(),
                    resolved.model.clone(),
                ));
                Ok((client, resolved))
            }
            Err(e) => Err(e.to_string()),
        }
    } else {
        let mut cfg = default_config.clone();
        if let Some(max_iter) = max_iterations {
            cfg.max_iterations = max_iter;
        }
        Ok((default_llm, cfg))
    }
}
