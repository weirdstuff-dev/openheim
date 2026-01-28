use actix_web::{web, HttpResponse, Responder};
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::agent::run_agent;
use crate::config::{AgentConfig, AppConfig};
use crate::core::models::AgentResult;
use crate::core::llm::{LlmClient, OpenAiCompatibleClient};
use crate::tools::ToolExecutor;

#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Optional model name. If provided, resolves against AppConfig to pick the right provider.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<AgentResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn execute_agent(
    req: web::Json<AgentRequest>,
    default_llm: web::Data<Arc<dyn LlmClient>>,
    executor: web::Data<Arc<dyn ToolExecutor>>,
    config: web::Data<AgentConfig>,
    app_config: web::Data<AppConfig>,
    http_client: web::Data<ReqwestClient>,
) -> impl Responder {
    // Resolve the LLM client: use per-request model if specified, otherwise use the default
    let (llm_client, agent_config) = if let Some(model_name) = &req.model {
        match app_config.resolve(Some(model_name)) {
            Ok(mut resolved) => {
                if let Some(max_iter) = req.max_iterations {
                    resolved.max_iterations = max_iter;
                }
                let client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatibleClient::new(
                    http_client.get_ref().clone(),
                    resolved.api_base.clone(),
                    resolved.api_key.clone(),
                    resolved.model.clone(),
                ));
                (client, resolved)
            }
            Err(e) => {
                return HttpResponse::BadRequest().json(AgentResponse {
                    success: false,
                    result: None,
                    error: Some(e.to_string()),
                });
            }
        }
    } else {
        let mut cfg = config.get_ref().clone();
        if let Some(max_iter) = req.max_iterations {
            cfg.max_iterations = max_iter;
        }
        (default_llm.get_ref().clone(), cfg)
    };

    let tool_executor = executor.get_ref().clone();

    match run_agent(llm_client, tool_executor, &agent_config, &req.prompt, false).await {
        Ok(result) => HttpResponse::Ok().json(AgentResponse {
            success: true,
            result: Some(result),
            error: None,
        }),
        Err(e) => HttpResponse::InternalServerError().json(AgentResponse {
            success: false,
            result: None,
            error: Some(e.to_string()),
        }),
    }
}
