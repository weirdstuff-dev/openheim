use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::agent::run_agent;
use crate::config::AgentConfig;
use crate::core::models::AgentResult;
use crate::core::llm::LlmClient;
use crate::tools::ToolExecutor;

#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    pub prompt: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

fn default_max_iterations() -> usize {
    10
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
    llm: web::Data<Arc<dyn LlmClient>>,
    executor: web::Data<Arc<dyn ToolExecutor>>,
    config: web::Data<AgentConfig>,
) -> impl Responder {
    let agent_config = config.with_max_iterations(req.max_iterations);

    let llm_client = llm.get_ref().clone();
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
