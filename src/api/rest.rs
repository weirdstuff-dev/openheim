use actix_web::{web, HttpResponse, Responder};
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::agent::run_agent_with_history;
use crate::config::{AgentConfig, AppConfig, resolve_client_and_config};
use crate::core::models::{AgentResult, Message};
use crate::core::llm::LlmClient;
use crate::rag::RagContext;
use crate::tools::ToolExecutor;

#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    pub prompt: String,
    /// Optional model name. If provided, resolves against AppConfig to pick the right provider.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub chat_id: Option<Uuid>,
    #[serde(default)]
    pub skills: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<AgentResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<Uuid>,
}

pub async fn execute_agent(
    req: web::Json<AgentRequest>,
    default_llm: web::Data<Arc<dyn LlmClient>>,
    executor: web::Data<Arc<dyn ToolExecutor>>,
    config: web::Data<AgentConfig>,
    app_config: web::Data<AppConfig>,
    http_client: web::Data<ReqwestClient>,
) -> impl Responder {
    let (llm_client, agent_config) = match resolve_client_and_config(
        req.model.as_deref(),
        req.max_iterations,
        app_config.get_ref(),
        http_client.get_ref(),
        default_llm.get_ref().clone(),
        config.get_ref(),
    ) {
        Ok((client, config)) => (client, config),
        Err(e) => {
            return HttpResponse::BadRequest().json(AgentResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
                chat_id: None,
            });
        }
    };

    let rag = match RagContext::new() {
        Ok(r) => r,
        Err(e) => {
            return HttpResponse::InternalServerError().json(AgentResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
                chat_id: None,
            });
        }
    };

    let skill_names = req.skills.clone().unwrap_or_default();
    let (mut conversation, prompt_builder) = match rag.prepare(
        req.chat_id,
        &skill_names,
        Some(agent_config.model.clone()),
        Some(agent_config.provider_name.clone()),
    ) {
        Ok(v) => v,
        Err(e) => {
            return HttpResponse::BadRequest().json(AgentResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
                chat_id: None,
            });
        }
    };

    conversation.messages.push(Message::user(req.prompt.clone()));

    let tool_executor = executor.get_ref().clone();
    let chat_id = conversation.meta.id;

    match run_agent_with_history(
        llm_client,
        tool_executor,
        &agent_config,
        &mut conversation.messages,
        false,
        Some(&prompt_builder),
    )
    .await
    {
        Ok(result) => {
            let _ = rag.history.save_conversation(&conversation);
            HttpResponse::Ok().json(AgentResponse {
                success: true,
                result: Some(result),
                error: None,
                chat_id: Some(chat_id),
            })
        }
        Err(e) => {
            let _ = rag.history.save_conversation(&conversation);
            HttpResponse::InternalServerError().json(AgentResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
                chat_id: Some(chat_id),
            })
        }
    }
}
