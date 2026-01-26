use actix_cors::Cors;
use actix_web::{middleware::Logger, web, App, HttpServer};
use anyhow::Result;
use reqwest::Client;
use std::sync::Arc;

use crate::{
    AgentConfig,
    llm::{LlmClient, OpenAiCompatibleClient},
    tools::{SystemToolExecutor, ToolExecutor},
};

pub mod rest;
pub mod ws;
pub mod ws_fs;

pub use rest::{execute_agent};
pub use ws::ws_handler;
pub use ws_fs::ws_fs_handler;

pub async fn start_api_server(
    host: String,
    port: u16,
    client: Client,
    config: AgentConfig,
) -> Result<()> {
    tracing::info!("🚀 Starting API server on {}:{}", host, port);
    tracing::info!("  POST /query           - Execute agent with prompt");
    tracing::info!("  WS   /ws              - WebSocket for streaming agent execution");
    tracing::info!("  WS   /ws/fs           - WebSocket for filesystem access (read/write/watch)");
    tracing::info!("");

    let llm_client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatibleClient::new(
        client.clone(),
        config.api_base.clone(),
        config.api_key.clone(),
        config.model.clone(),
    ));

    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(Logger::default())
            .wrap(cors)
            .app_data(web::Data::new(llm_client.clone()))
            .app_data(web::Data::new(tool_executor.clone()))
            .app_data(web::Data::new(config.clone()))
            .route("/query", web::post().to(execute_agent))
            .route("/ws", web::get().to(ws_handler))
            .route("/ws/fs", web::get().to(ws_fs_handler))
    })
    .bind((host.as_str(), port))?
    .run()
    .await?;

    Ok(())
}