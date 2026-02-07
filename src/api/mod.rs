use actix_cors::Cors;
use actix_web::{middleware::Logger, web, App, HttpServer};
use reqwest::Client;
use std::sync::Arc;

use crate::{
    AgentConfig, AppConfig,
    config::create_client,
    error::Result,
    llm::LlmClient,
    rag::RagContext,
    tools::{SystemToolExecutor, ToolExecutor},
};

pub mod rest;
pub mod ws;
pub mod ws_fs;

pub use rest::execute_agent;
pub use ws::ws_handler;
pub use ws_fs::ws_fs_handler;

pub async fn start_api_server(
    host: String,
    port: u16,
    client: Client,
    config: AgentConfig,
    app_config: AppConfig,
) -> Result<()> {
    tracing::info!("Starting API server on {}:{}", host, port);
    tracing::info!("  POST /query");
    tracing::info!("  WS   /ws");
    tracing::info!("  WS   /ws/fs");

    let llm_client: Arc<dyn LlmClient> = create_client(&config, &client);
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());
    let rag_context = RagContext::new()?;

    let server = HttpServer::new(move || {
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
            .app_data(web::Data::new(app_config.clone()))
            .app_data(web::Data::new(rag_context.clone()))
            .route("/query", web::post().to(execute_agent))
            .route("/ws", web::get().to(ws_handler))
            .route("/ws/fs", web::get().to(ws_fs_handler))
    })
    .bind((host.as_str(), port))
    .map_err(|e| crate::error::Error::Other(format!("Failed to bind server: {}", e)))?
    .run();

    let handle = server.handle();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("Received shutdown signal, gracefully stopping server...");
            handle.stop(true).await;
        }
    });

    server
        .await
        .map_err(|e| crate::error::Error::Other(format!("Server error: {}", e)))?;

    tracing::info!("Server stopped");
    Ok(())
}
