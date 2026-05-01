use actix_cors::Cors;
use actix_web::{middleware::Logger, web, App, HttpServer};
use reqwest::Client;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

pub use rest::{execute_agent, get_config_handler, restart_handler, update_config_handler};
pub use ws::{ws_handler, Shutdown};

/// Shared registry of active WebSocket connections.
pub type WsRegistry = Arc<Mutex<Vec<actix::Addr<ws::OpenheimWebSocket>>>>;

/// Notify all connected WS clients of an imminent restart, then exit.
pub async fn do_restart(registry: &WsRegistry) {
    let addrs = registry.lock().unwrap().clone();
    for addr in addrs {
        addr.do_send(Shutdown);
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    std::process::exit(0);
}

pub async fn start_api_server(
    host: String,
    port: u16,
    client: Client,
    config: AgentConfig,
    app_config: AppConfig,
) -> Result<()> {
    tracing::info!("Starting API server on {}:{}", host, port);
    tracing::info!("  POST /query");
    tracing::info!("  GET  /config");
    tracing::info!("  PUT  /config");
    tracing::info!("  POST /restart");
    tracing::info!("  WS   /ws");

    let llm_client: Arc<dyn LlmClient> = create_client(&config, &client);
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());
    let rag_context = RagContext::new()?;
    let registry: WsRegistry = Arc::new(Mutex::new(Vec::new()));

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
            .app_data(web::Data::new(registry.clone()))
            .route("/query", web::post().to(execute_agent))
            .route("/config", web::get().to(get_config_handler))
            .route("/config", web::put().to(update_config_handler))
            .route("/restart", web::post().to(restart_handler))
            .route("/ws", web::get().to(ws_handler))
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
