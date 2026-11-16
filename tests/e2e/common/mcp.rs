//! A stub MCP server over streamable HTTP on a loopback port, offering
//! `echo` (returns its `text` argument) and `fail` (always an error result).

use std::sync::Arc;

use rmcp::{
    ErrorData, RoleServer, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, InitializeResult,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, Tool,
    },
    service::RequestContext,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde_json::json;

#[derive(Clone)]
struct Stub;

impl ServerHandler for Stub {
    fn get_info(&self) -> InitializeResult {
        InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let schema = json!({"type": "object", "properties": {"text": {"type": "string"}}});
        let schema = Arc::new(schema.as_object().unwrap().clone());
        Ok(ListToolsResult::with_all_items(vec![
            Tool::new("echo", "Echoes its text argument", schema.clone()),
            Tool::new("fail", "Always fails", schema),
        ]))
    }

    async fn call_tool(
        &self,
        req: CallToolRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let arg = req
            .arguments
            .as_ref()
            .and_then(|a| a.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string();
        let result = match &*req.name {
            "echo" => CallToolResult::success(vec![ContentBlock::text(arg)]),
            _ => CallToolResult::error(vec![ContentBlock::text("stub failure")]),
        };
        Ok(CallToolResponse::Complete(result))
    }
}

/// Starts the server and returns its URL.
pub async fn start() -> String {
    let service = StreamableHttpService::new(
        || Ok(Stub),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/mcp")
}
