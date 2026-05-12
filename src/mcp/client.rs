use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, Content, RawContent, ResourceContents, Tool},
    service::{RoleClient, RunningService},
    transport::{TokioChildProcess, streamable_http_client::StreamableHttpClientTransport},
};

use crate::{
    config::McpServerConfig,
    error::{Error, Result},
};

pub struct McpClient {
    service: RunningService<RoleClient, ()>,
    pub server_name: String,
}

impl McpClient {
    pub async fn connect(name: &str, config: &McpServerConfig) -> Result<Self> {
        if let Some(ref url) = config.url {
            let transport = StreamableHttpClientTransport::from_uri(url.as_str());
            let service = ().serve(transport).await.map_err(|e| {
                Error::Other(format!("MCP HTTP connect to '{}' failed: {}", name, e))
            })?;
            Ok(Self {
                service,
                server_name: name.to_string(),
            })
        } else if let Some(ref command) = config.command {
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(&config.args);
            for (k, v) in &config.env {
                cmd.env(k, v);
            }
            let transport = TokioChildProcess::new(cmd)
                .map_err(|e| Error::Other(format!("MCP spawn '{}' failed: {}", name, e)))?;
            let service = ().serve(transport).await.map_err(|e| {
                Error::Other(format!("MCP stdio connect to '{}' failed: {}", name, e))
            })?;
            Ok(Self {
                service,
                server_name: name.to_string(),
            })
        } else {
            Err(Error::ConfigError(format!(
                "MCP server '{}' must have either 'command' (stdio) or 'url' (HTTP)",
                name
            )))
        }
    }

    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        self.service.list_all_tools().await.map_err(|e| {
            Error::Other(format!(
                "MCP list_tools failed for '{}': {}",
                self.server_name, e
            ))
        })
    }

    pub async fn call_tool(&self, name: &str, args_json: &str) -> Result<String> {
        let params = build_call_params(name, args_json)?;

        let result = self.service.peer().call_tool(params).await.map_err(|e| {
            Error::ToolExecutionError(format!(
                "MCP tool '{}' on '{}' failed: {}",
                name, self.server_name, e
            ))
        })?;

        if result.is_error.unwrap_or(false) {
            return Err(Error::ToolExecutionError(extract_text_content(
                &result.content,
            )));
        }

        Ok(extract_text_content(&result.content))
    }
}

fn build_call_params(name: &str, args_json: &str) -> Result<CallToolRequestParams> {
    let trimmed = args_json.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return Ok(CallToolRequestParams::new(name.to_string()));
    }
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(trimmed)?;
    Ok(CallToolRequestParams::new(name.to_string()).with_arguments(map))
}

fn extract_text_content(content: &[Content]) -> String {
    content
        .iter()
        .map(|item| match &**item {
            RawContent::Text(t) => t.text.clone(),
            RawContent::Image(i) => format!("[image: {}]", i.mime_type),
            RawContent::Audio(a) => format!("[audio: {}]", a.mime_type),
            RawContent::Resource(r) => match &r.resource {
                ResourceContents::TextResourceContents { text, .. } => text.clone(),
                ResourceContents::BlobResourceContents { uri, mime_type, .. } => {
                    format!(
                        "[blob: {} ({})]",
                        uri,
                        mime_type.as_deref().unwrap_or("unknown")
                    )
                }
            },
            RawContent::ResourceLink(l) => format!("[resource: {}]", l.uri),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
