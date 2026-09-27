use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use http::{HeaderName, HeaderValue};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, ContentBlock, ResourceContents, Tool},
    service::{RoleClient, RunningService},
    transport::{
        TokioChildProcess,
        streamable_http_client::{
            StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
        },
    },
};
use tokio::sync::Mutex;

use crate::{
    config::McpServerConfig,
    error::{Error, Result},
};

type Service = RunningService<RoleClient, ()>;

/// How long one tool call may take when the server's config sets no
/// `tool_timeout_secs`.
const DEFAULT_TOOL_TIMEOUT: Duration = Duration::from_secs(600);

/// How long a server whose connection closed gets to come back.
const RECONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// After a failed reconnect, how long calls fail with its error instead of
/// trying again, so calls queued behind it don't each wait out another
/// attempt.
const RECONNECT_BACKOFF: Duration = Duration::from_secs(30);

/// A server's connection, and the last failed attempt to replace it.
struct Connection {
    service: Arc<Service>,
    /// When the last reconnect failed, and why; cleared by a successful one.
    failed_reconnect: Option<(Instant, String)>,
}

/// Low-level MCP client wrapping an active [`rmcp`] service connection.
///
/// Created by [`McpClient::connect`] and shared via `Arc` across all
/// [`McpToolHandler`](super::tool_handler::McpToolHandler)s that belong to
/// the same server. If the connection closes (a stdio server exited, an HTTP
/// server dropped it), the next call reconnects.
pub struct McpClient {
    /// The current connection; replaced when its transport has closed.
    connection: Mutex<Connection>,
    /// What the connection was made from, to make it again.
    config: McpServerConfig,
    pub server_name: String,
    tool_timeout: Duration,
}

impl McpClient {
    /// Connects to an MCP server using the transport specified in `config`.
    ///
    /// - `config.url` set → connects via Streamable HTTP.
    /// - `config.command` set → spawns the process and connects via stdio.
    /// - Neither set, or `tool_timeout_secs = 0` → returns
    ///   [`Error::ConfigError`].
    pub async fn connect(name: &str, config: &McpServerConfig) -> Result<Self> {
        let tool_timeout = tool_timeout(name, config)?;
        let service = open(name, config).await?;
        Ok(Self {
            connection: Mutex::new(Connection {
                service: Arc::new(service),
                failed_reconnect: None,
            }),
            config: config.clone(),
            server_name: name.to_string(),
            tool_timeout,
        })
    }

    /// Returns all tools advertised by the MCP server.
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        let service = self.connection.lock().await.service.clone();
        service.list_all_tools().await.map_err(|e| {
            Error::Other(format!(
                "MCP list_tools failed for '{}': {}",
                self.server_name, e
            ))
        })
    }

    /// The connection to call through: the current one, or a new one if the
    /// current one's transport has closed. Concurrent callers wait for a
    /// single reconnect; for [`RECONNECT_BACKOFF`] after one fails, callers
    /// get its error without trying again.
    async fn live_service(&self) -> Result<Arc<Service>> {
        let mut connection = self.connection.lock().await;
        if !connection.service.peer().is_transport_closed() {
            return Ok(connection.service.clone());
        }
        if let Some((at, why)) = &connection.failed_reconnect
            && at.elapsed() < RECONNECT_BACKOFF
        {
            return Err(Error::ToolExecutionError(why.clone()));
        }

        tracing::warn!(server = %self.server_name, "MCP server connection closed; reconnecting");
        let reconnected =
            tokio::time::timeout(RECONNECT_TIMEOUT, open(&self.server_name, &self.config))
                .await
                .unwrap_or_else(|_| {
                    Err(Error::ToolExecutionError(format!(
                        "did not come back within {}s",
                        RECONNECT_TIMEOUT.as_secs()
                    )))
                });
        match reconnected {
            Ok(service) => {
                connection.service = Arc::new(service);
                connection.failed_reconnect = None;
                Ok(connection.service.clone())
            }
            Err(e) => {
                let why = format!(
                    "MCP server '{}' closed its connection and reconnecting failed: {e}",
                    self.server_name
                );
                connection.failed_reconnect = Some((Instant::now(), why.clone()));
                Err(Error::ToolExecutionError(why))
            }
        }
    }

    /// Invokes a tool by name with JSON-encoded arguments, failing once the
    /// server's tool timeout has passed, counting any wait for a reconnect.
    ///
    /// `args_json` must be a JSON object string (e.g. `{"path":"/tmp"}`). An empty
    /// string or `"{}"` is treated as no arguments.
    ///
    /// Returns the concatenated text content of all response blocks, or an error
    /// if the server reports `is_error: true`. A call that fails because the
    /// connection closed mid-call isn't retried, since the tool may already
    /// have run; the next call reconnects.
    pub async fn call_tool(&self, name: &str, args_json: &str) -> Result<String> {
        let params = build_call_params(name, args_json)?;

        let result = tokio::time::timeout(self.tool_timeout, async {
            let service = self.live_service().await?;
            service.peer().call_tool(params).await.map_err(|e| {
                Error::ToolExecutionError(format!(
                    "MCP tool '{}' on '{}' failed: {}",
                    name, self.server_name, e
                ))
            })
        })
        .await
        .map_err(|_| {
            Error::ToolExecutionError(format!(
                "MCP tool '{}' on '{}' did not finish within {}s",
                name,
                self.server_name,
                self.tool_timeout.as_secs()
            ))
        })??;

        if result.is_error.unwrap_or(false) {
            return Err(Error::ToolExecutionError(extract_text_content(
                &result.content,
            )));
        }

        Ok(extract_text_content(&result.content))
    }
}

/// The limit on one tool call to server `name`: its `tool_timeout_secs`, or
/// [`DEFAULT_TOOL_TIMEOUT`]. Zero is rejected; it would fail every call.
fn tool_timeout(name: &str, config: &McpServerConfig) -> Result<Duration> {
    match config.tool_timeout_secs {
        None => Ok(DEFAULT_TOOL_TIMEOUT),
        Some(0) => Err(Error::ConfigError(format!(
            "MCP server '{name}' has tool_timeout_secs = 0; it must be at least 1"
        ))),
        Some(secs) => Ok(Duration::from_secs(secs)),
    }
}

/// Opens a connection to server `name` as `config` describes.
async fn open(name: &str, config: &McpServerConfig) -> Result<Service> {
    if let Some(ref url) = config.url {
        let mut http_config = StreamableHttpClientTransportConfig::with_uri(url.as_str());
        if !config.headers.is_empty() {
            if url.starts_with("http://") {
                return Err(Error::ConfigError(format!(
                    "MCP server '{}' url '{}' uses http:// but has headers configured; \
                     credentials must not be sent over an unencrypted connection. Use https:// \
                     or drop the headers for keyless local servers",
                    name, url
                )));
            }
            let mut custom_headers = HashMap::with_capacity(config.headers.len());
            for (key, value) in &config.headers {
                let name = HeaderName::try_from(key.as_str()).map_err(|e| {
                    Error::ConfigError(format!(
                        "MCP server '{}' has an invalid header name '{}': {}",
                        name, key, e
                    ))
                })?;
                let value = HeaderValue::try_from(value.as_str()).map_err(|e| {
                    Error::ConfigError(format!(
                        "MCP server '{}' has an invalid value for header '{}': {}",
                        name, key, e
                    ))
                })?;
                custom_headers.insert(name, value);
            }
            http_config = http_config.custom_headers(custom_headers);
        }
        let transport = StreamableHttpClientTransport::from_config(http_config);
        ().serve(transport)
            .await
            .map_err(|e| Error::Other(format!("MCP HTTP connect to '{}' failed: {}", name, e)))
    } else if let Some(ref command) = config.command {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(&config.args);
        cmd.kill_on_drop(true);
        for (k, v) in &config.env {
            cmd.env(k, v);
        }
        let (transport, _) = TokioChildProcess::builder(cmd)
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| Error::Other(format!("MCP spawn '{}' failed: {}", name, e)))?;
        ().serve(transport)
            .await
            .map_err(|e| Error::Other(format!("MCP stdio connect to '{}' failed: {}", name, e)))
    } else {
        Err(Error::ConfigError(format!(
            "MCP server '{}' must have either 'command' (stdio) or 'url' (HTTP)",
            name
        )))
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

fn extract_text_content(content: &[ContentBlock]) -> String {
    content
        .iter()
        .map(|item| match item {
            ContentBlock::Text(t) => t.text.clone(),
            ContentBlock::Image(i) => format!("[image: {}]", i.mime_type),
            ContentBlock::Audio(a) => format!("[audio: {}]", a.mime_type),
            ContentBlock::Resource(r) => match &r.resource {
                ResourceContents::TextResourceContents { text, .. } => text.clone(),
                ResourceContents::BlobResourceContents { uri, mime_type, .. } => {
                    format!(
                        "[blob: {} ({})]",
                        uri,
                        mime_type.as_deref().unwrap_or("unknown")
                    )
                }
                _ => String::new(),
            },
            ContentBlock::ResourceLink(l) => format!("[resource: {}]", l.uri),
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_timeout_defaults_and_rejects_zero() {
        let server = McpServerConfig::stdio("srv", Vec::<String>::new());
        assert_eq!(tool_timeout("srv", &server).unwrap(), DEFAULT_TOOL_TIMEOUT);
        assert_eq!(
            tool_timeout("srv", &server.clone().with_tool_timeout_secs(5)).unwrap(),
            Duration::from_secs(5)
        );
        let err = tool_timeout("srv", &server.with_tool_timeout_secs(0)).unwrap_err();
        assert!(matches!(err, Error::ConfigError(_)), "{err}");
        assert!(err.to_string().contains("tool_timeout_secs"), "{err}");
    }
}
