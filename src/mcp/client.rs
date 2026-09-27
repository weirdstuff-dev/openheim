use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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

/// Low-level MCP client wrapping an active [`rmcp`] service connection.
///
/// Created by [`McpClient::connect`] and shared via `Arc` across all
/// [`McpToolHandler`](super::tool_handler::McpToolHandler)s that belong to
/// the same server. If the connection closes (a stdio server exited, an HTTP
/// server dropped it), the next call reconnects.
pub struct McpClient {
    /// The current connection; replaced when its transport has closed.
    service: Mutex<Arc<Service>>,
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
    /// - Neither set → returns [`Error::ConfigError`].
    pub async fn connect(name: &str, config: &McpServerConfig) -> Result<Self> {
        let service = open(name, config).await?;
        Ok(Self {
            service: Mutex::new(Arc::new(service)),
            config: config.clone(),
            server_name: name.to_string(),
            tool_timeout: config
                .tool_timeout_secs
                .map_or(DEFAULT_TOOL_TIMEOUT, Duration::from_secs),
        })
    }

    /// Returns all tools advertised by the MCP server.
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        let service = self.service.lock().await.clone();
        service.list_all_tools().await.map_err(|e| {
            Error::Other(format!(
                "MCP list_tools failed for '{}': {}",
                self.server_name, e
            ))
        })
    }

    /// The connection to call through: the current one, or a new one if the
    /// current one's transport has closed. Concurrent callers wait for a
    /// single reconnect.
    async fn live_service(&self) -> Result<Arc<Service>> {
        let mut service = self.service.lock().await;
        if service.peer().is_transport_closed() {
            tracing::warn!(server = %self.server_name, "MCP server connection closed; reconnecting");
            let fresh = tokio::time::timeout(
                RECONNECT_TIMEOUT,
                open(&self.server_name, &self.config),
            )
            .await
            .map_err(|_| {
                Error::ToolExecutionError(format!(
                    "MCP server '{}' closed its connection and did not come back within {}s",
                    self.server_name,
                    RECONNECT_TIMEOUT.as_secs()
                ))
            })??;
            *service = Arc::new(fresh);
        }
        Ok(service.clone())
    }

    /// Invokes a tool by name with JSON-encoded arguments, failing after the
    /// server's tool timeout.
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
        let service = self.live_service().await?;

        let result = tokio::time::timeout(self.tool_timeout, service.peer().call_tool(params))
            .await
            .map_err(|_| {
                Error::ToolExecutionError(format!(
                    "MCP tool '{}' on '{}' did not finish within {}s",
                    name,
                    self.server_name,
                    self.tool_timeout.as_secs()
                ))
            })?
            .map_err(|e| {
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A stdio MCP server with three tools: `echo` answers, `hang` never
    /// does, and `exit` makes the server exit mid-call.
    const STUB_SERVER: &str = r#"
import json, sys, time
def reply(id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id, "result": result}) + "\n")
    sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        reply(msg["id"], {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stub", "version": "1"},
        })
    elif method == "tools/list":
        tool = lambda name: {"name": name, "inputSchema": {"type": "object"}}
        reply(msg["id"], {"tools": [tool("echo"), tool("hang"), tool("exit")]})
    elif method == "tools/call":
        name = msg["params"]["name"]
        if name == "echo":
            reply(msg["id"], {"content": [{"type": "text", "text": "echoed"}]})
        elif name == "exit":
            sys.exit(0)
    elif "id" in msg:
        reply(msg["id"], {})
"#;

    async fn stub(dir: &tempfile::TempDir, tool_timeout_secs: u64) -> McpClient {
        let script = dir.path().join("stub.py");
        std::fs::write(&script, STUB_SERVER).unwrap();
        let config = McpServerConfig::stdio("python3", [script.to_string_lossy()])
            .with_tool_timeout_secs(tool_timeout_secs);
        McpClient::connect("stub", &config).await.unwrap()
    }

    #[tokio::test]
    async fn a_call_that_never_answers_times_out() {
        let dir = tempfile::tempdir().unwrap();
        let client = stub(&dir, 1).await;

        let err = tokio::time::timeout(Duration::from_secs(10), client.call_tool("hang", "{}"))
            .await
            .expect("the tool timeout should end the call")
            .unwrap_err();

        assert!(
            err.to_string().contains("did not finish within 1s"),
            "{err}"
        );
        assert_eq!(client.call_tool("echo", "{}").await.unwrap(), "echoed");
    }

    #[tokio::test]
    async fn a_server_that_exited_is_restarted_by_the_next_call() {
        let dir = tempfile::tempdir().unwrap();
        let client = stub(&dir, 10).await;

        // Not retried: the tool may already have run.
        assert!(client.call_tool("exit", "{}").await.is_err());

        assert_eq!(client.call_tool("echo", "{}").await.unwrap(), "echoed");
    }
}
