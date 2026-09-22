use std::collections::HashMap;
use std::sync::LazyLock;

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

use crate::{
    config::McpServerConfig,
    error::{Error, Result},
};

/// The `reqwest` client used for `url`-configured (Streamable HTTP) MCP
/// servers — built once and reused across every connection.
///
/// Deliberately *not* `reqwest::Client::new()` / `StreamableHttpClientTransport::from_config`'s
/// own default: those pull in `rustls-platform-verifier` (via the crate's
/// `default-tls` → `rustls` feature chain), which on Android checks
/// certificate revocation status and — per its own documented limitation —
/// treats a certificate that doesn't specify an OCSP responder or CRL as
/// **revoked** rather than "unknown", hard-failing the handshake even
/// though the certificate is perfectly valid. A remote MCP server not
/// stapling OCSP is common and not something this client can fix, so this
/// builds a plain webpki-roots-based `rustls::ClientConfig` instead —
/// static Mozilla root list, no OS integration, no revocation checking, and
/// no platform-specific surprises. Trade-off: this won't honor OS-level
/// trust decisions (an admin-installed corporate root CA, or a CA the OS
/// has since distrusted) the way the platform verifier would.
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    reqwest::Client::builder()
        .use_preconfigured_tls(tls_config)
        // Mirrors `StreamableHttpClientTransport`'s own default client
        // (see rmcp's `default_http_client`): avoids a ~40ms stall from TCP
        // Delayed ACK on Linux when pooling a connection whose previous
        // response body wasn't fully drained, and disables auto-redirects
        // so a caller-supplied custom header (e.g. `Authorization`) can't
        // get replayed to a redirect target.
        .pool_max_idle_per_host(0)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("webpki-roots TLS config is always valid for building a reqwest client")
});

/// Low-level MCP client wrapping an active [`rmcp`] service connection.
///
/// Created by [`McpClient::connect`] and shared via `Arc` across all
/// [`McpToolHandler`]s that belong to the same server.
pub struct McpClient {
    service: RunningService<RoleClient, ()>,
    pub server_name: String,
}

impl McpClient {
    /// Connects to an MCP server using the transport specified in `config`.
    ///
    /// - `config.url` set → connects via Streamable HTTP.
    /// - `config.command` set → spawns the process and connects via stdio.
    /// - Neither set → returns [`Error::ConfigError`].
    pub async fn connect(name: &str, config: &McpServerConfig) -> Result<Self> {
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
            let transport =
                StreamableHttpClientTransport::with_client(HTTP_CLIENT.clone(), http_config);
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
            cmd.kill_on_drop(true);
            for (k, v) in &config.env {
                cmd.env(k, v);
            }
            let (transport, _) = TokioChildProcess::builder(cmd)
                .stderr(std::process::Stdio::null())
                .spawn()
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

    /// Returns all tools advertised by the MCP server.
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        self.service.list_all_tools().await.map_err(|e| {
            Error::Other(format!(
                "MCP list_tools failed for '{}': {}",
                self.server_name, e
            ))
        })
    }

    /// Invokes a tool by name with JSON-encoded arguments.
    ///
    /// `args_json` must be a JSON object string (e.g. `{"path":"/tmp"}`). An empty
    /// string or `"{}"` is treated as no arguments.
    ///
    /// Returns the concatenated text content of all response blocks, or an error
    /// if the server reports `is_error: true`.
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
