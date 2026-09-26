mod client;
mod tool_handler;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use client::McpClient;
use serde::Serialize;
use tool_handler::McpToolHandler;

use crate::{
    config::McpServerConfig,
    error::{Error, Result},
    tools::ToolHandler,
};

/// Connection status and tool-count summary for a single MCP server.
///
/// Returned by [`crate::OpenheimClient::mcp_servers`] so callers can inspect which
/// servers connected successfully and how many tools each one exposed.
#[derive(Debug, Clone, Serialize)]
pub struct McpServerStatus {
    /// Name of the server as defined in the configuration.
    pub name: String,
    /// Transport type: `"stdio"`, `"http"`, or `"unknown"`.
    pub transport: &'static str,
    /// Spawn command for stdio servers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Base URL for HTTP servers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub connected: bool,
    pub tool_count: usize,
    /// Error message if the server failed to connect.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// How long a server gets to connect and list its tools. Generous, since a
/// stdio server launched through `npx`/`uvx` may download itself first; a
/// server that takes longer is reported as failed instead of holding up
/// startup indefinitely.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

/// Longest tool name every supported provider accepts.
const MAX_TOOL_NAME_LEN: usize = 64;

/// Connects to all configured MCP servers and returns their tool handlers and statuses.
///
/// Servers start concurrently, each within [`STARTUP_TIMEOUT`]. Connection
/// failures are non-fatal: a server that fails to connect (or times out)
/// produces a [`McpServerStatus`] with `connected: false` and an error
/// message, and the remaining servers are unaffected.
pub(crate) async fn load_mcp_tools(
    configs: &BTreeMap<String, McpServerConfig>,
) -> (Vec<Box<dyn ToolHandler>>, Vec<McpServerStatus>) {
    load_mcp_tools_within(configs, STARTUP_TIMEOUT).await
}

async fn load_mcp_tools_within(
    configs: &BTreeMap<String, McpServerConfig>,
    timeout: Duration,
) -> (Vec<Box<dyn ToolHandler>>, Vec<McpServerStatus>) {
    let mut handlers: Vec<Box<dyn ToolHandler>> = Vec::new();
    let mut statuses: Vec<McpServerStatus> = Vec::new();

    let connections = futures::future::join_all(configs.iter().map(|(name, config)| async move {
        let result = tokio::time::timeout(timeout, connect_server(name, config))
            .await
            .unwrap_or_else(|_| {
                Err(Error::Other(format!(
                    "MCP server '{name}' did not start within {}s",
                    timeout.as_secs()
                )))
            });
        (name, config, result)
    }))
    .await;

    for (name, config, result) in connections {
        let (transport, command, url) = if config.command.is_some() {
            ("stdio", config.command.clone(), None)
        } else if config.url.is_some() {
            ("http", None, config.url.clone())
        } else {
            ("unknown", None, None)
        };

        match result {
            Ok(server_handlers) => {
                tracing::info!(server = %name, count = server_handlers.len(), "MCP server connected");
                statuses.push(McpServerStatus {
                    name: name.clone(),
                    transport,
                    command,
                    url,
                    connected: true,
                    tool_count: server_handlers.len(),
                    error: None,
                });
                handlers.extend(server_handlers);
            }
            Err(e) => {
                tracing::warn!(server = %name, error = %e, "MCP server failed to connect");
                statuses.push(McpServerStatus {
                    name: name.clone(),
                    transport,
                    command,
                    url,
                    connected: false,
                    tool_count: 0,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    (handlers, statuses)
}

async fn connect_server(name: &str, config: &McpServerConfig) -> Result<Vec<Box<dyn ToolHandler>>> {
    let client = Arc::new(McpClient::connect(name, config).await?);
    let tools = client.list_tools().await?;

    let handlers = tools
        .iter()
        .map(|tool| -> Box<dyn ToolHandler> {
            let exposed = exposed_tool_name(name, &tool.name);
            Box::new(McpToolHandler::new(Arc::clone(&client), tool, exposed))
        })
        .collect();

    Ok(handlers)
}

/// The name MCP tool `tool` of server `server` is offered to the LLM under:
/// `{server}__{tool}`, made acceptable to every provider. The tool list goes
/// out with every request, so one name a provider rejects would fail them
/// all.
///
/// In the server name anything but ASCII letters, digits and `_` becomes
/// `_`; in the tool name `-` is kept too. A name starting with a digit gets a
/// leading `_`. A name over [`MAX_TOOL_NAME_LEN`] is shortened and ends in a
/// hash of the full name, so distinct long names stay distinct.
fn exposed_tool_name(server: &str, tool: &str) -> String {
    fn clean(s: &str, keep: impl Fn(char) -> bool) -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || keep(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
    let mut name = format!(
        "{}__{}",
        clean(server, |c| c == '_'),
        clean(tool, |c| c == '_' || c == '-')
    );
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    if name.len() > MAX_TOOL_NAME_LEN {
        let hash = format!("_{:08x}", fnv1a(&format!("{server}__{tool}")) as u32);
        name.truncate(MAX_TOOL_NAME_LEN - hash.len());
        name.push_str(&hash);
    }
    name
}

/// 64-bit FNV-1a: a hash that stays the same across builds and Rust
/// versions, so a shortened tool name does too.
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_valid_everywhere(name: &str) -> bool {
        name.len() <= MAX_TOOL_NAME_LEN
            && name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    }

    #[test]
    fn valid_names_are_only_prefixed() {
        assert_eq!(
            exposed_tool_name("github", "create_issue"),
            "github__create_issue"
        );
        assert_eq!(
            exposed_tool_name("my-server", "get-file"),
            "my_server__get-file"
        );
    }

    #[test]
    fn exposed_names_are_valid_for_every_provider() {
        for (server, tool) in [
            ("files", "fs.read"),
            ("café", "lire"),
            ("1password", "item/get"),
            ("srv", &"x".repeat(200)),
        ] {
            let name = exposed_tool_name(server, tool);
            assert!(is_valid_everywhere(&name), "{server}/{tool}: {name}");
        }
    }

    #[test]
    fn long_names_stay_distinct_and_stable() {
        let a = exposed_tool_name("srv", &format!("{}_a", "x".repeat(80)));
        let b = exposed_tool_name("srv", &format!("{}_b", "x".repeat(80)));
        assert_ne!(a, b);
        assert_eq!(a.len(), MAX_TOOL_NAME_LEN);
        assert_eq!(
            a,
            exposed_tool_name("srv", &format!("{}_a", "x".repeat(80)))
        );
    }

    // One server that never finishes its handshake is reported as failed
    // after the timeout, without holding up the others or startup.
    #[cfg(unix)]
    #[tokio::test(start_paused = true)]
    async fn hung_servers_time_out_together() {
        let hung = || McpServerConfig {
            command: Some("sleep".into()),
            args: vec!["3600".into()],
            env: Default::default(),
            url: None,
            headers: Default::default(),
        };
        let configs = BTreeMap::from([("a".to_string(), hung()), ("b".to_string(), hung())]);
        let started = tokio::time::Instant::now();

        let (handlers, statuses) = load_mcp_tools_within(&configs, Duration::from_secs(5)).await;

        assert!(handlers.is_empty());
        assert_eq!(statuses.len(), 2);
        for status in &statuses {
            assert!(!status.connected);
            let error = status.error.as_deref().unwrap();
            assert!(error.contains("did not start within 5s"), "{error}");
        }
        assert!(started.elapsed() < Duration::from_secs(10));
    }
}
