//! WebSocket + REST transport: serves the agent over an axum HTTP server.
//!
//! Exposes the following endpoints:
//!
//! | Endpoint | Description |
//! |----------|-------------|
//! | `GET /ws` | WebSocket endpoint; multiplexes ACP agent messages and filesystem events |
//! | `GET /acp` | WebSocket endpoint; bare ACP JSON-RPC, no envelope or fs sidecar |
//! | `GET /api/config` | Resolved configuration (providers, models) |
//! | `GET /api/models` | Available providers and their model lists |
//! | `GET /api/skills` | Installed skill names |
//! | `GET /api/tools` | Registered tool definitions |
//! | `GET /api/mcp-servers` | MCP server connection statuses |
//! | `GET /api/sessions` | Conversation history listing |
//! | `GET /api/sessions/{id}` | Single conversation by UUID |
//!
//! `/ws` messages are JSON-encoded with a `channel` discriminator:
//! - `{"channel":"agent","data":{…}}` — ACP protocol frames
//! - `{"channel":"fs","data":{…}}` — filesystem sidecar (watch / list / read / write / mkdir / delete / rename)
//!
//! The fs sidecar is rooted at the agent's resolved `work_dir` — the same
//! sandbox boundary the agent's own tools are held to. Every request path is
//! validated against it (relative paths resolve within it) and `watch` may
//! only select directories inside it. `delete` and `rename` also refuse the
//! root itself, so a client can't remove or move the whole workspace.
//!
//! `/acp` carries the same ACP protocol frames as `/ws`'s `agent` channel, but
//! unwrapped: each WebSocket text message is exactly one JSON-RPC object, with
//! no `channel` tag and no filesystem sidecar. Use this endpoint for generic
//! ACP clients that only speak the spec and don't know about openheim's `/ws`
//! envelope or `fs` channel.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{
        Path as AxumPath, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use futures::{
    SinkExt, StreamExt,
    channel::mpsc::{self, UnboundedSender},
};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tower_http::cors::{Any, CorsLayer};
use walkdir::WalkDir;

use agent_client_protocol::Lines;

use crate::{
    acp, client::OpenheimClient, core::runtime::AgentState, error::Error as AppError,
    tools::sandbox::validate_path,
};

#[derive(Deserialize)]
#[serde(tag = "channel", content = "data")]
enum WsInbound {
    #[serde(rename = "agent")]
    Agent(Value),
    #[serde(rename = "fs")]
    Fs(FsRequestEnvelope),
}

#[derive(Serialize)]
#[serde(tag = "channel", content = "data")]
enum WsOutbound {
    #[serde(rename = "agent")]
    Agent(Value),
    #[serde(rename = "fs")]
    Fs(FsReply),
}

/// An fs request plus an optional client-chosen `id`, which is echoed on its
/// reply so a client with several requests in flight can tell which reply
/// (or error) belongs to which: `{"action": "read", "path": "a.txt", "id": 7}`.
#[derive(Debug, Deserialize)]
pub struct FsRequestEnvelope {
    #[serde(default)]
    pub id: Option<Value>,
    #[serde(flatten)]
    pub request: FsRequest,
}

/// An fs message to the client. `id` is the request's, for a reply; absent
/// for unsolicited messages (the connection greeting, watcher events, an
/// unparseable payload).
#[derive(Debug, Serialize)]
pub struct FsReply {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(flatten)]
    pub response: FsResponse,
}

impl FsReply {
    fn unsolicited(response: FsResponse) -> WsOutbound {
        WsOutbound::Fs(Self { id: None, response })
    }
}

/// Entry in the file tree
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified: Option<u64>,
}

/// Requests from the frontend to the filesystem WebSocket
#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
pub enum FsRequest {
    /// Initialize watching on a workspace directory
    #[serde(rename = "watch")]
    Watch { path: String },

    /// Stop watching
    #[serde(rename = "unwatch")]
    Unwatch,

    /// List directory contents
    #[serde(rename = "list")]
    List {
        path: String,
        recursive: Option<bool>,
    },

    /// Read file contents
    #[serde(rename = "read")]
    Read { path: String },

    /// Write file contents
    #[serde(rename = "write")]
    Write { path: String, content: String },

    /// Create a directory
    #[serde(rename = "mkdir")]
    Mkdir { path: String },

    /// Delete a file or directory
    #[serde(rename = "delete")]
    Delete { path: String },

    /// Rename/move a file or directory
    #[serde(rename = "rename")]
    Rename { from: String, to: String },
}

/// Responses/events from the filesystem WebSocket to the frontend
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum FsResponse {
    #[serde(rename = "connected")]
    Connected { message: String },

    #[serde(rename = "watching")]
    Watching { path: String },

    #[serde(rename = "unwatched")]
    Unwatched,

    #[serde(rename = "file_list")]
    FileList {
        path: String,
        entries: Vec<FileEntry>,
    },

    #[serde(rename = "file_content")]
    FileContent { path: String, content: String },

    #[serde(rename = "write_success")]
    WriteSuccess { path: String },

    #[serde(rename = "mkdir_success")]
    MkdirSuccess { path: String },

    #[serde(rename = "delete_success")]
    DeleteSuccess { path: String },

    #[serde(rename = "rename_success")]
    RenameSuccess { from: String, to: String },

    /// File system change event (from watcher)
    #[serde(rename = "fs_event")]
    FsEvent {
        event_kind: String,
        paths: Vec<String>,
    },

    #[serde(rename = "error")]
    Error { message: String },
}

/// Starts the HTTP/WebSocket server for `client` — caller-built, so an
/// embedder with custom tools or a custom `LlmClient` can use this transport
/// too.
///
/// Blocks until a Ctrl-C signal is received, then shuts down gracefully.
pub async fn serve(client: OpenheimClient, host: String, port: u16) -> crate::error::Result<()> {
    let state = client.state().clone();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/acp", get(acp_ws_handler))
        .route("/api/config", get(config_handler))
        .route("/api/models", get(models_handler))
        .route("/api/skills", get(skills_handler))
        .route("/api/tools", get(tools_handler))
        .route("/api/mcp-servers", get(mcp_servers_handler))
        .route("/api/sessions", get(sessions_handler))
        .route("/api/sessions/{id}", get(session_handler))
        .layer(cors)
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| crate::error::Error::Other(format!("Failed to bind {addr}: {e}")))?;

    tracing::info!("WS server listening on ws://{addr}/ws (bare ACP also available at /acp)");
    tracing::info!("API available at http://{addr}/api/{{config,models,skills,tools,mcp-servers}}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("Shutdown signal received");
        })
        .await
        .map_err(|e| crate::error::Error::Other(format!("Server error: {e}")))?;

    Ok(())
}

async fn config_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.app_config.to_public(&state.work_dir))
}

async fn models_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.app_config.models_info())
}

async fn skills_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.memory.skills.list_skills().unwrap_or_default())
}

async fn tools_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.executor.list_tools())
}

async fn mcp_servers_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.mcp_statuses.clone())
}

async fn sessions_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    // History I/O is synchronous file access; run it off the runtime threads
    // (same as the ACP layer does) instead of blocking a worker.
    let history = state.memory.history.clone();
    match tokio::task::spawn_blocking(move || history.list_conversations()).await {
        Ok(Ok(metas)) => Json(metas).into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "failed to load conversations" })),
        )
            .into_response(),
    }
}

async fn session_handler(
    State(state): State<Arc<AgentState>>,
    AxumPath(id): AxumPath<String>,
) -> impl IntoResponse {
    let uuid = match uuid::Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid session id" })),
            )
                .into_response();
        }
    };
    let history = state.memory.history.clone();
    match tokio::task::spawn_blocking(move || history.load_conversation(&uuid)).await {
        Ok(Ok(conv)) => Json(conv).into_response(),
        Ok(Err(AppError::NotFound(_))) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "session not found" })),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "failed to load session" })),
        )
            .into_response(),
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AgentState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Starts an ACP server for one WebSocket connection, speaking JSON-RPC as
/// one line per message over a pair of channels. Returns the sender for
/// incoming lines (WS → ACP) and the receiver for outgoing ones (ACP → WS);
/// dropping the sender ends the server. Shared by both socket handlers so the
/// bridge can't drift between them.
fn spawn_acp_server(
    state: Arc<AgentState>,
) -> (
    UnboundedSender<std::io::Result<String>>,
    mpsc::UnboundedReceiver<String>,
) {
    let (out_tx, out_rx) = mpsc::unbounded::<String>();
    let (in_tx, in_rx) = mpsc::unbounded::<std::io::Result<String>>();
    let sink =
        out_tx.sink_map_err(|e| std::io::Error::new(std::io::ErrorKind::BrokenPipe, e.to_string()));
    tokio::spawn(acp::serve(Lines::new(sink, in_rx), state));
    (in_tx, out_rx)
}

async fn handle_socket(socket: WebSocket, state: Arc<AgentState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let work_dir = state.work_dir.clone();
    let (acp_in_tx, mut acp_out_rx) = spawn_acp_server(state);

    // FS sidecar: events and responses going back to the WS client
    let (fs_tx, mut fs_rx) = mpsc::unbounded::<WsOutbound>();

    let _ = fs_tx.unbounded_send(FsReply::unsolicited(FsResponse::Connected {
        message: "Connected to Openheim".to_string(),
    }));

    // Outbound task: merges ACP responses + FS events into WS frames
    let outbound = tokio::spawn(async move {
        loop {
            tokio::select! {
                msg = acp_out_rx.next() => {
                    match msg {
                        Some(line) => {
                            if let Ok(val) = serde_json::from_str::<Value>(&line)
                                && let Ok(text) = serde_json::to_string(&WsOutbound::Agent(val))
                                    && ws_tx.send(Message::Text(text.into())).await.is_err() {
                                        break;
                                    }
                        }
                        None => break,
                    }
                }
                msg = fs_rx.next() => {
                    match msg {
                        Some(env) => {
                            if let Ok(text) = serde_json::to_string(&env)
                                && ws_tx.send(Message::Text(text.into())).await.is_err() {
                                    break;
                                }
                        }
                        None => break,
                    }
                }
            }
        }
    });

    // Inbound: dispatch WS frames to ACP server or FS handler
    let mut fs_state = FsState::new(work_dir);
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => match serde_json::from_str::<WsInbound>(&text) {
                Ok(WsInbound::Agent(val)) => {
                    let line = serde_json::to_string(&val).unwrap_or_default();
                    let _ = acp_in_tx.unbounded_send(Ok(line));
                }
                Ok(WsInbound::Fs(FsRequestEnvelope { id, request })) => {
                    let response = fs_state.handle(request, fs_tx.clone()).await;
                    let _ = fs_tx.unbounded_send(WsOutbound::Fs(FsReply { id, response }));
                }
                Err(e) => {
                    tracing::warn!("invalid WS payload: {e}");
                    let _ = fs_tx.unbounded_send(FsReply::unsolicited(FsResponse::Error {
                        message: format!("Invalid payload: {e}"),
                    }));
                }
            },
            Message::Close(_) => break,
            _ => {}
        }
    }

    outbound.abort();
}

async fn acp_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AgentState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_acp_socket(socket, state))
}

/// Bare ACP over WebSocket: each text frame is exactly one JSON-RPC message,
/// no `{"channel":...}` envelope and no `fs` sidecar — see module docs.
async fn handle_acp_socket(socket: WebSocket, state: Arc<AgentState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (acp_in_tx, mut acp_out_rx) = spawn_acp_server(state);

    let outbound = tokio::spawn(async move {
        while let Some(line) = acp_out_rx.next().await {
            if ws_tx.send(Message::Text(line.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => {
                let _ = acp_in_tx.unbounded_send(Ok(text.to_string()));
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    outbound.abort();
}

// FS sidecar state

struct FsState {
    /// Sandbox boundary shared with the agent's own tools: every fs request
    /// is validated against this root, never against a client-chosen path.
    work_dir: PathBuf,
    /// The active `watch`, if any; dropping it stops the watch.
    watcher: Option<RecommendedWatcher>,
}

/// Shorthand for an error reply.
fn fs_error(message: impl Into<String>) -> FsResponse {
    FsResponse::Error {
        message: message.into(),
    }
}

impl FsState {
    fn new(work_dir: PathBuf) -> Self {
        Self {
            work_dir,
            watcher: None,
        }
    }

    /// Validates `path` against `work_dir` via the shared sandbox validator,
    /// or returns the error reply to send.
    fn validate(&self, path: &str) -> Result<PathBuf, FsResponse> {
        validate_path(path, &self.work_dir).map_err(|e| fs_error(e.to_string()))
    }

    /// Like [`Self::validate`], but also refuses the work directory itself:
    /// `delete`/`rename` on something *inside* the sandbox is fine, on the
    /// sandbox root would remove or move the whole workspace. `path` may
    /// name the root in many ways (`""`, `.`, `a/..`, its absolute path, a
    /// symlink to it); all of them resolve to the canonical root.
    fn validate_entry(&self, path: &str) -> Result<PathBuf, FsResponse> {
        let validated = self.validate(path)?;
        if self
            .work_dir
            .canonicalize()
            .is_ok_and(|root| root == validated)
        {
            return Err(fs_error(format!(
                "'{path}' is the work directory itself; it can't be deleted or renamed"
            )));
        }
        Ok(validated)
    }

    /// Carries out one request and returns its reply. `events` is where a
    /// `watch` sends its (unsolicited) change events from then on.
    async fn handle(&mut self, req: FsRequest, events: UnboundedSender<WsOutbound>) -> FsResponse {
        self.try_handle(req, events)
            .await
            .unwrap_or_else(|error_reply| error_reply)
    }

    async fn try_handle(
        &mut self,
        req: FsRequest,
        events: UnboundedSender<WsOutbound>,
    ) -> Result<FsResponse, FsResponse> {
        Ok(match req {
            FsRequest::Watch { path } => self.start_watching(path, events)?,
            FsRequest::Unwatch => {
                self.watcher = None;
                FsResponse::Unwatched
            }
            FsRequest::List { path, recursive } => {
                let validated = self.validate(&path)?;
                let entries = list_directory(&validated, recursive.unwrap_or(false)).await;
                FsResponse::FileList { path, entries }
            }
            FsRequest::Read { path } => {
                let validated = self.validate(&path)?;
                let content = fs::read_to_string(&validated)
                    .await
                    .map_err(|e| fs_error(format!("Failed to read: {e}")))?;
                FsResponse::FileContent { path, content }
            }
            FsRequest::Write { path, content } => {
                let validated = self.validate(&path)?;
                if let Some(parent) = validated.parent()
                    && !parent.exists()
                {
                    fs::create_dir_all(parent)
                        .await
                        .map_err(|e| fs_error(format!("Failed to create dirs: {e}")))?;
                }
                fs::write(&validated, content)
                    .await
                    .map_err(|e| fs_error(format!("Failed to write: {e}")))?;
                FsResponse::WriteSuccess { path }
            }
            FsRequest::Mkdir { path } => {
                let validated = self.validate(&path)?;
                fs::create_dir_all(&validated)
                    .await
                    .map_err(|e| fs_error(format!("Failed to mkdir: {e}")))?;
                FsResponse::MkdirSuccess { path }
            }
            FsRequest::Delete { path } => {
                let validated = self.validate_entry(&path)?;
                if validated.is_dir() {
                    fs::remove_dir_all(&validated)
                        .await
                        .map_err(|e| fs_error(format!("Failed to delete dir: {e}")))?;
                } else {
                    fs::remove_file(&validated)
                        .await
                        .map_err(|e| fs_error(format!("Failed to delete file: {e}")))?;
                }
                FsResponse::DeleteSuccess { path }
            }
            FsRequest::Rename { from, to } => {
                let (vf, vt) = (self.validate_entry(&from)?, self.validate_entry(&to)?);
                fs::rename(&vf, &vt)
                    .await
                    .map_err(|e| fs_error(format!("Failed to rename: {e}")))?;
                FsResponse::RenameSuccess { from, to }
            }
        })
    }

    fn start_watching(
        &mut self,
        path: String,
        events: UnboundedSender<WsOutbound>,
    ) -> Result<FsResponse, FsResponse> {
        let validated = self.validate(&path)?;
        if !validated.is_dir() {
            return Err(fs_error(format!("Invalid directory: {path}")));
        }

        self.watcher = None;

        let (notify_tx, mut notify_rx) = mpsc::unbounded::<notify::Result<Event>>();
        tokio::spawn(async move {
            while let Some(res) = notify_rx.next().await {
                let response = match res {
                    Ok(event) => FsResponse::FsEvent {
                        event_kind: format!("{:?}", event.kind),
                        paths: event
                            .paths
                            .iter()
                            .map(|p| p.to_string_lossy().to_string())
                            .collect(),
                    },
                    Err(e) => fs_error(format!("Watcher error: {e}")),
                };
                let _ = events.unbounded_send(FsReply::unsolicited(response));
            }
        });

        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = notify_tx.unbounded_send(res);
            },
            Config::default().with_poll_interval(Duration::from_secs(1)),
        )
        .map_err(|e| fs_error(format!("Failed to create watcher: {e}")))?;
        watcher
            .watch(&validated, RecursiveMode::Recursive)
            .map_err(|e| fs_error(format!("Failed to watch: {e}")))?;
        self.watcher = Some(watcher);
        Ok(FsResponse::Watching { path })
    }
}

async fn list_directory(path: &Path, recursive: bool) -> Vec<FileEntry> {
    if recursive {
        let path = path.to_path_buf();
        return tokio::task::spawn_blocking(move || {
            WalkDir::new(&path)
                .min_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter_map(|e| path_to_file_entry(e.path()))
                .collect()
        })
        .await
        .unwrap_or_default();
    }

    let mut entries = Vec::new();
    if let Ok(mut dir) = fs::read_dir(path).await {
        while let Ok(Some(e)) = dir.next_entry().await {
            if let Some(entry) = path_to_file_entry(&e.path()) {
                entries.push(entry);
            }
        }
    }
    entries
}

fn path_to_file_entry(path: &Path) -> Option<FileEntry> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let is_dir = path.is_dir();
    let metadata = path.metadata().ok();
    let size = metadata
        .as_ref()
        .and_then(|m| if m.is_file() { Some(m.len()) } else { None });
    let modified = metadata.as_ref().and_then(|m| {
        m.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
    });
    Some(FileEntry {
        path: path.to_string_lossy().to_string(),
        name,
        is_dir,
        size,
        modified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fs_state(work_dir: &Path) -> FsState {
        FsState::new(work_dir.to_path_buf())
    }

    /// Runs one fs request and returns its reply.
    async fn run_request(state: &mut FsState, req: FsRequest) -> FsResponse {
        let (events, _) = mpsc::unbounded::<WsOutbound>();
        state.handle(req, events).await
    }

    // Both socket handlers talk to ACP only through `spawn_acp_server`, so a
    // JSON-RPC round trip over its channels covers the bridge for both.
    #[tokio::test]
    async fn acp_server_answers_initialize_over_the_bridge() {
        let dir = tempfile::tempdir().unwrap();
        let client = crate::OpenheimClient::builder()
            .provider("openai")
            .api_key("test-key")
            .model("gpt-4o")
            .data_dir(dir.path())
            .work_dir(dir.path())
            .build()
            .await
            .unwrap();
        let (in_tx, mut out_rx) = spawn_acp_server(client.state().clone());

        in_tx
            .unbounded_send(Ok(
                r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}"#
                    .to_string(),
            ))
            .unwrap();
        let line = tokio::time::timeout(Duration::from_secs(5), out_rx.next())
            .await
            .expect("no response within 5s")
            .expect("server closed the channel");

        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], 1, "{response}");
        assert_eq!(
            response["result"]["agentInfo"]["name"], "openheim",
            "{response}"
        );
    }

    #[tokio::test]
    async fn fs_read_outside_work_dir_is_rejected() {
        let work = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "top secret").unwrap();

        let mut state = make_fs_state(work.path());
        let resp = run_request(
            &mut state,
            FsRequest::Read {
                path: secret.to_str().unwrap().to_string(),
            },
        )
        .await;

        assert!(
            matches!(&resp, FsResponse::Error { message } if message.contains("outside the work directory")),
            "unexpected response: {resp:?}"
        );
    }

    #[tokio::test]
    async fn fs_write_and_delete_outside_work_dir_are_rejected() {
        let work = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("evil.txt");

        let mut state = make_fs_state(work.path());
        let resp = run_request(
            &mut state,
            FsRequest::Write {
                path: target.to_str().unwrap().to_string(),
                content: "pwned".into(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::Error { .. }));
        assert!(!target.exists());

        let victim = outside.path().join("victim.txt");
        std::fs::write(&victim, "data").unwrap();
        let resp = run_request(
            &mut state,
            FsRequest::Delete {
                path: victim.to_str().unwrap().to_string(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::Error { .. }));
        assert!(victim.exists());
    }

    #[tokio::test]
    async fn fs_delete_and_rename_refuse_the_work_dir_itself() {
        let work = tempfile::tempdir().unwrap();
        std::fs::create_dir(work.path().join("sub")).unwrap();
        std::fs::write(work.path().join("keep.txt"), "data").unwrap();
        let mut state = make_fs_state(work.path());

        let mut names_for_root = vec![
            String::new(),
            ".".to_string(),
            "sub/..".to_string(),
            work.path().to_str().unwrap().to_string(),
        ];
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(work.path(), work.path().join("root_link")).unwrap();
            names_for_root.push("root_link".to_string());
        }

        for path in names_for_root {
            let resp = run_request(&mut state, FsRequest::Delete { path: path.clone() }).await;
            assert!(
                matches!(&resp, FsResponse::Error { message } if message.contains("work directory itself")),
                "delete {path:?}: {resp:?}"
            );
            let resp = run_request(
                &mut state,
                FsRequest::Rename {
                    from: path.clone(),
                    to: "moved".into(),
                },
            )
            .await;
            assert!(
                matches!(&resp, FsResponse::Error { message } if message.contains("work directory itself")),
                "rename {path:?}: {resp:?}"
            );
        }
        let resp = run_request(
            &mut state,
            FsRequest::Rename {
                from: "keep.txt".into(),
                to: ".".into(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::Error { .. }), "{resp:?}");
        assert!(work.path().join("keep.txt").exists());
        assert!(!work.path().join("moved").exists());

        // Entries inside the work dir can still be deleted.
        let resp = run_request(
            &mut state,
            FsRequest::Delete {
                path: "keep.txt".into(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::DeleteSuccess { .. }), "{resp:?}");
        assert!(!work.path().join("keep.txt").exists());
    }

    #[tokio::test]
    async fn fs_dotdot_traversal_is_rejected() {
        let work = tempfile::tempdir().unwrap();
        let mut state = make_fs_state(work.path());
        let resp = run_request(
            &mut state,
            FsRequest::Read {
                path: "../../etc/passwd".into(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::Error { .. }));
    }

    #[tokio::test]
    async fn fs_relative_path_resolves_against_work_dir() {
        let work = tempfile::tempdir().unwrap();
        let mut state = make_fs_state(work.path());

        let resp = run_request(
            &mut state,
            FsRequest::Write {
                path: "sub/file.txt".into(),
                content: "hello".into(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::WriteSuccess { .. }));
        assert_eq!(
            std::fs::read_to_string(work.path().join("sub/file.txt")).unwrap(),
            "hello"
        );

        let resp = run_request(
            &mut state,
            FsRequest::Read {
                path: "sub/file.txt".into(),
            },
        )
        .await;
        assert!(
            matches!(&resp, FsResponse::FileContent { content, .. } if content == "hello"),
            "unexpected response: {resp:?}"
        );
    }

    #[tokio::test]
    async fn fs_watch_outside_work_dir_is_rejected() {
        let work = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();

        let mut state = make_fs_state(work.path());
        let resp = run_request(
            &mut state,
            FsRequest::Watch {
                path: outside.path().to_str().unwrap().to_string(),
            },
        )
        .await;
        assert!(matches!(resp, FsResponse::Error { .. }));
        assert!(state.watcher.is_none());
    }

    #[tokio::test]
    async fn fs_watch_inside_work_dir_succeeds() {
        let work = tempfile::tempdir().unwrap();
        std::fs::create_dir(work.path().join("proj")).unwrap();

        let mut state = make_fs_state(work.path());
        let resp = run_request(
            &mut state,
            FsRequest::Watch {
                path: "proj".into(),
            },
        )
        .await;
        assert!(
            matches!(resp, FsResponse::Watching { .. }),
            "unexpected response: {resp:?}"
        );
        assert!(state.watcher.is_some());
    }

    #[test]
    fn fs_request_id_is_optional_and_echoed_on_the_reply() {
        let with_id: FsRequestEnvelope =
            serde_json::from_str(r#"{"action": "read", "path": "a.txt", "id": 7}"#).unwrap();
        assert_eq!(with_id.id, Some(serde_json::json!(7)));
        assert!(matches!(with_id.request, FsRequest::Read { path } if path == "a.txt"));

        let without_id: FsRequestEnvelope =
            serde_json::from_str(r#"{"action": "unwatch"}"#).unwrap();
        assert!(without_id.id.is_none());

        let reply = WsOutbound::Fs(FsReply {
            id: with_id.id,
            response: fs_error("nope"),
        });
        let json = serde_json::to_value(&reply).unwrap();
        assert_eq!(json["channel"], "fs");
        assert_eq!(json["data"]["id"], 7);
        assert_eq!(json["data"]["type"], "error");
        assert_eq!(json["data"]["message"], "nope");

        // Unsolicited messages carry no `id` key at all.
        let json = serde_json::to_value(FsReply::unsolicited(FsResponse::Unwatched)).unwrap();
        assert!(json["data"].get("id").is_none(), "{json}");
    }

    #[test]
    fn fs_request_deserializes_watch() {
        let json = r#"{"action": "watch", "path": "/tmp"}"#;
        let req: FsRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FsRequest::Watch { path } if path == "/tmp"));
    }

    #[test]
    fn fs_request_deserializes_write() {
        let json = r#"{"action": "write", "path": "a.txt", "content": "hello"}"#;
        let req: FsRequest = serde_json::from_str(json).unwrap();
        assert!(
            matches!(req, FsRequest::Write { path, content } if path == "a.txt" && content == "hello")
        );
    }

    #[test]
    fn fs_request_deserializes_rename() {
        let json = r#"{"action": "rename", "from": "a.txt", "to": "b.txt"}"#;
        let req: FsRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FsRequest::Rename { from, to } if from == "a.txt" && to == "b.txt"));
    }

    #[test]
    fn fs_response_serializes_with_type_tag() {
        let resp = FsResponse::Connected {
            message: "ok".into(),
        };
        let json: Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "connected");
        assert_eq!(json["message"], "ok");
    }

    #[test]
    fn fs_response_error_serializes() {
        let resp = FsResponse::Error {
            message: "not found".into(),
        };
        let json: Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["message"], "not found");
    }
}
