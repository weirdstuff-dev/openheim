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
//! The fs sidecar is confined to the agent's `work_dir`, like the agent's
//! own tools; relative paths resolve inside it. `delete` and `rename` act on
//! a symlink rather than its target, and refuse `work_dir` itself.
//!
//! `/acp` carries the same ACP frames as `/ws`'s `agent` channel, one
//! JSON-RPC object per text message with no envelope and no fs sidecar, for
//! clients that only speak ACP.

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
    acp,
    client::OpenheimClient,
    core::runtime::AgentState,
    error::Error as AppError,
    tools::sandbox::{validate_entry, validate_path},
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
pub(crate) struct FsRequestEnvelope {
    #[serde(default)]
    pub id: Option<Value>,
    #[serde(flatten)]
    pub request: FsRequest,
}

/// An fs message to the client. `id` is the request's, for a reply; absent
/// for unsolicited messages (the connection greeting, watcher events, an
/// unparseable payload).
#[derive(Debug, Serialize)]
pub(crate) struct FsReply {
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
pub(crate) struct FileEntry {
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
pub(crate) enum FsRequest {
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
pub(crate) enum FsResponse {
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

/// How long [`serve`] waits, once it stops accepting connections, for the
/// turns it cancelled to end.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Resolves on Ctrl-C or, on Unix, SIGTERM (how Docker, systemd and
/// Kubernetes stop a service).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut terminate) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = terminate.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

/// Serves `client` over HTTP and WebSocket until Ctrl-C or SIGTERM, then
/// shuts down gracefully: stops accepting connections, cancels the turns
/// in flight and waits up to 5 seconds for them to end, so their session
/// leases are released and their shell commands killed. A turn still
/// running after that keeps its lease until it ends or the runtime it runs
/// on is dropped (as `openheim serve` does on returning).
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
        .with_state(state.clone());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| crate::error::Error::Other(format!("Failed to bind {addr}: {e}")))?;

    tracing::info!("WS server listening on ws://{addr}/ws (bare ACP also available at /acp)");
    tracing::info!("API available at http://{addr}/api/{{config,models,skills,tools,mcp-servers}}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            shutdown_signal().await;
            tracing::info!("Shutdown signal received");
        })
        .await
        .map_err(|e| crate::error::Error::Other(format!("Server error: {e}")))?;

    // WebSocket connections outlive the HTTP server (they're upgraded off
    // it), so their turns are ended here.
    state.shutdown(SHUTDOWN_GRACE).await;
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
    // History I/O is blocking, so it runs off the async workers.
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

/// Starts an ACP server for one WebSocket connection, one JSON-RPC message
/// per line. Returns the sender for incoming lines and the receiver for
/// outgoing ones; dropping the sender ends the server.
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
    tokio::spawn(acp::serve_state(Lines::new(sink, in_rx), state));
    (in_tx, out_rx)
}

async fn handle_socket(socket: WebSocket, state: Arc<AgentState>) {
    let (ws_tx, ws_rx) = socket.split();
    relay(ws_tx, ws_rx, state).await;
}

/// Starts the fs sidecar for one connection: requests sent to the returned
/// sender are carried out one at a time, in order, and each reply goes to
/// `out`. A task of its own, so a slow request (a large recursive `list`, a
/// read from a slow mount) doesn't hold up the connection's ACP frames.
fn spawn_fs_worker(
    work_dir: PathBuf,
    out: UnboundedSender<WsOutbound>,
) -> (
    UnboundedSender<FsRequestEnvelope>,
    tokio::task::JoinHandle<()>,
) {
    let (requests_tx, mut requests) = mpsc::unbounded::<FsRequestEnvelope>();
    let worker = tokio::spawn(async move {
        let mut fs_state = FsState::new(work_dir);
        while let Some(FsRequestEnvelope { id, request }) = requests.next().await {
            let response = fs_state.handle(request, out.clone()).await;
            let _ = out.unbounded_send(WsOutbound::Fs(FsReply { id, response }));
        }
    });
    (requests_tx, worker)
}

/// Serves one `/ws` connection: `agent` frames go to and from its ACP
/// server, `fs` frames to its fs worker, until the client closes.
async fn relay<Tx, Rx, E>(mut ws_tx: Tx, mut ws_rx: Rx, state: Arc<AgentState>)
where
    Tx: futures::Sink<Message> + Unpin + Send + 'static,
    Rx: futures::Stream<Item = Result<Message, E>> + Unpin,
{
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

    // Inbound: dispatch WS frames to the ACP server or the fs worker. Neither
    // is awaited here, so one can't hold up the other.
    let (fs_requests, fs_worker) = spawn_fs_worker(work_dir, fs_tx.clone());
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => match serde_json::from_str::<WsInbound>(&text) {
                Ok(WsInbound::Agent(val)) => {
                    let line = serde_json::to_string(&val).unwrap_or_default();
                    let _ = acp_in_tx.unbounded_send(Ok(line));
                }
                Ok(WsInbound::Fs(envelope)) => {
                    let _ = fs_requests.unbounded_send(envelope);
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

    // Also drops the worker's `watch`, if any.
    fs_worker.abort();
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

    /// Validates `path` for `delete`/`rename` via the shared
    /// [`validate_entry`]: a symlink names the link itself, not its target,
    /// and the work directory itself is refused.
    fn validate_entry(&self, path: &str) -> Result<PathBuf, FsResponse> {
        validate_entry(path, &self.work_dir).map_err(|e| fs_error(e.to_string()))
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
                // The entry's own type: a symlink to a directory is removed
                // as a link, leaving the directory it points to alone.
                let metadata = fs::symlink_metadata(&validated)
                    .await
                    .map_err(|e| fs_error(format!("Failed to delete: {e}")))?;
                if metadata.is_dir() {
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

    /// The next frame from the relay on `channel` (`agent` or `fs`) that
    /// `wanted` accepts, skipping others; `None` after 5s without one.
    async fn next_frame(
        out: &mut mpsc::UnboundedReceiver<Message>,
        channel: &str,
        wanted: impl Fn(&Value) -> bool,
    ) -> Option<Value> {
        tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(Message::Text(text)) = out.next().await {
                let frame: Value = serde_json::from_str(&text).unwrap();
                if frame["channel"] == channel && wanted(&frame["data"]) {
                    return Some(frame["data"].clone());
                }
            }
            None
        })
        .await
        .ok()
        .flatten()
    }

    // An fs request that blocks (a read of a FIFO nobody writes to) doesn't
    // hold up the connection's ACP frames; its reply comes once it's done.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_slow_fs_request_does_not_hold_up_agent_frames() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("pipe");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        let client = crate::OpenheimClient::builder()
            .provider("openai")
            .api_key("test-key")
            .data_dir(dir.path())
            .work_dir(dir.path())
            .build()
            .await
            .unwrap();
        let (to_relay, from_client) =
            mpsc::unbounded::<Result<Message, std::convert::Infallible>>();
        let (to_client, mut out) = mpsc::unbounded::<Message>();
        tokio::spawn(relay(to_client, from_client, client.state().clone()));
        let send = |frame: &str| {
            to_relay
                .unbounded_send(Ok(Message::Text(frame.to_string().into())))
                .unwrap()
        };

        send(r#"{"channel":"fs","data":{"action":"read","path":"pipe","id":1}}"#);
        send(
            r#"{"channel":"agent","data":{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}}"#,
        );

        let reply = next_frame(&mut out, "agent", |data| data["id"] == 1).await;

        // Unblock the read before asserting, so a failure can't leave the
        // runtime's shutdown waiting on it. Its reply still arrives, with its
        // id.
        let writer = fifo.clone();
        tokio::task::spawn_blocking(move || std::fs::write(writer, "done"))
            .await
            .unwrap()
            .unwrap();
        assert!(reply.is_some(), "the ACP reply was held up by the fs read");
        let read = next_frame(&mut out, "fs", |data| data["id"] == 1)
            .await
            .expect("no fs reply");
        assert_eq!(read["type"], "file_content", "{read}");
        assert_eq!(read["content"], "done", "{read}");
    }
}
