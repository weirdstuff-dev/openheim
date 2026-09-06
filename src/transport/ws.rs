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
//! only select directories inside it.
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
    acp, config::load_config, core::runtime::AgentState, error::Error as AppError,
    memory::MemoryContext, tools::sandbox::validate_path,
};

#[derive(Deserialize)]
#[serde(tag = "channel", content = "data")]
enum WsInbound {
    #[serde(rename = "agent")]
    Agent(Value),
    #[serde(rename = "fs")]
    Fs(FsRequest),
}

#[derive(Serialize)]
#[serde(tag = "channel", content = "data")]
enum WsOutbound {
    #[serde(rename = "agent")]
    Agent(Value),
    #[serde(rename = "fs")]
    Fs(FsResponse),
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

/// Loads configuration, initialises the agent runtime, and starts the HTTP/WebSocket server.
///
/// Blocks until a Ctrl-C signal is received, then shuts down gracefully.
pub async fn serve(host: String, port: u16) -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;
    let memory = MemoryContext::new(app_config.default_skills.clone())?;
    let state = Arc::new(AgentState::new(agent_config, app_config, memory, vec![]).await?);

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
    Json(state.app_config.to_public_json(&state.work_dir))
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

async fn handle_socket(socket: WebSocket, state: Arc<AgentState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let work_dir = state.work_dir.clone();

    // ACP bridge: futures mpsc channels between the dispatch loop and the ACP server
    let (acp_out_tx, mut acp_out_rx) = mpsc::unbounded::<String>(); // ACP server → WS
    let (acp_in_tx, acp_in_rx) = mpsc::unbounded::<std::io::Result<String>>(); // WS → ACP server

    let acp_sink = acp_out_tx
        .sink_map_err(|e| std::io::Error::new(std::io::ErrorKind::BrokenPipe, e.to_string()));
    tokio::spawn(acp::serve(Lines::new(acp_sink, acp_in_rx), state));

    // FS sidecar: events and responses going back to the WS client
    let (fs_tx, mut fs_rx) = mpsc::unbounded::<WsOutbound>();

    let _ = fs_tx.unbounded_send(WsOutbound::Fs(FsResponse::Connected {
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
                Ok(WsInbound::Fs(req)) => {
                    fs_state.handle(req, fs_tx.clone()).await;
                }
                Err(e) => {
                    tracing::warn!("invalid WS payload: {e}");
                    let _ = fs_tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
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

    let (acp_out_tx, mut acp_out_rx) = mpsc::unbounded::<String>();
    let (acp_in_tx, acp_in_rx) = mpsc::unbounded::<std::io::Result<String>>();

    let acp_sink = acp_out_tx
        .sink_map_err(|e| std::io::Error::new(std::io::ErrorKind::BrokenPipe, e.to_string()));
    tokio::spawn(acp::serve(Lines::new(acp_sink, acp_in_rx), state));

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
    _watcher: Option<RecommendedWatcher>,
}

impl FsState {
    fn new(work_dir: PathBuf) -> Self {
        Self {
            work_dir,
            _watcher: None,
        }
    }

    /// Validates `path` against `work_dir` via the shared sandbox validator.
    /// On rejection the error is reported to the client and `None` returned.
    fn validate(&self, path: &str, tx: &UnboundedSender<WsOutbound>) -> Option<PathBuf> {
        match validate_path(path, &self.work_dir) {
            Ok(validated) => Some(validated),
            Err(e) => {
                let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                    message: e.to_string(),
                }));
                None
            }
        }
    }

    async fn handle(&mut self, req: FsRequest, tx: UnboundedSender<WsOutbound>) {
        match req {
            FsRequest::Watch { path } => self.start_watching(path, tx),
            FsRequest::Unwatch => {
                self.stop_watching();
                let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Unwatched));
            }
            FsRequest::List { path, recursive } => {
                if let Some(validated) = self.validate(&path, &tx) {
                    let entries = list_directory(&validated, recursive.unwrap_or(false)).await;
                    let _ =
                        tx.unbounded_send(WsOutbound::Fs(FsResponse::FileList { path, entries }));
                }
            }
            FsRequest::Read { path } => {
                if let Some(validated) = self.validate(&path, &tx) {
                    let resp = match fs::read_to_string(&validated).await {
                        Ok(content) => FsResponse::FileContent { path, content },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to read: {e}"),
                        },
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
            }
            FsRequest::Write { path, content } => {
                if let Some(validated) = self.validate(&path, &tx) {
                    if let Some(parent) = validated.parent()
                        && !parent.exists()
                        && let Err(e) = fs::create_dir_all(parent).await
                    {
                        let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                            message: format!("Failed to create dirs: {e}"),
                        }));
                        return;
                    }
                    let resp = match fs::write(&validated, content).await {
                        Ok(()) => FsResponse::WriteSuccess { path },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to write: {e}"),
                        },
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
            }
            FsRequest::Mkdir { path } => {
                if let Some(validated) = self.validate(&path, &tx) {
                    let resp = match fs::create_dir_all(&validated).await {
                        Ok(()) => FsResponse::MkdirSuccess { path },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to mkdir: {e}"),
                        },
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
            }
            FsRequest::Delete { path } => {
                if let Some(validated) = self.validate(&path, &tx) {
                    let resp = if validated.is_dir() {
                        match fs::remove_dir_all(&validated).await {
                            Ok(()) => FsResponse::DeleteSuccess { path },
                            Err(e) => FsResponse::Error {
                                message: format!("Failed to delete dir: {e}"),
                            },
                        }
                    } else {
                        match fs::remove_file(&validated).await {
                            Ok(()) => FsResponse::DeleteSuccess { path },
                            Err(e) => FsResponse::Error {
                                message: format!("Failed to delete file: {e}"),
                            },
                        }
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
            }
            FsRequest::Rename { from, to } => {
                if let (Some(vf), Some(vt)) = (self.validate(&from, &tx), self.validate(&to, &tx)) {
                    let resp = match fs::rename(&vf, &vt).await {
                        Ok(()) => FsResponse::RenameSuccess { from, to },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to rename: {e}"),
                        },
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
            }
        }
    }

    fn start_watching(&mut self, path: String, tx: UnboundedSender<WsOutbound>) {
        let Some(validated) = self.validate(&path, &tx) else {
            return;
        };
        if !validated.is_dir() {
            let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                message: format!("Invalid directory: {path}"),
            }));
            return;
        }

        self.stop_watching();

        let (notify_tx, mut notify_rx) = mpsc::unbounded::<notify::Result<Event>>();
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            while let Some(res) = notify_rx.next().await {
                match res {
                    Ok(event) => {
                        let _ = tx_clone.unbounded_send(WsOutbound::Fs(FsResponse::FsEvent {
                            event_kind: format!("{:?}", event.kind),
                            paths: event
                                .paths
                                .iter()
                                .map(|p| p.to_string_lossy().to_string())
                                .collect(),
                        }));
                    }
                    Err(e) => {
                        let _ = tx_clone.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                            message: format!("Watcher error: {e}"),
                        }));
                    }
                }
            }
        });

        let watcher_result = RecommendedWatcher::new(
            move |res| {
                let _ = notify_tx.unbounded_send(res);
            },
            Config::default().with_poll_interval(Duration::from_secs(1)),
        );

        match watcher_result {
            Ok(mut watcher) => {
                if let Err(e) = watcher.watch(&validated, RecursiveMode::Recursive) {
                    let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                        message: format!("Failed to watch: {e}"),
                    }));
                    return;
                }
                self._watcher = Some(watcher);
                let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Watching { path }));
            }
            Err(e) => {
                let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                    message: format!("Failed to create watcher: {e}"),
                }));
            }
        }
    }

    fn stop_watching(&mut self) {
        self._watcher = None;
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

    /// Runs one fs request and returns the first response sent to the client.
    async fn run_request(state: &mut FsState, req: FsRequest) -> FsResponse {
        let (tx, mut rx) = mpsc::unbounded::<WsOutbound>();
        state.handle(req, tx).await;
        // `tx` was dropped inside `handle`, so this yields the response or None.
        match rx.next().await {
            Some(WsOutbound::Fs(resp)) => resp,
            _ => panic!("expected an fs response"),
        }
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
        assert!(state._watcher.is_none());
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
        assert!(state._watcher.is_some());
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
