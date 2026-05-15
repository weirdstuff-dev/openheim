//! WebSocket + REST transport: serves the agent over an axum HTTP server.
//!
//! Exposes the following endpoints:
//!
//! | Endpoint | Description |
//! |----------|-------------|
//! | `GET /ws` | WebSocket endpoint; multiplexes ACP agent messages and filesystem events |
//! | `GET /api/config` | Resolved configuration (providers, models) |
//! | `GET /api/models` | Available providers and their model lists |
//! | `GET /api/skills` | Installed skill names |
//! | `GET /api/tools` | Registered tool definitions |
//! | `GET /api/mcp-servers` | MCP server connection statuses |
//! | `GET /api/sessions` | Conversation history listing |
//! | `GET /api/sessions/{id}` | Single conversation by UUID |
//!
//! WebSocket messages are JSON-encoded with a `channel` discriminator:
//! - `{"channel":"agent","data":{…}}` — ACP protocol frames
//! - `{"channel":"fs","data":{…}}` — filesystem sidecar (watch / list / read / write / mkdir / delete / rename)

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
    acp::{self, AgentState},
    config::load_config,
    core::models::{FileEntry, FsRequest, FsResponse},
    error::Error as AppError,
    rag::RagContext,
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

/// Loads configuration, initialises the agent runtime, and starts the HTTP/WebSocket server.
///
/// Blocks until a Ctrl-C signal is received, then shuts down gracefully.
pub async fn serve(host: String, port: u16) -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;
    let rag = RagContext::new()?;
    let state = Arc::new(AgentState::new(agent_config, app_config, rag).await?);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/ws", get(ws_handler))
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

    tracing::info!("WS server listening on ws://{addr}/ws");
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
    Json(state.app_config.to_public_json())
}

async fn models_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.app_config.models_info())
}

async fn skills_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.rag.skills.list_skills().unwrap_or_default())
}

async fn tools_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.executor.list_tools())
}

async fn mcp_servers_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    Json(state.mcp_statuses.clone())
}

async fn sessions_handler(State(state): State<Arc<AgentState>>) -> impl IntoResponse {
    match state.rag.history.list_conversations() {
        Ok(metas) => Json(metas).into_response(),
        Err(_) => (
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
    match state.rag.history.load_conversation(&uuid) {
        Ok(conv) => Json(conv).into_response(),
        Err(AppError::Other(_)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "session not found" })),
        )
            .into_response(),
        Err(_) => (
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
    let mut fs_state = FsState::new();
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

// FS sidecar state

struct FsState {
    workspace_root: Option<PathBuf>,
    _watcher: Option<RecommendedWatcher>,
}

impl FsState {
    fn new() -> Self {
        Self {
            workspace_root: None,
            _watcher: None,
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
                match validate_path_opt(&self.workspace_root, &path) {
                    Some(validated) => {
                        let entries = list_directory(&validated, recursive.unwrap_or(false)).await;
                        let _ = tx
                            .unbounded_send(WsOutbound::Fs(FsResponse::FileList { path, entries }));
                    }
                    None => send_path_error(&tx),
                }
            }
            FsRequest::Read { path } => match validate_path_opt(&self.workspace_root, &path) {
                Some(validated) => {
                    let resp = match fs::read_to_string(&validated).await {
                        Ok(content) => FsResponse::FileContent { path, content },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to read: {e}"),
                        },
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
                None => send_path_error(&tx),
            },
            FsRequest::Write { path, content } => {
                match validate_path_opt(&self.workspace_root, &path) {
                    Some(validated) => {
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
                    None => send_path_error(&tx),
                }
            }
            FsRequest::Mkdir { path } => match validate_path_opt(&self.workspace_root, &path) {
                Some(validated) => {
                    let resp = match fs::create_dir_all(&validated).await {
                        Ok(()) => FsResponse::MkdirSuccess { path },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to mkdir: {e}"),
                        },
                    };
                    let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                }
                None => send_path_error(&tx),
            },
            FsRequest::Delete { path } => match validate_path_opt(&self.workspace_root, &path) {
                Some(validated) => {
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
                None => send_path_error(&tx),
            },
            FsRequest::Rename { from, to } => {
                match (
                    validate_path_opt(&self.workspace_root, &from),
                    validate_path_opt(&self.workspace_root, &to),
                ) {
                    (Some(vf), Some(vt)) => {
                        let resp = match fs::rename(&vf, &vt).await {
                            Ok(()) => FsResponse::RenameSuccess { from, to },
                            Err(e) => FsResponse::Error {
                                message: format!("Failed to rename: {e}"),
                            },
                        };
                        let _ = tx.unbounded_send(WsOutbound::Fs(resp));
                    }
                    _ => send_path_error(&tx),
                }
            }
        }
    }

    fn start_watching(&mut self, path: String, tx: UnboundedSender<WsOutbound>) {
        let workspace_path = PathBuf::from(&path);
        if !workspace_path.exists() || !workspace_path.is_dir() {
            let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                message: format!("Invalid directory: {path}"),
            }));
            return;
        }

        let workspace_canonical = match workspace_path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                    message: format!("Failed to resolve path: {e}"),
                }));
                return;
            }
        };

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
                if let Err(e) = watcher.watch(&workspace_canonical, RecursiveMode::Recursive) {
                    let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
                        message: format!("Failed to watch: {e}"),
                    }));
                    return;
                }
                self.workspace_root = Some(workspace_canonical);
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
        self.workspace_root = None;
    }
}

fn send_path_error(tx: &UnboundedSender<WsOutbound>) {
    let _ = tx.unbounded_send(WsOutbound::Fs(FsResponse::Error {
        message: "Path not within workspace or does not exist".to_string(),
    }));
}

fn validate_path_opt(workspace: &Option<PathBuf>, path: &str) -> Option<PathBuf> {
    let workspace = workspace.as_ref()?;
    let workspace_canonical = workspace.canonicalize().ok()?;

    let requested = PathBuf::from(path);

    let canonical = if requested.is_absolute() {
        requested
    } else {
        if let Ok(cwd) = std::env::current_dir() {
            let from_cwd = cwd.join(&requested);
            if from_cwd.exists() {
                if let Ok(c) = from_cwd.canonicalize() {
                    if c.starts_with(&workspace_canonical) {
                        return Some(c);
                    }
                }
            }
        }
        // Fallback: treat path as relative to the workspace root
        workspace_canonical.join(&requested)
    };

    let check_path = if canonical.exists() {
        canonical.canonicalize().ok()?
    } else {
        let mut parent = canonical.as_path();
        loop {
            parent = parent.parent()?;
            if parent.exists() {
                let canonical_ancestor = parent.canonicalize().ok()?;
                if !canonical_ancestor.starts_with(&workspace_canonical) {
                    return None;
                }
                return Some(canonical);
            }
        }
    };

    if check_path.starts_with(&workspace_canonical) {
        Some(check_path)
    } else {
        None
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
