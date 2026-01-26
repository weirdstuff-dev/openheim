//! Filesystem WebSocket handler for workspace directory streaming.
//!
//! Provides real-time file system access to frontend clients:
//! - Watch directories for changes
//! - List directory contents
//! - Read/write files
//! - Create/delete/rename files and directories

use actix::{
    Actor, ActorContext, AsyncContext, Handler, Message as ActixMessage, StreamHandler,
    WrapFuture,
};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;
use tokio::fs;
use walkdir::WalkDir;

use crate::core::models::{FileEntry, FsRequest, FsResponse};

/// Actor managing a filesystem WebSocket connection
pub struct FsWebSocket {
    /// Current workspace root being watched (if any)
    workspace_root: Option<PathBuf>,
    /// File watcher handle (if active)
    watcher: Option<RecommendedWatcher>,
    /// Channel receiver for watcher events
    watcher_rx: Option<mpsc::Receiver<Result<Event, notify::Error>>>,
}

impl FsWebSocket {
    pub fn new() -> Self {
        Self {
            workspace_root: None,
            watcher: None,
            watcher_rx: None,
        }
    }

    fn send_json(&self, msg: &FsResponse, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(json) = serde_json::to_string(&msg) {
            ctx.text(json);
        }
    }

    fn send_error(&self, message: &str, ctx: &mut ws::WebsocketContext<Self>) {
        self.send_json(
            &FsResponse::Error {
                message: message.to_string(),
            },
            ctx,
        );
    }

    /// Validate that a path is within the workspace root (security check)
    fn validate_path(&self, path: &str) -> Option<PathBuf> {
        let workspace = self.workspace_root.as_ref()?;
        let requested = PathBuf::from(path);

        // Canonicalize to resolve .. and symlinks
        let canonical = if requested.is_absolute() {
            requested
        } else {
            workspace.join(&requested)
        };

        // For new files that don't exist yet, check the parent
        let check_path = if canonical.exists() {
            canonical.canonicalize().ok()?
        } else {
            // Check if parent exists and is within workspace
            let parent = canonical.parent()?;
            if !parent.exists() {
                return None;
            }
            let canonical_parent = parent.canonicalize().ok()?;
            let workspace_canonical = workspace.canonicalize().ok()?;
            if !canonical_parent.starts_with(&workspace_canonical) {
                return None;
            }
            return Some(canonical);
        };

        let workspace_canonical = workspace.canonicalize().ok()?;
        if check_path.starts_with(&workspace_canonical) {
            Some(check_path)
        } else {
            None
        }
    }

    /// Start watching a directory
    fn start_watching(&mut self, path: &str, ctx: &mut ws::WebsocketContext<Self>) {
        let workspace_path = PathBuf::from(path);

        if !workspace_path.exists() || !workspace_path.is_dir() {
            self.send_error(&format!("Invalid directory: {}", path), ctx);
            return;
        }

        // Stop any existing watcher
        self.stop_watching();

        // Create channel for watcher events
        let (tx, rx) = mpsc::channel();

        // Create the watcher
        let watcher_result = RecommendedWatcher::new(
            move |res| {
                let _ = tx.send(res);
            },
            Config::default().with_poll_interval(Duration::from_secs(1)),
        );

        match watcher_result {
            Ok(mut watcher) => {
                if let Err(e) = watcher.watch(&workspace_path, RecursiveMode::Recursive) {
                    self.send_error(&format!("Failed to watch directory: {}", e), ctx);
                    return;
                }

                self.workspace_root = Some(workspace_path.clone());
                self.watcher = Some(watcher);
                self.watcher_rx = Some(rx);

                // Start polling for file system events
                ctx.run_interval(Duration::from_millis(100), |act, ctx| {
                    act.poll_watcher_events(ctx);
                });

                self.send_json(
                    &FsResponse::Watching {
                        path: path.to_string(),
                    },
                    ctx,
                );
            }
            Err(e) => {
                self.send_error(&format!("Failed to create watcher: {}", e), ctx);
            }
        }
    }

    /// Stop watching the current directory
    fn stop_watching(&mut self) {
        self.watcher = None;
        self.watcher_rx = None;
        self.workspace_root = None;
    }

    /// Poll for file system events from the watcher
    fn poll_watcher_events(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        if let Some(rx) = &self.watcher_rx {
            // Drain all available events
            while let Ok(event_result) = rx.try_recv() {
                match event_result {
                    Ok(event) => {
                        let event_kind = format!("{:?}", event.kind);
                        let paths: Vec<String> = event
                            .paths
                            .iter()
                            .map(|p| p.to_string_lossy().to_string())
                            .collect();

                        self.send_json(&FsResponse::FsEvent { event_kind, paths }, ctx);
                    }
                    Err(e) => {
                        self.send_error(&format!("Watcher error: {}", e), ctx);
                    }
                }
            }
        }
    }
}

impl Default for FsWebSocket {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor for FsWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let msg = FsResponse::Connected {
            message: "Connected to Openheim Filesystem".to_string(),
        };
        self.send_json(&msg, ctx);
    }
}

// Messages for async operations

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct ListDir {
    path: String,
    recursive: bool,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct ReadFile {
    path: String,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct WriteFile {
    path: String,
    content: String,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct MkDir {
    path: String,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct DeletePath {
    path: String,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct RenamePath {
    from: String,
    to: String,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct SendResponse {
    response: FsResponse,
}

impl Handler<SendResponse> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: SendResponse, ctx: &mut Self::Context) {
        self.send_json(&msg.response, ctx);
    }
}

impl Handler<ListDir> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: ListDir, ctx: &mut Self::Context) {
        let validated_path = match self.validate_path(&msg.path) {
            Some(p) => p,
            None => {
                self.send_error("Path not within workspace or does not exist", ctx);
                return;
            }
        };

        let addr = ctx.address();
        let path_str = msg.path.clone();
        let recursive = msg.recursive;

        ctx.spawn(
            async move {
                let entries = list_directory(&validated_path, recursive).await;
                let response = FsResponse::FileList {
                    path: path_str,
                    entries,
                };
                addr.do_send(SendResponse { response });
            }
            .into_actor(self),
        );
    }
}

impl Handler<ReadFile> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: ReadFile, ctx: &mut Self::Context) {
        let validated_path = match self.validate_path(&msg.path) {
            Some(p) => p,
            None => {
                self.send_error("Path not within workspace or does not exist", ctx);
                return;
            }
        };

        let addr = ctx.address();
        let path_str = msg.path.clone();

        ctx.spawn(
            async move {
                let response = match fs::read_to_string(&validated_path).await {
                    Ok(content) => FsResponse::FileContent {
                        path: path_str,
                        content,
                    },
                    Err(e) => FsResponse::Error {
                        message: format!("Failed to read file: {}", e),
                    },
                };
                addr.do_send(SendResponse { response });
            }
            .into_actor(self),
        );
    }
}

impl Handler<WriteFile> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: WriteFile, ctx: &mut Self::Context) {
        let validated_path = match self.validate_path(&msg.path) {
            Some(p) => p,
            None => {
                self.send_error("Path not within workspace", ctx);
                return;
            }
        };

        let addr = ctx.address();
        let path_str = msg.path.clone();
        let content = msg.content;

        ctx.spawn(
            async move {
                // Create parent directories if needed
                if let Some(parent) = validated_path.parent()
                    && !parent.exists()
                        && let Err(e) = fs::create_dir_all(parent).await {
                            addr.do_send(SendResponse {
                                response: FsResponse::Error {
                                    message: format!("Failed to create directories: {}", e),
                                },
                            });
                            return;
                        }

                let response = match fs::write(&validated_path, content).await {
                    Ok(()) => FsResponse::WriteSuccess { path: path_str },
                    Err(e) => FsResponse::Error {
                        message: format!("Failed to write file: {}", e),
                    },
                };
                addr.do_send(SendResponse { response });
            }
            .into_actor(self),
        );
    }
}

impl Handler<MkDir> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: MkDir, ctx: &mut Self::Context) {
        let validated_path = match self.validate_path(&msg.path) {
            Some(p) => p,
            None => {
                self.send_error("Path not within workspace", ctx);
                return;
            }
        };

        let addr = ctx.address();
        let path_str = msg.path.clone();

        ctx.spawn(
            async move {
                let response = match fs::create_dir_all(&validated_path).await {
                    Ok(()) => FsResponse::MkdirSuccess { path: path_str },
                    Err(e) => FsResponse::Error {
                        message: format!("Failed to create directory: {}", e),
                    },
                };
                addr.do_send(SendResponse { response });
            }
            .into_actor(self),
        );
    }
}

impl Handler<DeletePath> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: DeletePath, ctx: &mut Self::Context) {
        let validated_path = match self.validate_path(&msg.path) {
            Some(p) => p,
            None => {
                self.send_error("Path not within workspace or does not exist", ctx);
                return;
            }
        };

        let addr = ctx.address();
        let path_str = msg.path.clone();

        ctx.spawn(
            async move {
                let response = if validated_path.is_dir() {
                    match fs::remove_dir_all(&validated_path).await {
                        Ok(()) => FsResponse::DeleteSuccess { path: path_str },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to delete directory: {}", e),
                        },
                    }
                } else {
                    match fs::remove_file(&validated_path).await {
                        Ok(()) => FsResponse::DeleteSuccess { path: path_str },
                        Err(e) => FsResponse::Error {
                            message: format!("Failed to delete file: {}", e),
                        },
                    }
                };
                addr.do_send(SendResponse { response });
            }
            .into_actor(self),
        );
    }
}

impl Handler<RenamePath> for FsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: RenamePath, ctx: &mut Self::Context) {
        let validated_from = match self.validate_path(&msg.from) {
            Some(p) => p,
            None => {
                self.send_error("Source path not within workspace or does not exist", ctx);
                return;
            }
        };

        let validated_to = match self.validate_path(&msg.to) {
            Some(p) => p,
            None => {
                self.send_error("Destination path not within workspace", ctx);
                return;
            }
        };

        let addr = ctx.address();
        let from_str = msg.from.clone();
        let to_str = msg.to.clone();

        ctx.spawn(
            async move {
                let response = match fs::rename(&validated_from, &validated_to).await {
                    Ok(()) => FsResponse::RenameSuccess {
                        from: from_str,
                        to: to_str,
                    },
                    Err(e) => FsResponse::Error {
                        message: format!("Failed to rename: {}", e),
                    },
                };
                addr.do_send(SendResponse { response });
            }
            .into_actor(self),
        );
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for FsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => match serde_json::from_str::<FsRequest>(&text) {
                Ok(req) => self.handle_request(req, ctx),
                Err(e) => {
                    self.send_error(&format!("Invalid request format: {}", e), ctx);
                }
            },
            Ok(ws::Message::Close(reason)) => {
                self.stop_watching();
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

impl FsWebSocket {
    fn handle_request(&mut self, req: FsRequest, ctx: &mut ws::WebsocketContext<Self>) {
        match req {
            FsRequest::Watch { path } => {
                self.start_watching(&path, ctx);
            }
            FsRequest::Unwatch => {
                self.stop_watching();
                self.send_json(&FsResponse::Unwatched, ctx);
            }
            FsRequest::List { path, recursive } => {
                ctx.notify(ListDir {
                    path,
                    recursive: recursive.unwrap_or(false),
                });
            }
            FsRequest::Read { path } => {
                ctx.notify(ReadFile { path });
            }
            FsRequest::Write { path, content } => {
                ctx.notify(WriteFile { path, content });
            }
            FsRequest::Mkdir { path } => {
                ctx.notify(MkDir { path });
            }
            FsRequest::Delete { path } => {
                ctx.notify(DeletePath { path });
            }
            FsRequest::Rename { from, to } => {
                ctx.notify(RenamePath { from, to });
            }
        }
    }
}

/// List directory contents
async fn list_directory(path: &Path, recursive: bool) -> Vec<FileEntry> {
    let mut entries = Vec::new();

    if recursive {
        for entry in WalkDir::new(path).min_depth(1).into_iter().filter_map(|e| e.ok()) {
            if let Some(file_entry) = path_to_file_entry(entry.path()) {
                entries.push(file_entry);
            }
        }
    } else if let Ok(mut dir) = fs::read_dir(path).await {
        while let Ok(Some(entry)) = dir.next_entry().await {
            if let Some(file_entry) = path_to_file_entry(&entry.path()) {
                entries.push(file_entry);
            }
        }
    }

    entries
}

/// Convert a path to a FileEntry
fn path_to_file_entry(path: &Path) -> Option<FileEntry> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let is_dir = path.is_dir();

    let metadata = path.metadata().ok();
    let size = metadata.as_ref().and_then(|m| {
        if m.is_file() {
            Some(m.len())
        } else {
            None
        }
    });
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

/// HTTP handler to upgrade to WebSocket
pub async fn ws_fs_handler(
    req: HttpRequest,
    stream: web::Payload,
) -> Result<HttpResponse, Error> {
    let ws = FsWebSocket::new();
    ws::start(ws, &req, stream)
}
