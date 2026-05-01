use actix::{Actor, ActorContext, AsyncContext, Handler, Message as ActixMessage, StreamHandler, WrapFuture};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use tokio::fs;
use walkdir::WalkDir;

use crate::api::WsRegistry;
use crate::config::{AgentConfig, AppConfig, resolve_client_and_config};
use crate::core::agent::run_agent_streaming_with_history;
use crate::core::llm::LlmClient;
use crate::core::models::{
    ClientEnvelope, FileEntry, FsRequest, FsResponse, Message, ServerEnvelope, StreamEvent,
    SystemEvent, WsRequest, WsResponse,
};
use crate::rag::RagContext;
use crate::tools::ToolExecutor;

pub struct OpenheimWebSocket {
    // Agent
    llm: Arc<dyn LlmClient>,
    executor: Arc<dyn ToolExecutor>,
    config: AgentConfig,
    app_config: AppConfig,
    rag: RagContext,
    // Filesystem
    workspace_root: Option<PathBuf>,
    watcher: Option<RecommendedWatcher>,
    watcher_rx: Option<mpsc::Receiver<Result<Event, notify::Error>>>,
    // Registry
    registry: WsRegistry,
    my_addr: Option<actix::Addr<Self>>,
}

impl OpenheimWebSocket {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        executor: Arc<dyn ToolExecutor>,
        config: AgentConfig,
        app_config: AppConfig,
        rag: RagContext,
        registry: WsRegistry,
    ) -> Self {
        Self {
            llm,
            executor,
            config,
            app_config,
            rag,
            workspace_root: None,
            watcher: None,
            watcher_rx: None,
            registry,
            my_addr: None,
        }
    }

    fn send_envelope(&self, envelope: &ServerEnvelope, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(json) = serde_json::to_string(envelope) {
            ctx.text(json);
        }
    }

    fn send_agent(&self, msg: WsResponse, ctx: &mut ws::WebsocketContext<Self>) {
        self.send_envelope(&ServerEnvelope::Agent(msg), ctx);
    }

    fn send_fs(&self, msg: FsResponse, ctx: &mut ws::WebsocketContext<Self>) {
        self.send_envelope(&ServerEnvelope::Fs(msg), ctx);
    }

    // --- Agent ---

    fn resolve_agent_config(
        &self,
        req: &WsRequest,
    ) -> crate::error::Result<(Arc<dyn LlmClient>, AgentConfig)> {
        resolve_client_and_config(
            req.model.as_deref(),
            req.max_iterations,
            &self.app_config,
            self.llm.clone(),
            &self.config,
        )
    }

    fn handle_agent_request(&mut self, req: WsRequest, ctx: &mut ws::WebsocketContext<Self>) {
        match self.resolve_agent_config(&req) {
            Ok((llm, config)) => {
                ctx.notify(ExecuteAgent {
                    llm,
                    config,
                    prompt: req.prompt,
                    chat_id: req.chat_id,
                    skills: req.skills.unwrap_or_default(),
                    rag: self.rag.clone(),
                });
            }
            Err(e) => {
                self.send_agent(WsResponse::Error { message: e.to_string() }, ctx);
            }
        }
    }

    // --- Filesystem ---

    /// Validates that a path is within the workspace root to prevent path traversal.
    fn validate_path(&self, path: &str) -> Option<PathBuf> {
        let workspace = self.workspace_root.as_ref()?;
        let requested = PathBuf::from(path);

        let canonical = if requested.is_absolute() {
            requested
        } else {
            workspace.join(&requested)
        };

        let check_path = if canonical.exists() {
            canonical.canonicalize().ok()?
        } else {
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

    fn start_watching(&mut self, path: &str, ctx: &mut ws::WebsocketContext<Self>) {
        let workspace_path = PathBuf::from(path);

        if !workspace_path.exists() || !workspace_path.is_dir() {
            self.send_fs(
                FsResponse::Error { message: format!("Invalid directory: {}", path) },
                ctx,
            );
            return;
        }

        self.stop_watching();

        let (tx, rx) = mpsc::channel();
        let watcher_result = RecommendedWatcher::new(
            move |res| {
                let _ = tx.send(res);
            },
            Config::default().with_poll_interval(Duration::from_secs(1)),
        );

        match watcher_result {
            Ok(mut watcher) => {
                if let Err(e) = watcher.watch(&workspace_path, RecursiveMode::Recursive) {
                    self.send_fs(
                        FsResponse::Error { message: format!("Failed to watch directory: {}", e) },
                        ctx,
                    );
                    return;
                }

                self.workspace_root = Some(workspace_path);
                self.watcher = Some(watcher);
                self.watcher_rx = Some(rx);

                ctx.run_interval(Duration::from_millis(100), |act, ctx| {
                    act.poll_watcher_events(ctx);
                });

                self.send_fs(FsResponse::Watching { path: path.to_string() }, ctx);
            }
            Err(e) => {
                self.send_fs(
                    FsResponse::Error { message: format!("Failed to create watcher: {}", e) },
                    ctx,
                );
            }
        }
    }

    fn stop_watching(&mut self) {
        self.watcher = None;
        self.watcher_rx = None;
        self.workspace_root = None;
    }

    fn poll_watcher_events(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        if let Some(rx) = &self.watcher_rx {
            while let Ok(event_result) = rx.try_recv() {
                match event_result {
                    Ok(event) => {
                        let event_kind = format!("{:?}", event.kind);
                        let paths: Vec<String> =
                            event.paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
                        self.send_fs(FsResponse::FsEvent { event_kind, paths }, ctx);
                    }
                    Err(e) => {
                        self.send_fs(
                            FsResponse::Error { message: format!("Watcher error: {}", e) },
                            ctx,
                        );
                    }
                }
            }
        }
    }

    fn handle_fs_request(&mut self, req: FsRequest, ctx: &mut ws::WebsocketContext<Self>) {
        match req {
            FsRequest::Watch { path } => self.start_watching(&path, ctx),
            FsRequest::Unwatch => {
                self.stop_watching();
                self.send_fs(FsResponse::Unwatched, ctx);
            }
            FsRequest::List { path, recursive } => {
                ctx.notify(FsOp::List { path, recursive: recursive.unwrap_or(false) });
            }
            FsRequest::Read { path } => ctx.notify(FsOp::Read { path }),
            FsRequest::Write { path, content } => ctx.notify(FsOp::Write { path, content }),
            FsRequest::Mkdir { path } => ctx.notify(FsOp::Mkdir { path }),
            FsRequest::Delete { path } => ctx.notify(FsOp::Delete { path }),
            FsRequest::Rename { from, to } => ctx.notify(FsOp::Rename { from, to }),
        }
    }
}

// --- Actor impl ---

impl Actor for OpenheimWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.my_addr = Some(addr.clone());
        self.registry.lock().unwrap().push(addr);
        self.send_envelope(
            &ServerEnvelope::System(SystemEvent::Connected {
                message: "Connected to Openheim".to_string(),
            }),
            ctx,
        );
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        if let Some(addr) = &self.my_addr {
            self.registry.lock().unwrap().retain(|a| a != addr);
        }
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for OpenheimWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => match serde_json::from_str::<ClientEnvelope>(&text) {
                Ok(ClientEnvelope::Agent(req)) => self.handle_agent_request(req, ctx),
                Ok(ClientEnvelope::Fs(req)) => self.handle_fs_request(req, ctx),
                Err(e) => {
                    self.send_envelope(
                        &ServerEnvelope::System(SystemEvent::Error {
                            message: format!("Invalid request: {}", e),
                        }),
                        ctx,
                    );
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

// --- Agent messages ---

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct ExecuteAgent {
    llm: Arc<dyn LlmClient>,
    config: AgentConfig,
    prompt: String,
    chat_id: Option<uuid::Uuid>,
    skills: Vec<String>,
    rag: RagContext,
}

impl Handler<ExecuteAgent> for OpenheimWebSocket {
    type Result = ();

    fn handle(&mut self, msg: ExecuteAgent, ctx: &mut Self::Context) {
        let llm = msg.llm;
        let executor = self.executor.clone();
        let config = msg.config;
        let prompt = msg.prompt;
        let chat_id = msg.chat_id;
        let skills = msg.skills;
        let rag = msg.rag;
        let addr = ctx.address();
        let addr_for_closure = addr.clone();

        ctx.spawn(
            async move {
                let (mut conversation, prompt_builder) = match rag.prepare(
                    chat_id,
                    &skills,
                    Some(config.model.clone()),
                    Some(config.provider_name.clone()),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Agent(WsResponse::Error {
                                message: e.to_string(),
                            })),
                        });
                        return;
                    }
                };

                conversation.messages.push(Message::user(prompt));
                let conv_id = conversation.meta.id;

                let result = run_agent_streaming_with_history(
                    llm,
                    executor,
                    &config,
                    &mut conversation.messages,
                    Some(&prompt_builder),
                    move |event: StreamEvent| {
                        addr_for_closure.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Agent(WsResponse::Event {
                                data: event,
                            })),
                        });
                    },
                )
                .await;

                if let Err(e) = rag.history.save_conversation(&conversation) {
                    tracing::warn!("Failed to save conversation: {e}");
                }

                match result {
                    Ok(_) => addr.do_send(SendText {
                        text: serialize_envelope(&ServerEnvelope::Agent(WsResponse::Done {
                            chat_id: Some(conv_id.to_string()),
                        })),
                    }),
                    Err(e) => addr.do_send(SendText {
                        text: serialize_envelope(&ServerEnvelope::Agent(WsResponse::Error {
                            message: e.to_string(),
                        })),
                    }),
                }
            }
            .into_actor(self),
        );
    }
}

// --- Filesystem messages ---

#[derive(ActixMessage)]
#[rtype(result = "()")]
enum FsOp {
    List { path: String, recursive: bool },
    Read { path: String },
    Write { path: String, content: String },
    Mkdir { path: String },
    Delete { path: String },
    Rename { from: String, to: String },
}

impl Handler<FsOp> for OpenheimWebSocket {
    type Result = ();

    fn handle(&mut self, op: FsOp, ctx: &mut Self::Context) {
        let addr = ctx.address();

        macro_rules! require_path {
            ($path:expr) => {
                match self.validate_path(&$path) {
                    Some(p) => p,
                    None => {
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(FsResponse::Error {
                                message: "Path not within workspace or does not exist".to_string(),
                            })),
                        });
                        return;
                    }
                }
            };
        }

        match op {
            FsOp::List { path, recursive } => {
                let validated = require_path!(path.clone());
                ctx.spawn(
                    async move {
                        let entries = list_directory(&validated, recursive).await;
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(FsResponse::FileList {
                                path,
                                entries,
                            })),
                        });
                    }
                    .into_actor(self),
                );
            }
            FsOp::Read { path } => {
                let validated = require_path!(path.clone());
                ctx.spawn(
                    async move {
                        let response = match fs::read_to_string(&validated).await {
                            Ok(content) => FsResponse::FileContent { path, content },
                            Err(e) => {
                                FsResponse::Error { message: format!("Failed to read file: {}", e) }
                            }
                        };
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(response)),
                        });
                    }
                    .into_actor(self),
                );
            }
            FsOp::Write { path, content } => {
                let validated = require_path!(path.clone());
                ctx.spawn(
                    async move {
                        if let Some(parent) = validated.parent()
                            && !parent.exists()
                            && let Err(e) = fs::create_dir_all(parent).await
                        {
                            addr.do_send(SendText {
                                text: serialize_envelope(&ServerEnvelope::Fs(FsResponse::Error {
                                    message: format!("Failed to create directories: {}", e),
                                })),
                            });
                            return;
                        }

                        let response = match fs::write(&validated, content).await {
                            Ok(()) => FsResponse::WriteSuccess { path },
                            Err(e) => {
                                FsResponse::Error { message: format!("Failed to write file: {}", e) }
                            }
                        };
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(response)),
                        });
                    }
                    .into_actor(self),
                );
            }
            FsOp::Mkdir { path } => {
                let validated = require_path!(path.clone());
                ctx.spawn(
                    async move {
                        let response = match fs::create_dir_all(&validated).await {
                            Ok(()) => FsResponse::MkdirSuccess { path },
                            Err(e) => FsResponse::Error {
                                message: format!("Failed to create directory: {}", e),
                            },
                        };
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(response)),
                        });
                    }
                    .into_actor(self),
                );
            }
            FsOp::Delete { path } => {
                let validated = require_path!(path.clone());
                ctx.spawn(
                    async move {
                        let response = if validated.is_dir() {
                            match fs::remove_dir_all(&validated).await {
                                Ok(()) => FsResponse::DeleteSuccess { path },
                                Err(e) => FsResponse::Error {
                                    message: format!("Failed to delete directory: {}", e),
                                },
                            }
                        } else {
                            match fs::remove_file(&validated).await {
                                Ok(()) => FsResponse::DeleteSuccess { path },
                                Err(e) => FsResponse::Error {
                                    message: format!("Failed to delete file: {}", e),
                                },
                            }
                        };
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(response)),
                        });
                    }
                    .into_actor(self),
                );
            }
            FsOp::Rename { from, to } => {
                let validated_from = require_path!(from.clone());
                let validated_to = require_path!(to.clone());
                ctx.spawn(
                    async move {
                        let response = match fs::rename(&validated_from, &validated_to).await {
                            Ok(()) => FsResponse::RenameSuccess { from, to },
                            Err(e) => {
                                FsResponse::Error { message: format!("Failed to rename: {}", e) }
                            }
                        };
                        addr.do_send(SendText {
                            text: serialize_envelope(&ServerEnvelope::Fs(response)),
                        });
                    }
                    .into_actor(self),
                );
            }
        }
    }
}

// --- Shutdown message ---

#[derive(ActixMessage)]
#[rtype(result = "()")]
pub struct Shutdown;

impl Handler<Shutdown> for OpenheimWebSocket {
    type Result = ();

    fn handle(&mut self, _: Shutdown, ctx: &mut Self::Context) {
        ctx.close(Some(ws::CloseReason {
            code: ws::CloseCode::Restart,
            description: None,
        }));
        ctx.stop();
    }
}

// --- Shared message for sending pre-serialized text from spawned futures ---

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct SendText {
    text: String,
}

impl Handler<SendText> for OpenheimWebSocket {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        ctx.text(msg.text);
    }
}

// --- Helpers ---

fn serialize_envelope(envelope: &ServerEnvelope) -> String {
    serde_json::to_string(envelope).unwrap_or_else(|_| r#"{"channel":"system","data":{"type":"error","message":"serialization error"}}"#.to_string())
}

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

fn path_to_file_entry(path: &Path) -> Option<FileEntry> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let is_dir = path.is_dir();

    let metadata = path.metadata().ok();
    let size = metadata.as_ref().and_then(|m| if m.is_file() { Some(m.len()) } else { None });
    let modified = metadata.as_ref().and_then(|m| {
        m.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
    });

    Some(FileEntry { path: path.to_string_lossy().to_string(), name, is_dir, size, modified })
}

// --- HTTP handler ---

pub async fn ws_handler(
    req: HttpRequest,
    stream: web::Payload,
    llm: web::Data<Arc<dyn LlmClient>>,
    executor: web::Data<Arc<dyn ToolExecutor>>,
    config: web::Data<AgentConfig>,
    app_config: web::Data<AppConfig>,
    rag: web::Data<RagContext>,
    registry: web::Data<WsRegistry>,
) -> Result<HttpResponse, Error> {
    let ws = OpenheimWebSocket::new(
        llm.get_ref().clone(),
        executor.get_ref().clone(),
        config.get_ref().clone(),
        app_config.get_ref().clone(),
        rag.get_ref().clone(),
        registry.get_ref().clone(),
    );
    ws::start(ws, &req, stream)
}
