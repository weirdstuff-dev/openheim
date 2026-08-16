pub mod session;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Dispatch, Handled, on_receive_dispatch,
    on_receive_notification, on_receive_request,
    schema::{
        AgentCapabilities, CancelNotification, ClientCapabilities, ContentBlock as AcpContentBlock,
        ContentChunk, Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest,
        ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, ModelInfo,
        NewSessionRequest, NewSessionResponse, PermissionOption, PermissionOptionKind,
        PromptCapabilities, PromptRequest, PromptResponse, ReadTextFileRequest,
        ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionRequest,
        RequestPermissionResponse, SessionCapabilities, SessionInfo, SessionListCapabilities,
        SessionMode, SessionModeState, SessionModelState, SessionNotification, SessionUpdate,
        SetSessionModeRequest, SetSessionModeResponse, SetSessionModelRequest,
        SetSessionModelResponse, StopReason, TextContent, ToolCall as AcpToolCall, ToolCallStatus,
        ToolCallUpdate, ToolCallUpdateFields, ToolKind, WriteTextFileRequest,
        WriteTextFileResponse,
    },
    util::internal_error,
};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::{AgentConfig, AppConfig, build_http_client, create_client},
    core::{
        agent::run_agent_streaming_with_history,
        client_io::ClientIo,
        models::{ContentBlock, Message, Role, StopReason as CoreStopReason, StreamEvent},
        permission::{PermissionDecision, PermissionGate, approval_key},
        turn::TurnContext,
    },
    error::{Error, Result},
    llm::LlmClient,
    rag::{Conversation, RagContext},
    subagents::SubagentLoader,
    tools::{
        SandboxedExecutor, ScopedExecutor, SystemToolExecutor, ToolExecutor, ToolHandler,
        with_delegation,
    },
};

use session::{
    MAX_LIVE_SESSIONS, SESSION_IDLE_EVICTION_AFTER, SessionState, evict_idle_sessions,
    insert_or_keep_live,
};

type Sessions = Arc<RwLock<HashMap<String, SessionState>>>;

/// Which tool policy a session runs under, set via `session/set_mode`.
/// [`Self::as_str`] gives the ACP wire-level mode id; [`Self::parse`] is the
/// inverse, for the boundary where that id arrives as a `&str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentMode {
    /// Full tool access; tool calls go through the permission gate as normal.
    #[default]
    Code,
    /// Read-only: only `read_file` is offered to the LLM, so nothing
    /// mutating can run. `read_file` calls still go through the permission
    /// gate and can trigger a `session/request_permission` prompt unless
    /// already approved.
    Architect,
}

impl AgentMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            AgentMode::Code => "code",
            AgentMode::Architect => "architect",
        }
    }

    pub fn parse(mode_id: &str) -> Result<Self> {
        match mode_id {
            "code" => Ok(AgentMode::Code),
            "architect" => Ok(AgentMode::Architect),
            other => Err(Error::ParseError(format!("unknown session mode: {other}"))),
        }
    }
}

fn session_mode_state(current_mode: AgentMode) -> SessionModeState {
    SessionModeState::new(
        current_mode.as_str().to_string(),
        vec![
            SessionMode::new(AgentMode::Code.as_str(), "Code")
                .description("Full tool access; tool calls request permission."),
            SessionMode::new(AgentMode::Architect.as_str(), "Architect")
                .description("Read-only: inspects and plans without editing or executing."),
        ],
    )
}

pub struct AgentState {
    pub llm: Arc<dyn LlmClient>,
    pub executor: Arc<dyn ToolExecutor>,
    pub config: AgentConfig,
    pub app_config: AppConfig,
    pub rag: RagContext,
    pub mcp_statuses: Vec<crate::mcp::McpServerStatus>,
    /// Resolved work directory used as the sandbox boundary for every session.
    pub work_dir: PathBuf,
    /// Whether shell command execution is enabled for the LLM.
    pub allow_shell: bool,
    sessions: Sessions,
}

impl AgentState {
    /// `custom_tools` are registered alongside the built-ins (`execute_command`,
    /// `read_file`, `write_file`) and any MCP-sourced tools, before the
    /// sandbox/delegation wrappers are applied — so custom tools are subject
    /// to the same `work_dir`/`allow_shell` boundary as everything else.
    pub async fn new(
        config: AgentConfig,
        app_config: AppConfig,
        rag: RagContext,
        custom_tools: Vec<Box<dyn ToolHandler>>,
    ) -> Result<Self> {
        let http_client = build_http_client(config.timeout_secs)?;
        let llm = create_client(&config, &http_client);
        let allow_shell = app_config.allow_shell;
        let work_dir = match app_config.work_dir.clone() {
            Some(wd) => wd,
            None => std::env::current_dir().map_err(|e| {
                crate::error::Error::ConfigError(format!(
                    "failed to determine current directory for work_dir: {e}"
                ))
            })?,
        };
        let (mut sys_executor, mcp_statuses) =
            SystemToolExecutor::build(&app_config.mcp_servers, allow_shell).await;
        for tool in custom_tools {
            sys_executor.register(tool);
        }
        let executor = Arc::new(sys_executor) as Arc<dyn ToolExecutor>;

        let profiles = SubagentLoader::new()?.load()?;
        let executor = with_delegation(
            executor,
            work_dir.clone(),
            allow_shell,
            profiles,
            llm.clone(),
            app_config.clone(),
            config.clone(),
        );

        Ok(Self {
            llm,
            executor,
            config,
            app_config,
            rag,
            mcp_statuses,
            work_dir,
            allow_shell,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn acp_new_session(
        &self,
        model: Option<&str>,
        skills: Vec<String>,
        cwd: PathBuf,
    ) -> Result<String> {
        let chat_id = Uuid::new_v4();
        let session_key = chat_id.to_string();
        let config = model
            .and_then(|m| self.app_config.resolve(Some(m)).ok())
            .unwrap_or_else(|| self.config.clone());
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(
                session_key.clone(),
                SessionState {
                    chat_id,
                    config,
                    cwd,
                    skills,
                    cancel: CancellationToken::new(),
                    approved_tools: HashMap::new(),
                    mode: AgentMode::Code,
                    prompt_lock: Arc::new(Mutex::new(())),
                    last_active: Instant::now(),
                },
            );
            // Bound the map on every insert; a brand-new session has the
            // freshest `last_active`, so the sweep can only claim others.
            evict_idle_sessions(
                &mut sessions,
                Instant::now(),
                SESSION_IDLE_EVICTION_AFTER,
                MAX_LIVE_SESSIONS,
            );
        }
        Ok(session_key)
    }

    /// Cancels the currently active prompt turn for `session_id`, if any.
    /// No-op if the session doesn't exist or has no turn in flight.
    pub async fn cancel_session(&self, session_id: &str) {
        // Write lock: bumping `last_active` marks the session as recently
        // used so the eviction sweep can't claim an actively used session.
        if let Some(s) = self.sessions.write().await.get_mut(session_id) {
            s.last_active = Instant::now();
            s.cancel.cancel();
        }
    }

    /// Swaps a live session's [`AgentConfig`], returning its `(provider, model)`.
    /// Shared by the two public model-switch entry points below.
    async fn apply_session_config(
        &self,
        session_id: &str,
        new_config: AgentConfig,
    ) -> Result<(String, String)> {
        let provider_name = new_config.provider_name.clone();
        let model_name = new_config.model.clone();
        let mut sessions = self.sessions.write().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
        s.config = new_config;
        s.last_active = Instant::now();
        Ok((provider_name, model_name))
    }

    pub async fn acp_update_session_model(
        &self,
        session_id: &str,
        provider: &str,
        model: &str,
    ) -> Result<(String, String)> {
        let new_config = self.app_config.resolve_with_provider(provider, model)?;
        self.apply_session_config(session_id, new_config).await
    }

    pub async fn acp_set_session_model(
        &self,
        session_id: &str,
        model_id: &str,
    ) -> Result<(String, String)> {
        let new_config = self.app_config.resolve(Some(model_id))?;
        self.apply_session_config(session_id, new_config).await
    }

    pub async fn acp_set_session_mode(&self, session_id: &str, mode_id: &str) -> Result<()> {
        let mode = AgentMode::parse(mode_id)?;
        let mut sessions = self.sessions.write().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
        s.mode = mode;
        s.last_active = Instant::now();
        Ok(())
    }

    pub fn session_model_state(&self, current_model: &str) -> SessionModelState {
        let available_models = self
            .app_config
            .providers
            .iter()
            .flat_map(|(provider_name, p)| {
                p.models.iter().map(move |m| {
                    let mut meta = serde_json::Map::new();
                    meta.insert(
                        "provider".to_string(),
                        serde_json::Value::String(provider_name.clone()),
                    );
                    ModelInfo::new(m.clone(), m.clone()).meta(meta)
                })
            })
            .collect();
        SessionModelState::new(current_model.to_string(), available_models)
    }

    /// Persists `conv`'s full current state off the async runtime thread,
    /// logging (not propagating) any failure — history durability is
    /// best-effort and must never fail a turn that otherwise succeeded.
    /// `context` is folded into the warning log line to identify which of
    /// this method's call sites failed.
    async fn persist_conversation(&self, conv: &Conversation, context: &str) {
        let history = self.rag.history.clone();
        let conv = conv.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || history.save_conversation(&conv))
            .await
            .unwrap_or_else(|e| Err(Error::from(e)))
        {
            tracing::warn!("failed to {context}: {e}");
        }
    }

    /// Runs one prompt turn to completion and returns why it stopped, so the
    /// caller can map it to an ACP [`StopReason`] directly instead of having
    /// to reverse-engineer it (e.g. by polling session state for cancellation
    /// after the fact).
    pub async fn acp_prompt<F>(
        &self,
        session_id: &str,
        prompt: Vec<AcpContentBlock>,
        permission_gate: Arc<dyn PermissionGate>,
        client_io: Arc<dyn ClientIo>,
        mut on_update: F,
    ) -> Result<CoreStopReason>
    where
        F: FnMut(SessionUpdate) + Send,
    {
        let (llm, executor, config, chat_id, skills, cwd, cancel, _prompt_guard) = {
            // Write lock: each new prompt turn gets a fresh cancellation token,
            // since a token can only ever transition uncancelled -> cancelled
            // and must not leak a previous turn's cancellation into this one.
            let mut sessions = self.sessions.write().await;
            let s = sessions
                .get_mut(session_id)
                .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
            // Held until this function returns (success, error, or cancellation);
            // a second overlapping `session/prompt` on the same session would
            // otherwise race this one to reset `cancel` and to save history.
            let prompt_guard = s.try_acquire_prompt_lock(session_id)?;
            s.cancel = CancellationToken::new();
            s.last_active = Instant::now();
            let llm = crate::config::client_for_config(&s.config, &self.config, &self.llm)?;
            let base: Arc<dyn ToolExecutor> = if s.mode == AgentMode::Architect {
                Arc::new(ScopedExecutor::new(
                    self.executor.clone(),
                    vec!["read_file".to_string()],
                ))
            } else {
                self.executor.clone()
            };
            let sandboxed = Arc::new(SandboxedExecutor::new(
                base,
                self.work_dir.clone(),
                self.allow_shell,
                client_io,
            )) as Arc<dyn ToolExecutor>;
            (
                llm,
                sandboxed,
                s.config.clone(),
                s.chat_id,
                s.skills.clone(),
                s.cwd.clone(),
                s.cancel.clone(),
                prompt_guard,
            )
        };

        let (mut conversation, prompt_builder) = self.rag.prepare(
            Some(chat_id),
            &skills,
            Some(config.model.clone()),
            Some(config.provider_name.clone()),
        )?;

        conversation.meta.cwd = Some(cwd);
        conversation.messages.push(Message {
            role: Role::User,
            content: convert_prompt_blocks(&prompt)?,
        });

        // Full checkpoint before the turn starts: durably records this
        // turn's new user message even if the turn crashes before producing
        // anything else, and — since `save_conversation` always rewrites the
        // message log from scratch — transparently upgrades a pre-split-
        // format conversation (see `rag::history::HistoryManager`'s doc
        // comment) so the `append_message` calls below have a `.jsonl` log
        // that already reflects everything up to this point to append onto.
        self.persist_conversation(&conversation, "persist conversation before turn start")
            .await;

        let history_for_append = self.rag.history.clone();
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
        };
        let run_result = run_agent_streaming_with_history(
            llm,
            executor,
            &config,
            &mut conversation.messages,
            Some(&prompt_builder),
            &turn,
            move |event| match event {
                // Blocking I/O called synchronously (not via `spawn_blocking`)
                // deliberately: appends must land in the log in the same
                // order messages are produced, and this closure already runs
                // strictly sequentially with the rest of the turn, so a
                // small, fast local-disk append here doesn't race anything —
                // spawning it would only risk two concurrent appends landing
                // out of order.
                StreamEvent::MessageAppended { message } => {
                    if let Err(e) = history_for_append.append_message(&chat_id, &message) {
                        tracing::warn!("failed to append message to history: {e}");
                    }
                }
                StreamEvent::LlmResponse { content } => {
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::from(content),
                    )));
                }
                StreamEvent::ThinkingContent { content } => {
                    // Tunnel thinking through ContentBlock::Text using a meta tag so
                    // it survives the ACP layer (ACP's ContentBlock has no Thinking variant).
                    let mut meta = serde_json::Map::new();
                    meta.insert(
                        "kind".to_string(),
                        serde_json::Value::String("thinking".to_string()),
                    );
                    let text = TextContent::new(content).meta(meta);
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::Text(text),
                    )));
                }
                StreamEvent::ToolCall {
                    id,
                    tool_name,
                    arguments,
                } => {
                    // Pending, not InProgress: the permission gate (invoked by the
                    // agent loop right after this event) hasn't authorized
                    // execution yet at this point.
                    let raw_input = serde_json::from_str(&arguments).ok();
                    on_update(SessionUpdate::ToolCall(
                        AcpToolCall::new(id, &*tool_name)
                            .kind(tool_kind_for(&tool_name))
                            .status(ToolCallStatus::Pending)
                            .raw_input(raw_input),
                    ));
                }
                StreamEvent::ToolResult {
                    id,
                    result,
                    is_error,
                    ..
                } => {
                    let status = if is_error {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    };
                    on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                        id,
                        ToolCallUpdateFields::new()
                            .status(status)
                            .raw_output(serde_json::Value::String(result)),
                    )));
                }
                _ => {}
            },
        )
        .await;

        // Final full checkpoint: reconciles whatever `append_message` calls
        // landed above into one consistent, complete log, and is the only
        // save at all for a turn that produced no messages (cancelled or
        // errored before the first LLM response).
        self.persist_conversation(&conversation, "save conversation")
            .await;

        run_result.map(|r| r.stop_reason)
    }

    pub async fn acp_list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<SessionInfo>> {
        let history = self.rag.history.clone();
        let metas = tokio::task::spawn_blocking(move || history.list_conversations())
            .await
            .map_err(Error::from)??;
        Ok(metas
            .iter()
            .filter(|m| cwd.is_none_or(|filter| m.cwd.as_deref() == Some(filter)))
            .map(|m| {
                let path = m.cwd.clone().unwrap_or_else(|| PathBuf::from("/"));
                let mut info = SessionInfo::new(m.id.to_string(), path);
                if let Some(t) = &m.title {
                    info = info.title(t.clone());
                }
                info.updated_at(m.updated_at.to_rfc3339())
            })
            .collect())
    }

    pub async fn acp_load_session<F>(
        &self,
        session_id: &str,
        cwd: PathBuf,
        mut on_update: F,
    ) -> Result<()>
    where
        F: FnMut(SessionUpdate) + Send,
    {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| Error::ParseError("invalid session id format".to_string()))?;

        let history = self.rag.history.clone();
        let conversation = tokio::task::spawn_blocking(move || history.load_conversation(&uuid))
            .await
            .map_err(Error::from)??;

        let mut session_config = self.config.clone();
        if let Some(provider_name) = &conversation.meta.provider {
            // Same resolution (and validation) as every other config path;
            // a session whose saved provider/model no longer resolves —
            // removed from the config, model dropped from the allowlist —
            // falls back to the default provider rather than failing the load.
            let resolved = match &conversation.meta.model {
                Some(model) => self.app_config.resolve_with_provider(provider_name, model),
                None => self.app_config.resolve_provider_default(provider_name),
            };
            match resolved {
                Ok(config) => session_config = config,
                Err(e) => {
                    let warning = format!(
                        "[warning] Could not restore this session's provider '{}' ({e}). Falling back to the default provider '{}'.",
                        provider_name, session_config.provider_name
                    );
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::from(warning),
                    )));
                }
            }
        } else if let Some(model) = &conversation.meta.model {
            session_config.model = model.clone();
        }

        let fresh = SessionState {
            chat_id: uuid,
            config: session_config,
            cwd,
            skills: conversation.meta.skills.clone(),
            cancel: CancellationToken::new(),
            approved_tools: HashMap::new(),
            mode: AgentMode::Code,
            prompt_lock: Arc::new(Mutex::new(())),
            last_active: Instant::now(),
        };
        {
            let mut sessions = self.sessions.write().await;
            // A second connection attaching to an already-live session
            // must not replace its control state — a fresh `cancel` token
            // would orphan an in-flight turn, wiping `approved_tools` loses
            // remembered AllowAlways decisions, and a fresh `prompt_lock`
            // would let two turns overlap on one chat. The live entry (if
            // any) is also newer than the disk snapshot above; the history
            // replay below still streams to *this* connection either way.
            if !insert_or_keep_live(&mut sessions, session_id, fresh) {
                tracing::debug!("session {session_id} is already live; keeping live control state");
            }
            evict_idle_sessions(
                &mut sessions,
                Instant::now(),
                SESSION_IDLE_EVICTION_AFTER,
                MAX_LIVE_SESSIONS,
            );
        }

        for msg in &conversation.messages {
            match msg.role {
                Role::User => {
                    if let Some(text) = msg.text() {
                        on_update(SessionUpdate::UserMessageChunk(ContentChunk::new(
                            AcpContentBlock::from(text),
                        )));
                    }
                }
                Role::Assistant => {
                    if let Some(text) = msg.text() {
                        on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                            AcpContentBlock::from(text),
                        )));
                    }
                    for tc in msg.tool_calls() {
                        let raw_input = match serde_json::from_str(&tc.arguments) {
                            Ok(v) => Some(v),
                            Err(e) => {
                                tracing::warn!(
                                    tool_call_id = %tc.id,
                                    tool_name = %tc.name,
                                    "failed to parse tool call arguments: {e}"
                                );
                                None
                            }
                        };
                        on_update(SessionUpdate::ToolCall(
                            AcpToolCall::new(tc.id.clone(), &tc.name)
                                .status(ToolCallStatus::InProgress)
                                .raw_input(raw_input),
                        ));
                    }
                }
                Role::Tool => {
                    if let Some(tr) = msg.tool_result_block() {
                        let status = if tr.is_error {
                            ToolCallStatus::Failed
                        } else {
                            ToolCallStatus::Completed
                        };
                        on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                            tr.tool_call_id,
                            ToolCallUpdateFields::new()
                                .status(status)
                                .raw_output(serde_json::Value::String(tr.content)),
                        )));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Maps a builtin/MCP tool name to the closest ACP [`ToolKind`], purely for
/// client UI treatment (icons etc.) — has no bearing on execution.
fn tool_kind_for(tool_name: &str) -> ToolKind {
    match tool_name {
        "execute_command" => ToolKind::Execute,
        "read_file" => ToolKind::Read,
        "write_file" => ToolKind::Edit,
        _ => ToolKind::Other,
    }
}

/// Maps core's own [`CoreStopReason`] onto ACP's `StopReason`.
fn map_stop_reason(reason: CoreStopReason) -> StopReason {
    match reason {
        CoreStopReason::EndTurn => StopReason::EndTurn,
        CoreStopReason::MaxIterations => StopReason::MaxTurnRequests,
        CoreStopReason::Cancelled => StopReason::Cancelled,
        // ACP has no "the model returned nothing usable" variant; `EndTurn`
        // is the least misleading fit (it's not cancellation or exhaustion).
        CoreStopReason::NoContent => StopReason::EndTurn,
    }
}

/// [`PermissionGate`] backed by ACP's `session/request_permission`. Lives here
/// (not in `core`) because it depends on the live client connection.
struct AcpPermissionGate {
    cx: ConnectionTo<Client>,
    session_id: String,
    state: Arc<AgentState>,
}

#[async_trait::async_trait]
impl PermissionGate for AcpPermissionGate {
    async fn check(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision {
        let key = approval_key(tool_name, arguments);
        if let Some(remembered) = self
            .state
            .sessions
            .read()
            .await
            .get(&self.session_id)
            .and_then(|s| s.approved_tools.get(&key).copied())
        {
            return remembered;
        }

        let raw_input = serde_json::from_str(arguments).ok();
        let tool_call = ToolCallUpdate::new(
            tool_call_id.to_string(),
            ToolCallUpdateFields::new()
                .title(tool_name)
                .kind(tool_kind_for(tool_name))
                .status(ToolCallStatus::Pending)
                .raw_input(raw_input),
        );
        let options = vec![
            PermissionOption::new("allow_once", "Allow Once", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                "allow_always",
                "Allow Always",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(
                "reject_once",
                "Reject Once",
                PermissionOptionKind::RejectOnce,
            ),
            PermissionOption::new(
                "reject_always",
                "Reject Always",
                PermissionOptionKind::RejectAlways,
            ),
        ];

        let response = self
            .cx
            .send_request(RequestPermissionRequest::new(
                self.session_id.clone(),
                tool_call,
                options,
            ))
            .block_task()
            .await;

        let decision = match response {
            Ok(RequestPermissionResponse {
                outcome: RequestPermissionOutcome::Selected(selected),
                ..
            }) => match selected.option_id.0.as_ref() {
                "allow_once" => PermissionDecision::AllowOnce,
                "allow_always" => PermissionDecision::AllowAlways,
                "reject_always" => PermissionDecision::RejectAlways,
                _ => PermissionDecision::RejectOnce,
            },
            Ok(RequestPermissionResponse {
                outcome: RequestPermissionOutcome::Cancelled,
                ..
            }) => PermissionDecision::RejectOnce,
            // `RequestPermissionOutcome` is #[non_exhaustive]; treat any future
            // variant conservatively, same as an explicit rejection.
            Ok(_) => PermissionDecision::RejectOnce,
            Err(e) => {
                tracing::warn!("session/request_permission failed: {e}");
                PermissionDecision::RejectOnce
            }
        };

        if matches!(
            decision,
            PermissionDecision::AllowAlways | PermissionDecision::RejectAlways
        ) && let Some(s) = self.state.sessions.write().await.get_mut(&self.session_id)
        {
            s.approved_tools.insert(key, decision);
        }

        decision
    }
}

/// [`ClientIo`] backed by ACP's `fs/read_text_file` / `fs/write_text_file`.
/// Only attempts them when the client actually advertised the corresponding
/// capability at `initialize` time; otherwise defers to local I/O.
struct AcpClientIo {
    cx: ConnectionTo<Client>,
    session_id: String,
    client_capabilities: Arc<RwLock<ClientCapabilities>>,
}

#[async_trait::async_trait]
impl ClientIo for AcpClientIo {
    async fn read_file(&self, path: &std::path::Path) -> Option<Result<String>> {
        if !self.client_capabilities.read().await.fs.read_text_file {
            return None;
        }
        let response = self
            .cx
            .send_request(ReadTextFileRequest::new(
                self.session_id.clone(),
                path.to_path_buf(),
            ))
            .block_task()
            .await;
        Some(match response {
            Ok(ReadTextFileResponse { content, .. }) => Ok(content),
            Err(e) => Err(Error::Other(format!("fs/read_text_file failed: {e}"))),
        })
    }

    async fn write_file(&self, path: &std::path::Path, content: &str) -> Option<Result<()>> {
        if !self.client_capabilities.read().await.fs.write_text_file {
            return None;
        }
        let response = self
            .cx
            .send_request(WriteTextFileRequest::new(
                self.session_id.clone(),
                path.to_path_buf(),
                content,
            ))
            .block_task()
            .await;
        Some(match response {
            Ok(WriteTextFileResponse { .. }) => Ok(()),
            Err(e) => Err(Error::Other(format!("fs/write_text_file failed: {e}"))),
        })
    }
}

/// Converts an ACP `session/prompt` payload into content blocks for a new
/// user [`Message`].
///
/// `Text` and `Image` pass through directly (images require the `image`
/// prompt capability, which `initialize` declares). `ResourceLink` agents
/// must support unconditionally per the ACP spec, but embedding its content
/// isn't implemented — it's surfaced as a text pointer instead, which the
/// agent's own file tools can follow if needed. `Audio` and embedded
/// `Resource` blocks (and anything the `#[non_exhaustive]` enum might add
/// later) aren't supported at all: reject loudly instead of silently
/// dropping part of the user's input.
fn convert_prompt_blocks(blocks: &[AcpContentBlock]) -> Result<Vec<ContentBlock>> {
    let mut out = Vec::new();
    for block in blocks {
        match block {
            AcpContentBlock::Text(t) => {
                out.push(ContentBlock::Text {
                    text: t.text.clone(),
                });
            }
            AcpContentBlock::Image(img) => {
                out.push(ContentBlock::Image {
                    data: img.data.clone(),
                    mime_type: img.mime_type.clone(),
                });
            }
            AcpContentBlock::ResourceLink(link) => {
                out.push(ContentBlock::Text {
                    text: format!("[referenced resource: {} ({})]", link.name, link.uri),
                });
            }
            AcpContentBlock::Audio(_) => {
                return Err(Error::Other(
                    "audio prompt content is not supported".to_string(),
                ));
            }
            AcpContentBlock::Resource(_) => {
                return Err(Error::Other(
                    "embedded resource prompt content is not supported".to_string(),
                ));
            }
            _ => {
                return Err(Error::Other("unsupported prompt content type".to_string()));
            }
        }
    }
    Ok(out)
}

pub async fn serve(
    transport: impl ConnectTo<Agent>,
    state: Arc<AgentState>,
) -> agent_client_protocol::Result<()> {
    let state_init = state.clone();
    let state_session = state.clone();
    let state_prompt = state.clone();
    let state_list = state.clone();
    let state_load = state.clone();
    let state_set_model = state.clone();
    let state_set_mode = state.clone();
    let state_cancel = state.clone();

    // Client capabilities are per-connection (learned once, at `initialize`),
    // not per-session — `serve()` runs fresh per connection even though
    // `AgentState` itself is shared, so this is scoped correctly here.
    let client_capabilities = Arc::new(RwLock::new(ClientCapabilities::default()));
    let client_capabilities_init = client_capabilities.clone();
    let client_capabilities_prompt = client_capabilities.clone();

    Agent
        .builder()
        .name("openheim")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx: ConnectionTo<Client>| {
                *client_capabilities_init.write().await = req.client_capabilities.clone();
                let mut meta = serde_json::Map::new();
                if let Ok(val) = serde_json::to_value(state_init.app_config.models_info()) {
                    meta.insert("models".to_string(), val);
                }
                if let Ok(val) = serde_json::to_value(&state_init.mcp_statuses) {
                    meta.insert("mcp_servers".to_string(), val);
                }
                if let Ok(skills) = state_init.rag.skills.list_skills()
                    && let Ok(val) = serde_json::to_value(skills)
                {
                    meta.insert("skills".to_string(), val);
                }
                if let Ok(val) = serde_json::to_value(state_init.executor.list_tools()) {
                    meta.insert("tools".to_string(), val);
                }
                // Advertise that thinking content arrives as AgentMessageChunk with
                // content._meta.kind == "thinking" (ACP _meta extensibility).
                meta.insert(
                    "thinking".to_string(),
                    serde_json::json!({ "meta_key": "kind", "meta_value": "thinking" }),
                );
                responder.respond(
                    InitializeResponse::new(req.protocol_version)
                        .agent_capabilities(
                            AgentCapabilities::new()
                                .load_session(true)
                                .prompt_capabilities(PromptCapabilities::new().image(true))
                                .session_capabilities(
                                    SessionCapabilities::new().list(SessionListCapabilities::new()),
                                ),
                        )
                        .agent_info(Implementation::new("openheim", env!("CARGO_PKG_VERSION")))
                        .meta(meta),
                )
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: NewSessionRequest, responder, _cx: ConnectionTo<Client>| {
                let skills: Vec<String> = req
                    .meta
                    .as_ref()
                    .and_then(|m| m.get("skills"))
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                let model = req
                    .meta
                    .as_ref()
                    .and_then(|m| m.get("model"))
                    .and_then(|v| v.as_str())
                    .map(String::from);

                let current_model = model
                    .as_deref()
                    .unwrap_or(&state_session.config.model)
                    .to_string();
                let model_state = state_session.session_model_state(&current_model);

                match state_session
                    .acp_new_session(model.as_deref(), skills, req.cwd)
                    .await
                {
                    Ok(session_key) => responder.respond(
                        NewSessionResponse::new(session_key)
                            .models(model_state)
                            .modes(session_mode_state(AgentMode::Code)),
                    ),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                let session_key = req.session_id.to_string();
                let prompt_blocks = req.prompt;
                let cx_cb = cx.clone();
                let session_id_cb = req.session_id.clone();
                let state = state_prompt.clone();
                let permission_gate = Arc::new(AcpPermissionGate {
                    cx: cx.clone(),
                    session_id: session_key.clone(),
                    state: state.clone(),
                }) as Arc<dyn PermissionGate>;
                let client_io = Arc::new(AcpClientIo {
                    cx: cx.clone(),
                    session_id: session_key.clone(),
                    client_capabilities: client_capabilities_prompt.clone(),
                }) as Arc<dyn ClientIo>;

                // The prompt turn can run for a long time (many LLM/tool round-trips)
                // and, once permission requests land, will itself await replies from
                // the client. Handlers run on the single-task event loop, which can't
                // read new messages (including those replies, or a session/cancel
                // notification) while a handler is executing — so this must be moved
                // off the event loop via `cx.spawn`, with the response sent from
                // inside the spawned task once the turn actually finishes.
                cx.spawn(async move {
                    let result = state
                        .acp_prompt(
                            &session_key,
                            prompt_blocks,
                            permission_gate,
                            client_io,
                            move |update| {
                                let _ = cx_cb.send_notification(SessionNotification::new(
                                    session_id_cb.clone(),
                                    update,
                                ));
                            },
                        )
                        .await;

                    // A spawned task returning an error shuts down the whole
                    // connection, so per-turn failures must never propagate past
                    // here — they're reported via the response instead.
                    let respond_result = match result {
                        Ok(stop_reason) => {
                            responder.respond(PromptResponse::new(map_stop_reason(stop_reason)))
                        }
                        Err(e) => {
                            tracing::error!("agent loop error: {e}");
                            responder.respond_with_internal_error(e.to_string())
                        }
                    };
                    if let Err(e) = respond_result {
                        tracing::warn!("failed to send prompt response: {e}");
                    }
                    Ok(())
                })
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: ListSessionsRequest, responder, _cx: ConnectionTo<Client>| {
                match state_list.acp_list_sessions(req.cwd.as_deref()).await {
                    Ok(sessions) => responder.respond(ListSessionsResponse::new(sessions)),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                let session_id_str = req.session_id.0.as_ref().to_string();
                let cx_cb = cx.clone();
                let session_id_cb = req.session_id.clone();

                let result = state_load
                    .acp_load_session(&session_id_str, req.cwd.clone(), move |update| {
                        let _ = cx_cb.send_notification(SessionNotification::new(
                            session_id_cb.clone(),
                            update,
                        ));
                    })
                    .await;

                match result {
                    Ok(()) => responder.respond(
                        LoadSessionResponse::new().modes(session_mode_state(AgentMode::Code)),
                    ),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: SetSessionModelRequest, responder, _cx: ConnectionTo<Client>| {
                let session_id = req.session_id.0.as_ref().to_string();
                let model_id = req.model_id.0.as_ref();
                match state_set_model
                    .acp_set_session_model(&session_id, model_id)
                    .await
                {
                    Ok(_) => responder.respond(SetSessionModelResponse::new()),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: SetSessionModeRequest, responder, _cx: ConnectionTo<Client>| {
                let session_id = req.session_id.0.as_ref().to_string();
                let mode_id = req.mode_id.0.as_ref();
                match state_set_mode
                    .acp_set_session_mode(&session_id, mode_id)
                    .await
                {
                    Ok(()) => responder.respond(SetSessionModeResponse::new()),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
        )
        .on_receive_notification(
            async move |notif: CancelNotification, _cx: ConnectionTo<Client>| {
                let session_id = notif.session_id.0.as_ref().to_string();
                state_cancel.cancel_session(&session_id).await;
                Ok(())
            },
            on_receive_notification!(),
        )
        .on_receive_dispatch(
            async move |message: Dispatch,
                        cx: ConnectionTo<Client>|
                        -> agent_client_protocol::Result<Handled<Dispatch>> {
                // Responses to requests *this agent* sent (e.g.
                // session/request_permission, fs/read_text_file) are also
                // routed through here if nothing else claims them first.
                // `respond_with_error` would convert a legitimate success
                // response into an error delivered to whatever is awaiting
                // it — so those must be declined, not rejected, letting the
                // crate's own default handling forward the real result.
                if matches!(message, Dispatch::Response(..)) {
                    return Ok(Handled::No {
                        message,
                        retry: false,
                    });
                }
                message.respond_with_error(internal_error("unsupported method"), cx)?;
                Ok(Handled::Yes)
            },
            on_receive_dispatch!(),
        )
        .connect_to(transport)
        .await
}

#[cfg(test)]
mod prompt_block_tests {
    use super::*;
    use agent_client_protocol::schema::{
        AudioContent, EmbeddedResource, EmbeddedResourceResource, ImageContent, ResourceLink,
        TextResourceContents,
    };

    #[test]
    fn text_passes_through() {
        let blocks = vec![AcpContentBlock::Text(TextContent::new("hello"))];
        let result = convert_prompt_blocks(&blocks).unwrap();
        assert_eq!(
            result,
            vec![ContentBlock::Text {
                text: "hello".into()
            }]
        );
    }

    #[test]
    fn image_passes_through() {
        let blocks = vec![AcpContentBlock::Image(ImageContent::new(
            "base64data",
            "image/png",
        ))];
        let result = convert_prompt_blocks(&blocks).unwrap();
        assert_eq!(
            result,
            vec![ContentBlock::Image {
                data: "base64data".into(),
                mime_type: "image/png".into(),
            }]
        );
    }

    #[test]
    fn resource_link_becomes_text_hint() {
        let blocks = vec![AcpContentBlock::ResourceLink(ResourceLink::new(
            "notes.txt",
            "file:///tmp/notes.txt",
        ))];
        let result = convert_prompt_blocks(&blocks).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(
            &result[0],
            ContentBlock::Text { text }
                if text.contains("notes.txt") && text.contains("file:///tmp/notes.txt")
        ));
    }

    #[test]
    fn audio_is_rejected() {
        let blocks = vec![AcpContentBlock::Audio(AudioContent::new(
            "base64data",
            "audio/wav",
        ))];
        assert!(convert_prompt_blocks(&blocks).is_err());
    }

    #[test]
    fn embedded_resource_is_rejected() {
        let blocks = vec![AcpContentBlock::Resource(EmbeddedResource::new(
            EmbeddedResourceResource::TextResourceContents(TextResourceContents::new(
                "content",
                "file:///tmp/notes.txt",
            )),
        ))];
        assert!(convert_prompt_blocks(&blocks).is_err());
    }
}
