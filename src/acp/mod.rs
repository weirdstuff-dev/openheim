pub mod session;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Dispatch, on_receive_dispatch, on_receive_notification,
    on_receive_request,
    schema::{
        AgentCapabilities, CancelNotification, ContentBlock, ContentChunk, Implementation,
        InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
        LoadSessionRequest, LoadSessionResponse, ModelInfo, NewSessionRequest, NewSessionResponse,
        PermissionOption, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority,
        PlanEntryStatus, PromptRequest, PromptResponse, RequestPermissionOutcome,
        RequestPermissionRequest, RequestPermissionResponse, SessionCapabilities, SessionInfo,
        SessionListCapabilities, SessionMode, SessionModeState, SessionModelState,
        SessionNotification, SessionUpdate, SetSessionModeRequest, SetSessionModeResponse,
        SetSessionModelRequest, SetSessionModelResponse, StopReason, TextContent,
        ToolCall as AcpToolCall, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
    },
    util::internal_error,
};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::{AgentConfig, AppConfig, build_http_client, create_client},
    core::{
        agent::{TurnContext, run_agent_streaming_with_history},
        models::{Message, PlanStepStatus, Role, StreamEvent},
        permission::{PermissionDecision, PermissionGate},
    },
    error::{Error, Result},
    llm::LlmClient,
    rag::RagContext,
    subagents::SubagentLoader,
    tools::{SandboxedExecutor, ScopedExecutor, SystemToolExecutor, ToolExecutor, with_delegation},
};

use session::SessionState;

type Sessions = Arc<RwLock<HashMap<String, SessionState>>>;

/// Full tool access; tool calls go through the permission gate as normal.
pub const MODE_CODE: &str = "code";
/// Read-only: only `read_file` is offered to the LLM, regardless of
/// permission decisions. No `session/request_permission` prompts occur
/// since nothing mutating is ever on the tool list.
pub const MODE_ARCHITECT: &str = "architect";

fn session_mode_state(current_mode_id: &str) -> SessionModeState {
    SessionModeState::new(
        current_mode_id.to_string(),
        vec![
            SessionMode::new(MODE_CODE, "Code")
                .description("Full tool access; tool calls request permission."),
            SessionMode::new(MODE_ARCHITECT, "Architect")
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
    pub async fn new(config: AgentConfig, app_config: AppConfig, rag: RagContext) -> Result<Self> {
        let http_client = build_http_client(config.timeout_secs)?;
        let llm = create_client(&config, &http_client);
        let allow_shell = app_config.allow_shell;
        let work_dir = match app_config.work_dir.clone() {
            Some(wd) => wd,
            None => std::env::current_dir().map_err(|e| {
                crate::error::Error::Other(format!(
                    "failed to determine current directory for work_dir: {e}"
                ))
            })?,
        };
        let (sys_executor, mcp_statuses) =
            SystemToolExecutor::build(&app_config.mcp_servers, allow_shell).await;
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
        self.sessions.write().await.insert(
            session_key.clone(),
            SessionState {
                chat_id,
                config,
                cwd,
                skills,
                cancel: CancellationToken::new(),
                approved_tools: HashMap::new(),
                mode: MODE_CODE.to_string(),
            },
        );
        Ok(session_key)
    }

    /// Cancels the currently active prompt turn for `session_id`, if any.
    /// No-op if the session doesn't exist or has no turn in flight.
    pub async fn cancel_session(&self, session_id: &str) {
        if let Some(s) = self.sessions.read().await.get(session_id) {
            s.cancel.cancel();
        }
    }

    /// Whether the most recent (or currently running) prompt turn for
    /// `session_id` was cancelled via [`Self::cancel_session`].
    pub async fn is_session_cancelled(&self, session_id: &str) -> bool {
        self.sessions
            .read()
            .await
            .get(session_id)
            .map(|s| s.cancel.is_cancelled())
            .unwrap_or(false)
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
            .ok_or_else(|| Error::Other(format!("session not found: {session_id}")))?;
        s.config = new_config;
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
        if mode_id != MODE_CODE && mode_id != MODE_ARCHITECT {
            return Err(Error::Other(format!("unknown session mode: {mode_id}")));
        }
        let mut sessions = self.sessions.write().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Other(format!("session not found: {session_id}")))?;
        s.mode = mode_id.to_string();
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

    pub async fn acp_prompt<F>(
        &self,
        session_id: &str,
        text: String,
        permission_gate: Arc<dyn PermissionGate>,
        mut on_update: F,
    ) -> Result<()>
    where
        F: FnMut(SessionUpdate) + Send,
    {
        let (llm, executor, config, chat_id, skills, cwd, cancel) = {
            // Write lock: each new prompt turn gets a fresh cancellation token,
            // since a token can only ever transition uncancelled -> cancelled
            // and must not leak a previous turn's cancellation into this one.
            let mut sessions = self.sessions.write().await;
            let s = sessions
                .get_mut(session_id)
                .ok_or_else(|| Error::Other(format!("session not found: {session_id}")))?;
            s.cancel = CancellationToken::new();
            let llm = crate::config::client_for_config(&s.config, &self.config, &self.llm)?;
            let base: Arc<dyn ToolExecutor> = if s.mode == MODE_ARCHITECT {
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
            )) as Arc<dyn ToolExecutor>;
            (
                llm,
                sandboxed,
                s.config.clone(),
                s.chat_id,
                s.skills.clone(),
                s.cwd.clone(),
                s.cancel.clone(),
            )
        };

        let (mut conversation, prompt_builder) = self.rag.prepare(
            Some(chat_id),
            &skills,
            Some(config.model.clone()),
            Some(config.provider_name.clone()),
        )?;

        conversation.meta.cwd = Some(cwd);
        conversation.messages.push(Message::user(text));

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
                StreamEvent::LlmResponse { content } => {
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        ContentBlock::from(content),
                    )));
                }
                StreamEvent::ThinkingContent { content } => {
                    // Tunnel thinking through ContentBlock::Text using a meta tag so
                    // it survives the ACP layer (ContentBlock has no Thinking variant).
                    let mut meta = serde_json::Map::new();
                    meta.insert(
                        "kind".to_string(),
                        serde_json::Value::String("thinking".to_string()),
                    );
                    let text = TextContent::new(content).meta(meta);
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        ContentBlock::Text(text),
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
                StreamEvent::PlanUpdate { entries } => {
                    on_update(SessionUpdate::Plan(Plan::new(
                        entries
                            .into_iter()
                            .map(|step| {
                                let status = match step.status {
                                    PlanStepStatus::Pending => PlanEntryStatus::Pending,
                                    PlanStepStatus::InProgress => PlanEntryStatus::InProgress,
                                    PlanStepStatus::Completed => PlanEntryStatus::Completed,
                                };
                                PlanEntry::new(step.content, PlanEntryPriority::Medium, status)
                            })
                            .collect(),
                    )));
                }
                _ => {}
            },
        )
        .await;

        let history = self.rag.history.clone();
        let conv_to_save = conversation.clone();
        if let Err(e) =
            tokio::task::spawn_blocking(move || history.save_conversation(&conv_to_save))
                .await
                .unwrap_or_else(|e| Err(Error::Other(e.to_string())))
        {
            tracing::warn!("failed to save conversation: {e}");
        }

        run_result.map(|_| ())
    }

    pub async fn acp_list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<SessionInfo>> {
        let history = self.rag.history.clone();
        let metas = tokio::task::spawn_blocking(move || history.list_conversations())
            .await
            .map_err(|e| Error::Other(e.to_string()))??;
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
            .map_err(|_| Error::Other("invalid session id format".to_string()))?;

        let history = self.rag.history.clone();
        let conversation = tokio::task::spawn_blocking(move || history.load_conversation(&uuid))
            .await
            .map_err(|e| Error::Other(e.to_string()))??;

        let mut session_config = self.config.clone();
        if let Some(provider_name) = &conversation.meta.provider {
            if let Some(provider_cfg) = self.app_config.providers.get(provider_name) {
                session_config.provider_name = provider_name.clone();
                session_config.api_base = provider_cfg.api_base.clone();
                session_config.api_key = provider_cfg.resolve_api_key();
                session_config.timeout_secs = provider_cfg.timeout_secs.unwrap_or(120);
                session_config.max_tokens = provider_cfg.max_tokens;
                session_config.model = conversation
                    .meta
                    .model
                    .clone()
                    .unwrap_or_else(|| provider_cfg.default_model.clone());
            } else {
                let warning = format!(
                    "[warning] Provider '{}' from this session is not configured. Falling back to the default provider '{}'.",
                    provider_name, session_config.provider_name
                );
                on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                    ContentBlock::from(warning),
                )));
            }
        } else if let Some(model) = &conversation.meta.model {
            session_config.model = model.clone();
        }

        self.sessions.write().await.insert(
            session_id.to_string(),
            SessionState {
                chat_id: uuid,
                config: session_config,
                cwd,
                skills: conversation.meta.skills.clone(),
                cancel: CancellationToken::new(),
                approved_tools: HashMap::new(),
                mode: MODE_CODE.to_string(),
            },
        );

        for msg in &conversation.messages {
            match msg.role {
                Role::User => {
                    let text = msg.content.clone().unwrap_or_default();
                    if !text.is_empty() {
                        on_update(SessionUpdate::UserMessageChunk(ContentChunk::new(
                            ContentBlock::from(text),
                        )));
                    }
                }
                Role::Assistant => {
                    let text = msg.content.clone().unwrap_or_default();
                    if !text.is_empty() {
                        on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                            ContentBlock::from(text),
                        )));
                    }
                    if let Some(tool_calls) = &msg.tool_calls {
                        for tc in tool_calls {
                            let raw_input = match serde_json::from_str(&tc.function.arguments) {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    tracing::warn!(
                                        tool_call_id = %tc.id,
                                        tool_name = %tc.function.name,
                                        "failed to parse tool call arguments: {e}"
                                    );
                                    None
                                }
                            };
                            on_update(SessionUpdate::ToolCall(
                                AcpToolCall::new(tc.id.clone(), &tc.function.name)
                                    .status(ToolCallStatus::InProgress)
                                    .raw_input(raw_input),
                            ));
                        }
                    }
                }
                Role::Tool => {
                    if let (Some(id), Some(content)) = (&msg.tool_call_id, &msg.content) {
                        let status = if msg.is_error {
                            ToolCallStatus::Failed
                        } else {
                            ToolCallStatus::Completed
                        };
                        on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                            id.clone(),
                            ToolCallUpdateFields::new()
                                .status(status)
                                .raw_output(serde_json::Value::String(content.clone())),
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
        if let Some(remembered) = self
            .state
            .sessions
            .read()
            .await
            .get(&self.session_id)
            .and_then(|s| s.approved_tools.get(tool_name).copied())
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
            s.approved_tools.insert(tool_name.to_string(), decision);
        }

        decision
    }
}

fn extract_prompt_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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

    Agent
        .builder()
        .name("openheim")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx: ConnectionTo<Client>| {
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
                            .modes(session_mode_state(MODE_CODE)),
                    ),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                let session_key = req.session_id.to_string();
                let text = extract_prompt_text(&req.prompt);
                let cx_cb = cx.clone();
                let session_id_cb = req.session_id.clone();
                let state = state_prompt.clone();
                let permission_gate = Arc::new(AcpPermissionGate {
                    cx: cx.clone(),
                    session_id: session_key.clone(),
                    state: state.clone(),
                }) as Arc<dyn PermissionGate>;

                // The prompt turn can run for a long time (many LLM/tool round-trips)
                // and, once permission requests land, will itself await replies from
                // the client. Handlers run on the single-task event loop, which can't
                // read new messages (including those replies, or a session/cancel
                // notification) while a handler is executing — so this must be moved
                // off the event loop via `cx.spawn`, with the response sent from
                // inside the spawned task once the turn actually finishes.
                cx.spawn(async move {
                    let result = state
                        .acp_prompt(&session_key, text, permission_gate, move |update| {
                            let _ = cx_cb.send_notification(SessionNotification::new(
                                session_id_cb.clone(),
                                update,
                            ));
                        })
                        .await;

                    // A spawned task returning an error shuts down the whole
                    // connection, so per-turn failures must never propagate past
                    // here — they're reported via the response instead.
                    let respond_result = match result {
                        Ok(()) => {
                            let stop_reason = if state.is_session_cancelled(&session_key).await {
                                StopReason::Cancelled
                            } else {
                                StopReason::EndTurn
                            };
                            responder.respond(PromptResponse::new(stop_reason))
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
                    Ok(()) => responder
                        .respond(LoadSessionResponse::new().modes(session_mode_state(MODE_CODE))),
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
            async move |message: Dispatch, cx: ConnectionTo<Client>| {
                message.respond_with_error(internal_error("unsupported method"), cx)
            },
            on_receive_dispatch!(),
        )
        .connect_to(transport)
        .await
}
