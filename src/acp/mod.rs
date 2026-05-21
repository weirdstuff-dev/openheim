pub mod session;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Dispatch, on_receive_dispatch, on_receive_request,
    schema::{
        AgentCapabilities, ContentBlock, ContentChunk, Implementation, InitializeRequest,
        InitializeResponse, ListSessionsRequest, ListSessionsResponse, LoadSessionRequest,
        LoadSessionResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
        SessionCapabilities, SessionInfo, SessionListCapabilities, SessionNotification,
        SessionUpdate, StopReason, ToolCall as AcpToolCall, ToolCallStatus, ToolCallUpdate,
        ToolCallUpdateFields,
    },
    util::internal_error,
};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::{AgentConfig, AppConfig, build_http_client, create_client},
    core::{
        agent::run_agent_streaming_with_history,
        models::{Message, Role, StreamEvent},
    },
    error::{Error, Result},
    llm::LlmClient,
    rag::RagContext,
    tools::{SystemToolExecutor, ToolExecutor},
};

use session::SessionState;

type Sessions = Arc<RwLock<HashMap<String, SessionState>>>;

pub struct AgentState {
    pub llm: Arc<dyn LlmClient>,
    pub executor: Arc<dyn ToolExecutor>,
    pub config: AgentConfig,
    pub app_config: AppConfig,
    pub rag: RagContext,
    pub mcp_statuses: Vec<crate::mcp::McpServerStatus>,
    sessions: Sessions,
}

impl AgentState {
    pub async fn new(config: AgentConfig, app_config: AppConfig, rag: RagContext) -> Result<Self> {
        let http_client = build_http_client(config.timeout_secs)?;
        let llm = create_client(&config, &http_client);
        let (sys_executor, mcp_statuses) = SystemToolExecutor::build(&app_config.mcp_servers).await;
        let executor = Arc::new(sys_executor) as Arc<dyn ToolExecutor>;
        Ok(Self {
            llm,
            executor,
            config,
            app_config,
            rag,
            mcp_statuses,
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
            },
        );
        Ok(session_key)
    }

    pub async fn acp_prompt<F>(
        &self,
        session_id: &str,
        text: String,
        mut on_update: F,
    ) -> Result<()>
    where
        F: FnMut(SessionUpdate) + Send,
    {
        let (llm, executor, config, chat_id, skills, cwd) = {
            let sessions = self.sessions.read().await;
            let s = sessions
                .get(session_id)
                .ok_or_else(|| Error::Other(format!("session not found: {session_id}")))?;
            (
                self.llm.clone(),
                self.executor.clone(),
                s.config.clone(),
                s.chat_id,
                s.skills.clone(),
                s.cwd.clone(),
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

        let mut last_tool_call_id: Option<String> = None;

        let run_result = run_agent_streaming_with_history(
            llm,
            executor,
            &config,
            &mut conversation.messages,
            Some(&prompt_builder),
            move |event| match event {
                StreamEvent::LlmResponse { content } => {
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        ContentBlock::from(content),
                    )));
                }
                StreamEvent::ToolCall {
                    tool_name,
                    arguments,
                } => {
                    let id = Uuid::new_v4().to_string();
                    last_tool_call_id = Some(id.clone());
                    let raw_input = serde_json::from_str(&arguments).ok();
                    on_update(SessionUpdate::ToolCall(
                        AcpToolCall::new(id, &*tool_name)
                            .status(ToolCallStatus::InProgress)
                            .raw_input(raw_input),
                    ));
                }
                StreamEvent::ToolResult {
                    result, is_error, ..
                } => {
                    if let Some(id) = last_tool_call_id.take() {
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
        if let Some(model) = &conversation.meta.model {
            session_config.model = model.clone();
        }
        if let Some(provider) = &conversation.meta.provider {
            session_config.provider_name = provider.clone();
        }

        self.sessions.write().await.insert(
            session_id.to_string(),
            SessionState {
                chat_id: uuid,
                config: session_config,
                cwd,
                skills: conversation.meta.skills.clone(),
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

                match state_session
                    .acp_new_session(model.as_deref(), skills, req.cwd)
                    .await
                {
                    Ok(session_key) => responder.respond(NewSessionResponse::new(session_key)),
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

                let result = state_prompt
                    .acp_prompt(&session_key, text, move |update| {
                        let _ = cx_cb.send_notification(SessionNotification::new(
                            session_id_cb.clone(),
                            update,
                        ));
                    })
                    .await;

                match result {
                    Ok(()) => responder.respond(PromptResponse::new(StopReason::EndTurn)),
                    Err(e) => {
                        tracing::error!("agent loop error: {e}");
                        responder.respond_with_internal_error(e.to_string())
                    }
                }
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
                    Ok(()) => responder.respond(LoadSessionResponse::new()),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            on_receive_request!(),
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
