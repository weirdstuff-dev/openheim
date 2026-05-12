pub mod session;

use std::{collections::HashMap, sync::Arc};

use agent_client_protocol::{
    Agent, Client, ConnectionTo, Dispatch, ConnectTo,
    on_receive_dispatch, on_receive_request,
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
    core::{agent::run_agent_streaming_with_history, models::{Role, StreamEvent}},
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
    pub async fn new(
        config: AgentConfig,
        app_config: AppConfig,
        rag: RagContext,
    ) -> crate::error::Result<Self> {
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
                if let Ok(skills) = state_init.rag.skills.list_skills() {
                    if let Ok(val) = serde_json::to_value(skills) {
                        meta.insert("skills".to_string(), val);
                    }
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
                                    SessionCapabilities::new()
                                        .list(SessionListCapabilities::new()),
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
                let chat_id = Uuid::new_v4();
                let session_key = chat_id.to_string();

                let skills: Vec<String> = req.meta
                    .as_ref()
                    .and_then(|m| m.get("skills"))
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();

                let config = req.meta
                    .as_ref()
                    .and_then(|m| m.get("model"))
                    .and_then(|v| v.as_str())
                    .and_then(|model| state_session.app_config.resolve(Some(model)).ok())
                    .unwrap_or_else(|| state_session.config.clone());

                state_session.sessions.write().await.insert(
                    session_key.clone(),
                    SessionState {
                        chat_id,
                        config,
                        cwd: req.cwd,
                        skills,
                    },
                );

                responder.respond(NewSessionResponse::new(session_key))
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                let session_key = req.session_id.to_string();
                let text = extract_prompt_text(&req.prompt);

                let (llm, executor, config, rag, chat_id, skills, cwd) = {
                    let sessions = state_prompt.sessions.read().await;
                    match sessions.get(&session_key) {
                        Some(s) => (
                            state_prompt.llm.clone(),
                            state_prompt.executor.clone(),
                            s.config.clone(),
                            state_prompt.rag.clone(),
                            s.chat_id,
                            s.skills.clone(),
                            s.cwd.clone(),
                        ),
                        None => {
                            return responder.respond_with_internal_error(
                                format!("session not found: {session_key}"),
                            );
                        }
                    }
                };

                let (mut conversation, prompt_builder) = match rag.prepare(
                    Some(chat_id),
                    &skills,
                    Some(config.model.clone()),
                    Some(config.provider_name.clone()),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        return responder.respond_with_internal_error(
                            format!("failed to prepare conversation: {e}"),
                        );
                    }
                };

                conversation.meta.cwd = Some(cwd);
                conversation
                    .messages
                    .push(crate::core::models::Message::user(text));

                let cx_cb = cx.clone();
                let session_id_cb = req.session_id.clone();
                let mut last_tool_call_id: Option<String> = None;

                let run_result = run_agent_streaming_with_history(
                    llm,
                    executor,
                    &config,
                    &mut conversation.messages,
                    Some(&prompt_builder),
                    move |event| match event {
                        StreamEvent::LlmResponse { content } => {
                            let _ = cx_cb.send_notification(SessionNotification::new(
                                session_id_cb.clone(),
                                SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                    ContentBlock::from(content),
                                )),
                            ));
                        }
                        StreamEvent::ToolCall { tool_name, arguments } => {
                            let id = Uuid::new_v4().to_string();
                            last_tool_call_id = Some(id.clone());
                            let raw_input = serde_json::from_str(&arguments).ok();
                            let _ = cx_cb.send_notification(SessionNotification::new(
                                session_id_cb.clone(),
                                SessionUpdate::ToolCall(
                                    AcpToolCall::new(id, &*tool_name)
                                        .status(ToolCallStatus::InProgress)
                                        .raw_input(raw_input),
                                ),
                            ));
                        }
                        StreamEvent::ToolResult { result, .. } => {
                            if let Some(id) = last_tool_call_id.take() {
                                let _ = cx_cb.send_notification(SessionNotification::new(
                                    session_id_cb.clone(),
                                    SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                                        id,
                                        ToolCallUpdateFields::new()
                                            .status(ToolCallStatus::Completed)
                                            .raw_output(serde_json::Value::String(result)),
                                    )),
                                ));
                            }
                        }
                        _ => {}
                    },
                )
                .await;

                if let Err(e) = rag.history.save_conversation(&conversation) {
                    tracing::warn!("failed to save conversation: {e}");
                }

                match run_result {
                    Ok(_) => responder.respond(PromptResponse::new(StopReason::EndTurn)),
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
                let metas = match state_list.rag.history.list_conversations() {
                    Ok(m) => m,
                    Err(e) => return responder.respond_with_internal_error(e.to_string()),
                };

                let sessions: Vec<SessionInfo> = metas
                    .iter()
                    .filter(|m| {
                        req.cwd.as_ref().map_or(true, |filter| {
                            m.cwd.as_deref().map_or(false, |c| c == filter.as_path())
                        })
                    })
                    .map(|m| {
                        let cwd = m.cwd.clone().unwrap_or_else(|| std::path::PathBuf::from("/"));
                        let mut info = SessionInfo::new(m.id.to_string(), cwd);
                        if let Some(t) = &m.title {
                            info = info.title(t.clone());
                        }
                        info.updated_at(m.updated_at.to_rfc3339())
                    })
                    .collect();

                responder.respond(ListSessionsResponse::new(sessions))
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                let session_id_str = req.session_id.0.as_ref();
                let uuid = match uuid::Uuid::parse_str(session_id_str) {
                    Ok(u) => u,
                    Err(_) => {
                        return responder
                            .respond_with_internal_error("invalid session id format");
                    }
                };

                let conversation = match state_load.rag.history.load_conversation(&uuid) {
                    Ok(c) => c,
                    Err(e) => return responder.respond_with_internal_error(e.to_string()),
                };

                let mut session_config = state_load.config.clone();
                if let Some(model) = &conversation.meta.model {
                    session_config.model = model.clone();
                }
                if let Some(provider) = &conversation.meta.provider {
                    session_config.provider_name = provider.clone();
                }
                state_load.sessions.write().await.insert(
                    req.session_id.0.to_string(),
                    SessionState {
                        chat_id: uuid,
                        config: session_config,
                        cwd: req.cwd.clone(),
                        skills: conversation.meta.skills.clone(),
                    },
                );

                for msg in &conversation.messages {
                    let text = msg.content.clone().unwrap_or_default();
                    if text.is_empty() {
                        continue;
                    }
                    let update = match msg.role {
                        Role::User => SessionUpdate::UserMessageChunk(ContentChunk::new(
                            ContentBlock::from(text),
                        )),
                        Role::Assistant => SessionUpdate::AgentMessageChunk(ContentChunk::new(
                            ContentBlock::from(text),
                        )),
                        _ => continue,
                    };
                    let _ = cx.send_notification(SessionNotification::new(
                        req.session_id.clone(),
                        update,
                    ));
                }

                responder.respond(LoadSessionResponse::new())
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
