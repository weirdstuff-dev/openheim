pub mod session;

use std::{collections::HashMap, sync::Arc};

use agent_client_protocol::{
    Agent, Client, ConnectionTo, Dispatch, ConnectTo,
    on_receive_dispatch, on_receive_request,
    schema::{
        AgentCapabilities, ContentBlock, ContentChunk, Implementation, InitializeRequest,
        InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
        SessionNotification, SessionUpdate, StopReason, ToolCall as AcpToolCall, ToolCallStatus,
        ToolCallUpdate, ToolCallUpdateFields,
    },
    util::internal_error,
};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::{AgentConfig, AppConfig, build_http_client, create_client},
    core::{agent::run_agent_streaming_with_history, models::StreamEvent},
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
        let executor =
            Arc::new(SystemToolExecutor::build(&app_config.mcp_servers).await) as Arc<dyn ToolExecutor>;
        Ok(Self {
            llm,
            executor,
            config,
            app_config,
            rag,
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
    let state_session = state.clone();
    let state_prompt = state.clone();

    Agent
        .builder()
        .name("openheim")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx: ConnectionTo<Client>| {
                responder.respond(
                    InitializeResponse::new(req.protocol_version)
                        .agent_capabilities(AgentCapabilities::new())
                        .agent_info(Implementation::new("openheim", env!("CARGO_PKG_VERSION"))),
                )
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: NewSessionRequest, responder, _cx: ConnectionTo<Client>| {
                let chat_id = Uuid::new_v4();
                let session_key = chat_id.to_string();

                state_session.sessions.write().await.insert(
                    session_key.clone(),
                    SessionState {
                        chat_id,
                        config: state_session.config.clone(),
                        cwd: req.cwd,
                        skills: vec![],
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

                let (llm, executor, config, rag, chat_id, skills) = {
                    let sessions = state_prompt.sessions.read().await;
                    match sessions.get(&session_key) {
                        Some(s) => (
                            state_prompt.llm.clone(),
                            state_prompt.executor.clone(),
                            s.config.clone(),
                            state_prompt.rag.clone(),
                            s.chat_id,
                            s.skills.clone(),
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

                if let Err(e) = run_result {
                    tracing::error!("agent loop error: {e}");
                }

                responder.respond(PromptResponse::new(StopReason::EndTurn))
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
