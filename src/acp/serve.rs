//! The ACP connection loop: wires each protocol method to an [`AgentState`]
//! call and dispatches client responses back to whichever request sent them.

use std::sync::Arc;

use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Dispatch, Handled, on_receive_dispatch,
    on_receive_notification, on_receive_request,
    schema::{
        AgentCapabilities, CancelNotification, ClientCapabilities, Implementation,
        InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
        LoadSessionRequest, LoadSessionResponse, NewSessionRequest, NewSessionResponse,
        PromptCapabilities, PromptRequest, PromptResponse, SessionCapabilities,
        SessionListCapabilities, SessionNotification, SetSessionModeRequest,
        SetSessionModeResponse, SetSessionModelRequest, SetSessionModelResponse,
    },
    util::internal_error,
};
use tokio::sync::RwLock;

use crate::{core::client_io::ClientIo, core::permission::PermissionGate, error::Error};

use super::{
    AgentMode, AgentState,
    client_io::AcpClientIo,
    permission::AcpPermissionGate,
    util::{map_stop_reason, session_mode_state},
};

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
                if let Ok(skills) = state_init.memory.skills.list_skills()
                    && let Ok(val) = serde_json::to_value(skills)
                {
                    meta.insert("skills".to_string(), val);
                }
                if let Ok(val) = serde_json::to_value(state_init.executor.list_tools()) {
                    meta.insert("tools".to_string(), val);
                }
                // Resolved sandbox root, shared by every session this connection
                // opens — not per-session, so `initialize` (not `session/new`)
                // is the natural home for it.
                meta.insert(
                    "work_dir".to_string(),
                    serde_json::Value::String(state_init.work_dir.display().to_string()),
                );
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
                            // SessionLocked carries structured fields a caller
                            // needs to build a "busy, retry" UX instead of a
                            // generic failure — encode them in the JSON-RPC
                            // error's `data` so they survive the trip back to
                            // the client instead of collapsing to `e.to_string()`.
                            let acp_error = match &e {
                                Error::SessionLocked {
                                    session_id,
                                    pid,
                                    host,
                                } => agent_client_protocol::Error::internal_error().data(
                                    serde_json::json!({
                                        "kind": "session_locked",
                                        "session_id": session_id,
                                        "pid": pid,
                                        "host": host,
                                    }),
                                ),
                                _ => internal_error(e.to_string()),
                            };
                            responder.respond_with_error(acp_error)
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
                    Ok(mode) => responder
                        .respond(LoadSessionResponse::new().modes(session_mode_state(mode))),
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
