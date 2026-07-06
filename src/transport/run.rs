//! Headless transport: runs a single agent prompt in-process and streams output to stdout.
//!
//! Used by `openheim run "<prompt>"`. Spins up a fully-featured ACP server on an
//! in-process duplex pipe, connects an ACP client to it, sends the prompt, and
//! prints streamed text chunks to stdout as they arrive.

use std::io::Write as _;
use std::sync::Arc;

use agent_client_protocol::{
    Agent, ByteStreams, Client, ConnectionTo, SessionMessage, on_receive_request,
    schema::{
        ContentBlock, InitializeRequest, PermissionOptionKind, ProtocolVersion,
        RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
        SelectedPermissionOutcome, SessionNotification, SessionUpdate,
    },
    util::MatchDispatch,
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::{
    acp::{self, AgentState},
    config::load_config,
    rag::RagContext,
};

/// Runs the agent against `prompt` using an in-process ACP session and prints
/// the streamed response to stdout.
///
/// `model` overrides the default model from the configuration file. If `None`,
/// the provider's configured default is used.
pub async fn run_headless(prompt: String, model: Option<String>) -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(model.as_deref())?;
    let rag = RagContext::new(app_config.default_skills.clone())?;
    let state = Arc::new(AgentState::new(agent_config, app_config, rag, vec![]).await?);

    let (server_half, client_half) = tokio::io::duplex(65536);
    let (server_read, server_write) = tokio::io::split(server_half);
    let (client_read, client_write) = tokio::io::split(client_half);

    let server_transport = ByteStreams::new(server_write.compat_write(), server_read.compat());
    let client_transport = ByteStreams::new(client_write.compat_write(), client_read.compat());

    let server_handle = tokio::spawn(acp::serve(server_transport, state));

    Client
        .builder()
        // `openheim run` is a one-shot, non-interactive CLI invocation with no
        // human to prompt — the user already consented to this run by invoking
        // it. Without this handler, the agent's session/request_permission
        // requests go unclaimed and every tool call is treated as denied.
        // Auto-allow so a headless run behaves like it did before
        // session/request_permission existed.
        .on_receive_request(
            async |req: RequestPermissionRequest, responder, _cx: ConnectionTo<Agent>| {
                let option_id = req
                    .options
                    .iter()
                    .find(|o| o.kind == PermissionOptionKind::AllowOnce)
                    .map(|o| o.option_id.clone());
                let outcome = match option_id {
                    Some(id) => {
                        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id))
                    }
                    None => RequestPermissionOutcome::Cancelled,
                };
                responder.respond(RequestPermissionResponse::new(outcome))
            },
            on_receive_request!(),
        )
        .connect_with(client_transport, async |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            cx.build_session_cwd()?
                .block_task()
                .run_until(async |mut session| {
                    session.send_prompt(&prompt)?;
                    loop {
                        match session.read_update().await? {
                            SessionMessage::StopReason(_) => break,
                            SessionMessage::SessionMessage(dispatch) => {
                                MatchDispatch::new(dispatch)
                                    .if_notification(async |notif: SessionNotification| {
                                        if let SessionUpdate::AgentMessageChunk(chunk) =
                                            notif.update
                                            && let ContentBlock::Text(t) = chunk.content
                                        {
                                            print!("{}", t.text);
                                            let _ = std::io::stdout().flush();
                                        }
                                        Ok(())
                                    })
                                    .await
                                    .otherwise_ignore()?;
                            }
                            _ => {}
                        }
                    }
                    println!();
                    Ok(())
                })
                .await
        })
        .await
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;

    server_handle.abort();
    Ok(())
}
