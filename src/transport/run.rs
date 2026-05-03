use std::io::Write as _;
use std::sync::Arc;

use agent_client_protocol::{
    ByteStreams, Client,
    schema::{
        ContentBlock, InitializeRequest, ProtocolVersion, SessionNotification, SessionUpdate,
    },
    util::MatchDispatch,
    SessionMessage,
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::{
    acp::{self, AgentState},
    config::load_config,
    rag::RagContext,
};

pub async fn run_headless(prompt: String, model: Option<String>) -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(model.as_deref())?;
    let rag = RagContext::new()?;
    let state = Arc::new(AgentState::new(agent_config, app_config, rag).await?);

    let (server_half, client_half) = tokio::io::duplex(65536);
    let (server_read, server_write) = tokio::io::split(server_half);
    let (client_read, client_write) = tokio::io::split(client_half);

    let server_transport = ByteStreams::new(server_write.compat_write(), server_read.compat());
    let client_transport = ByteStreams::new(client_write.compat_write(), client_read.compat());

    tokio::spawn(acp::serve(server_transport, state));

    Client
        .builder()
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
                                    .if_notification(
                                        async |notif: SessionNotification| {
                                            if let SessionUpdate::AgentMessageChunk(chunk) =
                                                notif.update
                                            {
                                                if let ContentBlock::Text(t) = chunk.content {
                                                    print!("{}", t.text);
                                                    let _ = std::io::stdout().flush();
                                                }
                                            }
                                            Ok(())
                                        },
                                    )
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
        .map_err(|e| crate::error::Error::Other(e.to_string()))
}
