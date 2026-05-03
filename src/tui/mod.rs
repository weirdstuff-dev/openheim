mod app;
pub mod views;

pub use app::AgentUpdate;

use std::io;
use std::sync::Arc;

use agent_client_protocol::{
    ByteStreams, Client,
    schema::{
        ContentBlock, InitializeRequest, ProtocolVersion, SessionNotification, SessionUpdate,
        ToolCallStatus,
    },
    util::MatchDispatch,
    SessionMessage,
};
use crossterm::{
    event::{EventStream},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::{
    acp::{self, AgentState},
    config::load_config,
    rag::RagContext,
};

pub async fn run() -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;
    let rag = RagContext::new()?;
    let state = Arc::new(AgentState::new(agent_config.clone(), app_config.clone(), rag).await?);

    let (prompt_tx, prompt_rx) = mpsc::channel::<String>(1);
    let (update_tx, update_rx) = mpsc::channel::<AgentUpdate>(64);

    let (server_half, client_half) = tokio::io::duplex(65536);
    let (server_read, server_write) = tokio::io::split(server_half);
    let (client_read, client_write) = tokio::io::split(client_half);

    let server_transport = ByteStreams::new(server_write.compat_write(), server_read.compat());
    let client_transport = ByteStreams::new(client_write.compat_write(), client_read.compat());

    tokio::spawn(acp::serve(server_transport, state));
    tokio::spawn(run_acp_client(client_transport, prompt_rx, update_tx));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = app::App::new(prompt_tx, agent_config, app_config);
    app.load_sessions();

    let result = event_loop(&mut terminal, &mut app, update_rx).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_acp_client(
    transport: ByteStreams<
        tokio_util::compat::Compat<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
        tokio_util::compat::Compat<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    >,
    mut prompt_rx: mpsc::Receiver<String>,
    update_tx: mpsc::Sender<AgentUpdate>,
) {
    let result = Client
        .builder()
        .connect_with(transport, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            cx.build_session_cwd()?
                .block_task()
                .run_until(async move |mut session| {
                    while let Some(prompt) = prompt_rx.recv().await {
                        session.send_prompt(&prompt)?;
                        loop {
                            match session.read_update().await? {
                                SessionMessage::StopReason(_) => {
                                    let _ = update_tx.send(AgentUpdate::Done).await;
                                    break;
                                }
                                SessionMessage::SessionMessage(dispatch) => {
                                    let tx = update_tx.clone();
                                    MatchDispatch::new(dispatch)
                                        .if_notification(
                                            async move |notif: SessionNotification| {
                                                match notif.update {
                                                    SessionUpdate::AgentMessageChunk(chunk) => {
                                                        if let ContentBlock::Text(t) = chunk.content
                                                        {
                                                            let _ = tx
                                                                .send(AgentUpdate::TextChunk(
                                                                    t.text,
                                                                ))
                                                                .await;
                                                        }
                                                    }
                                                    SessionUpdate::ToolCall(tc) => {
                                                        let args = tc
                                                            .raw_input
                                                            .as_ref()
                                                            .map(|v| v.to_string())
                                                            .unwrap_or_default();
                                                        let _ = tx
                                                            .send(AgentUpdate::ToolCall {
                                                                name: tc.title.clone(),
                                                                args,
                                                            })
                                                            .await;
                                                    }
                                                    SessionUpdate::ToolCallUpdate(tcu) => {
                                                        if matches!(
                                                            tcu.fields.status,
                                                            Some(ToolCallStatus::Completed)
                                                                | Some(ToolCallStatus::Failed)
                                                        ) {
                                                            let result = match tcu.fields.raw_output
                                                            {
                                                                Some(serde_json::Value::String(
                                                                    s,
                                                                )) => s,
                                                                Some(v) => v.to_string(),
                                                                None => String::new(),
                                                            };
                                                            let _ = tx
                                                                .send(AgentUpdate::ToolResult(
                                                                    result,
                                                                ))
                                                                .await;
                                                        }
                                                    }
                                                    _ => {}
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
                    }
                    Ok(())
                })
                .await
        })
        .await;

    if let Err(e) = result {
        tracing::error!("TUI ACP client error: {e}");
    }
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut app::App,
    mut update_rx: mpsc::Receiver<AgentUpdate>,
) -> crate::error::Result<()> {
    let mut events = EventStream::new();

    loop {
        terminal.draw(|f| app.render(f))?;

        tokio::select! {
            maybe_event = events.next() => {
                match maybe_event {
                    Some(Ok(event)) => {
                        if app.handle_event(event) {
                            break;
                        }
                    }
                    Some(Err(e)) => return Err(crate::error::Error::IoError(e)),
                    None => break,
                }
            }
            Some(update) = update_rx.recv() => {
                app.handle_agent_update(update);
            }
        }
    }

    Ok(())
}
