mod app;
mod render;
mod types;

use std::io;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::{ContentBlock, SessionUpdate, ToolCallStatus};
use crossterm::{
    event::{Event, EventStream},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use crate::{
    client::OpenheimClient,
    config::load_config,
};

use app::App;
use types::AgentUpdate;

pub async fn run(skills: Vec<String>) -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;

    let client = OpenheimClient::builder()
        .build()
        .await
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;

    let session = Arc::new(
        client
            .new_session()
            .skills(skills.clone())
            .start()
            .await
            .map_err(|e| crate::error::Error::Other(e.to_string()))?,
    );

    let (update_tx, mut update_rx) = mpsc::unbounded_channel::<AgentUpdate>();
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<String>();

    {
        let session = Arc::clone(&session);
        let update_tx = update_tx.clone();
        tokio::spawn(async move {
            while let Some(prompt) = prompt_rx.recv().await {
                let tx_cb = update_tx.clone();
                let result = session
                    .prompt(&prompt, move |update| convert_update(&tx_cb, update))
                    .await;
                match result {
                    Ok(()) => {
                        let _ = update_tx.send(AgentUpdate::Done);
                    }
                    Err(e) => {
                        let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                    }
                }
            }
        });
    }

    let mut app = App::new(agent_config, app_config, skills, prompt_tx);

    enable_raw_mode().map_err(|e| crate::error::Error::Other(e.to_string()))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal =
        Terminal::new(backend).map_err(|e| crate::error::Error::Other(e.to_string()))?;

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal
            .draw(|f| app.draw(f))
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;

        if app.should_quit {
            break;
        }

        tokio::select! {
            _ = tick.tick() => {
                if app.status != types::Status::Idle {
                    app.spinner_frame = app.spinner_frame.wrapping_add(1);
                }
            }
            maybe = events.next() => {
                match maybe {
                    Some(Ok(Event::Key(key))) => app.handle_key(key),
                    Some(Ok(Event::Resize(_, _))) => app.cached_width = 0,
                    Some(Err(_)) | None => break,
                    _ => {}
                }
            }
            Some(update) = update_rx.recv() => {
                app.handle_update(update);
            }
        }
    }

    disable_raw_mode().map_err(|e| crate::error::Error::Other(e.to_string()))?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;
    terminal
        .show_cursor()
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;

    Ok(())
}

fn convert_update(tx: &mpsc::UnboundedSender<AgentUpdate>, update: SessionUpdate) {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(t) = chunk.content {
                let _ = tx.send(AgentUpdate::TextChunk(t.text));
            }
        }
        SessionUpdate::ToolCall(tc) => {
            let args = tc.raw_input.as_ref().map(|v| v.to_string()).unwrap_or_default();
            let _ = tx.send(AgentUpdate::ToolCall { name: tc.title.clone(), args });
        }
        SessionUpdate::ToolCallUpdate(tcu) => {
            if matches!(
                tcu.fields.status,
                Some(ToolCallStatus::Completed) | Some(ToolCallStatus::Failed)
            ) {
                let is_error = matches!(tcu.fields.status, Some(ToolCallStatus::Failed));
                let result = match tcu.fields.raw_output {
                    Some(serde_json::Value::String(s)) => s,
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                let _ = tx.send(AgentUpdate::ToolResult { result, is_error });
            }
        }
        _ => {}
    }
}
