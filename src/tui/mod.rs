mod app;
mod permission;
mod render;
mod types;

use std::io;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::{ContentBlock, SessionUpdate, ToolCallStatus};
use crossterm::{
    cursor::Show,
    event::{
        Event, EventStream, KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use crate::{client::OpenheimClient, core::permission::PermissionGate};

use app::App;
use permission::TuiPermissionGate;
use types::{AgentUpdate, ChatItem};

struct TerminalGuard {
    kbd_enhanced: bool,
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.kbd_enhanced {
            let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
        }
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), Show);
    }
}

pub async fn run(skills: Vec<String>) -> crate::error::Result<()> {
    let client = OpenheimClient::builder().build().await?;
    // Snapshots for `:config`/`:models` — read once here instead of a second
    // `load_config()` duplicating the one `OpenheimClient::builder().build()`
    // already did internally.
    let agent_config = client.state().config.clone();
    let app_config = client.state().app_config.clone();

    let (permission_tx, mut permission_rx) =
        mpsc::unbounded_channel::<permission::PermissionRequest>();
    let permission_gate: Arc<dyn PermissionGate> = Arc::new(TuiPermissionGate::new(permission_tx));

    let session = client
        .new_session()
        .skills(skills.clone())
        .start()
        .await?
        .permission_gate(permission_gate.clone());

    let (update_tx, mut update_rx) = mpsc::unbounded_channel::<AgentUpdate>();
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<String>();
    let (switch_model_tx, mut switch_model_rx) = mpsc::unbounded_channel::<(String, String)>();
    let (switch_session_tx, mut switch_session_rx) =
        mpsc::unbounded_channel::<(String, std::path::PathBuf)>();
    let (list_sessions_tx, mut list_sessions_rx) = mpsc::unbounded_channel::<()>();
    let (new_session_tx, mut new_session_rx) = mpsc::unbounded_channel::<()>();

    let agent_handle = {
        let update_tx = update_tx.clone();
        // Captured separately from the `skills` moved into `App::new` below —
        // this copy lives inside the agent task so a `:new` command can spin
        // up another session with the same skills, same as startup did.
        let session_skills = skills.clone();
        tokio::spawn(async move {
            let mut session = session;
            loop {
                tokio::select! {
                    maybe_prompt = prompt_rx.recv() => {
                        match maybe_prompt {
                            Some(prompt) => {
                                let tx_cb = update_tx.clone();
                                // `StreamEvent::Finished`/`Usage` arrive as part of
                                // this stream and drive the status/footer directly
                                // (see `App::handle_stream_event`) — no separate
                                // "done" signal or post-turn context-usage re-read
                                // needed here.
                                let result = session
                                    .prompt_events(&prompt, move |event| {
                                        let _ = tx_cb.send(AgentUpdate::Stream(event));
                                    })
                                    .await;
                                if let Err(e) = result {
                                    let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                                }
                            }
                            None => break,
                        }
                    }
                    maybe_model = switch_model_rx.recv() => {
                        match maybe_model {
                            Some((provider, model)) => {
                                match session.switch_model(&provider, &model).await {
                                    Ok((provider, model)) => {
                                        let _ = update_tx.send(AgentUpdate::ModelChanged { provider, model });
                                    }
                                    Err(e) => {
                                        let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                    maybe_switch = switch_session_rx.recv() => {
                        match maybe_switch {
                            Some((session_id, cwd)) => {
                                // Collected rather than streamed one at a time: a
                                // history replay isn't "live" the way a turn is,
                                // and batching means the app only clears/repaints
                                // once instead of on every historical message.
                                let mut history = Vec::new();
                                match session
                                    .restore(&session_id, cwd, |update| {
                                        history.extend(session_update_to_chat_item(update));
                                    })
                                    .await
                                {
                                    Ok(restored) => {
                                        history.push(ChatItem::SystemInfo(
                                            "─── session restored".to_string(),
                                        ));
                                        let _ = update_tx.send(AgentUpdate::History(history));
                                        // Refreshes the footer's context size to
                                        // the restored session's own snapshot
                                        // instead of leaving the previous
                                        // session's stale. `Ok(None)` is sent
                                        // through too, explicitly clearing the
                                        // footer rather than leaving it showing
                                        // the prior session's usage.
                                        if let Ok(usage) = restored.context_usage().await {
                                            let _ = update_tx.send(AgentUpdate::Usage(usage));
                                        }
                                        session = restored;
                                    }
                                    Err(e) => {
                                        let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                    maybe_new = new_session_rx.recv() => {
                        match maybe_new {
                            Some(()) => {
                                match client
                                    .new_session()
                                    .skills(session_skills.clone())
                                    .start()
                                    .await
                                {
                                    Ok(new_session) => {
                                        session = new_session.permission_gate(permission_gate.clone());
                                        let _ = update_tx.send(AgentUpdate::History(vec![
                                            ChatItem::SystemInfo("─── session started".to_string()),
                                        ]));
                                        let _ = update_tx.send(AgentUpdate::Usage(None));
                                    }
                                    Err(e) => {
                                        let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                    maybe_list = list_sessions_rx.recv() => {
                        match maybe_list {
                            Some(()) => {
                                match client.list_all_sessions().await {
                                    Ok(metas) => {
                                        let _ = update_tx.send(AgentUpdate::SessionList(metas));
                                    }
                                    Err(e) => {
                                        let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
        })
    };

    let mut app = App::new(
        agent_config,
        app_config,
        skills,
        prompt_tx,
        switch_model_tx,
        switch_session_tx,
        list_sessions_tx,
        new_session_tx,
    );

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    // Enable keyboard enhancement on supporting terminals so that arrow-key
    // escape sequences (\x1b[B etc.) are never ambiguously split into a
    // spurious Esc + characters, which caused `[B` to appear in the input.
    let kbd_enhanced = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    if kbd_enhanced {
        execute!(
            stdout,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES,
            )
        )
        .ok();
    }

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let _guard = TerminalGuard { kbd_enhanced };

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        original_hook(info);
    }));

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal.draw(|f| app.draw(f))?;

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
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        app.handle_key(key);
                    }
                    Some(Ok(Event::Resize(_, _))) => app.cached_width = 0,
                    Some(Err(_)) | None => break,
                    _ => {}
                }
            }
            Some(update) = update_rx.recv() => {
                app.handle_update(update);
            }
            Some(request) = permission_rx.recv() => {
                app.handle_permission_request(request);
            }
        }
    }

    // Drop app first so all channel senders close, signaling the agent task to exit.
    drop(app);
    agent_handle.abort();
    let _ = agent_handle.await;

    Ok(())
}

/// Converts one `SessionUpdate` from a history replay (`SessionHandle::restore`'s
/// `on_history` callback) into the `ChatItem` a live turn would have produced
/// for the equivalent content — the same mapping `App::handle_stream_event`
/// applies to a live turn's `StreamEvent`s, just starting from the ACP shape
/// `replay_history_messages` replays persisted messages as instead (there's
/// no live `StreamEvent` for "here's a message from a past turn"). Unlike the
/// raw `Message`-block walk this replaces, an image attachment isn't silently
/// dropped — it renders as a placeholder line, since the terminal can't
/// inline it.
fn session_update_to_chat_item(update: SessionUpdate) -> Option<ChatItem> {
    match update {
        SessionUpdate::UserMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(t) => Some(ChatItem::UserMessage(t.text)),
            ContentBlock::Image(_) => Some(ChatItem::SystemInfo("[image attached]".to_string())),
            _ => None,
        },
        SessionUpdate::AgentMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(t) => {
                let is_thinking = t
                    .meta
                    .as_ref()
                    .and_then(|m| m.get("kind"))
                    .and_then(|v| v.as_str())
                    == Some("thinking");
                if is_thinking {
                    Some(ChatItem::Thinking(t.text))
                } else {
                    Some(ChatItem::AssistantMessage(t.text))
                }
            }
            _ => None,
        },
        SessionUpdate::ToolCall(tc) => {
            let args = tc
                .raw_input
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default();
            Some(ChatItem::ToolCall {
                name: tc.title.clone(),
                args,
            })
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
                Some(ChatItem::ToolResult { result, is_error })
            } else {
                None
            }
        }
        _ => None,
    }
}
