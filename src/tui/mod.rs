mod app;
mod permission;
mod render;
mod types;

use std::io;
use std::sync::Arc;
use std::time::Duration;

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

/// Runs the TUI against `client` — caller-built, so an embedder with custom
/// tools or a custom `LlmClient` can use this transport too.
pub async fn run(client: OpenheimClient, skills: Vec<String>) -> crate::error::Result<()> {
    // Snapshots for `:config`/`:models` — read once here instead of a second
    // `load_config()` duplicating the one `OpenheimClient::builder().build()`
    // the caller did.
    let agent_config = client.state().config.clone();
    let app_config = client.state().app_config.clone();

    let (permission_tx, mut permission_rx) =
        mpsc::unbounded_channel::<permission::PermissionRequest>();
    let permission_gate: Arc<dyn PermissionGate> = Arc::new(TuiPermissionGate::new(
        permission_tx,
        client.state().executor.clone(),
    ));

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
        // up another session with the same skills, matching startup.
        let session_skills = skills.clone();
        // `client.new_session()` always starts from the client's original
        // default config (see `AgentState::new_session`'s fallback), not
        // whatever `:model`/`:models` had switched this session to — tracked
        // here so a `:new` replacement can re-apply it instead of silently
        // regressing to the default.
        let default_provider = agent_config.provider_name.clone();
        let default_model = agent_config.model.clone();
        tokio::spawn(async move {
            let mut session = session;
            let mut current_provider = default_provider.clone();
            let mut current_model = default_model.clone();
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
                                        current_provider = provider.clone();
                                        current_model = model.clone();
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
                                match session.resume(&session_id, cwd).await {
                                    Ok((restored, loaded)) => {
                                        let mut history = Vec::new();
                                        if let Some(warning) = loaded.warning {
                                            history.push(ChatItem::AssistantMessage(warning));
                                        }
                                        for msg in &loaded.messages {
                                            history.extend(message_to_chat_items(msg));
                                        }
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
                                        let new_session =
                                            new_session.permission_gate(permission_gate.clone());
                                        // Re-apply the active model on top of the
                                        // fresh session's default. If it's no
                                        // longer valid (e.g. removed from the
                                        // config file since startup), fall back to
                                        // the default and let the UI know so the
                                        // footer doesn't keep showing the stale one.
                                        match new_session.switch_model(&current_provider, &current_model).await {
                                            Ok(_) => {}
                                            Err(_) => {
                                                current_provider = default_provider.clone();
                                                current_model = default_model.clone();
                                                let _ = update_tx.send(AgentUpdate::ModelChanged {
                                                    provider: default_provider.clone(),
                                                    model: default_model.clone(),
                                                });
                                            }
                                        }
                                        session = new_session;
                                        let _ = update_tx.send(AgentUpdate::NewSession(vec![
                                            ChatItem::SystemInfo("─── new session".to_string()),
                                        ]));
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
                                match client.list_sessions(None).await {
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

/// Converts one persisted [`Message`](crate::core::models::Message) from a
/// history replay (`SessionHandle::resume`'s `LoadedSession::messages`) into
/// the `ChatItem`s a live turn would have produced for the equivalent
/// content — the same mapping `App::handle_stream_event` applies to a live
/// turn's `StreamEvent`s, just walking the message's content blocks directly
/// instead of decoding ACP's `SessionUpdate` vocabulary (there's no live
/// `StreamEvent` for "here's a message from a past turn"), so this works
/// without the `acp` feature. An image attachment isn't silently dropped —
/// it renders as a placeholder line, since the terminal can't inline it.
fn message_to_chat_items(msg: &crate::core::models::Message) -> Vec<ChatItem> {
    use crate::core::models::{ContentBlock, Role};

    let mut items = Vec::new();
    match msg.role {
        Role::User => {
            // Every block, not just `msg.text()` (which only concatenates
            // `Text` blocks), so an image attached to the prompt is
            // restored alongside the text instead of silently dropped.
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => items.push(ChatItem::UserMessage(text.clone())),
                    ContentBlock::Image { .. } => {
                        items.push(ChatItem::SystemInfo("[image attached]".to_string()));
                    }
                    _ => {}
                }
            }
        }
        Role::Assistant => {
            for block in &msg.content {
                if let ContentBlock::Thinking { thinking, .. } = block {
                    items.push(ChatItem::Thinking(thinking.clone()));
                }
            }
            if let Some(text) = msg.text() {
                items.push(ChatItem::AssistantMessage(text));
            }
            for tc in msg.tool_calls() {
                items.push(ChatItem::ToolCall {
                    name: tc.name,
                    args: tc.arguments,
                });
            }
        }
        Role::Tool => {
            if let Some(tr) = msg.tool_result_block() {
                items.push(ChatItem::ToolResult {
                    result: tr.content,
                    is_error: tr.is_error,
                });
            }
        }
        Role::System => {}
    }
    items
}
