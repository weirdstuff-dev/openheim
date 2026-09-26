mod app;
mod permission;
mod render;
mod state;
mod types;

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

use crate::{
    client::OpenheimClient,
    core::{models::StopReason, permission::PermissionGate},
};

use app::App;
use permission::TuiPermissionGate;
use state::AgentChannels;
use types::{AgentUpdate, ChatItem};

type PanicHook = dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync;

/// Puts the terminal back the way `run` found it. Called from both
/// `TerminalGuard::drop` and the panic hook; only the first call does
/// anything, so the keyboard flags pushed at startup are popped exactly once.
struct TerminalRestore {
    kbd_enhanced: bool,
    done: AtomicBool,
}

impl TerminalRestore {
    fn restore(&self) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        if self.kbd_enhanced {
            let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
        }
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), Show);
    }

    fn is_restored(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }
}

/// Restores the terminal when `run` returns, and on a panic in any thread
/// before that, so the panic message is printed to the normal screen with
/// raw mode off instead of vanishing with the alternate screen. The panic
/// hook in place before `run` is put back when the guard drops.
struct TerminalGuard {
    terminal: Arc<TerminalRestore>,
    previous_hook: Arc<PanicHook>,
}

impl TerminalGuard {
    fn install(kbd_enhanced: bool) -> Self {
        let terminal = Arc::new(TerminalRestore {
            kbd_enhanced,
            done: AtomicBool::new(false),
        });
        let previous_hook: Arc<PanicHook> = Arc::from(std::panic::take_hook());
        {
            let terminal = terminal.clone();
            let previous_hook = previous_hook.clone();
            std::panic::set_hook(Box::new(move |info| {
                terminal.restore();
                previous_hook(info);
            }));
        }
        Self {
            terminal,
            previous_hook,
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.terminal.restore();
        // `set_hook` panics on a panicking thread; the process is going down
        // with our hook in place anyway.
        if !std::thread::panicking() {
            let previous_hook = self.previous_hook.clone();
            std::panic::set_hook(Box::new(move |info| previous_hook(info)));
        }
    }
}

/// Runs the TUI against `client`.
pub async fn run(client: OpenheimClient, skills: Vec<String>) -> crate::error::Result<()> {
    // Snapshots for `:config`/`:models`.
    let agent_config = client.state().config().clone();
    let app_config = client.state().app_config.clone();
    let paths = client.state().paths().clone();

    let (permission_tx, mut permission_rx) =
        mpsc::unbounded_channel::<permission::PendingPermission>();
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
    let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel::<()>();

    let agent_handle = {
        let update_tx = update_tx.clone();
        // For `:new`: the same skills as at startup, and the model the
        // current session is on (a new session starts on the default).
        let session_skills = skills.clone();
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
                                // A cancel sent while no turn was running
                                // (e.g. during `:new`, or just as the last
                                // turn ended) isn't meant for this one.
                                while cancel_rx.try_recv().is_ok() {}
                                let tx_cb = update_tx.clone();
                                let turn = session.prompt(prompt, move |event| {
                                    let _ = tx_cb.send(AgentUpdate::Stream(event));
                                });
                                tokio::pin!(turn);
                                // The turn is polled first, so it queues for the
                                // session lock (where it resets its cancel
                                // token) ahead of a cancel arriving with it.
                                let result = loop {
                                    tokio::select! {
                                        biased;
                                        result = &mut turn => break result,
                                        Some(()) = cancel_rx.recv() => session.cancel().await,
                                    }
                                };
                                match result {
                                    Ok(StopReason::Cancelled) => {
                                        let _ = update_tx
                                            .send(AgentUpdate::Notice("turn cancelled".to_string()));
                                    }
                                    Ok(stop_reason) => {
                                        if let Some(notice) = stop_reason.notice() {
                                            let _ = update_tx
                                                .send(AgentUpdate::Notice(notice.to_string()));
                                        }
                                    }
                                    Err(e) => {
                                        let _ = update_tx.send(AgentUpdate::Error(e.to_string()));
                                    }
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
                                // Sent as one batch, so the app repaints once.
                                match client.resume_session(&session_id, cwd).await {
                                    Ok((restored, loaded)) => {
                                        let restored =
                                            restored.permission_gate(permission_gate.clone());
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
                                        // The restored session's context size;
                                        // `None` clears the previous one's.
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
                                        // Keep the active model; if it no longer
                                        // resolves, tell the UI it's the default.
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
        paths,
        skills,
        AgentChannels {
            prompt: prompt_tx,
            switch_model: switch_model_tx,
            switch_session: switch_session_tx,
            list_sessions: list_sessions_tx,
            new_session: new_session_tx,
            cancel: cancel_tx,
        },
    );

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    // Enable keyboard enhancement on supporting terminals so that arrow-key
    // escape sequences (\x1b[B etc.) are never split into a spurious Esc
    // plus characters that land in the input.
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

    let guard = TerminalGuard::install(kbd_enhanced);
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        // A panic (e.g. in the agent task) has already restored the
        // terminal; drawing again would paint over its message.
        if guard.terminal.is_restored() {
            break;
        }
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
                    Some(Ok(Event::Resize(_, _))) => app.transcript.invalidate(),
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
    match agent_handle.await {
        Err(e) if e.is_panic() => Err(crate::error::Error::Other(
            "the TUI's agent task panicked".to_string(),
        )),
        _ => Ok(()),
    }
}

/// The `ChatItem`s a live turn would have shown for one saved message.
/// [`Message::transcript`] decides what is shown; an image becomes a
/// placeholder line.
///
/// [`Message::transcript`]: crate::core::models::Message::transcript
fn message_to_chat_items(msg: &crate::core::models::Message) -> Vec<ChatItem> {
    use crate::core::models::TranscriptEntry;

    msg.transcript()
        .map(|entry| match entry {
            TranscriptEntry::UserText(text) => ChatItem::UserMessage(text.to_string()),
            TranscriptEntry::UserImage { .. } => {
                ChatItem::SystemInfo("[image attached]".to_string())
            }
            TranscriptEntry::Thinking(thinking) => ChatItem::Thinking(thinking.to_string()),
            TranscriptEntry::AssistantText(text) => ChatItem::AssistantMessage(text.to_string()),
            TranscriptEntry::ToolCall {
                name, arguments, ..
            } => ChatItem::ToolCall {
                name: name.to_string(),
                args: arguments.to_string(),
            },
            TranscriptEntry::ToolResult {
                content, is_error, ..
            } => ChatItem::ToolResult {
                result: content.to_string(),
                is_error,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::models::{ContentBlock, Message, Role};

    // Interleaved thinking stores several thinking blocks between text and
    // tool calls; a restored session shows them in that order.
    #[test]
    fn restored_assistant_turn_keeps_its_block_order() {
        let thinking = |t: &str| ContentBlock::Thinking {
            thinking: t.into(),
            signature: Some("sig".into()),
        };
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                thinking("first"),
                ContentBlock::from("Let me look."),
                ContentBlock::RedactedThinking { data: "enc".into() },
                thinking("second"),
                ContentBlock::tool_use("toolu_1", "read_file", "{}"),
            ],
        };
        assert_eq!(
            message_to_chat_items(&msg),
            [
                ChatItem::Thinking("first".into()),
                ChatItem::AssistantMessage("Let me look.".into()),
                ChatItem::Thinking("second".into()),
                ChatItem::ToolCall {
                    name: "read_file".into(),
                    args: "{}".into(),
                },
            ]
        );
    }
}
