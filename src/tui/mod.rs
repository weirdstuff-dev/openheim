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
        DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyEventKind,
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use crate::{
    client::{OpenheimClient, SessionHandle},
    core::{models::StopReason, permission::PermissionGate},
};

use app::App;
use permission::TuiPermissionGate;
use state::{AgentChannels, AgentCommand};
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
        let _ = execute!(io::stdout(), DisableBracketedPaste);
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
    let (commands_tx, commands_rx) = mpsc::unbounded_channel::<AgentCommand>();
    let (cancel_tx, cancel_rx) = mpsc::unbounded_channel::<()>();

    let agent_handle = tokio::spawn(
        AgentTask {
            client: client.clone(),
            permission_gate,
            skills: skills.clone(),
            default_provider: agent_config.provider_name.clone(),
            default_model: agent_config.model.clone(),
            updates: update_tx,
        }
        .run(session, commands_rx, cancel_rx),
    );

    let mut app = App::new(
        agent_config,
        app_config,
        paths,
        skills,
        AgentChannels {
            commands: commands_tx,
            cancel: cancel_tx,
        },
    );

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    // A paste then arrives as one `Event::Paste` instead of keystrokes, so
    // its newlines don't press Enter. Terminals without it type the paste.
    execute!(stdout, EnableBracketedPaste).ok();

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
                    Some(Ok(Event::Paste(text))) => app.handle_paste(&text),
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

/// The TUI's agent side: runs the UI's [`AgentCommand`]s against the
/// client and reports back as [`AgentUpdate`]s.
struct AgentTask {
    client: OpenheimClient,
    permission_gate: Arc<dyn PermissionGate>,
    /// For `:new`: the same skills as at startup.
    skills: Vec<String>,
    /// The model a new session falls back to when the active one no
    /// longer resolves.
    default_provider: String,
    default_model: String,
    updates: mpsc::UnboundedSender<AgentUpdate>,
}

impl AgentTask {
    /// Handles `commands` one at a time, in the order sent, until the UI
    /// drops its sender. `cancel` is only read while a turn runs.
    async fn run(
        self,
        mut session: SessionHandle,
        mut commands: mpsc::UnboundedReceiver<AgentCommand>,
        mut cancel: mpsc::UnboundedReceiver<()>,
    ) {
        // A new session starts on the model the current one is on.
        let mut current_provider = self.default_provider.clone();
        let mut current_model = self.default_model.clone();
        while let Some(command) = commands.recv().await {
            match command {
                AgentCommand::Prompt(prompt) => {
                    self.prompt(&session, prompt, &mut cancel).await;
                }
                AgentCommand::SwitchModel { provider, model } => {
                    match session.switch_model(&provider, &model).await {
                        Ok((provider, model)) => {
                            current_provider = provider.clone();
                            current_model = model.clone();
                            self.send(AgentUpdate::ModelChanged { provider, model });
                        }
                        Err(e) => self.send(AgentUpdate::Error(e.to_string())),
                    }
                }
                AgentCommand::SwitchSession { id, cwd } => {
                    if let Some(restored) = self.resume(&id, cwd).await {
                        session = restored;
                    }
                }
                AgentCommand::NewSession => {
                    let started = self
                        .new_session(&mut current_provider, &mut current_model)
                        .await;
                    if let Some(new_session) = started {
                        session = new_session;
                    }
                }
                AgentCommand::ListSessions => match self.client.list_sessions(None).await {
                    Ok(metas) => self.send(AgentUpdate::SessionList(metas)),
                    Err(e) => self.send(AgentUpdate::Error(e.to_string())),
                },
            }
        }
    }

    fn send(&self, update: AgentUpdate) {
        let _ = self.updates.send(update);
    }

    /// Runs one turn, cancelling it if `cancel` fires meanwhile.
    async fn prompt(
        &self,
        session: &SessionHandle,
        prompt: String,
        cancel: &mut mpsc::UnboundedReceiver<()>,
    ) {
        // A cancel sent while no turn was running (e.g. during `:new`, or
        // just as the last turn ended) isn't meant for this one.
        while cancel.try_recv().is_ok() {}
        let updates = self.updates.clone();
        let turn = session.prompt(prompt, move |event| {
            let _ = updates.send(AgentUpdate::Stream(event));
        });
        tokio::pin!(turn);
        // The turn is polled first, so it queues for the session lock (where
        // it resets its cancel token) ahead of a cancel arriving with it.
        let result = loop {
            tokio::select! {
                biased;
                result = &mut turn => break result,
                Some(()) = cancel.recv() => session.cancel().await,
            }
        };
        match result {
            Ok(StopReason::Cancelled) => {
                self.send(AgentUpdate::Notice("turn cancelled".to_string()));
            }
            Ok(stop_reason) => {
                if let Some(notice) = stop_reason.notice() {
                    self.send(AgentUpdate::Notice(notice.to_string()));
                }
            }
            Err(e) => self.send(AgentUpdate::Error(e.to_string())),
        }
    }

    /// Resumes saved session `id` and sends its history, or reports why it
    /// couldn't.
    async fn resume(&self, id: &str, cwd: std::path::PathBuf) -> Option<SessionHandle> {
        let (restored, loaded) = match self.client.resume_session(id, cwd).await {
            Ok(resumed) => resumed,
            Err(e) => {
                self.send(AgentUpdate::Error(e.to_string()));
                return None;
            }
        };
        let restored = restored.permission_gate(self.permission_gate.clone());
        // Sent as one batch, so the app repaints once.
        let mut history = Vec::new();
        if let Some(warning) = loaded.warning {
            history.push(ChatItem::AssistantMessage(warning));
        }
        for msg in &loaded.messages {
            history.extend(message_to_chat_items(msg));
        }
        history.push(ChatItem::SystemInfo("─── session restored".to_string()));
        self.send(AgentUpdate::History(history));
        // The restored session's context size; `None` clears the previous
        // one's.
        if let Ok(usage) = restored.context_usage().await {
            self.send(AgentUpdate::Usage(usage));
        }
        Some(restored)
    }

    /// Starts a new session on the active model (`provider`/`model`), or on
    /// the default one if that no longer resolves, updating both to match.
    async fn new_session(
        &self,
        provider: &mut String,
        model: &mut String,
    ) -> Option<SessionHandle> {
        let new_session = match self
            .client
            .new_session()
            .skills(self.skills.clone())
            .start()
            .await
        {
            Ok(new_session) => new_session.permission_gate(self.permission_gate.clone()),
            Err(e) => {
                self.send(AgentUpdate::Error(e.to_string()));
                return None;
            }
        };
        if new_session.switch_model(provider, model).await.is_err() {
            provider.clone_from(&self.default_provider);
            model.clone_from(&self.default_model);
            self.send(AgentUpdate::ModelChanged {
                provider: self.default_provider.clone(),
                model: self.default_model.clone(),
            });
        }
        self.send(AgentUpdate::NewSession(vec![ChatItem::SystemInfo(
            "─── new session".to_string(),
        )]));
        Some(new_session)
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
    use crate::config::{AppConfig, ProviderConfig};
    use crate::core::models::{ContentBlock, Message, Role};

    // Commands run in the order the UI sent them: a model switch sent right
    // after a session switch applies to the resumed session, and each
    // command's updates come before the next command's.
    #[tokio::test]
    async fn agent_task_runs_commands_in_the_order_sent() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig::new("local").with_provider(
            "local",
            ProviderConfig::new("http://127.0.0.1:1/v1", "m1").with_models(["m1", "m2"]),
        );
        let client = OpenheimClient::builder()
            .app_config(config)
            .data_dir(dir.path())
            .work_dir(dir.path())
            .build()
            .await
            .unwrap();
        let history = client.state().memory.history.clone();
        let mut saved = history
            .create_conversation(Some("m1".into()), Some("local".into()), vec![])
            .unwrap();
        saved.messages.push(Message::user("hi"));
        history.save_conversation(&saved).unwrap();
        let saved_id = saved.meta.id.to_string();

        let (updates_tx, mut updates) = mpsc::unbounded_channel();
        let (commands_tx, commands) = mpsc::unbounded_channel();
        let (_cancel_tx, cancel) = mpsc::unbounded_channel();
        for command in [
            AgentCommand::SwitchSession {
                id: saved_id.clone(),
                cwd: dir.path().to_path_buf(),
            },
            AgentCommand::ListSessions,
            AgentCommand::SwitchModel {
                provider: "local".into(),
                model: "m2".into(),
            },
        ] {
            commands_tx.send(command).unwrap();
        }
        drop(commands_tx);

        let task = AgentTask {
            client: client.clone(),
            permission_gate: Arc::new(crate::core::permission::AllowAll),
            skills: vec![],
            default_provider: "local".into(),
            default_model: "m1".into(),
            updates: updates_tx,
        };
        let session = client.new_session().start().await.unwrap();
        task.run(session, commands, cancel).await;

        let mut kinds = Vec::new();
        while let Ok(update) = updates.try_recv() {
            kinds.push(match update {
                AgentUpdate::History(_) => "history",
                AgentUpdate::Usage(_) => "usage",
                AgentUpdate::SessionList(_) => "sessions",
                AgentUpdate::ModelChanged { .. } => "model",
                other => panic!("unexpected update {other:?}"),
            });
        }
        assert_eq!(kinds, ["history", "usage", "sessions", "model"]);

        // The live session behind `saved_id` is the one on the new model.
        let (_, loaded) = client
            .resume_session(&saved_id, dir.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(loaded.model, "m2");
    }

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
