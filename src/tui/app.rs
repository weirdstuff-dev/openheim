use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
};

use crate::{
    config::{AgentConfig, AppConfig, RuntimePaths},
    core::{models::StreamEvent, permission::PermissionDecision},
    memory::{ConversationMeta, SkillsManager},
};

use super::permission::PendingPermission;
use super::render::{self, FooterLabels};
use super::state::{AgentChannels, InputLine, Overlay, PermissionQueue, Theme, Transcript};
use super::types::{AgentUpdate, ChatItem, ConfigRow, Screen, Status};

/// How long after a Ctrl-C a second one quits.
const QUIT_CONFIRM_WINDOW: Duration = Duration::from_secs(2);

/// Formats a token count for the footer: exact below 1000, `k`-suffixed with
/// one decimal place above it (`1.2k`, `84.0k`) so the label stays short
/// enough to sit next to the provider/model name.
fn format_token_count(tokens: u64) -> String {
    if tokens < 1000 {
        format!("{tokens} ctx")
    } else {
        format!("{:.1}k ctx", tokens as f64 / 1000.0)
    }
}

/// Moves a list selection up or down within `len` entries.
fn move_selection(selected: &mut usize, len: usize, code: KeyCode) {
    match code {
        KeyCode::Up => *selected = selected.saturating_sub(1),
        KeyCode::Down => *selected = (*selected + 1).min(len.saturating_sub(1)),
        _ => {}
    }
}

/// Scrolls a read-only viewer.
fn move_scroll(scroll: &mut usize, code: KeyCode) {
    match code {
        KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => *scroll = scroll.saturating_add(1),
        KeyCode::PageUp => *scroll = scroll.saturating_sub(5),
        KeyCode::PageDown => *scroll = scroll.saturating_add(5),
        _ => {}
    }
}

/// The terminal UI's state: a base screen (welcome or chat), at most one
/// popup over it, and the permission-prompt queue over everything.
pub(super) struct App {
    pub(super) transcript: Transcript,
    input: InputLine,
    pub(super) spinner_frame: usize,
    pub(super) status: Status,
    pub(super) should_quit: bool,
    /// Set by a Ctrl-C: another one before this instant quits.
    quit_armed_until: Option<Instant>,
    /// The screen under any popup.
    screen: Screen,
    overlay: Option<Overlay>,
    permissions: PermissionQueue,
    agent_config: AgentConfig,
    app_config: AppConfig,
    /// Where `:skills` looks and `:theme` writes.
    paths: RuntimePaths,
    skills: Vec<String>,
    theme: Theme,
    channels: AgentChannels,
    /// Current context size: the most recent LLM call's usage, i.e. how
    /// full the context window is right now — not a cumulative session
    /// total. Refreshed after each completed turn and on session switch.
    /// `None` until a turn has completed (never sent for a brand-new,
    /// never-prompted session).
    context_usage: Option<crate::core::models::Usage>,
}

impl App {
    pub(super) fn new(
        agent_config: AgentConfig,
        app_config: AppConfig,
        paths: RuntimePaths,
        skills: Vec<String>,
        channels: AgentChannels,
    ) -> Self {
        let theme = Theme::named(app_config.tui.theme_color.as_deref().unwrap_or("gray"));
        Self {
            transcript: Transcript::new(),
            input: InputLine::default(),
            spinner_frame: 0,
            status: Status::Idle,
            should_quit: false,
            quit_armed_until: None,
            screen: Screen::Welcome,
            overlay: None,
            permissions: PermissionQueue::default(),
            agent_config,
            app_config,
            paths,
            skills,
            theme,
            channels,
            context_usage: None,
        }
    }

    fn push(&mut self, item: ChatItem) {
        self.transcript.push(item);
    }

    pub(super) fn handle_update(&mut self, update: AgentUpdate) {
        match update {
            AgentUpdate::Stream(event) => self.handle_stream_event(event),
            AgentUpdate::Error(e) => {
                self.status = Status::Idle;
                self.push(ChatItem::Err(e));
            }
            AgentUpdate::Notice(notice) => {
                self.push(ChatItem::SystemInfo(notice));
            }
            AgentUpdate::Usage(usage) => {
                self.context_usage = usage;
            }
            AgentUpdate::ModelChanged { provider, model } => {
                self.agent_config.provider_name = provider.clone();
                self.agent_config.model = model.clone();
                self.push(ChatItem::SystemInfo(format!(
                    "switched to {provider} / {model}"
                )));
            }
            AgentUpdate::SessionList(sessions) => {
                if sessions.is_empty() {
                    self.push(ChatItem::SystemInfo("no sessions yet".to_string()));
                } else {
                    self.overlay = Some(Overlay::SessionPicker {
                        sessions,
                        selected: 0,
                    });
                }
            }
            // Appended, not replacing the transcript — `open_session` already
            // cleared it and pushed the header/warning synchronously, so this
            // is just the replayed message history (plus a trailing
            // "restored" marker) arriving once the load completes.
            AgentUpdate::History(items) => {
                for item in items {
                    self.push(item);
                }
            }
            // Only now — creation confirmed — is it safe to drop the old
            // session's transcript; see `start_new_session`.
            AgentUpdate::NewSession(items) => {
                self.transcript.clear();
                self.context_usage = None;
                self.status = Status::Idle;
                for item in items {
                    self.push(item);
                }
            }
        }
    }

    /// Handles one raw event from a live turn. `IterationStart` and
    /// `MessageAppended` have no UI presence and are ignored.
    fn handle_stream_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::LlmResponse { content } => {
                self.status = Status::Streaming;
                self.transcript.append_assistant_text(content);
            }
            StreamEvent::ThinkingContent { content } => {
                self.status = Status::Streaming;
                self.transcript.append_thinking(content);
            }
            StreamEvent::ToolCall {
                tool_name,
                arguments,
                ..
            } => {
                self.status = Status::Thinking;
                self.push(ChatItem::ToolCall {
                    name: tool_name,
                    args: arguments,
                });
            }
            StreamEvent::ToolResult {
                result, is_error, ..
            } => {
                self.push(ChatItem::ToolResult { result, is_error });
            }
            // The current context size, refreshed live as each LLM call in
            // the turn completes rather than once at the end, so the footer
            // never needs a separate disk read after the turn finishes.
            StreamEvent::Usage { usage } => {
                self.context_usage = Some(usage);
            }
            StreamEvent::Finished { .. } => {
                self.status = Status::Idle;
            }
            StreamEvent::IterationStart { .. } | StreamEvent::MessageAppended { .. } => {}
        }
    }

    /// The first Ctrl-C cancels the running turn, if any; a second one
    /// within [`QUIT_CONFIRM_WINDOW`] quits. Any other key in between starts
    /// over.
    fn handle_ctrl_c(&mut self, now: Instant) {
        if self.quit_armed(now) {
            self.should_quit = true;
            return;
        }
        if self.status != Status::Idle {
            let _ = self.channels.cancel.send(());
        }
        self.quit_armed_until = Some(now + QUIT_CONFIRM_WINDOW);
    }

    /// Whether a Ctrl-C now would quit.
    fn quit_armed(&self, now: Instant) -> bool {
        self.quit_armed_until.is_some_and(|until| now < until)
    }

    pub(super) fn handle_permission_request(&mut self, request: PendingPermission) {
        self.permissions.push(request);
    }

    /// Answers the permission prompt on screen and notes the decision in the
    /// transcript.
    fn resolve_permission(&mut self, decision: PermissionDecision) {
        if let Some(call) = self.permissions.resolve(decision) {
            let label = match decision {
                PermissionDecision::AllowOnce => "allowed",
                PermissionDecision::AllowAlways => "always allowed",
                PermissionDecision::RejectOnce => "rejected",
                PermissionDecision::RejectAlways => "always rejected",
            };
            self.push(ChatItem::SystemInfo(format!("{label} {call}")));
        }
    }

    pub(super) fn handle_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.handle_ctrl_c(Instant::now());
            return;
        }
        self.quit_armed_until = None;
        // Topmost layer first: a permission prompt, then a popup, then the
        // input line.
        self.permissions.prune_stale();
        if !self.permissions.is_empty() {
            self.handle_permission_key(key);
        } else if self.overlay.is_some() {
            self.handle_overlay_key(key);
        } else {
            self.handle_input_key(key);
        }
    }

    fn handle_permission_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.permissions.select_previous(),
            KeyCode::Down => self.permissions.select_next(),
            KeyCode::Enter => self.resolve_permission(self.permissions.selected_decision()),
            KeyCode::Char('y') => self.resolve_permission(PermissionDecision::AllowOnce),
            KeyCode::Char('a') => self.resolve_permission(PermissionDecision::AllowAlways),
            KeyCode::Char('n') => self.resolve_permission(PermissionDecision::RejectOnce),
            KeyCode::Char('r') => self.resolve_permission(PermissionDecision::RejectAlways),
            // No Esc-to-dismiss: unlike popups, a permission prompt needs an
            // explicit decision — the agent task is blocked on it.
            _ => {}
        }
    }

    /// Esc closes any popup; Enter in a picker acts on the selection and
    /// closes it too. The popup is taken out of `self` while handling so the
    /// actions below can use the rest of `App` freely.
    fn handle_overlay_key(&mut self, key: KeyEvent) {
        let Some(mut overlay) = self.overlay.take() else {
            return;
        };
        if key.code == KeyCode::Esc {
            return;
        }
        let keep_open = match &mut overlay {
            Overlay::ModelPicker { items, selected } => {
                if key.code == KeyCode::Enter {
                    if let Some((provider, model)) = items.get(*selected) {
                        let _ = self
                            .channels
                            .switch_model
                            .send((provider.clone(), model.clone()));
                    }
                    self.screen = Screen::Chat;
                    false
                } else {
                    move_selection(selected, items.len(), key.code);
                    true
                }
            }
            Overlay::SessionPicker { sessions, selected } => {
                if key.code == KeyCode::Enter {
                    if let Some(meta) = sessions.get(*selected).cloned() {
                        self.screen = Screen::Chat;
                        self.open_session(&meta);
                        false
                    } else {
                        true
                    }
                } else {
                    move_selection(selected, sessions.len(), key.code);
                    true
                }
            }
            Overlay::ThemePicker { selected } => {
                if key.code == KeyCode::Enter {
                    let name = render::THEME_COLORS[*selected];
                    self.screen = Screen::Chat;
                    self.apply_theme(name);
                    false
                } else {
                    move_selection(selected, render::THEME_COLORS.len(), key.code);
                    true
                }
            }
            Overlay::ConfigViewer { scroll, .. }
            | Overlay::McpViewer { scroll, .. }
            | Overlay::SkillsViewer { scroll, .. } => {
                move_scroll(scroll, key.code);
                true
            }
        };
        if keep_open {
            self.overlay = Some(overlay);
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.clear();
            }
            KeyCode::Enter => {
                if self.status != Status::Idle {
                    return;
                }
                let line = self.input.text().trim().to_string();
                if line.is_empty() {
                    return;
                }
                self.input.clear();
                self.screen = Screen::Chat;
                if let Some(rest) = line.strip_prefix(':') {
                    self.handle_command(rest.trim());
                } else {
                    self.push(ChatItem::UserMessage(line.clone()));
                    self.status = Status::Thinking;
                    self.transcript.pin();
                    let _ = self.channels.prompt.send(line);
                }
            }
            KeyCode::Char(c) => self.input.insert(c),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Up if self.screen == Screen::Chat => self.transcript.scroll_up(1),
            KeyCode::Down if self.screen == Screen::Chat => self.transcript.scroll_down(1),
            KeyCode::PageUp if self.screen == Screen::Chat => self.transcript.scroll_up(20),
            KeyCode::PageDown if self.screen == Screen::Chat => self.transcript.scroll_down(20),
            _ => {}
        }
    }

    fn apply_theme(&mut self, name: &str) {
        self.theme = Theme::named(name);
        self.transcript.invalidate();
        self.app_config.tui.theme_color = Some(name.to_string());
        match crate::config::save_theme_to_config_at(&self.paths.config_path, name) {
            Ok(()) => self.push(ChatItem::SystemInfo(format!("theme set to {name}"))),
            Err(e) => self.push(ChatItem::SystemInfo(format!(
                "theme set to {name} (could not save: {e})"
            ))),
        }
    }

    fn handle_command(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match name {
            "q" | "quit" => self.should_quit = true,
            "help" => self.push(ChatItem::SystemInfo(
                ":help              show this\n\
                 :q / :quit         exit\n\
                 :new               start a new session\n\
                 :sessions          browse and restore saved sessions\n\
                 :config            current config\n\
                 :models            list available models\n\
                 :models <name>     switch to model mid-session\n\
                 :mcp               MCP servers\n\
                 :skills            available skills\n\
                 :theme             change accent color\n\
                 :theme <name>      apply color directly\n\n\
                 ↑/↓  scroll · PgUp/PgDn  page\n\
                 Ctrl+C  cancel the running turn · Ctrl+C twice  quit"
                    .to_string(),
            )),
            "new" => self.start_new_session(),
            "sessions" => {
                let _ = self.channels.list_sessions.send(());
            }
            "config" => {
                self.overlay = Some(Overlay::ConfigViewer {
                    rows: self.config_rows(),
                    scroll: 0,
                });
            }
            "mcp" => {
                if self.app_config.mcp_servers.is_empty() {
                    self.push(ChatItem::SystemInfo(
                        "no MCP servers configured\n\
                         add [mcp_servers.<name>] to ~/.openheim/config.toml"
                            .to_string(),
                    ));
                } else {
                    self.overlay = Some(Overlay::McpViewer {
                        rows: self.mcp_rows(),
                        scroll: 0,
                    });
                }
            }
            "models" => {
                if arg.is_empty() {
                    let info = self.app_config.models_info();
                    let mut items: Vec<(String, String)> = info
                        .providers
                        .into_iter()
                        .flat_map(|(provider, p)| {
                            p.models.into_iter().map(move |m| (provider.clone(), m))
                        })
                        .collect();
                    items.sort();
                    let selected = items
                        .iter()
                        .position(|(_, m)| m == &self.agent_config.model)
                        .unwrap_or(0);
                    self.overlay = Some(Overlay::ModelPicker { items, selected });
                } else {
                    match self.app_config.resolve(Some(arg)) {
                        Ok(config) => {
                            let _ = self
                                .channels
                                .switch_model
                                .send((config.provider_name, config.model));
                        }
                        Err(e) => {
                            self.push(ChatItem::SystemInfo(format!("unknown model: {e}")));
                        }
                    }
                }
            }
            "skills" => {
                match SkillsManager::with_dir(self.paths.data_dir.join("skills")).list_skills() {
                    Ok(names) if names.is_empty() => {
                        self.push(ChatItem::SystemInfo(
                            "no skills available\n\
                             add <name>.md files to ~/.openheim/skills/"
                                .to_string(),
                        ));
                    }
                    Ok(items) => {
                        self.overlay = Some(Overlay::SkillsViewer { items, scroll: 0 });
                    }
                    Err(e) => self.push(ChatItem::Err(e.to_string())),
                }
            }
            "theme" => {
                if arg.is_empty() {
                    let selected = render::THEME_COLORS
                        .iter()
                        .position(|&n| n == self.theme.name)
                        .unwrap_or(0);
                    self.overlay = Some(Overlay::ThemePicker { selected });
                } else if render::THEME_COLORS.contains(&arg) {
                    self.apply_theme(arg);
                } else {
                    self.push(ChatItem::SystemInfo(format!(
                        ":{arg}: unknown theme  (available: {})",
                        render::THEME_COLORS.join(", ")
                    )));
                }
            }
            unknown => self.push(ChatItem::SystemInfo(format!(
                ":{unknown}: unknown command  (try :help)"
            ))),
        }
    }

    /// Rows for the `:config` viewer.
    fn config_rows(&self) -> Vec<ConfigRow> {
        let ac = &self.agent_config;
        let mut rows = vec![
            ConfigRow::Entry {
                key: "Provider".to_string(),
                val: ac.provider_name.clone(),
            },
            ConfigRow::Entry {
                key: "Model".to_string(),
                val: ac.model.clone(),
            },
            ConfigRow::Entry {
                key: "Max iterations".to_string(),
                val: ac.max_iterations.to_string(),
            },
            ConfigRow::Entry {
                key: "Timeout".to_string(),
                val: format!("{}s", ac.timeout_secs),
            },
        ];
        if !self.app_config.providers.is_empty() {
            rows.push(ConfigRow::Blank);
            rows.push(ConfigRow::Header("Providers".to_string()));
            for (pname, p) in &self.app_config.providers {
                let label = if pname == &self.app_config.default_provider {
                    format!("{pname}  (default)")
                } else {
                    pname.clone()
                };
                rows.push(ConfigRow::Entry {
                    key: label,
                    val: p.default_model.clone(),
                });
            }
        }
        if !self.app_config.mcp_servers.is_empty() {
            rows.push(ConfigRow::Blank);
            rows.push(ConfigRow::Header("MCP Servers".to_string()));
            for sname in self.app_config.mcp_servers.keys() {
                rows.push(ConfigRow::Item(sname.clone()));
            }
        }
        rows
    }

    /// Rows for the `:mcp` viewer.
    fn mcp_rows(&self) -> Vec<ConfigRow> {
        let mut rows = Vec::new();
        let mut iter = self.app_config.mcp_servers.iter().peekable();
        while let Some((sname, server)) = iter.next() {
            rows.push(ConfigRow::Header(sname.clone()));
            if let Some(cmd) = &server.command {
                let args_str = server.args.join(" ");
                let val = if args_str.is_empty() {
                    cmd.clone()
                } else {
                    format!("{cmd} {args_str}")
                };
                rows.push(ConfigRow::Entry {
                    key: "stdio".to_string(),
                    val,
                });
            }
            if let Some(url) = &server.url {
                rows.push(ConfigRow::Entry {
                    key: "http".to_string(),
                    val: url.clone(),
                });
            }
            if iter.peek().is_some() {
                rows.push(ConfigRow::Blank);
            }
        }
        rows
    }

    /// Asks the agent task to start a brand-new, unsaved session — the same
    /// `SessionBuilder::start` path `run()` uses on startup, just triggered
    /// mid-session instead. Deliberately leaves the current transcript and
    /// `Status::Idle` check gates prompt submission (see the `KeyCode::Enter`
    /// handler) until then, keeping the request transactional: if
    /// `client.new_session().start()` fails, `mod.rs` reports it as a plain
    /// `AgentUpdate::Error`, `Status` falls back to `Idle`, and the old
    /// session — still live in the agent task — is exactly where it was.
    fn start_new_session(&mut self) {
        self.status = Status::Thinking;
        if self.channels.new_session.send(()).is_err() {
            self.status = Status::Idle;
            self.push(ChatItem::Err(
                "failed to start new session: agent task is gone".to_string(),
            ));
        }
    }

    /// Clears the transcript and requests the agent task load `meta`'s full
    /// history via `OpenheimClient::resume_session`, converted straight to `ChatItem`s
    /// (see `message_to_chat_items`), so thinking blocks and image
    /// attachments show up instead of being silently dropped. That also
    /// keeps the history read off the UI task and on the agent task, where
    /// the rest of I/O lives — the actual message items arrive later as
    /// `AgentUpdate::History` once the load completes.
    fn open_session(&mut self, meta: &ConversationMeta) {
        self.transcript.clear();
        self.status = Status::Idle;

        let title = meta.title.as_deref().unwrap_or("(untitled)");
        self.push(ChatItem::SystemInfo(format!("─── {title}")));

        if let Some(provider_name) = &meta.provider
            && !self
                .app_config
                .providers
                .contains_key(provider_name.as_str())
        {
            self.push(ChatItem::SystemInfo(format!(
                "warning: provider '{}' is not configured; using default provider instead.",
                provider_name
            )));
        }

        let cwd = meta
            .cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| "/".into()));
        let _ = self
            .channels
            .switch_session
            .send((meta.id.to_string(), cwd));
    }

    pub(super) fn draw(&mut self, f: &mut Frame) {
        self.permissions.prune_stale();

        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Fill(1), Constraint::Length(3)])
            .split(area);
        let [content_area, input_area] = [chunks[0], chunks[1]];
        let theme = self.theme.color;

        match self.screen {
            Screen::Welcome => render::render_welcome(
                f,
                content_area,
                &self.agent_config.model,
                &self.agent_config.provider_name,
                &self.skills,
                theme,
            ),
            Screen::Chat => self.transcript.draw(f, content_area, theme),
        }

        const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame = SPINNER[self.spinner_frame % SPINNER.len()];
        let labels = FooterLabels {
            left: match &self.status {
                _ if self.quit_armed(Instant::now()) => Some("Ctrl+C again to quit".to_string()),
                Status::Idle => None,
                Status::Thinking => Some(format!("{frame} thinking… · Ctrl+C to cancel")),
                Status::Streaming => Some(format!("{frame} streaming… · Ctrl+C to cancel")),
            },
            right: match &self.context_usage {
                Some(usage) => format!(
                    "{} · {} · {}",
                    self.agent_config.provider_name,
                    self.agent_config.model,
                    format_token_count(usage.total())
                ),
                None => format!(
                    "{} · {}",
                    self.agent_config.provider_name, self.agent_config.model
                ),
            },
        };
        let input_has_focus = self.overlay.is_none() && self.permissions.is_empty();
        render::render_input_bar(f, input_area, &self.input, &labels, input_has_focus, theme);

        match &self.overlay {
            Some(Overlay::ModelPicker { items, selected }) => {
                render::render_model_picker(f, area, items, *selected, theme)
            }
            Some(Overlay::SessionPicker { sessions, selected }) => {
                render::render_session_picker(f, area, sessions, *selected, theme)
            }
            Some(Overlay::ConfigViewer { rows, scroll }) => {
                render::render_config_viewer(f, area, rows, *scroll, theme)
            }
            Some(Overlay::McpViewer { rows, scroll }) => {
                render::render_mcp_viewer(f, area, rows, *scroll, theme)
            }
            Some(Overlay::SkillsViewer { items, scroll }) => {
                render::render_skills_viewer(f, area, items, *scroll, theme)
            }
            Some(Overlay::ThemePicker { selected }) => {
                render::render_theme_picker(f, area, *selected, &self.theme.name, theme)
            }
            None => {}
        }

        if let Some(request) = self.permissions.front() {
            render::render_permission_prompt(
                f,
                area,
                &request.tool_name,
                request.subagent.as_deref(),
                &request.arguments,
                self.permissions.selected(),
                theme,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::{mpsc, oneshot};

    use super::*;

    fn test_app() -> App {
        test_app_with_cancel().0
    }

    /// A test app plus the receiving end of its cancel channel.
    fn test_app_with_cancel() -> (App, mpsc::UnboundedReceiver<()>) {
        let (prompt, _) = mpsc::unbounded_channel();
        let (switch_model, _) = mpsc::unbounded_channel();
        let (switch_session, _) = mpsc::unbounded_channel();
        let (list_sessions, _) = mpsc::unbounded_channel();
        let (new_session, _) = mpsc::unbounded_channel();
        let (cancel, cancel_rx) = mpsc::unbounded_channel();
        let app = App::new(
            AgentConfig::default(),
            AppConfig::for_tests("mock"),
            RuntimePaths {
                data_dir: "/nonexistent/openheim".into(),
                config_path: "/nonexistent/openheim/config.toml".into(),
            },
            vec![],
            AgentChannels {
                prompt,
                switch_model,
                switch_session,
                list_sessions,
                new_session,
                cancel,
            },
        );
        (app, cancel_rx)
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    #[test]
    fn ctrl_c_during_a_turn_cancels_it_and_a_second_one_quits() {
        let (mut app, mut cancel_rx) = test_app_with_cancel();
        app.status = Status::Thinking;

        app.handle_key(ctrl_c());
        assert!(
            cancel_rx.try_recv().is_ok(),
            "first Ctrl-C cancels the turn"
        );
        assert!(!app.should_quit);

        app.handle_key(ctrl_c());
        assert!(app.should_quit);
    }

    #[test]
    fn ctrl_c_when_idle_cancels_nothing_and_still_needs_a_second_press() {
        let (mut app, mut cancel_rx) = test_app_with_cancel();

        app.handle_key(ctrl_c());
        assert!(cancel_rx.try_recv().is_err());
        assert!(!app.should_quit);

        app.handle_key(ctrl_c());
        assert!(app.should_quit);
    }

    #[test]
    fn another_key_or_the_window_passing_disarms_quit() {
        let mut app = test_app();

        app.handle_key(ctrl_c());
        app.handle_key(key(KeyCode::Char('x')));
        app.handle_key(ctrl_c());
        assert!(!app.should_quit, "a key in between starts over");

        let now = Instant::now();
        app.handle_ctrl_c(now + QUIT_CONFIRM_WINDOW);
        assert!(!app.should_quit, "the window has passed");
        app.handle_ctrl_c(now + QUIT_CONFIRM_WINDOW + Duration::from_millis(500));
        assert!(app.should_quit);
    }

    fn permission_request() -> (PendingPermission, oneshot::Receiver<PermissionDecision>) {
        let (respond_to, rx) = oneshot::channel();
        (
            PendingPermission {
                tool_name: "read_file".into(),
                arguments: "{}".into(),
                subagent: None,
                respond_to,
            },
            rx,
        )
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn prune_drops_requests_whose_receiver_is_gone() {
        let mut app = test_app();

        let (live_request, _live_rx) = permission_request();
        let (dead_request, dead_rx) = permission_request();
        drop(dead_rx); // agent task gave up waiting (e.g. turn cancelled)

        app.handle_permission_request(dead_request);
        app.handle_permission_request(live_request);
        app.permissions.prune_stale();

        // A live request remains, so the prompt stays up.
        assert_eq!(app.permissions.len(), 1);
        assert_eq!(app.permissions.front().unwrap().tool_name, "read_file");
    }

    #[test]
    fn keys_go_to_the_input_once_stale_prompts_are_pruned() {
        let mut app = test_app();
        let (dead_request, dead_rx) = permission_request();
        drop(dead_rx);
        app.handle_permission_request(dead_request);

        // The prompt is stale, so the key reaches the input line instead of
        // answering (and being swallowed by) a prompt nobody is waiting on.
        app.handle_key(key(KeyCode::Char('y')));
        assert!(app.permissions.is_empty());
        assert_eq!(app.input.text(), "y");
    }

    // Pruning a stale request *behind* the prompt on screen keeps its
    // highlight; resetting it to "Allow Once" would let Enter allow a call
    // the user had moved to "Reject Once".
    #[test]
    fn highlight_survives_a_stale_request_behind_the_prompt() {
        let mut app = test_app();
        let (front, front_rx) = permission_request();
        let (behind, behind_rx) = permission_request();
        app.handle_permission_request(front);
        app.handle_permission_request(behind);

        // Allow Once → Allow Always → Reject Once.
        app.handle_key(key(KeyCode::Down));
        app.handle_key(key(KeyCode::Down));
        drop(behind_rx); // the request behind the prompt goes stale
        app.permissions.prune_stale(); // as the next frame's draw would
        app.handle_key(key(KeyCode::Enter));

        assert_eq!(front_rx.blocking_recv(), Ok(PermissionDecision::RejectOnce));
    }

    #[test]
    fn answering_a_prompt_sends_the_decision_and_shows_the_next() {
        let mut app = test_app();
        let (first, first_rx) = permission_request();
        let (second, _second_rx) = permission_request();
        app.handle_permission_request(first);
        app.handle_permission_request(second);

        app.handle_key(key(KeyCode::Char('a')));

        assert_eq!(
            first_rx.blocking_recv(),
            Ok(PermissionDecision::AllowAlways)
        );
        assert_eq!(app.permissions.len(), 1);
    }

    // A permission prompt over an open popup doesn't replace what Esc
    // returns to: after answering it, Esc still closes the popup.
    #[test]
    fn esc_closes_a_popup_after_a_permission_prompt_over_it() {
        let mut app = test_app();
        app.handle_key(key(KeyCode::Char(':')));
        for c in "theme".chars() {
            app.handle_key(key(KeyCode::Char(c)));
        }
        app.handle_key(key(KeyCode::Enter));
        assert!(matches!(app.overlay, Some(Overlay::ThemePicker { .. })));

        let (request, _rx) = permission_request();
        app.handle_permission_request(request);
        app.handle_key(key(KeyCode::Char('y')));
        // Back on the theme picker…
        assert!(matches!(app.overlay, Some(Overlay::ThemePicker { .. })));

        // …and Esc closes it.
        app.handle_key(key(KeyCode::Esc));
        assert!(app.overlay.is_none());
        assert_eq!(app.screen, Screen::Chat);
    }
}
