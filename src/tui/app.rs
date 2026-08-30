use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use tokio::sync::mpsc;

use crate::{
    config::{AgentConfig, AppConfig},
    core::permission::PermissionDecision,
    rag::{ConversationMeta, RagContext, SkillsManager},
};

use super::permission::{PERMISSION_OPTIONS, PermissionRequest};
use super::render;
use super::types::{AgentUpdate, ChatItem, ConfigRow, Screen, Status};

fn save_theme_to_config(name: &str) -> crate::error::Result<()> {
    let path = crate::config::config_path()?;
    let contents = std::fs::read_to_string(&path)?;
    let new_line = format!("theme_color = \"{name}\"");
    let has_theme = contents
        .lines()
        .any(|l| l.trim_start().starts_with("theme_color"));
    let updated: String = if has_theme {
        contents
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("theme_color") {
                    new_line.clone()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let mut lines: Vec<String> = contents.lines().map(String::from).collect();
        let insert_pos = lines
            .iter()
            .rposition(|l| {
                l.trim_start().starts_with("max_iterations")
                    || l.trim_start().starts_with("default_provider")
            })
            .map(|i| i + 1)
            .unwrap_or(0);
        lines.insert(insert_pos, new_line);
        lines.join("\n")
    };
    let trailing = if contents.ends_with('\n') { "\n" } else { "" };
    std::fs::write(&path, format!("{updated}{trailing}"))?;
    Ok(())
}

pub(super) struct App {
    pub(super) items: Vec<ChatItem>,
    pub(super) input: String,
    pub(super) cursor: usize,
    pub(super) scroll: usize,
    pub(super) pinned: bool,
    pub(super) spinner_frame: usize,
    pub(super) status: Status,
    pub(super) should_quit: bool,
    screen: Screen,
    pre_picker_screen: Screen,
    agent_config: AgentConfig,
    app_config: AppConfig,
    skills: Vec<String>,
    sessions: Vec<ConversationMeta>,
    cached_lines: Vec<Line<'static>>,
    pub(super) cached_width: u16,
    prompt_tx: mpsc::UnboundedSender<String>,
    switch_model_tx: mpsc::UnboundedSender<(String, String)>,
    switch_session_tx: mpsc::UnboundedSender<(String, std::path::PathBuf)>,
    picker_items: Vec<(String, String)>,
    picker_selected: usize,
    config_rows: Vec<ConfigRow>,
    config_scroll: usize,
    skills_items: Vec<String>,
    skills_scroll: usize,
    mcp_rows: Vec<ConfigRow>,
    mcp_scroll: usize,
    theme_color: Color,
    theme_color_name: String,
    theme_selected: usize,
    /// A queue rather than a single slot — with tool calls now checked
    /// concurrently (see agent.rs's Phase 1b), several requests can arrive
    /// before the first is answered. Shown one at a time; the rest wait
    /// their turn instead of overwriting (and orphaning) an earlier one.
    pending_permissions: VecDeque<PermissionRequest>,
    permission_selected: usize,
}

impl App {
    pub(super) fn new(
        agent_config: AgentConfig,
        app_config: AppConfig,
        skills: Vec<String>,
        prompt_tx: mpsc::UnboundedSender<String>,
        switch_model_tx: mpsc::UnboundedSender<(String, String)>,
        switch_session_tx: mpsc::UnboundedSender<(String, std::path::PathBuf)>,
    ) -> Self {
        let theme_name = app_config
            .theme_color
            .as_deref()
            .unwrap_or("gray")
            .to_string();
        let theme_color = render::theme_color(&theme_name);
        Self {
            items: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            pinned: true,
            spinner_frame: 0,
            status: Status::Idle,
            should_quit: false,
            screen: Screen::Welcome,
            pre_picker_screen: Screen::Welcome,
            agent_config,
            app_config,
            skills,
            sessions: Vec::new(),
            cached_lines: Vec::new(),
            cached_width: 0,
            prompt_tx,
            switch_model_tx,
            switch_session_tx,
            picker_items: Vec::new(),
            picker_selected: 0,
            config_rows: Vec::new(),
            config_scroll: 0,
            skills_items: Vec::new(),
            skills_scroll: 0,
            mcp_rows: Vec::new(),
            mcp_scroll: 0,
            theme_color,
            theme_color_name: theme_name,
            theme_selected: 0,
            pending_permissions: VecDeque::new(),
            permission_selected: 0,
        }
    }

    pub(super) fn push(&mut self, item: ChatItem) {
        self.items.push(item);
        self.cached_width = 0;
    }

    pub(super) fn handle_update(&mut self, update: AgentUpdate) {
        match update {
            AgentUpdate::TextChunk(text) => {
                self.status = Status::Streaming;
                match self.items.last_mut() {
                    Some(ChatItem::AssistantMessage(existing)) => existing.push_str(&text),
                    _ => self.items.push(ChatItem::AssistantMessage(text)),
                }
                self.cached_width = 0;
            }
            AgentUpdate::ThinkingChunk(text) => {
                self.status = Status::Streaming;
                match self.items.last_mut() {
                    Some(ChatItem::Thinking(existing)) => existing.push_str(&text),
                    _ => self.items.push(ChatItem::Thinking(text)),
                }
                self.cached_width = 0;
            }
            AgentUpdate::ToolCall { name, args } => {
                self.status = Status::Thinking;
                self.push(ChatItem::ToolCall { name, args });
            }
            AgentUpdate::ToolResult { result, is_error } => {
                self.push(ChatItem::ToolResult { result, is_error });
            }
            AgentUpdate::Done => {
                self.status = Status::Idle;
            }
            AgentUpdate::Error(e) => {
                self.status = Status::Idle;
                self.push(ChatItem::Err(e));
            }
            AgentUpdate::ModelChanged { provider, model } => {
                self.agent_config.provider_name = provider.clone();
                self.agent_config.model = model.clone();
                self.push(ChatItem::SystemInfo(format!(
                    "switched to {provider} / {model}"
                )));
            }
        }
    }

    fn handle_session_picker_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up => {
                self.picker_selected = self.picker_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                if !self.sessions.is_empty() {
                    self.picker_selected = (self.picker_selected + 1).min(self.sessions.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(meta) = self.sessions.get(self.picker_selected).cloned() {
                    self.screen = Screen::Chat;
                    self.open_session(&meta);
                }
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    fn handle_scroll_key(
        key: KeyEvent,
        scroll: &mut usize,
        should_quit: &mut bool,
        screen: &mut Screen,
        prev: Screen,
    ) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                *should_quit = true;
            }
            KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => *scroll = scroll.saturating_add(1),
            KeyCode::PageUp => *scroll = scroll.saturating_sub(5),
            KeyCode::PageDown => *scroll = scroll.saturating_add(5),
            KeyCode::Esc => *screen = prev,
            _ => {}
        }
    }

    fn handle_config_viewer_key(&mut self, key: KeyEvent) {
        Self::handle_scroll_key(
            key,
            &mut self.config_scroll,
            &mut self.should_quit,
            &mut self.screen,
            self.pre_picker_screen,
        );
    }

    fn handle_skills_viewer_key(&mut self, key: KeyEvent) {
        Self::handle_scroll_key(
            key,
            &mut self.skills_scroll,
            &mut self.should_quit,
            &mut self.screen,
            self.pre_picker_screen,
        );
    }

    fn handle_mcp_viewer_key(&mut self, key: KeyEvent) {
        Self::handle_scroll_key(
            key,
            &mut self.mcp_scroll,
            &mut self.should_quit,
            &mut self.screen,
            self.pre_picker_screen,
        );
    }

    fn handle_theme_picker_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up => {
                self.theme_selected = self.theme_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                self.theme_selected = (self.theme_selected + 1).min(render::THEME_COLORS.len() - 1);
            }
            KeyCode::Enter => {
                let name = render::THEME_COLORS[self.theme_selected].to_string();
                self.screen = Screen::Chat;
                self.apply_theme(&name);
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    pub(super) fn handle_permission_request(&mut self, request: PermissionRequest) {
        // Only the first arrival needs to switch screens (and remember what
        // to return to) — later ones just join the queue behind it and
        // surface once the ones ahead are resolved.
        let already_showing = !self.pending_permissions.is_empty();
        self.pending_permissions.push_back(request);
        if !already_showing {
            self.permission_selected = 0;
            self.push_screen(Screen::PermissionPrompt);
        }
    }

    /// Answers the front-of-queue request (if any), then either advances to
    /// the next queued one or returns to the prior screen once the queue is
    /// empty. A dropped `respond_to` (send fails) means the agent task
    /// already gave up waiting — nothing to do beyond discarding the request.
    fn resolve_permission(&mut self, decision: PermissionDecision) {
        if let Some(request) = self.pending_permissions.pop_front() {
            let label = match decision {
                PermissionDecision::AllowOnce => "allowed",
                PermissionDecision::AllowAlways => "always allowed",
                PermissionDecision::RejectOnce => "rejected",
                PermissionDecision::RejectAlways => "always rejected",
            };
            self.push(ChatItem::SystemInfo(format!(
                "{label} '{}'",
                request.tool_name
            )));
            let _ = request.respond_to.send(decision);
        }
        if self.pending_permissions.is_empty() {
            self.screen = self.pre_picker_screen;
        } else {
            self.permission_selected = 0;
        }
    }

    /// Drops queued requests whose `respond_to` receiver is already gone —
    /// the agent task gave up waiting (e.g. the turn was cancelled) before
    /// the user ever answered. Restores the screen shown before the prompt
    /// popped up if that empties the queue, instead of leaving a stale
    /// prompt on screen with nothing left to resolve.
    fn prune_stale_permissions(&mut self) {
        let before = self.pending_permissions.len();
        self.pending_permissions
            .retain(|request| !request.respond_to.is_closed());
        if self.pending_permissions.len() == before {
            return;
        }
        if self.pending_permissions.is_empty() {
            self.screen = self.pre_picker_screen;
        } else {
            self.permission_selected = 0;
        }
    }

    fn handle_permission_prompt_key(&mut self, key: KeyEvent) {
        self.prune_stale_permissions();
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up => {
                self.permission_selected = self.permission_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                self.permission_selected =
                    (self.permission_selected + 1).min(PERMISSION_OPTIONS.len() - 1);
            }
            KeyCode::Enter => {
                let (_, decision) = PERMISSION_OPTIONS[self.permission_selected];
                self.resolve_permission(decision);
            }
            KeyCode::Char('y') => self.resolve_permission(PermissionDecision::AllowOnce),
            KeyCode::Char('a') => self.resolve_permission(PermissionDecision::AllowAlways),
            KeyCode::Char('n') => self.resolve_permission(PermissionDecision::RejectOnce),
            KeyCode::Char('r') => self.resolve_permission(PermissionDecision::RejectAlways),
            // No Esc-to-dismiss: unlike other overlays, a permission prompt
            // needs an explicit decision — the agent task is blocked on it.
            _ => {}
        }
    }

    fn apply_theme(&mut self, name: &str) {
        self.theme_color = render::theme_color(name);
        self.theme_color_name = name.to_string();
        self.cached_width = 0;
        self.app_config.theme_color = Some(name.to_string());
        match save_theme_to_config(name) {
            Ok(()) => self.push(ChatItem::SystemInfo(format!("theme set to {name}"))),
            Err(e) => self.push(ChatItem::SystemInfo(format!(
                "theme set to {name} (could not save: {e})"
            ))),
        }
    }

    fn handle_model_picker_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up => {
                self.picker_selected = self.picker_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                if !self.picker_items.is_empty() {
                    self.picker_selected =
                        (self.picker_selected + 1).min(self.picker_items.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some((provider, model)) = self.picker_items.get(self.picker_selected) {
                    let _ = self.switch_model_tx.send((provider.clone(), model.clone()));
                }
                self.screen = Screen::Chat;
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    fn push_screen(&mut self, next: Screen) {
        self.pre_picker_screen = self.screen;
        self.screen = next;
    }

    pub(super) fn handle_key(&mut self, key: KeyEvent) {
        match self.screen {
            Screen::ModelPicker => {
                self.handle_model_picker_key(key);
                return;
            }
            Screen::ConfigViewer => {
                self.handle_config_viewer_key(key);
                return;
            }
            Screen::SessionPicker => {
                self.handle_session_picker_key(key);
                return;
            }
            Screen::SkillsViewer => {
                self.handle_skills_viewer_key(key);
                return;
            }
            Screen::McpViewer => {
                self.handle_mcp_viewer_key(key);
                return;
            }
            Screen::ThemePicker => {
                self.handle_theme_picker_key(key);
                return;
            }
            Screen::PermissionPrompt => {
                self.handle_permission_prompt_key(key);
                return;
            }
            Screen::Welcome | Screen::Chat => {}
        }
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.clear();
                self.cursor = 0;
            }
            KeyCode::Enter => {
                if self.status != Status::Idle {
                    return;
                }
                let line = self.input.trim().to_string();
                if line.is_empty() {
                    return;
                }
                self.input.clear();
                self.cursor = 0;
                self.screen = Screen::Chat;
                if let Some(rest) = line.strip_prefix(':') {
                    self.handle_command(rest.trim());
                } else {
                    self.push(ChatItem::UserMessage(line.clone()));
                    self.status = Status::Thinking;
                    self.pinned = true;
                    let _ = self.prompt_tx.send(line);
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += c.len_utf8();
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let prev = self.input.floor_char_boundary(self.cursor - 1);
                    self.input.drain(prev..self.cursor);
                    self.cursor = prev;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    let next = self.input.ceil_char_boundary(self.cursor + 1);
                    self.input.drain(self.cursor..next);
                }
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.input.floor_char_boundary(self.cursor - 1);
                }
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor = self.input.ceil_char_boundary(self.cursor + 1);
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up if self.screen == Screen::Chat => {
                self.scroll = self.scroll.saturating_sub(1);
                self.pinned = false;
            }
            KeyCode::Down if self.screen == Screen::Chat => {
                self.scroll = self.scroll.saturating_add(1);
            }
            KeyCode::PageUp if self.screen == Screen::Chat => {
                self.scroll = self.scroll.saturating_sub(20);
                self.pinned = false;
            }
            KeyCode::PageDown if self.screen == Screen::Chat => {
                self.scroll = self.scroll.saturating_add(20);
            }
            _ => {}
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
                 :sessions          browse and restore saved sessions\n\
                 :config            current config\n\
                 :models            list available models\n\
                 :models <name>     switch to model mid-session\n\
                 :mcp               MCP servers\n\
                 :skills            available skills\n\
                 :theme             change accent color\n\
                 :theme <name>      apply color directly\n\n\
                 ↑/↓  scroll · PgUp/PgDn  page · Ctrl+C  quit"
                    .to_string(),
            )),
            "sessions" => {
                match RagContext::new(vec![]).and_then(|r| r.history.list_conversations()) {
                    Ok(metas) if metas.is_empty() => {
                        self.push(ChatItem::SystemInfo("no sessions yet".to_string()));
                    }
                    Ok(metas) => {
                        self.sessions = metas;
                        self.picker_selected = 0;
                        self.push_screen(Screen::SessionPicker);
                    }
                    Err(e) => self.push(ChatItem::Err(e.to_string())),
                }
            }
            "config" => {
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
                self.config_rows = rows;
                self.config_scroll = 0;
                self.push_screen(Screen::ConfigViewer);
            }
            "mcp" => {
                if self.app_config.mcp_servers.is_empty() {
                    self.push(ChatItem::SystemInfo(
                        "no MCP servers configured\n\
                         add [mcp_servers.<name>] to ~/.openheim/config.toml"
                            .to_string(),
                    ));
                } else {
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
                    self.mcp_rows = rows;
                    self.mcp_scroll = 0;
                    self.push_screen(Screen::McpViewer);
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
                    self.picker_items = items;
                    self.picker_selected = selected;
                    self.push_screen(Screen::ModelPicker);
                } else {
                    match self.app_config.resolve(Some(arg)) {
                        Ok(config) => {
                            let _ = self
                                .switch_model_tx
                                .send((config.provider_name, config.model));
                        }
                        Err(e) => {
                            self.push(ChatItem::SystemInfo(format!("unknown model: {e}")));
                        }
                    }
                }
            }
            "skills" => match SkillsManager::new().and_then(|m| m.list_skills()) {
                Ok(names) if names.is_empty() => {
                    self.push(ChatItem::SystemInfo(
                        "no skills available\n\
                         add <name>.md files to ~/.openheim/skills/"
                            .to_string(),
                    ));
                }
                Ok(names) => {
                    self.skills_items = names;
                    self.skills_scroll = 0;
                    self.push_screen(Screen::SkillsViewer);
                }
                Err(e) => self.push(ChatItem::Err(e.to_string())),
            },
            "theme" => {
                if arg.is_empty() {
                    self.theme_selected = render::THEME_COLORS
                        .iter()
                        .position(|&n| n == self.theme_color_name)
                        .unwrap_or(0);
                    self.push_screen(Screen::ThemePicker);
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

    fn open_session(&mut self, meta: &ConversationMeta) {
        use crate::core::models::Role;

        self.items.clear();
        self.status = Status::Idle;
        self.scroll = 0;
        self.pinned = true;

        let title = meta.title.as_deref().unwrap_or("(untitled)");
        self.push(ChatItem::SystemInfo(format!("─── {title}")));

        match RagContext::new(vec![]).and_then(|r| r.history.load_conversation(&meta.id)) {
            Ok(conv) => {
                for msg in &conv.messages {
                    match msg.role {
                        Role::System => {}
                        Role::User => {
                            if let Some(content) = msg.text() {
                                self.push(ChatItem::UserMessage(content));
                            }
                        }
                        Role::Assistant => {
                            if let Some(content) = msg.text() {
                                self.push(ChatItem::AssistantMessage(content));
                            }
                            for tc in msg.tool_calls() {
                                self.push(ChatItem::ToolCall {
                                    name: tc.name,
                                    args: tc.arguments,
                                });
                            }
                        }
                        Role::Tool => {
                            if let Some(tr) = msg.tool_result_block() {
                                self.push(ChatItem::ToolResult {
                                    result: tr.content,
                                    is_error: tr.is_error,
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => self.push(ChatItem::Err(e.to_string())),
        }

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

        self.push(ChatItem::SystemInfo("─── session restored".to_string()));

        let cwd = meta
            .cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| "/".into()));
        let _ = self.switch_session_tx.send((meta.id.to_string(), cwd));
    }

    pub(super) fn draw(&mut self, f: &mut Frame) {
        self.prune_stale_permissions();

        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Fill(1), Constraint::Length(3)])
            .split(area);

        let [content_area, input_area] = [chunks[0], chunks[1]];

        let bg_screen = if self.screen.is_overlay() {
            self.pre_picker_screen
        } else {
            self.screen
        };

        if bg_screen == Screen::Welcome {
            render::render_welcome(
                f,
                content_area,
                &self.agent_config.model,
                &self.agent_config.provider_name,
                &self.skills,
                self.theme_color,
            );
        } else {
            self.draw_chat(f, content_area);
        }

        const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame = SPINNER[self.spinner_frame % SPINNER.len()];
        let left_label = match &self.status {
            Status::Idle => None,
            Status::Thinking => Some(format!("{frame} thinking…")),
            Status::Streaming => Some(format!("{frame} streaming…")),
        };
        let right_label = format!(
            "{} · {}",
            self.agent_config.provider_name, self.agent_config.model
        );

        let theme = self.theme_color;
        render::render_input_bar(
            f,
            input_area,
            &self.input,
            self.cursor,
            left_label.as_deref(),
            &right_label,
            !self.screen.is_overlay(),
            theme,
        );

        match self.screen {
            Screen::ModelPicker => {
                render::render_model_picker(
                    f,
                    area,
                    &self.picker_items,
                    self.picker_selected,
                    theme,
                );
            }
            Screen::ConfigViewer => {
                render::render_config_viewer(f, area, &self.config_rows, self.config_scroll, theme);
            }
            Screen::SessionPicker => {
                render::render_session_picker(f, area, &self.sessions, self.picker_selected, theme);
            }
            Screen::SkillsViewer => {
                render::render_skills_viewer(
                    f,
                    area,
                    &self.skills_items,
                    self.skills_scroll,
                    theme,
                );
            }
            Screen::McpViewer => {
                render::render_mcp_viewer(f, area, &self.mcp_rows, self.mcp_scroll, theme);
            }
            Screen::ThemePicker => {
                render::render_theme_picker(
                    f,
                    area,
                    self.theme_selected,
                    &self.theme_color_name,
                    theme,
                );
            }
            Screen::PermissionPrompt => {
                if let Some(request) = self.pending_permissions.front() {
                    render::render_permission_prompt(
                        f,
                        area,
                        &request.tool_name,
                        &request.arguments,
                        self.permission_selected,
                        theme,
                    );
                }
            }
            Screen::Welcome | Screen::Chat => {}
        }
    }

    fn draw_chat(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        let chat_w = area.width;
        if self.cached_width != chat_w {
            self.cached_lines = render::build_lines(&self.items, chat_w, self.theme_color);
            self.cached_width = chat_w;
        }

        let total = self.cached_lines.len();
        let visible_h = area.height as usize;
        let max_scroll = total.saturating_sub(visible_h);

        if self.pinned {
            self.scroll = max_scroll;
        } else {
            self.scroll = self.scroll.min(max_scroll);
            if self.scroll >= max_scroll {
                self.pinned = true;
            }
        }

        let start = self.scroll;
        let end = (start + visible_h).min(total);
        let visible: Vec<Line<'static>> = if start < end {
            self.cached_lines[start..end].to_vec()
        } else {
            vec![]
        };

        let scroll_hint = if !self.pinned && max_scroll > 0 {
            format!(" {}% ↑ ", (self.scroll * 100) / max_scroll)
        } else {
            String::new()
        };

        let chat_block = Block::default()
            .borders(Borders::NONE)
            .title_bottom(Line::from(Span::styled(
                scroll_hint,
                Style::default().fg(self.theme_color),
            )));
        let chat_inner = chat_block.inner(area);
        f.render_widget(chat_block, area);
        f.render_widget(Paragraph::new(visible), chat_inner);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tokio::sync::oneshot;

    use super::*;
    use crate::config::AgentConfig;

    fn test_app_config() -> AppConfig {
        AppConfig {
            default_provider: "mock".into(),
            max_iterations: 10,
            theme_color: None,
            providers: BTreeMap::new(),
            mcp_servers: BTreeMap::new(),
            default_skills: vec![],
            work_dir: None,
            allow_shell: false,
        }
    }

    fn test_app() -> App {
        let (prompt_tx, _) = mpsc::unbounded_channel();
        let (switch_model_tx, _) = mpsc::unbounded_channel();
        let (switch_session_tx, _) = mpsc::unbounded_channel();
        App::new(
            AgentConfig::default(),
            test_app_config(),
            vec![],
            prompt_tx,
            switch_model_tx,
            switch_session_tx,
        )
    }

    fn permission_request() -> (PermissionRequest, oneshot::Receiver<PermissionDecision>) {
        let (respond_to, rx) = oneshot::channel();
        (
            PermissionRequest {
                tool_name: "read_file".into(),
                arguments: "{}".into(),
                respond_to,
            },
            rx,
        )
    }

    #[test]
    fn prune_drops_requests_whose_receiver_is_gone() {
        let mut app = test_app();
        app.screen = Screen::Chat;

        let (live_request, _live_rx) = permission_request();
        let (dead_request, dead_rx) = permission_request();
        drop(dead_rx); // agent task gave up waiting (e.g. turn cancelled)

        app.handle_permission_request(dead_request);
        app.handle_permission_request(live_request);
        assert_eq!(app.screen, Screen::PermissionPrompt);

        app.prune_stale_permissions();

        assert_eq!(app.pending_permissions.len(), 1);
        assert_eq!(app.pending_permissions[0].tool_name, "read_file");
        // A live request remains, so the prompt stays up.
        assert_eq!(app.screen, Screen::PermissionPrompt);
    }

    #[test]
    fn prune_restores_prior_screen_once_queue_empties() {
        let mut app = test_app();
        app.screen = Screen::Chat;

        let (dead_request, dead_rx) = permission_request();
        drop(dead_rx);

        app.handle_permission_request(dead_request);
        assert_eq!(app.screen, Screen::PermissionPrompt);

        app.prune_stale_permissions();

        assert!(app.pending_permissions.is_empty());
        assert_eq!(app.screen, Screen::Chat);
    }
}
