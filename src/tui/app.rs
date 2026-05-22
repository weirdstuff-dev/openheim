use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
};
use tokio::sync::mpsc;

use crate::{
    config::{AgentConfig, AppConfig},
    rag::{ConversationMeta, RagContext, SkillsManager},
};

use super::render;
use super::types::{AgentUpdate, ChatItem, ConfigRow, Screen, Status};

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
    switch_model_tx: mpsc::UnboundedSender<String>,
    switch_session_tx: mpsc::UnboundedSender<(String, std::path::PathBuf)>,
    picker_items: Vec<(String, String)>,
    picker_selected: usize,
    config_rows: Vec<ConfigRow>,
    config_scroll: usize,
    skills_items: Vec<String>,
    skills_scroll: usize,
    mcp_rows: Vec<ConfigRow>,
    mcp_scroll: usize,
}

impl App {
    pub(super) fn new(
        agent_config: AgentConfig,
        app_config: AppConfig,
        skills: Vec<String>,
        prompt_tx: mpsc::UnboundedSender<String>,
        switch_model_tx: mpsc::UnboundedSender<String>,
        switch_session_tx: mpsc::UnboundedSender<(String, std::path::PathBuf)>,
    ) -> Self {
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
                self.push(ChatItem::SystemInfo(format!("switched to {provider} / {model}")));
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
                    self.picker_selected =
                        (self.picker_selected + 1).min(self.sessions.len() - 1);
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

    fn handle_config_viewer_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.config_scroll = self.config_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.config_scroll += 1;
            }
            KeyCode::PageUp => {
                self.config_scroll = self.config_scroll.saturating_sub(5);
            }
            KeyCode::PageDown => {
                self.config_scroll += 5;
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    fn handle_skills_viewer_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.skills_scroll = self.skills_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.skills_scroll += 1;
            }
            KeyCode::PageUp => {
                self.skills_scroll = self.skills_scroll.saturating_sub(5);
            }
            KeyCode::PageDown => {
                self.skills_scroll += 5;
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    fn handle_mcp_viewer_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.mcp_scroll = self.mcp_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.mcp_scroll += 1;
            }
            KeyCode::PageUp => {
                self.mcp_scroll = self.mcp_scroll.saturating_sub(5);
            }
            KeyCode::PageDown => {
                self.mcp_scroll += 5;
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    fn handle_picker_key(&mut self, key: KeyEvent) {
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
                if let Some((_, model)) = self.picker_items.get(self.picker_selected) {
                    let _ = self.switch_model_tx.send(model.clone());
                }
                self.screen = Screen::Chat;
            }
            KeyCode::Esc => {
                self.screen = self.pre_picker_screen;
            }
            _ => {}
        }
    }

    pub(super) fn handle_key(&mut self, key: KeyEvent) {
        if self.screen == Screen::ModelPicker {
            self.handle_picker_key(key);
            return;
        }
        if self.screen == Screen::ConfigViewer {
            self.handle_config_viewer_key(key);
            return;
        }
        if self.screen == Screen::SessionPicker {
            self.handle_session_picker_key(key);
            return;
        }
        if self.screen == Screen::SkillsViewer {
            self.handle_skills_viewer_key(key);
            return;
        }
        if self.screen == Screen::McpViewer {
            self.handle_mcp_viewer_key(key);
            return;
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
                    let prev = prev_char_boundary(&self.input, self.cursor);
                    self.input.drain(prev..self.cursor);
                    self.cursor = prev;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    let next = next_char_boundary(&self.input, self.cursor);
                    self.input.drain(self.cursor..next);
                }
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = prev_char_boundary(&self.input, self.cursor);
                }
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor = next_char_boundary(&self.input, self.cursor);
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
                 :skills            available skills\n\n\
                 ↑/↓  scroll · PgUp/PgDn  page · Ctrl+C  quit"
                    .to_string(),
            )),
            "sessions" => match RagContext::new().and_then(|r| r.history.list_conversations()) {
                Ok(metas) if metas.is_empty() => {
                    self.push(ChatItem::SystemInfo("no sessions yet".to_string()));
                }
                Ok(metas) => {
                    self.sessions = metas;
                    self.picker_selected = 0;
                    self.pre_picker_screen = self.screen;
                    self.screen = Screen::SessionPicker;
                }
                Err(e) => self.push(ChatItem::Err(e.to_string())),
            },
            "config" => {
                let ac = &self.agent_config;
                let mut rows = vec![
                    ConfigRow::Entry { key: "Provider".to_string(), val: ac.provider_name.clone() },
                    ConfigRow::Entry { key: "Model".to_string(), val: ac.model.clone() },
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
                        rows.push(ConfigRow::Entry { key: label, val: p.default_model.clone() });
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
                self.pre_picker_screen = self.screen;
                self.screen = Screen::ConfigViewer;
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
                    self.pre_picker_screen = self.screen;
                    self.screen = Screen::McpViewer;
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
                    self.pre_picker_screen = self.screen;
                    self.screen = Screen::ModelPicker;
                } else {
                    let _ = self.switch_model_tx.send(arg.to_string());
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
                    self.pre_picker_screen = self.screen;
                    self.screen = Screen::SkillsViewer;
                }
                Err(e) => self.push(ChatItem::Err(e.to_string())),
            },
            unknown => self.push(ChatItem::SystemInfo(format!(
                ":{unknown}: unknown command  (try :help)"
            ))),
        }
    }

    fn open_session(&mut self, meta: &ConversationMeta) {
        use crate::core::models::Role;

        let title = meta.title.as_deref().unwrap_or("(untitled)");
        self.push(ChatItem::SystemInfo(format!("─── {title}")));

        match RagContext::new().and_then(|r| r.history.load_conversation(&meta.id)) {
            Ok(conv) => {
                for msg in &conv.messages {
                    match msg.role {
                        Role::System => {}
                        Role::User => {
                            if let Some(content) = &msg.content {
                                if !content.is_empty() {
                                    self.push(ChatItem::UserMessage(content.clone()));
                                }
                            }
                        }
                        Role::Assistant => {
                            if let Some(content) = &msg.content {
                                if !content.is_empty() {
                                    self.push(ChatItem::AssistantMessage(content.clone()));
                                }
                            }
                            if let Some(tool_calls) = &msg.tool_calls {
                                for tc in tool_calls {
                                    self.push(ChatItem::ToolCall {
                                        name: tc.function.name.clone(),
                                        args: tc.function.arguments.clone(),
                                    });
                                }
                            }
                        }
                        Role::Tool => {
                            if let Some(content) = &msg.content {
                                self.push(ChatItem::ToolResult {
                                    result: content.clone(),
                                    is_error: msg.is_error,
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => self.push(ChatItem::Err(e.to_string())),
        }

        self.push(ChatItem::SystemInfo("─── session restored".to_string()));

        let cwd = meta
            .cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| "/".into()));
        let _ = self.switch_session_tx.send((meta.id.to_string(), cwd));
    }

    pub(super) fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Fill(1), Constraint::Length(3)])
            .split(area);

        let [content_area, input_area] = [chunks[0], chunks[1]];

        let bg_screen = match self.screen {
            Screen::ModelPicker
            | Screen::ConfigViewer
            | Screen::SessionPicker
            | Screen::SkillsViewer
            | Screen::McpViewer => self.pre_picker_screen,
            s => s,
        };

        if bg_screen == Screen::Welcome {
            let model = self.agent_config.model.clone();
            let provider = self.agent_config.provider_name.clone();
            let skills = self.skills.clone();
            render::render_welcome(f, content_area, &model, &provider, &skills);
        } else {
            self.draw_chat(f, content_area);
        }

        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let left_label = match &self.status {
            Status::Idle => None,
            Status::Thinking => {
                Some(format!("{} thinking…", spinner[self.spinner_frame % spinner.len()]))
            }
            Status::Streaming => {
                Some(format!("{} streaming…", spinner[self.spinner_frame % spinner.len()]))
            }
        };
        let right_label =
            format!("{} · {}", self.agent_config.provider_name, self.agent_config.model);

        let input = self.input.clone();
        render::render_input_bar(
            f,
            input_area,
            &input,
            self.cursor,
            left_label.as_deref(),
            &right_label,
            self.screen != Screen::ModelPicker
                && self.screen != Screen::ConfigViewer
                && self.screen != Screen::SessionPicker
                && self.screen != Screen::SkillsViewer
                && self.screen != Screen::McpViewer,
        );

        if self.screen == Screen::ModelPicker {
            render::render_model_picker(f, area, &self.picker_items, self.picker_selected);
        }
        if self.screen == Screen::ConfigViewer {
            render::render_config_viewer(f, area, &self.config_rows, self.config_scroll);
        }
        if self.screen == Screen::SessionPicker {
            render::render_session_picker(f, area, &self.sessions, self.picker_selected);
        }
        if self.screen == Screen::SkillsViewer {
            render::render_skills_viewer(f, area, &self.skills_items, self.skills_scroll);
        }
        if self.screen == Screen::McpViewer {
            render::render_mcp_viewer(f, area, &self.mcp_rows, self.mcp_scroll);
        }
    }

    fn draw_chat(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        let chat_w = area.width;
        if self.cached_width != chat_w {
            self.cached_lines = render::build_lines(&self.items, chat_w);
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
        let visible: Vec<Line<'static>> =
            if start < end { self.cached_lines[start..end].to_vec() } else { vec![] };

        let scroll_hint = if !self.pinned && max_scroll > 0 {
            format!(" {}% ↑ ", (self.scroll * 100) / max_scroll)
        } else {
            String::new()
        };

        use ratatui::widgets::{Block, Borders};
        let chat_block = Block::default()
            .borders(Borders::NONE)
            .title_bottom(Line::from(
                Span::styled(scroll_hint, Style::default().fg(Color::DarkGray)),
            ));
        let chat_inner = chat_block.inner(area);
        f.render_widget(chat_block, area);
        f.render_widget(Paragraph::new(visible), chat_inner);
    }
}

fn prev_char_boundary(s: &str, pos: usize) -> usize {
    let mut p = pos;
    loop {
        if p == 0 {
            return 0;
        }
        p -= 1;
        if s.is_char_boundary(p) {
            return p;
        }
    }
}

fn next_char_boundary(s: &str, pos: usize) -> usize {
    let mut p = pos + 1;
    while p <= s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p.min(s.len())
}
