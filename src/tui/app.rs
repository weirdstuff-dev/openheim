use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Paragraph, Tabs},
};
use tokio::sync::mpsc;

use crate::config::{AgentConfig, AppConfig};

use super::views::{chat, config, mcp, sessions};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Chat,
    Sessions,
    Config,
    Mcp,
}

impl View {
    const ALL: &'static [View] = &[View::Chat, View::Sessions, View::Config, View::Mcp];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&v| v == self).unwrap_or(0)
    }

    fn label(self) -> &'static str {
        match self {
            View::Chat => "Chat",
            View::Sessions => "Sessions",
            View::Config => "Config",
            View::Mcp => "MCP",
        }
    }

    fn next(self) -> Self {
        Self::ALL[(self.index() + 1) % Self::ALL.len()]
    }

    fn prev(self) -> Self {
        Self::ALL[(self.index() + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone)]
pub enum AgentUpdate {
    TextChunk(String),
    ToolCall { name: String, args: String },
    ToolResult(String),
    Done,
    Error(String),
}

pub struct App {
    pub active_view: View,
    pub chat: chat::ChatState,
    pub sessions: sessions::SessionsState,
    pub config: config::ConfigState,
    pub mcp: mcp::McpState,
    prompt_tx: mpsc::Sender<String>,
}

impl App {
    pub fn new(
        prompt_tx: mpsc::Sender<String>,
        agent_config: AgentConfig,
        app_config: AppConfig,
    ) -> Self {
        Self {
            active_view: View::Chat,
            chat: chat::ChatState::default(),
            sessions: sessions::SessionsState::default(),
            config: config::ConfigState::new(agent_config, app_config.clone()),
            mcp: mcp::McpState::new(app_config),
            prompt_tx,
        }
    }

    pub fn load_sessions(&mut self) {
        if let Ok(rag) = crate::rag::RagContext::new() {
            if let Ok(metas) = rag.history.list_conversations() {
                self.sessions.set_conversations(metas);
            }
        }
    }

    pub fn handle_agent_update(&mut self, update: AgentUpdate) {
        self.chat.handle_update(update);
    }

    /// Returns true if the app should quit.
    pub fn handle_event(&mut self, event: Event) -> bool {
        if let Event::Key(key) = event {
            self.handle_key(key)
        } else {
            false
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }

        if key.code == KeyCode::Tab {
            self.active_view = self.active_view.next();
            return false;
        }
        if key.code == KeyCode::BackTab {
            self.active_view = self.active_view.prev();
            return false;
        }

        match self.active_view {
            View::Chat => {
                if let Some(prompt) = self.chat.handle_key(key) {
                    let _ = self.prompt_tx.try_send(prompt);
                }
            }
            View::Sessions => {
                if self.sessions.handle_key(key).is_some() {
                    // Session resumption requires extending ACP to pass a chat_id —
                    // for now just switch to chat view where a fresh session is active.
                    self.active_view = View::Chat;
                }
            }
            View::Config => self.config.handle_key(key),
            View::Mcp => self.mcp.handle_key(key),
        }

        false
    }

    pub fn render(&mut self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .split(area);

        let tab_labels: Vec<&str> = View::ALL.iter().map(|v| v.label()).collect();
        let tabs = Tabs::new(tab_labels)
            .select(self.active_view.index())
            .block(Block::default().borders(Borders::BOTTOM))
            .style(Style::default().fg(Color::DarkGray))
            .highlight_style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            );
        f.render_widget(tabs, chunks[0]);

        match self.active_view {
            View::Chat => chat::render(f, chunks[1], &mut self.chat),
            View::Sessions => sessions::render(f, chunks[1], &mut self.sessions),
            View::Config => config::render(f, chunks[1], &self.config),
            View::Mcp => mcp::render(f, chunks[1], &self.mcp),
        }

        let status = Paragraph::new(
            " Tab: next  Shift+Tab: prev  Enter: send  ↑↓: scroll  Ctrl+C: quit",
        )
        .style(Style::default().fg(Color::DarkGray));
        f.render_widget(status, chunks[2]);
    }
}
