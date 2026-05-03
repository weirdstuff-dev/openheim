use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::config::{AgentConfig, AppConfig};

pub struct ConfigState {
    agent_config: AgentConfig,
    app_config: AppConfig,
    scroll: u16,
}

impl ConfigState {
    pub fn new(agent_config: AgentConfig, app_config: AppConfig) -> Self {
        Self {
            agent_config,
            app_config,
            scroll: 0,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll = self.scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll += 1;
            }
            _ => {}
        }
    }
}

pub fn render(f: &mut Frame, area: Rect, state: &ConfigState) {
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(""));
    lines.push(kv("  Provider", &state.agent_config.provider_name));
    lines.push(kv("  Model", &state.agent_config.model));
    lines.push(kv(
        "  Max iterations",
        &state.agent_config.max_iterations.to_string(),
    ));
    lines.push(kv(
        "  Timeout",
        &format!("{}s", state.agent_config.timeout_secs),
    ));

    if !state.app_config.providers.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  Providers",
            Style::default().fg(Color::Yellow),
        )]));
        for (name, provider) in &state.app_config.providers {
            let is_default = name == &state.app_config.default_provider;
            let suffix = if is_default { "  (default)" } else { "" };
            lines.push(Line::from(vec![
                Span::raw(format!("    {name}{suffix}  ")),
                Span::styled(
                    provider.default_model.clone(),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }
    }

    if !state.app_config.mcp_servers.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  MCP Servers",
            Style::default().fg(Color::Yellow),
        )]));
        for name in state.app_config.mcp_servers.keys() {
            lines.push(Line::from(format!("    {name}")));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  Edit config: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "~/.openheim/config.toml",
            Style::default().fg(Color::White),
        ),
    ]));

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" config "))
        .scroll((state.scroll, 0));
    f.render_widget(para, area);
}

fn kv(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{key:<22}"),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(value.to_string(), Style::default().fg(Color::White)),
    ])
}
