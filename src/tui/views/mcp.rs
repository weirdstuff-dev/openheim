use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::config::AppConfig;

pub struct McpState {
    app_config: AppConfig,
    scroll: u16,
}

impl McpState {
    pub fn new(app_config: AppConfig) -> Self {
        Self {
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

pub fn render(f: &mut Frame, area: Rect, state: &McpState) {
    let mut lines: Vec<Line> = Vec::new();

    if state.app_config.mcp_servers.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  No MCP servers configured.",
            Style::default().fg(Color::DarkGray),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "  Add [mcp_servers.<name>] entries to ~/.openheim/config.toml",
            Style::default().fg(Color::DarkGray),
        )]));
    } else {
        for (name, server) in &state.app_config.mcp_servers {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("  ● ", Style::default().fg(Color::Green)),
                Span::styled(
                    name.clone(),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            if let Some(cmd) = &server.command {
                let args_str = server.args.join(" ");
                let cmd_line = if args_str.is_empty() {
                    cmd.clone()
                } else {
                    format!("{cmd} {args_str}")
                };
                lines.push(Line::from(vec![
                    Span::styled("    stdio  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(cmd_line, Style::default().fg(Color::DarkGray)),
                ]));
            }

            if let Some(url) = &server.url {
                lines.push(Line::from(vec![
                    Span::styled("    http   ", Style::default().fg(Color::DarkGray)),
                    Span::styled(url.clone(), Style::default().fg(Color::DarkGray)),
                ]));
            }

            if !server.env.is_empty() {
                for (k, _) in &server.env {
                    lines.push(Line::from(vec![
                        Span::styled("    env    ", Style::default().fg(Color::DarkGray)),
                        Span::styled(k.clone(), Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }
        }
    }

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" mcp servers "))
        .scroll((state.scroll, 0));
    f.render_widget(para, area);
}
