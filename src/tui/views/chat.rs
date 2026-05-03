use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::tui::app::AgentUpdate;

#[derive(Debug, Clone)]
pub enum ChatMessage {
    User(String),
    AgentText(String),
    ToolCall { name: String, args: String },
    ToolResult(String),
    Error(String),
}

pub struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub cursor: usize,
    pub scroll: u16,
    pub is_thinking: bool,
    auto_scroll: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            is_thinking: false,
            auto_scroll: true,
        }
    }
}

impl ChatState {
    pub fn handle_update(&mut self, update: AgentUpdate) {
        match update {
            AgentUpdate::TextChunk(text) => {
                self.is_thinking = false;
                if let Some(ChatMessage::AgentText(s)) = self.messages.last_mut() {
                    s.push_str(&text);
                } else {
                    self.messages.push(ChatMessage::AgentText(text));
                }
                self.auto_scroll = true;
            }
            AgentUpdate::ToolCall { name, args } => {
                let preview: String = args.chars().take(60).collect();
                let preview = if args.chars().count() > 60 {
                    format!("{preview}…")
                } else {
                    preview
                };
                self.messages.push(ChatMessage::ToolCall { name, args: preview });
                self.auto_scroll = true;
            }
            AgentUpdate::ToolResult(result) => {
                let flat: String = result
                    .chars()
                    .take(100)
                    .collect::<String>()
                    .replace('\n', " ");
                let flat = flat.trim().to_string();
                self.messages.push(ChatMessage::ToolResult(flat));
                self.auto_scroll = true;
            }
            AgentUpdate::Done => {
                self.is_thinking = false;
                self.auto_scroll = true;
            }
            AgentUpdate::Error(e) => {
                self.is_thinking = false;
                self.messages.push(ChatMessage::Error(e));
                self.auto_scroll = true;
            }
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<String> {
        if self.is_thinking {
            return None;
        }

        match key.code {
            KeyCode::Enter => {
                let prompt = self.input.trim().to_string();
                if !prompt.is_empty() {
                    self.messages.push(ChatMessage::User(prompt.clone()));
                    self.input.clear();
                    self.cursor = 0;
                    self.is_thinking = true;
                    self.auto_scroll = true;
                    return Some(prompt);
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += c.len_utf8();
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let prev = self.prev_boundary();
                    self.input.drain(prev..self.cursor);
                    self.cursor = prev;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    let next = self.next_boundary();
                    self.input.drain(self.cursor..next);
                }
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.prev_boundary();
                }
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor = self.next_boundary();
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => {
                self.scroll = self.scroll.saturating_sub(1);
                self.auto_scroll = false;
            }
            KeyCode::Down => {
                self.scroll += 1;
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(10);
                self.auto_scroll = false;
            }
            KeyCode::PageDown => {
                self.scroll += 10;
            }
            _ => {}
        }

        None
    }

    fn prev_boundary(&self) -> usize {
        let mut pos = self.cursor - 1;
        while pos > 0 && !self.input.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }

    fn next_boundary(&self) -> usize {
        let mut pos = self.cursor + 1;
        while pos < self.input.len() && !self.input.is_char_boundary(pos) {
            pos += 1;
        }
        pos
    }
}

pub fn render(f: &mut Frame, area: Rect, state: &mut ChatState) {
    let chunks = Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).split(area);

    let mut lines: Vec<Line> = Vec::new();

    if state.messages.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  Start a conversation — type below and press Enter",
            Style::default().fg(Color::DarkGray),
        )]));
    }

    for msg in &state.messages {
        match msg {
            ChatMessage::User(text) => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        "  you      ",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(text.clone()),
                ]));
            }
            ChatMessage::AgentText(text) => {
                for (i, part) in text.split('\n').enumerate() {
                    if i == 0 {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "  openheim ",
                                Style::default()
                                    .fg(Color::Yellow)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::raw(part.to_string()),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("             "),
                            Span::raw(part.to_string()),
                        ]));
                    }
                }
            }
            ChatMessage::ToolCall { name, args } => {
                lines.push(Line::from(vec![
                    Span::styled("  ┌ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        name.clone(),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  {args}"),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
            ChatMessage::ToolResult(result) => {
                lines.push(Line::from(vec![
                    Span::styled("  └ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(result.clone(), Style::default().fg(Color::DarkGray)),
                ]));
            }
            ChatMessage::Error(e) => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        "  error    ",
                        Style::default()
                            .fg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(e.clone()),
                ]));
            }
        }
    }

    if state.is_thinking {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  thinking…",
            Style::default().fg(Color::DarkGray),
        )]));
    }

    let content_height = lines.len() as u16;
    let visible_height = chunks[0].height.saturating_sub(2);
    let max_scroll = content_height.saturating_sub(visible_height);

    if state.auto_scroll {
        state.scroll = max_scroll;
    } else {
        state.scroll = state.scroll.min(max_scroll);
    }

    let messages_widget = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title(" openheim "))
        .wrap(Wrap { trim: false })
        .scroll((state.scroll, 0));
    f.render_widget(messages_widget, chunks[0]);

    // Input box
    let (input_title, input_style) = if state.is_thinking {
        (
            " thinking… ",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        (" prompt ", Style::default().fg(Color::White))
    };

    let input_widget = Paragraph::new(state.input.as_str())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(input_title)
                .border_style(if state.is_thinking {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default().fg(Color::White)
                }),
        )
        .style(input_style);
    f.render_widget(input_widget, chunks[1]);

    if !state.is_thinking {
        let display_offset = state.input[..state.cursor].chars().count() as u16;
        f.set_cursor_position((chunks[1].x + 1 + display_offset, chunks[1].y + 1));
    }
}
