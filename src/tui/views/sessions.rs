use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, ListState},
};
use uuid::Uuid;

use crate::rag::ConversationMeta;

#[derive(Default)]
pub struct SessionsState {
    conversations: Vec<ConversationMeta>,
    pub list_state: ListState,
}

impl SessionsState {
    pub fn set_conversations(&mut self, metas: Vec<ConversationMeta>) {
        self.conversations = metas;
        if !self.conversations.is_empty() {
            self.list_state.select(Some(0));
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<Uuid> {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                let i = self.list_state.selected().unwrap_or(0);
                if i > 0 {
                    self.list_state.select(Some(i - 1));
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let len = self.conversations.len();
                if len > 0 {
                    let i = self.list_state.selected().unwrap_or(0);
                    if i + 1 < len {
                        self.list_state.select(Some(i + 1));
                    }
                }
            }
            KeyCode::Enter => {
                if let Some(i) = self.list_state.selected() {
                    if let Some(meta) = self.conversations.get(i) {
                        return Some(meta.id);
                    }
                }
            }
            _ => {}
        }
        None
    }
}

pub fn render(f: &mut Frame, area: Rect, state: &mut SessionsState) {
    let items: Vec<ListItem> = state
        .conversations
        .iter()
        .map(|meta| {
            let title = meta.title.as_deref().unwrap_or("(untitled)");
            let date = meta.updated_at.format("%Y-%m-%d %H:%M").to_string();
            let model = meta.model.as_deref().unwrap_or("?");
            ListItem::new(format!("  {title}  ·  {date}  ·  {model}"))
        })
        .collect();

    let empty_msg;
    let list = if items.is_empty() {
        empty_msg = vec![ListItem::new("  No conversations yet. Start chatting!")
            .style(Style::default().fg(Color::DarkGray))];
        List::new(empty_msg)
    } else {
        List::new(items)
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▶ ")
    }
    .block(Block::default().borders(Borders::ALL).title(" sessions "));

    f.render_stateful_widget(list, area, &mut state.list_state);
}
