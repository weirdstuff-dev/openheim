//! The pieces of TUI state `App` is made of: the transcript, the input line,
//! the popup currently open, the permission-prompt queue, the theme, and the
//! channels to the agent task. Each owns its own fields and the operations on
//! them; `App` wires them to keys, agent updates and drawing.

use std::collections::VecDeque;

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use tokio::sync::mpsc;

use crate::{core::permission::PermissionDecision, memory::ConversationMeta};

use super::permission::{PERMISSION_OPTIONS, PendingPermission};
use super::render;
use super::types::{ChatItem, ConfigRow};

/// The chat history on screen, plus its scroll position and the rendered
/// lines cached for the current width.
pub(super) struct Transcript {
    items: Vec<ChatItem>,
    scroll: usize,
    /// Follow the newest line; cleared by scrolling up, restored by
    /// scrolling back to the bottom.
    pinned: bool,
    cached_lines: Vec<Line<'static>>,
    /// Width `cached_lines` was built for; 0 means stale.
    cached_width: u16,
}

impl Transcript {
    pub(super) fn new() -> Self {
        Self {
            items: Vec::new(),
            scroll: 0,
            pinned: true,
            cached_lines: Vec::new(),
            cached_width: 0,
        }
    }

    pub(super) fn push(&mut self, item: ChatItem) {
        self.items.push(item);
        self.invalidate();
    }

    /// Appends streamed reply text to the reply in progress, or starts one.
    pub(super) fn append_assistant_text(&mut self, content: String) {
        match self.items.last_mut() {
            Some(ChatItem::AssistantMessage(existing)) => existing.push_str(&content),
            _ => self.items.push(ChatItem::AssistantMessage(content)),
        }
        self.invalidate();
    }

    /// Appends streamed thinking to the block in progress, or starts one.
    pub(super) fn append_thinking(&mut self, content: String) {
        match self.items.last_mut() {
            Some(ChatItem::Thinking(existing)) => existing.push_str(&content),
            _ => self.items.push(ChatItem::Thinking(content)),
        }
        self.invalidate();
    }

    /// Empties the transcript and jumps back to following the bottom.
    pub(super) fn clear(&mut self) {
        self.items.clear();
        self.scroll = 0;
        self.pinned = true;
        self.invalidate();
    }

    /// Forces the lines to be rebuilt on the next draw (new content, theme
    /// change, terminal resize).
    pub(super) fn invalidate(&mut self) {
        self.cached_width = 0;
    }

    pub(super) fn pin(&mut self) {
        self.pinned = true;
    }

    pub(super) fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
        self.pinned = false;
    }

    /// Scrolling down never re-pins by itself; `draw` re-pins once the view
    /// reaches the bottom.
    pub(super) fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines);
    }

    pub(super) fn draw(&mut self, f: &mut Frame, area: Rect, theme: Color) {
        if self.cached_width != area.width {
            self.cached_lines = render::build_lines(&self.items, area.width, theme);
            self.cached_width = area.width;
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

        let block = Block::default()
            .borders(Borders::NONE)
            .title_bottom(Line::from(Span::styled(
                scroll_hint,
                Style::default().fg(theme),
            )));
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(Paragraph::new(visible), inner);
    }
}

/// The prompt being typed. `cursor` is a byte offset that always sits on a
/// character boundary.
#[derive(Default)]
pub(super) struct InputLine {
    text: String,
    cursor: usize,
}

impl InputLine {
    pub(super) fn text(&self) -> &str {
        &self.text
    }

    pub(super) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(super) fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub(super) fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub(super) fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.text.floor_char_boundary(self.cursor - 1);
            self.text.drain(prev..self.cursor);
            self.cursor = prev;
        }
    }

    pub(super) fn delete(&mut self) {
        if self.cursor < self.text.len() {
            let next = self.text.ceil_char_boundary(self.cursor + 1);
            self.text.drain(self.cursor..next);
        }
    }

    pub(super) fn left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.text.floor_char_boundary(self.cursor - 1);
        }
    }

    pub(super) fn right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor = self.text.ceil_char_boundary(self.cursor + 1);
        }
    }

    pub(super) fn home(&mut self) {
        self.cursor = 0;
    }

    pub(super) fn end(&mut self) {
        self.cursor = self.text.len();
    }
}

/// The popup currently shown over the base screen, carrying its own data.
/// Closing it (Esc, or picking an entry) always returns to the base screen.
pub(super) enum Overlay {
    ModelPicker {
        /// `(provider, model)` pairs.
        items: Vec<(String, String)>,
        selected: usize,
    },
    SessionPicker {
        sessions: Vec<ConversationMeta>,
        selected: usize,
    },
    ConfigViewer {
        rows: Vec<ConfigRow>,
        scroll: usize,
    },
    McpViewer {
        rows: Vec<ConfigRow>,
        scroll: usize,
    },
    SkillsViewer {
        items: Vec<String>,
        scroll: usize,
    },
    ThemePicker {
        selected: usize,
    },
}

/// Tool-call approvals waiting on the user, shown one at a time on top of
/// everything else. A queue rather than a single slot: tool calls are checked
/// concurrently (see `run_agent`), so several requests can arrive before the
/// first is answered; later ones wait their turn.
#[derive(Default)]
pub(super) struct PermissionQueue {
    pending: VecDeque<PendingPermission>,
    /// Highlighted option in [`PERMISSION_OPTIONS`] for the front request.
    selected: usize,
}

impl PermissionQueue {
    pub(super) fn push(&mut self, request: PendingPermission) {
        if self.pending.is_empty() {
            self.selected = 0;
        }
        self.pending.push_back(request);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.pending.len()
    }

    /// The request currently shown, if any.
    pub(super) fn front(&self) -> Option<&PendingPermission> {
        self.pending.front()
    }

    pub(super) fn selected(&self) -> usize {
        self.selected
    }

    pub(super) fn select_previous(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub(super) fn select_next(&mut self) {
        self.selected = (self.selected + 1).min(PERMISSION_OPTIONS.len() - 1);
    }

    pub(super) fn selected_decision(&self) -> PermissionDecision {
        PERMISSION_OPTIONS[self.selected].1
    }

    /// Sends `decision` for the front request and moves on to the next one.
    /// Returns the answered call's [`PendingPermission::describe`]. A failed
    /// send means the agent task already gave up waiting; the request is
    /// dropped either way.
    pub(super) fn resolve(&mut self, decision: PermissionDecision) -> Option<String> {
        let request = self.pending.pop_front()?;
        self.selected = 0;
        let description = request.describe();
        let _ = request.respond_to.send(decision);
        Some(description)
    }

    /// Drops requests nobody is waiting on any more (the agent task gave up,
    /// e.g. because the turn was cancelled) so a stale prompt isn't left on
    /// screen. The highlight resets only if the prompt on screen was one of
    /// them: this runs every frame, and moving the user's highlight (say,
    /// off "Reject") because a request *behind* it expired would make the
    /// next Enter do something they didn't choose.
    pub(super) fn prune_stale(&mut self) {
        let front_is_stale = self
            .pending
            .front()
            .is_some_and(|request| request.respond_to.is_closed());
        self.pending
            .retain(|request| !request.respond_to.is_closed());
        if front_is_stale {
            self.selected = 0;
        }
    }
}

/// The accent color and the name it was chosen by.
pub(super) struct Theme {
    pub(super) color: Color,
    pub(super) name: String,
}

impl Theme {
    pub(super) fn named(name: &str) -> Self {
        Self {
            color: render::theme_color(name),
            name: name.to_string(),
        }
    }
}

/// Requests from the UI to the agent task (see `tui::run`).
pub(super) struct AgentChannels {
    pub(super) prompt: mpsc::UnboundedSender<String>,
    /// `(provider, model)`.
    pub(super) switch_model: mpsc::UnboundedSender<(String, String)>,
    /// `(session id, cwd)`.
    pub(super) switch_session: mpsc::UnboundedSender<(String, std::path::PathBuf)>,
    pub(super) list_sessions: mpsc::UnboundedSender<()>,
    pub(super) new_session: mpsc::UnboundedSender<()>,
}
