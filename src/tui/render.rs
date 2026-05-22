use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

use super::types::{ChatItem, ConfigRow};

pub(crate) fn build_lines(items: &[ChatItem], width: u16) -> Vec<Line<'static>> {
    let inner_w = width.saturating_sub(2) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();

    for item in items {
        match item {
            ChatItem::UserMessage(text) => {
                lines.extend(user_bubble(text, width));
            }
            ChatItem::AssistantMessage(text) => {
                for wl in word_wrap(text, inner_w) {
                    lines.push(Line::raw(format!("  {wl}")));
                }
                lines.push(Line::default());
            }
            ChatItem::ToolCall { name, args } => {
                let used = 4 + name.chars().count();
                let preview_w = inner_w.saturating_sub(used + 1);
                let preview: String = args.chars().take(preview_w).collect();
                let preview = if args.chars().count() > preview_w {
                    format!("{preview}…")
                } else {
                    preview
                };
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("⚙ ", Style::default().fg(Color::Cyan)),
                    Span::styled(
                        name.clone(),
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(preview, Style::default().fg(Color::DarkGray)),
                ]));
            }
            ChatItem::ToolResult { result, is_error } => {
                let flat: String =
                    result.chars().take(200).collect::<String>().replace('\n', " ");
                let style = if *is_error {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                lines.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled("→ ", style),
                    Span::styled(flat.trim().to_string(), style),
                ]));
            }
            ChatItem::SystemInfo(text) => {
                for line in text.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
                lines.push(Line::default());
            }
            ChatItem::Err(text) => {
                lines.push(Line::from(Span::styled(
                    format!("  error: {text}"),
                    Style::default().fg(Color::Red),
                )));
                lines.push(Line::default());
            }
        }
    }

    lines
}

// Right-aligned chat bubble for user messages.
//
//                             ╭──────────────────╮
//                             │ list files in    │
//                             │ src/             │
//                             ╰──────────────────╯
fn user_bubble(text: &str, width: u16) -> Vec<Line<'static>> {
    let content_max = 50usize.min(width.saturating_sub(8) as usize).max(1);

    let wrapped = word_wrap(text, content_max);
    let content_w = wrapped.iter().map(|l| l.chars().count()).max().unwrap_or(0).max(1);

    let border = Style::default().fg(Color::Green);
    let mut out: Vec<Line<'static>> = Vec::new();

    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("╭{}╮", "─".repeat(content_w + 2)), border),
    ]));

    for content_line in &wrapped {
        let gap = " ".repeat(content_w - content_line.chars().count());
        out.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│ ".to_string(), border),
            Span::raw(content_line.clone()),
            Span::raw(gap),
            Span::styled(" │".to_string(), border),
        ]));
    }

    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("╰{}╯", "─".repeat(content_w + 2)), border),
    ]));

    out.push(Line::default());
    out
}

pub(crate) fn render_welcome(
    f: &mut Frame,
    area: Rect,
    model: &str,
    provider: &str,
    skills: &[String],
) {
    const VERSION: &str = env!("CARGO_PKG_VERSION");

    const COMMANDS: &[(&str, &str)] = &[
        (":help", "show all commands"),
        (":config", "current config"),
        (":sessions", "past sessions"),
        (":skills", "available skills"),
        (":mcp", "MCP servers"),
        (":q", "quit"),
    ];

    let subtitle = if skills.is_empty() {
        format!("{model}  ·  {provider}")
    } else {
        format!("{model}  ·  {provider}  ·  skills: {}", skills.join(", "))
    };
    let hint = "type a message to start";

    // title + blank + subtitle + blank*2 + hint + blank + commands
    let content_h = 1 + 1 + 1 + 2 + 1 + 1 + COMMANDS.len();
    let top_pad = (area.height as usize).saturating_sub(content_h) / 2;
    let w = area.width as usize;

    let center = |text_w: usize| " ".repeat(w.saturating_sub(text_w) / 2);

    let title = format!("openheim  v{VERSION}");
    let title_pad = center(title.chars().count());
    let mut lines: Vec<Line<'static>> = (0..top_pad).map(|_| Line::default()).collect();
    lines.push(Line::from(vec![
        Span::raw(title_pad),
        Span::styled(
            "openheim".to_string(),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  v{VERSION}"),
            Style::default().fg(Color::DarkGray),
        ),
    ]));

    lines.push(Line::default());

    let sub_pad = center(subtitle.chars().count());
    lines.push(Line::styled(
        format!("{sub_pad}{subtitle}"),
        Style::default().fg(Color::DarkGray),
    ));

    lines.push(Line::default());
    lines.push(Line::default());

    let hint_pad = center(hint.chars().count());
    lines.push(Line::styled(
        format!("{hint_pad}{hint}"),
        Style::default().fg(Color::DarkGray),
    ));

    lines.push(Line::default());

    let cmd_key_w = COMMANDS.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(0);
    let cmd_desc_w = COMMANDS.iter().map(|(_, d)| d.chars().count()).max().unwrap_or(0);
    let cmd_block_w = cmd_key_w + 6 + cmd_desc_w;
    let cmd_pad = center(cmd_block_w);

    for &(key, desc) in COMMANDS {
        let gap = " ".repeat(cmd_key_w - key.chars().count() + 6);
        lines.push(Line::from(vec![
            Span::raw(cmd_pad.clone()),
            Span::styled(key.to_string(), Style::default().fg(Color::White)),
            Span::raw(gap),
            Span::styled(desc.to_string(), Style::default().fg(Color::DarkGray)),
        ]));
    }

    f.render_widget(Paragraph::new(lines), area);
}

pub(crate) fn render_input_bar(
    f: &mut Frame,
    area: Rect,
    input: &str,
    cursor: usize,
    left_label: Option<&str>,
    right_label: &str,
    show_cursor: bool,
) {
    let dim = Style::default().fg(Color::DarkGray);
    let mut block = Block::default().borders(Borders::TOP).border_style(dim);

    if let Some(left) = left_label {
        block = block
            .title_top(Line::from(Span::styled(format!("─── {left} "), dim)).left_aligned());
    }
    block = block.title_top(
        Line::from(Span::styled(format!(" {right_label} ───"), dim)).right_aligned(),
    );

    let inner = block.inner(area);
    f.render_widget(block, area);

    let prompt_prefix = "  › ";
    f.render_widget(Paragraph::new(format!("{prompt_prefix}{input}")), inner);

    if show_cursor {
        let cursor_col = inner.x
            + prompt_prefix.chars().count() as u16
            + input[..cursor].chars().count() as u16;
        f.set_cursor_position((
            cursor_col.min(inner.x + inner.width.saturating_sub(1)),
            inner.y,
        ));
    }
}

pub(crate) fn render_model_picker(
    f: &mut Frame,
    area: Rect,
    items: &[(String, String)],
    selected: usize,
) {
    let max_label = items
        .iter()
        .map(|(p, m)| p.chars().count() + 2 + m.chars().count())
        .max()
        .unwrap_or(20);

    let popup_w = ((max_label + 6) as u16).max(32).min(area.width.saturating_sub(4));
    let popup_h = ((items.len() + 2) as u16).max(5).min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(popup_w)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_h)) / 2;
    let popup_rect = Rect::new(x, y, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(Color::DarkGray);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                " models ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ))
            .centered(),
        )
        .title_bottom(
            Line::from(Span::styled(" ↑/↓  enter  esc ", dim)).centered(),
        )
        .borders(Borders::ALL)
        .border_style(dim);

    let inner = block.inner(popup_rect);
    f.render_widget(block, popup_rect);

    let visible_h = inner.height as usize;
    if visible_h == 0 {
        return;
    }
    let start = selected.saturating_sub(visible_h.saturating_sub(1));
    let end = (start + visible_h).min(items.len());
    let start = start.min(end);
    let inner_w = inner.width as usize;

    let lines: Vec<Line<'static>> = items[start..end]
        .iter()
        .enumerate()
        .map(|(i, (provider, model))| {
            let idx = start + i;
            if idx == selected {
                let label = format!("  {provider}  {model}");
                let truncated: String = label.chars().take(inner_w).collect();
                let padding = " ".repeat(inner_w.saturating_sub(truncated.chars().count()));
                Line::styled(
                    format!("{truncated}{padding}"),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(provider.clone(), Style::default().fg(Color::DarkGray)),
                    Span::raw("  "),
                    Span::styled(model.clone(), Style::default().fg(Color::White)),
                ])
            }
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

pub(crate) fn render_config_viewer(
    f: &mut Frame,
    area: Rect,
    rows: &[ConfigRow],
    scroll: usize,
) {
    let entry_key_w = rows
        .iter()
        .filter_map(|r| {
            if let ConfigRow::Entry { key, .. } = r { Some(key.chars().count()) } else { None }
        })
        .max()
        .unwrap_or(10);
    let entry_val_w = rows
        .iter()
        .filter_map(|r| {
            if let ConfigRow::Entry { val, .. } = r { Some(val.chars().count()) } else { None }
        })
        .max()
        .unwrap_or(10);
    let item_w = rows
        .iter()
        .filter_map(|r| {
            if let ConfigRow::Item(s) = r { Some(s.chars().count() + 4) } else { None }
        })
        .max()
        .unwrap_or(0);
    let header_w = rows
        .iter()
        .filter_map(|r| {
            if let ConfigRow::Header(h) = r { Some(h.chars().count() + 2) } else { None }
        })
        .max()
        .unwrap_or(0);
    let content_w = (entry_key_w + 4 + entry_val_w).max(item_w).max(header_w);

    let popup_w = ((content_w + 6) as u16).max(36).min(area.width.saturating_sub(4));
    let popup_h = ((rows.len() + 2) as u16).max(6).min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(popup_w)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_h)) / 2;
    let popup_rect = Rect::new(x, y, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(Color::DarkGray);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                " config ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ))
            .centered(),
        )
        .title_bottom(Line::from(Span::styled(" ↑/↓  esc ", dim)).centered())
        .borders(Borders::ALL)
        .border_style(dim);

    let inner = block.inner(popup_rect);
    f.render_widget(block, popup_rect);

    let visible_h = inner.height as usize;
    if visible_h == 0 {
        return;
    }
    let scroll = scroll.min(rows.len().saturating_sub(visible_h));
    let end = (scroll + visible_h).min(rows.len());

    let lines: Vec<Line<'static>> = rows[scroll..end]
        .iter()
        .map(|row| match row {
            ConfigRow::Blank => Line::default(),
            ConfigRow::Header(h) => Line::from(Span::styled(
                format!("  {h}"),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            ConfigRow::Entry { key, val } => {
                let gap = " ".repeat(entry_key_w.saturating_sub(key.chars().count()) + 2);
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(key.clone(), Style::default().fg(Color::DarkGray)),
                    Span::raw(gap),
                    Span::styled(val.clone(), Style::default().fg(Color::White)),
                ])
            }
            ConfigRow::Item(s) => Line::from(vec![
                Span::raw("    "),
                Span::styled(s.clone(), Style::default().fg(Color::White)),
            ]),
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

// Word-wraps `text` to `width` chars, preserving newlines as paragraph breaks.
pub(crate) fn word_wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return text.lines().map(String::from).collect();
    }
    let mut out = Vec::new();
    for para in text.split('\n') {
        if para.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut current_len = 0usize;
        for word in para.split_whitespace() {
            let wlen = word.chars().count();
            if current.is_empty() {
                current.push_str(word);
                current_len = wlen;
            } else if current_len + 1 + wlen <= width {
                current.push(' ');
                current.push_str(word);
                current_len += 1 + wlen;
            } else {
                out.push(current);
                current = word.to_string();
                current_len = wlen;
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}
