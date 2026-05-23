use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

use super::types::{ChatItem, ConfigRow};

pub(crate) const THEME_COLORS: &[&str] = &[
    "white", "gray", "blue", "cyan", "magenta", "green", "yellow", "red", "pink",
];

pub(crate) fn theme_color(name: &str) -> Color {
    match name {
        "white" => Color::White,
        "gray" => Color::DarkGray,
        "blue" => Color::Blue,
        "cyan" => Color::Cyan,
        "magenta" => Color::LightMagenta,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "red" => Color::Red,
        "pink" => Color::LightRed,
        _ => Color::White,
    }
}

fn centered_popup(area: Rect, w: u16, h: u16) -> Rect {
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect::new(x, y, w, h)
}

fn highlight_row(label: &str, inner_w: usize) -> Line<'static> {
    let truncated: String = label.chars().take(inner_w).collect();
    let padding = " ".repeat(inner_w.saturating_sub(truncated.chars().count()));
    Line::styled(
        format!("{truncated}{padding}"),
        Style::default()
            .fg(Color::Black)
            .bg(Color::White)
            .add_modifier(Modifier::BOLD),
    )
}

pub(crate) fn build_lines(items: &[ChatItem], width: u16, theme: Color) -> Vec<Line<'static>> {
    let inner_w = width.saturating_sub(2) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();

    for (idx, item) in items.iter().enumerate() {
        match item {
            ChatItem::UserMessage(text) => {
                lines.extend(user_bubble(text, width, theme));
            }
            ChatItem::AssistantMessage(text) => {
                for wl in word_wrap(text, inner_w) {
                    lines.push(Line::from(Span::styled(
                        format!("  {wl}"),
                        Style::default().fg(Color::White),
                    )));
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
                let call_color = match items.get(idx + 1) {
                    Some(ChatItem::ToolResult {
                        is_error: false, ..
                    }) => Color::Green,
                    Some(ChatItem::ToolResult { is_error: true, .. }) => Color::Red,
                    _ => theme,
                };
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("⚙ ", Style::default().fg(call_color)),
                    Span::styled(
                        name.clone(),
                        Style::default().fg(call_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(preview, Style::default().fg(Color::DarkGray)),
                ]));
            }
            ChatItem::ToolResult { result, .. } => {
                let flat: String = result
                    .chars()
                    .take(200)
                    .map(|c| if c == '\n' { ' ' } else { c })
                    .collect();
                lines.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled("→ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        flat.trim().to_string(),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
            ChatItem::SystemInfo(text) => {
                for line in text.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(theme),
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
fn user_bubble(text: &str, width: u16, theme: Color) -> Vec<Line<'static>> {
    let content_max = 50usize.min(width.saturating_sub(8) as usize).max(1);

    let wrapped = word_wrap(text, content_max);
    let content_w = wrapped
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0)
        .max(1);

    let border = Style::default().fg(theme);
    let text_style = Style::default().fg(theme);
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
            Span::styled(content_line.clone(), text_style),
            Span::styled(gap, text_style),
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
    theme: Color,
) {
    const VERSION: &str = env!("CARGO_PKG_VERSION");

    const COMMANDS: &[(&str, &str)] = &[
        (":help", "show all commands"),
        (":config", "current config"),
        (":sessions", "past sessions"),
        (":skills", "available skills"),
        (":mcp", "MCP servers"),
        (":theme", "change accent color"),
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
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  v{VERSION}"), Style::default().fg(theme)),
    ]));

    lines.push(Line::default());

    let sub_pad = center(subtitle.chars().count());
    lines.push(Line::styled(
        format!("{sub_pad}{subtitle}"),
        Style::default().fg(theme),
    ));

    lines.push(Line::default());
    lines.push(Line::default());

    let hint_pad = center(hint.chars().count());
    lines.push(Line::styled(
        format!("{hint_pad}{hint}"),
        Style::default().fg(theme),
    ));

    lines.push(Line::default());

    let cmd_key_w = COMMANDS
        .iter()
        .map(|(k, _)| k.chars().count())
        .max()
        .unwrap_or(0);
    let cmd_desc_w = COMMANDS
        .iter()
        .map(|(_, d)| d.chars().count())
        .max()
        .unwrap_or(0);
    let cmd_block_w = cmd_key_w + 6 + cmd_desc_w;
    let cmd_pad = center(cmd_block_w);

    for &(key, desc) in COMMANDS {
        let gap = " ".repeat(cmd_key_w - key.chars().count() + 6);
        lines.push(Line::from(vec![
            Span::raw(cmd_pad.clone()),
            Span::styled(key.to_string(), Style::default().fg(Color::White)),
            Span::raw(gap),
            Span::styled(desc.to_string(), Style::default().fg(theme)),
        ]));
    }

    f.render_widget(Paragraph::new(lines), area);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_input_bar(
    f: &mut Frame,
    area: Rect,
    input: &str,
    cursor: usize,
    left_label: Option<&str>,
    right_label: &str,
    show_cursor: bool,
    theme: Color,
) {
    let dim = Style::default().fg(theme);
    let mut block = Block::default().borders(Borders::TOP).border_style(dim);

    if let Some(left) = left_label {
        block =
            block.title_top(Line::from(Span::styled(format!("─── {left} "), dim)).left_aligned());
    }
    block = block
        .title_top(Line::from(Span::styled(format!(" {right_label} ───"), dim)).right_aligned());

    let inner = block.inner(area);
    f.render_widget(block, area);

    let prompt_prefix = "  › ";
    f.render_widget(
        Paragraph::new(format!("{prompt_prefix}{input}")).style(Style::default().fg(Color::White)),
        inner,
    );

    if show_cursor {
        let cursor_col =
            inner.x + prompt_prefix.chars().count() as u16 + input[..cursor].chars().count() as u16;
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
    theme: Color,
) {
    let max_label = items
        .iter()
        .map(|(p, m)| p.chars().count() + 2 + m.chars().count())
        .max()
        .unwrap_or(20);

    let popup_w = ((max_label + 6) as u16)
        .max(32)
        .min(area.width.saturating_sub(4));
    let popup_h = ((items.len() + 2) as u16)
        .max(5)
        .min(area.height.saturating_sub(4));
    let popup_rect = centered_popup(area, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(theme);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                " models ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
            .centered(),
        )
        .title_bottom(Line::from(Span::styled(" ↑/↓  enter  esc ", dim)).centered())
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
                highlight_row(&format!("  {provider}  {model}"), inner_w)
            } else {
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(provider.clone(), Style::default().fg(theme)),
                    Span::raw("  "),
                    Span::styled(model.clone(), Style::default().fg(Color::White)),
                ])
            }
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

pub(crate) fn render_session_picker(
    f: &mut Frame,
    area: Rect,
    items: &[crate::rag::ConversationMeta],
    selected: usize,
    theme: Color,
) {
    if items.is_empty() {
        return;
    }

    let max_title = items
        .iter()
        .map(|m| m.title.as_deref().unwrap_or("(untitled)").chars().count())
        .max()
        .unwrap_or(10);

    let max_meta = items
        .iter()
        .map(|m| {
            let date = m.updated_at.format("%Y-%m-%d %H:%M").to_string();
            let model = m.model.as_deref().unwrap_or("?");
            date.len() + 5 + model.len()
        })
        .max()
        .unwrap_or(20);

    let content_w = max_title + 4 + max_meta;
    let popup_w = ((content_w + 6) as u16)
        .max(40)
        .min(area.width.saturating_sub(4));
    let popup_h = ((items.len() + 2) as u16)
        .max(5)
        .min(area.height.saturating_sub(4));
    let popup_rect = centered_popup(area, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(theme);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                " sessions ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
            .centered(),
        )
        .title_bottom(Line::from(Span::styled(" ↑/↓  enter  esc ", dim)).centered())
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
        .map(|(i, meta)| {
            let idx = start + i;
            let title = meta.title.as_deref().unwrap_or("(untitled)").to_string();
            let date = meta.updated_at.format("%Y-%m-%d %H:%M").to_string();
            let model = meta.model.as_deref().unwrap_or("?").to_string();
            let meta_str = format!("{date}  ·  {model}");

            if idx == selected {
                highlight_row(&format!("  {title}  {meta_str}"), inner_w)
            } else {
                let gap = " ".repeat(max_title.saturating_sub(title.chars().count()) + 4);
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(title, Style::default().fg(Color::White)),
                    Span::raw(gap),
                    Span::styled(meta_str, Style::default().fg(theme)),
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
    theme: Color,
) {
    render_rows_popup(f, area, rows, scroll, " config ", 36, "  ", theme);
}

pub(crate) fn render_mcp_viewer(
    f: &mut Frame,
    area: Rect,
    rows: &[ConfigRow],
    scroll: usize,
    theme: Color,
) {
    render_rows_popup(f, area, rows, scroll, " mcp servers ", 40, "    ", theme);
}

#[allow(clippy::too_many_arguments)]
fn render_rows_popup(
    f: &mut Frame,
    area: Rect,
    rows: &[ConfigRow],
    scroll: usize,
    title: &'static str,
    min_popup_w: u16,
    entry_prefix: &'static str,
    theme: Color,
) {
    let (entry_key_w, entry_val_w, item_w, header_w) =
        rows.iter()
            .fold((0, 0, 0, 0), |(ek, ev, iw, hw), row| match row {
                ConfigRow::Entry { key, val } => (
                    ek.max(key.chars().count()),
                    ev.max(val.chars().count()),
                    iw,
                    hw,
                ),
                ConfigRow::Item(s) => (ek, ev, iw.max(s.chars().count() + 4), hw),
                ConfigRow::Header(h) => (ek, ev, iw, hw.max(h.chars().count() + 2)),
                ConfigRow::Blank => (ek, ev, iw, hw),
            });

    let content_w = (entry_key_w + 4 + entry_val_w).max(item_w).max(header_w);
    let popup_w = ((content_w + 6) as u16)
        .max(min_popup_w)
        .min(area.width.saturating_sub(4));
    let popup_h = ((rows.len() + 2) as u16)
        .max(6)
        .min(area.height.saturating_sub(4));
    let popup_rect = centered_popup(area, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(theme);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                title,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
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
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            ConfigRow::Entry { key, val } => {
                let gap = " ".repeat(entry_key_w.saturating_sub(key.chars().count()) + 2);
                Line::from(vec![
                    Span::raw(entry_prefix),
                    Span::styled(key.clone(), Style::default().fg(theme)),
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

pub(crate) fn render_skills_viewer(
    f: &mut Frame,
    area: Rect,
    items: &[String],
    scroll: usize,
    theme: Color,
) {
    let max_w = items.iter().map(|s| s.chars().count()).max().unwrap_or(10);
    let content_w = max_w + 4;
    let popup_w = ((content_w + 6) as u16)
        .max(36)
        .min(area.width.saturating_sub(4));
    let popup_h = ((items.len() + 4) as u16)
        .max(6)
        .min(area.height.saturating_sub(4));
    let popup_rect = centered_popup(area, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(theme);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                " skills ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
            .centered(),
        )
        .title_bottom(
            Line::from(Span::styled(
                " activate: openheim --skills <name>,... · ↑/↓  esc ",
                dim,
            ))
            .centered(),
        )
        .borders(Borders::ALL)
        .border_style(dim);

    let inner = block.inner(popup_rect);
    f.render_widget(block, popup_rect);

    let visible_h = inner.height as usize;
    if visible_h == 0 || items.is_empty() {
        return;
    }
    let scroll = scroll.min(items.len().saturating_sub(visible_h));
    let end = (scroll + visible_h).min(items.len());

    let lines: Vec<Line<'static>> = items[scroll..end]
        .iter()
        .map(|name| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(name.clone(), Style::default().fg(Color::White)),
            ])
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

pub(crate) fn render_theme_picker(
    f: &mut Frame,
    area: Rect,
    selected: usize,
    current_name: &str,
    theme: Color,
) {
    let max_w = THEME_COLORS.iter().map(|n| n.len()).max().unwrap_or(10);
    let popup_w = ((max_w + 8) as u16)
        .max(24)
        .min(area.width.saturating_sub(4));
    let popup_h = (THEME_COLORS.len() as u16 + 2)
        .max(5)
        .min(area.height.saturating_sub(4));
    let popup_rect = centered_popup(area, popup_w, popup_h);

    f.render_widget(Clear, popup_rect);

    let dim = Style::default().fg(theme);
    let block = Block::default()
        .title(
            Line::from(Span::styled(
                " theme ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
            .centered(),
        )
        .title_bottom(Line::from(Span::styled(" ↑/↓  enter  esc ", dim)).centered())
        .borders(Borders::ALL)
        .border_style(dim);

    let inner = block.inner(popup_rect);
    f.render_widget(block, popup_rect);

    if inner.height == 0 {
        return;
    }

    let lines: Vec<Line<'static>> = THEME_COLORS
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let color = theme_color(name);
            let is_selected = i == selected;
            let is_current = name == current_name;
            let marker = if is_current { "·" } else { " " };
            if is_selected {
                Line::from(vec![
                    Span::styled("> ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{name} {marker}"),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("{name} {marker}"), Style::default().fg(color)),
                ])
            }
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

fn word_wrap(text: &str, width: usize) -> Vec<String> {
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
