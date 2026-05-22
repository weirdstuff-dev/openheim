use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use super::types::ChatItem;

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
    #[rustfmt::skip]
    const LOGO: &[&str] = &[
        "  ___  _ __   ___ _ __  | |__   ___(_)_ __ ___  ",
        r" / _ \| '_ \ / _ \ '_ \ | '_ \ / _ \ | '_ ` _ \ ",
        "| (_) | |_) |  __/ | | || | | |  __/ | | | | | |",
        r" \___/| .__/ \___|_| |_||_| |_|\___|_|_| |_| |_|",
        "      |_|                                        ",
    ];

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

    // logo + blank + subtitle + blank*2 + hint + blank + commands
    let content_h = LOGO.len() + 1 + 1 + 2 + 1 + 1 + COMMANDS.len();
    let top_pad = (area.height as usize).saturating_sub(content_h) / 2;
    let w = area.width as usize;

    let center = |text_w: usize| " ".repeat(w.saturating_sub(text_w) / 2);

    let mut lines: Vec<Line<'static>> = (0..top_pad).map(|_| Line::default()).collect();

    let logo_w = LOGO.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let logo_pad = center(logo_w);
    for &logo_line in LOGO {
        lines.push(Line::styled(
            format!("{logo_pad}{logo_line}"),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
    }

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

pub(crate) fn render_input_bar(f: &mut Frame, area: Rect, input: &str, cursor: usize) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let prompt_prefix = "  › ";
    f.render_widget(Paragraph::new(format!("{prompt_prefix}{input}")), inner);

    let cursor_col = inner.x
        + prompt_prefix.chars().count() as u16
        + input[..cursor].chars().count() as u16;
    f.set_cursor_position((
        cursor_col.min(inner.x + inner.width.saturating_sub(1)),
        inner.y,
    ));
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
