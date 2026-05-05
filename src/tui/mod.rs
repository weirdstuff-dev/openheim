use std::io::{Write, stdout};
use std::sync::Arc;

use agent_client_protocol::{
    ByteStreams, Client,
    schema::{
        ContentBlock, InitializeRequest, NewSessionRequest, ProtocolVersion, SessionNotification,
        SessionUpdate, ToolCallStatus,
    },
    util::MatchDispatch,
    SessionMessage,
};
use colored::Colorize;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::{
    acp::{self, AgentState},
    config::{AgentConfig, AppConfig, load_config},
    rag::{ConversationMeta, RagContext, SkillsManager},
};

#[derive(Debug, Clone)]
enum AgentUpdate {
    TextChunk(String),
    ToolCall { name: String, args: String },
    ToolResult(String),
    Done,
    Error(String),
}

pub async fn run(skills: Vec<String>) -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;
    let rag = RagContext::new()?;
    let state = Arc::new(AgentState::new(agent_config.clone(), app_config.clone(), rag).await?);

    let (prompt_tx, prompt_rx) = mpsc::channel::<String>(1);
    let (update_tx, mut update_rx) = mpsc::channel::<AgentUpdate>(64);

    let (server_half, client_half) = tokio::io::duplex(65536);
    let (server_read, server_write) = tokio::io::split(server_half);
    let (client_read, client_write) = tokio::io::split(client_half);

    let server_transport = ByteStreams::new(server_write.compat_write(), server_read.compat());
    let client_transport = ByteStreams::new(client_write.compat_write(), client_read.compat());

    tokio::spawn(acp::serve(server_transport, state));
    tokio::spawn(run_acp_client(client_transport, prompt_rx, update_tx, skills.clone()));

    let mut editor = DefaultEditor::new()
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;

    println!("{}", "openheim".yellow().bold());
    if skills.is_empty() {
        println!("{}", "type a message or :help for commands".dimmed());
    } else {
        println!("{}", "type a message or :help for commands".dimmed());
        println!("  {} {}", "skills:".dimmed(), skills.join(", ").yellow());
    }
    println!();

    let mut sessions: Vec<ConversationMeta> = Vec::new();

    loop {
        let line = tokio::task::block_in_place(|| editor.readline("› "));
        match line {
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => break,
            Err(e) => return Err(crate::error::Error::Other(e.to_string())),
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                if let Some(rest) = line.strip_prefix(':') {
                    let mut parts = rest.trim().splitn(2, ' ');
                    let cmd = parts.next().unwrap_or("");
                    let arg = parts.next().unwrap_or("").trim();
                    if handle_command(cmd, arg, &agent_config, &app_config, &mut sessions) {
                        break;
                    }
                } else {
                    let _ = editor.add_history_entry(&line);
                    prompt_tx.send(line).await.ok();
                    stream_response(&mut update_rx).await;
                }
            }
        }
    }

    Ok(())
}

fn handle_command(
    cmd: &str,
    arg: &str,
    agent_config: &AgentConfig,
    app_config: &AppConfig,
    sessions: &mut Vec<ConversationMeta>,
) -> bool {
    match cmd {
        "q" | "quit" => return true,
        "sessions" => {
            if let Ok(rag) = RagContext::new() {
                if let Ok(metas) = rag.history.list_conversations() {
                    if metas.is_empty() {
                        println!("{}", "  no sessions yet".dimmed());
                    } else {
                        println!();
                        for (i, meta) in metas.iter().enumerate() {
                            let title = meta.title.as_deref().unwrap_or("(untitled)");
                            let date = meta.updated_at.format("%Y-%m-%d %H:%M").to_string();
                            let model = meta.model.as_deref().unwrap_or("?");
                            println!(
                                "  {}  {}  ·  {}  ·  {}",
                                format!("{}", i + 1).dimmed(),
                                title,
                                date.dimmed(),
                                model.dimmed(),
                            );
                        }
                        *sessions = metas;
                        println!();
                        println!("{}", "  :open <n> to load a session".dimmed());
                    }
                    println!();
                }
            }
        }
        "open" => {
            if let Ok(n) = arg.parse::<usize>() {
                if n == 0 || n > sessions.len() {
                    println!(
                        "{}",
                        format!("  no session {n}  (run :sessions first)").red()
                    );
                } else {
                    open_session(&sessions[n - 1]);
                }
            } else {
                println!("{}", "  usage: :open <number>".dimmed());
            }
        }
        "config" => {
            println!();
            let k = |s: &str| format!("{:<20}", s).dimmed().to_string();
            println!("  {}  {}", k("Provider"), agent_config.provider_name);
            println!("  {}  {}", k("Model"), agent_config.model);
            println!("  {}  {}", k("Max iterations"), agent_config.max_iterations);
            println!("  {}  {}s", k("Timeout"), agent_config.timeout_secs);
            if !app_config.providers.is_empty() {
                println!();
                println!("  {}", "Providers".yellow());
                for (name, provider) in &app_config.providers {
                    let suffix = if name == &app_config.default_provider {
                        "  (default)"
                    } else {
                        ""
                    };
                    println!(
                        "    {}{}  {}",
                        name,
                        suffix,
                        provider.default_model.dimmed()
                    );
                }
            }
            if !app_config.mcp_servers.is_empty() {
                println!();
                println!("  {}", "MCP Servers".yellow());
                for name in app_config.mcp_servers.keys() {
                    println!("    {name}");
                }
            }
            println!();
            println!(
                "  {}  {}",
                "edit config:".dimmed(),
                "~/.openheim/config.toml"
            );
            println!();
        }
        "mcp" => {
            if app_config.mcp_servers.is_empty() {
                println!("{}", "  no MCP servers configured".dimmed());
                println!(
                    "{}",
                    "  add [mcp_servers.<name>] to ~/.openheim/config.toml".dimmed()
                );
            } else {
                for (name, server) in &app_config.mcp_servers {
                    println!();
                    println!("  {} {}", "●".green(), name.bold());
                    if let Some(cmd) = &server.command {
                        let args_str = server.args.join(" ");
                        let cmd_line = if args_str.is_empty() {
                            cmd.clone()
                        } else {
                            format!("{cmd} {args_str}")
                        };
                        println!("    {}  {}", "stdio".dimmed(), cmd_line.dimmed());
                    }
                    if let Some(url) = &server.url {
                        println!("    {}  {}", "http ".dimmed(), url.dimmed());
                    }
                    for (k, _) in &server.env {
                        println!("    {}  {}", "env  ".dimmed(), k.dimmed());
                    }
                }
            }
            println!();
        }
        "skills" => {
            match SkillsManager::new() {
                Ok(mgr) => match mgr.list_skills() {
                    Ok(names) if names.is_empty() => {
                        println!("{}", "  no skills available".dimmed());
                        println!(
                            "{}",
                            "  add <name>.md files to ~/.openheim/skills/".dimmed()
                        );
                    }
                    Ok(names) => {
                        println!();
                        for name in &names {
                            println!("  {}", name);
                        }
                        println!();
                        println!(
                            "{}",
                            "  activate with: openheim --skills <name>,<name>,...".dimmed()
                        );
                    }
                    Err(e) => println!("{}", format!("  error listing skills: {e}").red()),
                },
                Err(e) => println!("{}", format!("  error: {e}").red()),
            }
            println!();
        }
        "help" => {
            println!();
            let c = |s: &str| format!("{:<16}", s).bold().to_string();
            println!("  {}  show this help", c(":help"));
            println!("  {}  quit openheim", c(":q / :quit"));
            println!("  {}  list saved sessions", c(":sessions"));
            println!("  {}  load session n", c(":open <n>"));
            println!("  {}  show current config", c(":config"));
            println!("  {}  show MCP servers", c(":mcp"));
            println!("  {}  list available skills", c(":skills"));
            println!();
        }
        unknown => {
            println!("{}", format!(":{unknown}: unknown command  (try :help)").red());
        }
    }
    false
}

fn open_session(meta: &ConversationMeta) {
    use crate::core::models::Role;

    let title = meta.title.as_deref().unwrap_or("(untitled)");
    let divider = "─".repeat(52);
    println!();
    println!("  {}  {}", divider.dimmed(), title.bold());
    println!();

    if let Ok(rag) = RagContext::new() {
        if let Ok(conv) = rag.history.load_conversation(&meta.id) {
            for msg in &conv.messages {
                match msg.role {
                    Role::System => {}
                    Role::User => {
                        if let Some(content) = &msg.content {
                            if !content.is_empty() {
                                println!("  {}", "you".green().bold());
                                for line in content.split('\n') {
                                    println!("  {line}");
                                }
                                println!();
                            }
                        }
                    }
                    Role::Assistant => {
                        let mut printed_header = false;
                        if let Some(content) = &msg.content {
                            if !content.is_empty() {
                                println!("  {}", "openheim".yellow().bold());
                                printed_header = true;
                                for line in content.split('\n') {
                                    println!("  {line}");
                                }
                            }
                        }
                        if let Some(tool_calls) = &msg.tool_calls {
                            for tc in tool_calls {
                                if !printed_header {
                                    println!("  {}", "openheim".yellow().bold());
                                    printed_header = true;
                                }
                                let args = &tc.function.arguments;
                                let preview: String = args.chars().take(60).collect();
                                let preview = if args.chars().count() > 60 {
                                    format!("{preview}…")
                                } else {
                                    preview
                                };
                                println!(
                                    "  {} {} {}",
                                    "⚙".cyan(),
                                    tc.function.name.cyan().bold(),
                                    preview.dimmed()
                                );
                            }
                        }
                        if printed_header {
                            println!();
                        }
                    }
                    Role::Tool => {
                        if let Some(content) = &msg.content {
                            let flat: String = content
                                .chars()
                                .take(100)
                                .collect::<String>()
                                .replace('\n', " ");
                            let flat = flat.trim().to_string();
                            if !flat.is_empty() {
                                println!("    {} {}", "→".dimmed(), flat.dimmed());
                            }
                        }
                    }
                }
            }
        }
    }

    println!("  {}", divider.dimmed());
    println!();
}

async fn stream_response(update_rx: &mut mpsc::Receiver<AgentUpdate>) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop_flag);

    let spinner = tokio::spawn(async move {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut i = 0usize;
        tokio::time::sleep(Duration::from_millis(120)).await;
        while !stop_clone.load(Ordering::Relaxed) {
            print!("\r  {} {}", frames[i % frames.len()], "thinking…".dimmed());
            let _ = stdout().flush();
            i += 1;
            tokio::time::sleep(Duration::from_millis(80)).await;
        }
        print!("\r{}\r", " ".repeat(24));
        let _ = stdout().flush();
    });

    let mut agent_started = false;
    let mut spinner = Some(spinner);

    loop {
        let update = update_rx.recv().await;

        if let Some(handle) = spinner.take() {
            stop_flag.store(true, Ordering::Relaxed);
            handle.await.ok();
        }

        match update {
            Some(AgentUpdate::TextChunk(text)) => {
                if !agent_started {
                    print!("  {}  ", "openheim".yellow().bold());
                    agent_started = true;
                }
                print!("{text}");
                let _ = stdout().flush();
            }
            Some(AgentUpdate::ToolCall { name, args }) => {
                if !agent_started {
                    println!("  {}", "openheim".yellow().bold());
                    agent_started = true;
                } else {
                    println!();
                }
                let preview: String = args.chars().take(60).collect();
                let preview = if args.chars().count() > 60 {
                    format!("{preview}…")
                } else {
                    preview
                };
                println!(
                    "  {} {} {}",
                    "⚙".cyan(),
                    name.cyan().bold(),
                    preview.dimmed()
                );
            }
            Some(AgentUpdate::ToolResult(result)) => {
                let flat: String = result
                    .chars()
                    .take(100)
                    .collect::<String>()
                    .replace('\n', " ");
                println!("    {} {}", "→".dimmed(), flat.trim().dimmed());
            }
            Some(AgentUpdate::Done) | None => {
                if agent_started {
                    println!();
                }
                println!();
                break;
            }
            Some(AgentUpdate::Error(e)) => {
                if agent_started {
                    println!();
                }
                println!("  {} {}", "error".red().bold(), e);
                println!();
                break;
            }
        }
    }
}

async fn run_acp_client(
    transport: ByteStreams<
        tokio_util::compat::Compat<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
        tokio_util::compat::Compat<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    >,
    mut prompt_rx: mpsc::Receiver<String>,
    update_tx: mpsc::Sender<AgentUpdate>,
    skills: Vec<String>,
) {
    let error_tx = update_tx.clone();
    let result = Client
        .builder()
        .connect_with(transport, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let cwd = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            let request = if skills.is_empty() {
                NewSessionRequest::new(cwd)
            } else {
                let mut meta = serde_json::Map::new();
                meta.insert("skills".to_string(), serde_json::json!(skills));
                NewSessionRequest::new(cwd).meta(meta)
            };

            cx.build_session_from(request)
                .block_task()
                .run_until(async move |mut session| {
                    while let Some(prompt) = prompt_rx.recv().await {
                        session.send_prompt(&prompt)?;
                        loop {
                            match session.read_update().await? {
                                SessionMessage::StopReason(_) => {
                                    let _ = update_tx.send(AgentUpdate::Done).await;
                                    break;
                                }
                                SessionMessage::SessionMessage(dispatch) => {
                                    let tx = update_tx.clone();
                                    MatchDispatch::new(dispatch)
                                        .if_notification(
                                            async move |notif: SessionNotification| {
                                                match notif.update {
                                                    SessionUpdate::AgentMessageChunk(chunk) => {
                                                        if let ContentBlock::Text(t) = chunk.content
                                                        {
                                                            let _ = tx
                                                                .send(AgentUpdate::TextChunk(
                                                                    t.text,
                                                                ))
                                                                .await;
                                                        }
                                                    }
                                                    SessionUpdate::ToolCall(tc) => {
                                                        let args = tc
                                                            .raw_input
                                                            .as_ref()
                                                            .map(|v| v.to_string())
                                                            .unwrap_or_default();
                                                        let _ = tx
                                                            .send(AgentUpdate::ToolCall {
                                                                name: tc.title.clone(),
                                                                args,
                                                            })
                                                            .await;
                                                    }
                                                    SessionUpdate::ToolCallUpdate(tcu) => {
                                                        if matches!(
                                                            tcu.fields.status,
                                                            Some(ToolCallStatus::Completed)
                                                                | Some(ToolCallStatus::Failed)
                                                        ) {
                                                            let result = match tcu.fields.raw_output
                                                            {
                                                                Some(
                                                                    serde_json::Value::String(s),
                                                                ) => s,
                                                                Some(v) => v.to_string(),
                                                                None => String::new(),
                                                            };
                                                            let _ = tx
                                                                .send(AgentUpdate::ToolResult(
                                                                    result,
                                                                ))
                                                                .await;
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                                Ok(())
                                            },
                                        )
                                        .await
                                        .otherwise_ignore()?;
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(())
                })
                .await
        })
        .await;

    if let Err(e) = result {
        tracing::error!("TUI ACP client error: {e}");
        let _ = error_tx.send(AgentUpdate::Error(e.to_string())).await;
    }
}
