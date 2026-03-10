use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use crate::{
    agent::run_agent_streaming_with_history,
    AgentConfig, AppConfig, LlmClient, Message, StreamEvent,
    config::{resolve_client_and_config, create_client},
    error::{Error, Result},
    rag::{Conversation, PromptBuilder, RagContext},
    tools::{SystemToolExecutor, ToolExecutor},
};

#[derive(Parser, Debug)]
#[command(name = "openheim")]
#[command(about = "A basic LLM agent", long_about = None)]
pub struct Args {
    #[arg(short, long, help = "The prompt to send to the LLM (CLI mode)")]
    pub query: Option<String>,

    #[arg(long, help = "Maximum number of agent iterations (overrides config)")]
    pub max_iterations: Option<usize>,

    #[arg(long, help = "Run in API server mode")]
    pub api_mode: bool,

    #[arg(long, help = "Run in agent mode on the CLI")]
    pub agent_mode: bool,

    #[arg(long, default_value = "0.0.0.0", help = "API server bind address")]
    pub host: String,

    #[arg(long, default_value = "8080", help = "API server port")]
    pub port: u16,

    #[arg(long, help = "Model name to use (must be configured in a provider)")]
    pub model: Option<String>,

    #[arg(long, help = "List all configured providers and models")]
    pub list: bool,

    #[arg(long, help = "Initialize config file at ~/.openheim/config.toml")]
    pub init: bool,

    #[arg(long, help = "Continue a specific conversation by its ID")]
    pub chat_id: Option<String>,

    #[arg(long, help = "Continue the most recent conversation")]
    pub continue_last: bool,

    #[arg(long, value_delimiter = ',', help = "Skills to activate (comma-separated)")]
    pub skills: Option<Vec<String>>,

    #[arg(long, help = "List all available skills")]
    pub list_skills: bool,
}

// Holds all resolved state needed to run an agent interaction.
struct SessionContext {
    llm_client: Arc<dyn LlmClient>,
    config: AgentConfig,
    tool_executor: Arc<dyn ToolExecutor>,
    rag: RagContext,
    conversation: Conversation,
    prompt_builder: PromptBuilder,
}

impl SessionContext {
    fn new(
        client: &Client,
        agent_config: &AgentConfig,
        app_config: &AppConfig,
        model_name: Option<&str>,
        max_iterations: Option<usize>,
        chat_id: Option<Uuid>,
        skill_names: Vec<String>,
    ) -> Result<Self> {
        let (llm_client, config) = resolve_client_and_config(
            model_name,
            max_iterations,
            app_config,
            create_client(agent_config, client),
            agent_config,
        )?;
        let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());
        let rag = RagContext::new()?;
        let (conversation, prompt_builder) = rag.prepare(
            chat_id,
            &skill_names,
            Some(config.model.clone()),
            Some(config.provider_name.clone()),
        )?;
        Ok(Self { llm_client, config, tool_executor, rag, conversation, prompt_builder })
    }
}

fn parse_chat_id(id_str: &str) -> Result<Uuid> {
    Uuid::parse_str(id_str)
        .map_err(|e| Error::ConfigError(format!("Invalid chat ID '{}': {}", id_str, e)))
}

fn resolve_chat_id(chat_id: Option<&str>, continue_last: bool) -> Result<Option<Uuid>> {
    if let Some(id_str) = chat_id {
        Ok(Some(parse_chat_id(id_str)?))
    } else if continue_last {
        let rag = RagContext::new()?;
        Ok(rag.history.get_last_conversation()?.map(|c| c.meta.id))
    } else {
        Ok(None)
    }
}

fn make_spinner() -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
            .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

fn make_event_handler(pb: ProgressBar) -> impl FnMut(StreamEvent) {
    move |event| match event {
        StreamEvent::IterationStart { .. } => {
            pb.set_message("thinking...");
        }
        StreamEvent::ToolCall { tool_name, arguments } => {
            let args_preview: String = arguments.chars().take(48).collect();
            let args_preview = if arguments.chars().count() > 48 {
                format!("{}…", args_preview)
            } else {
                arguments.clone()
            };
            pb.println(format!(
                "  {} {}  {}",
                "┌".dimmed(),
                tool_name.cyan().bold(),
                args_preview.dimmed()
            ));
            pb.set_message(format!("running {}...", tool_name));
        }
        StreamEvent::ToolResult { result, .. } => {
            let flat: String = result.chars().take(80).collect();
            let flat = flat.replace('\n', " ");
            let flat = flat.trim();
            let preview = if result.chars().count() > 80 {
                format!("{}…", flat)
            } else {
                flat.to_string()
            };
            pb.println(format!("  {} {}", "└".dimmed(), preview.dimmed()));
            pb.set_message("thinking...");
        }
        StreamEvent::LlmResponse { .. } | StreamEvent::Finished { .. } => {
            pb.finish_and_clear();
        }
    }
}

pub async fn run_agent_mode(
    client: &Client,
    config: &AgentConfig,
    app_config: &AppConfig,
    model_name: Option<&str>,
    max_iterations: Option<usize>,
    chat_id: Option<&str>,
    continue_last: bool,
    skill_names: Vec<String>,
) -> Result<()> {
    let chat_id = resolve_chat_id(chat_id, continue_last)?;
    let mut session = SessionContext::new(
        client, config, app_config, model_name, max_iterations, chat_id, skill_names,
    )?;

    let chat_id_short = &session.conversation.meta.id.to_string()[..8];
    let chat_status = if session.conversation.messages.is_empty() {
        "new".to_string()
    } else {
        format!("{} msgs", session.conversation.messages.len())
    };
    println!();
    println!(
        "  {} {}",
        "◆".truecolor(255, 140, 0),
        "openheim".bold()
    );
    println!(
        "  {}",
        format!(
            "{}  ·  {}  ·  chat {}  ·  {}",
            session.config.model, session.config.provider_name, chat_id_short, chat_status
        )
        .dimmed()
    );
    println!("  {}", "─".repeat(48).dimmed());
    println!();

    let mut rl = DefaultEditor::new()
        .map_err(|e| Error::Other(format!("Failed to initialize readline: {}", e)))?;

    loop {
        let readline = rl.readline(&format!("{} ", "you >".bold().green()));
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                if input == "exit" || input == "quit" || input == ":q" {
                    println!("  {}", "goodbye".dimmed());
                    break;
                }

                let _ = rl.add_history_entry(input);
                session.conversation.messages.push(Message::user(input.to_string()));

                let pb = make_spinner();
                pb.set_message("thinking...");
                let pb_cb = pb.clone();

                let result = run_agent_streaming_with_history(
                    session.llm_client.clone(),
                    session.tool_executor.clone(),
                    &session.config,
                    &mut session.conversation.messages,
                    Some(&session.prompt_builder),
                    make_event_handler(pb_cb),
                )
                .await;

                match result {
                    Ok(result) => {
                        if let Err(e) = session.rag.history.save_conversation(&session.conversation) {
                            tracing::warn!("Failed to save conversation: {e}");
                        }
                        println!("{} {}", "agent >".bold().truecolor(255, 140, 0), result.final_response);
                        println!(
                            "  {}",
                            format!("{} iterations  ·  saved", result.iterations_used).dimmed()
                        );
                        println!();
                    }
                    Err(e) => {
                        pb.finish_and_clear();
                        if let Err(e) = session.rag.history.save_conversation(&session.conversation) {
                            tracing::warn!("Failed to save conversation: {e}");
                        }
                        eprintln!("  {} {}", "error:".red().bold(), e);
                        println!();
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(err) => {
                eprintln!("  {} {:?}", "error:".red().bold(), err);
                break;
            }
        }
    }

    Ok(())
}

pub async fn run_single_prompt(
    client: &Client,
    config: &AgentConfig,
    app_config: &AppConfig,
    model_name: Option<&str>,
    max_iterations: Option<usize>,
    args: &Args,
) -> Result<()> {
    let prompt = args.query.as_ref().ok_or_else(|| {
        Error::ConfigError(
            "Prompt is required in CLI mode. Use --query, --agent-mode, or switch to API mode with --api-mode".to_string()
        )
    })?;

    let chat_id = args.chat_id.as_deref().map(parse_chat_id).transpose()?;
    let skill_names = args.skills.clone().unwrap_or_default();

    let mut session = SessionContext::new(
        client, config, app_config, model_name, max_iterations, chat_id, skill_names,
    )?;

    session.conversation.messages.push(Message::user(prompt.clone()));

    let pb = make_spinner();
    pb.set_message("thinking...");
    let pb_cb = pb.clone();

    let result = run_agent_streaming_with_history(
        session.llm_client,
        session.tool_executor,
        &session.config,
        &mut session.conversation.messages,
        Some(&session.prompt_builder),
        make_event_handler(pb_cb),
    )
    .await;

    pb.finish_and_clear();

    match result {
        Ok(result) => {
            if let Err(e) = session.rag.history.save_conversation(&session.conversation) {
                tracing::warn!("Failed to save conversation: {e}");
            }
            println!("{} {}", "agent >".bold().truecolor(255, 140, 0), result.final_response);
            println!(
                "  {}",
                format!(
                    "chat: {}  ·  {} iterations",
                    &session.conversation.meta.id.to_string()[..8],
                    result.iterations_used
                )
                .dimmed()
            );
        }
        Err(e) => {
            eprintln!("  {} {}", "error:".red().bold(), e);
            return Err(e);
        }
    }

    Ok(())
}
