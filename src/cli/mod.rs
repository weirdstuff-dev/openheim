use clap::Parser;
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    agent::run_agent_with_history,
    AgentConfig, AppConfig, Message,
    config::{resolve_client_and_config, create_client},
    error::{Error, Result},
    rag::RagContext,
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
    tracing::info!("Starting agent mode - persistent conversation");
    tracing::info!("Type your messages and press Enter. Type 'exit', 'quit' or :q to end the conversation.\n");

    let (llm_client, resolved_config) = resolve_client_and_config(
        model_name,
        max_iterations,
        app_config,
        client,
        create_client(config, client),
        config,
    )?;

    let config = resolved_config;
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());

    let rag = RagContext::new()?;

    let resolved_chat_id = if let Some(id_str) = chat_id {
        Some(
            Uuid::parse_str(id_str)
                .map_err(|e| Error::ConfigError(format!("Invalid chat ID '{}': {}", id_str, e)))?,
        )
    } else if continue_last {
        rag.history
            .get_last_conversation()?
            .map(|c| c.meta.id)
    } else {
        None
    };

    let (mut conversation, prompt_builder) = rag.prepare(
        resolved_chat_id,
        &skill_names,
        Some(config.model.clone()),
        Some(config.provider_name.clone()),
    )?;

    println!("Chat ID: {}", conversation.meta.id);
    if !conversation.messages.is_empty() {
        println!(
            "Loaded {} messages from previous conversation.",
            conversation.messages.len()
        );
    }

    let mut rl = DefaultEditor::new()
        .map_err(|e| Error::Other(format!("Failed to initialize readline: {}", e)))?;

    loop {
        let readline = rl.readline("You: ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                if input == "exit" || input == "quit" || input == ":q" {
                    println!("Goodbye!");
                    break;
                }

                let _ = rl.add_history_entry(input);

                conversation.messages.push(Message::user(input.to_string()));

                match run_agent_with_history(
                    llm_client.clone(),
                    tool_executor.clone(),
                    &config,
                    &mut conversation.messages,
                    true,
                    Some(&prompt_builder),
                )
                .await
                {
                    Ok(result) => {
                        if let Err(e) = rag.history.save_conversation(&conversation) {
                            tracing::warn!("Failed to save conversation: {e}");
                        }
                        println!("\n=== Agent Response ===");
                        println!("{}", result.final_response);
                        println!("Iterations: {}\n", result.iterations_used);
                    }
                    Err(e) => {
                        if let Err(e) = rag.history.save_conversation(&conversation) {
                            tracing::warn!("Failed to save conversation: {e}");
                        }
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("^D");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
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

    let (llm_client, resolved_config) = resolve_client_and_config(
        model_name,
        max_iterations,
        app_config,
        client,
        create_client(config, client),
        config,
    )?;

    let config = resolved_config;
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());

    let skill_names = args.skills.clone().unwrap_or_default();

    let rag = RagContext::new()?;

    let resolved_chat_id = if let Some(id_str) = &args.chat_id {
        Some(
            Uuid::parse_str(id_str)
                .map_err(|e| Error::ConfigError(format!("Invalid chat ID '{}': {}", id_str, e)))?,
        )
    } else {
        None
    };

    let (mut conversation, prompt_builder) = rag.prepare(
        resolved_chat_id,
        &skill_names,
        Some(config.model.clone()),
        Some(config.provider_name.clone()),
    )?;

    conversation.messages.push(Message::user(prompt.clone()));

    let result = run_agent_with_history(
        llm_client,
        tool_executor,
        &config,
        &mut conversation.messages,
        true,
        Some(&prompt_builder),
    )
    .await?;

    if let Err(e) = rag.history.save_conversation(&conversation) {
        tracing::warn!("Failed to save conversation: {e}");
    }

    println!("\n=== Final Result ===");
    println!("{}", result.final_response);
    println!("\nChat ID: {}", conversation.meta.id);
    println!("Completed in {} iterations", result.iterations_used);

    Ok(())
}
