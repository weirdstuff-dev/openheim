use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;

use crate::{
    agent::{run_agent, run_agent_with_history},
    AgentConfig, AppConfig, Message,
    config::{resolve_client_and_config, create_client},
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

    #[arg(long, help = "Run in API server mode instead of CLI mode")]
    pub api_mode: bool,

    #[arg(long, help = "Run in continue mode for persistent conversations (CLI only)")]
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
}

pub async fn run_agent_mode(
    client: &Client,
    config: &AgentConfig,
    app_config: &AppConfig,
    model_name: Option<&str>,
    max_iterations: Option<usize>,
) -> Result<()> {
    tracing::info!("🔄 Starting continue mode - persistent conversation");
    tracing::info!("Type your messages and press Enter. Type 'exit', 'quit' or :q to end the conversation.\n");

    let (llm_client, resolved_config) = resolve_client_and_config(
        model_name,
        max_iterations,
        app_config,
        client,
        create_client(config, client),
        config,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    let config = resolved_config;
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());

    let mut messages: Vec<Message> = Vec::new();
    let mut rl = DefaultEditor::new()?;

    loop {
        let readline = rl.readline("You: ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                if input == "exit" || input == "quit" || input == ":q" {
                    println!("👋 Goodbye!");
                    break;
                }

                rl.add_history_entry(input)?;

                messages.push(Message::user(input.to_string()));

                match run_agent_with_history(
                    llm_client.clone(),
                    tool_executor.clone(),
                    &config,
                    &mut messages,
                    true,
                )
                .await
                {
                    Ok(result) => {
                        println!("\n=== Agent Response ===");
                        println!("{}", result.final_response);
                        println!("Iterations: {}\n", result.iterations_used);
                    }
                    Err(e) => {
                        eprintln!("❌ Error: {}", e);
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
    let prompt = args.query.as_ref().context(
        "Prompt is required in CLI mode. Use --prompt, --agent-mode, or switch to API mode with --api-mode"
    )?;

    let (llm_client, resolved_config) = resolve_client_and_config(
        model_name,
        max_iterations,
        app_config,
        client,
        create_client(config, client),
        config,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    let config = resolved_config;
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());

    let result = run_agent(
        llm_client,
        tool_executor,
        &config,
        prompt,
        true,
    )
    .await?;

    println!("\n=== Final Result ===");
    println!("{}", result.final_response);
    println!("\nCompleted in {} iterations", result.iterations_used);

    Ok(())
}
