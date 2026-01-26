use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;

use crate::{
    agent::{run_agent, run_agent_with_history},
    AgentConfig, Message,
    llm::{LlmClient, OpenAiCompatibleClient},
    tools::{SystemToolExecutor, ToolExecutor},
};

#[derive(Parser, Debug)]
#[command(name = "openheim")]
#[command(about = "A basic LLM agent", long_about = None)]
pub struct Args {
    #[arg(short, long, help = "The prompt to send to the LLM (CLI mode)")]
    pub query: Option<String>,

    #[arg(long, default_value = "10", help = "Maximum number of agent iterations")]
    pub max_iterations: usize,

    #[arg(long, help = "Run in API server mode instead of CLI mode")]
    pub api_mode: bool,

    #[arg(long, help = "Run in continue mode for persistent conversations (CLI only)")]
    pub agent_mode: bool,

    #[arg(long, default_value = "0.0.0.0", help = "API server bind address")]
    pub host: String,

    #[arg(long, default_value = "8080", help = "API server port")]
    pub port: u16,

    #[arg(long, default_value = "", help = "LLM API base URL (or set OPENAI_API_BASE)")]
    pub api_base: String,

    #[arg(long, default_value = "", help = "LLM API key (or set OPENAI_API_KEY)")]
    pub api_key: String,

    #[arg(long, default_value = "", help = "LLM model name (or set OPENAI_MODEL)")]
    pub model: String,
}

pub async fn run_agent_mode(client: &Client, config: &AgentConfig) -> Result<()> {
    tracing::info!("🔄 Starting continue mode - persistent conversation");
    tracing::info!("Type your messages and press Enter. Type 'exit', 'quit' or :q to end the conversation.\n");

    let llm_client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatibleClient::new(
        client.clone(),
        config.api_base.clone(),
        config.api_key.clone(),
        config.model.clone(),
    ));
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
                    config,
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

pub async fn run_single_prompt(client: &Client, config: &AgentConfig, args: &Args) -> Result<()> {
    let prompt = args.query.as_ref().context(
        "Prompt is required in CLI mode. Use --prompt, --agent-mode, or switch to API mode with --api-mode"
    )?;

    let llm_client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatibleClient::new(
        client.clone(),
        config.api_base.clone(),
        config.api_key.clone(),
        config.model.clone(),
    ));
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(SystemToolExecutor::new());

    let result = run_agent(
        llm_client,
        tool_executor,
        config,
        prompt,
        true,
    )
    .await?;

    println!("\n=== Final Result ===");
    println!("{}", result.final_response);
    println!("\nCompleted in {} iterations", result.iterations_used);

    Ok(())
}
