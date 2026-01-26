use anyhow::Result;
use clap::Parser;
use reqwest::Client;
use std::env;
use tracing_subscriber::{fmt, EnvFilter};

use openheim::{
    api,
    cli::{self, Args},
    AgentConfig,
};

#[actix_web::main]
async fn main() -> Result<()> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt::Subscriber::builder().with_env_filter(env_filter).init();

    let args = Args::parse();
    let client = Client::new();

    let api_base = env::var("OPENAI_API_BASE").unwrap_or_else(|_| args.api_base.clone());
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| args.api_key.clone());
    if api_key.trim().is_empty() {
        eprintln!("Error: OPENAI_API_KEY must be provided via environment variable OPENAI_API_KEY or the --api-key flag.");
        std::process::exit(1);
    }
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| args.model.clone());

    let config = AgentConfig::new(api_base, api_key, model, args.max_iterations);

    if args.api_mode {
        api::start_api_server(args.host, args.port, client, config).await?;
    } else if args.agent_mode {
        cli::run_agent_mode(&client, &config).await?;
    } else {
        cli::run_single_prompt(&client, &config, &args).await?;
    }

    Ok(())
}

