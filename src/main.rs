use anyhow::Result;
use clap::Parser;
use reqwest::Client;
use tracing_subscriber::{fmt, EnvFilter};

use openheim::{
    api,
    cli::{self, Args},
    config::{init_config, load_config},
};

#[actix_web::main]
async fn main() -> Result<()> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt::Subscriber::builder().with_env_filter(env_filter).init();

    let args = Args::parse();

    if args.init {
        match init_config() {
            Ok(path) => {
                println!("Config file created at {}", path.display());
                println!("Edit it to configure your LLM providers.");
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    let app_config = match load_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    if args.list {
        print!("{}", app_config.list_models());
        return Ok(());
    }

    let mut agent_config = match app_config.resolve(args.model.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    if let Some(max_iter) = args.max_iterations {
        agent_config.max_iterations = max_iter;
    }

    let client = Client::new();

    if args.api_mode {
        api::start_api_server(args.host, args.port, client, agent_config, app_config).await?;
    } else if args.agent_mode {
        cli::run_agent_mode(&client, &agent_config).await?;
    } else {
        cli::run_single_prompt(&client, &agent_config, &args).await?;
    }

    Ok(())
}
