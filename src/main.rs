use clap::Parser;
use reqwest::Client;
use tracing_subscriber::{fmt, EnvFilter};

use openheim::{
    api,
    cli::{self, Args},
    config::{init_config, load_config},
    rag::SkillsManager,
};

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        match app_config.list_models() {
            Ok(output) => print!("{}", output),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    if args.list_skills {
        match SkillsManager::new() {
            Ok(mgr) => match mgr.list_skills() {
                Ok(skills) => {
                    if skills.is_empty() {
                        println!("No skills found. Add .md files to ~/.openheim/skills/");
                    } else {
                        println!("Available skills:");
                        for name in &skills {
                            println!("  - {}", name);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    let agent_config = match app_config.resolve(None) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let client = Client::new();

    if args.api_mode {
        api::start_api_server(args.host, args.port, client, agent_config, app_config).await?;
    } else if args.agent_mode {
        cli::run_agent_mode(
            &client,
            &agent_config,
            &app_config,
            args.model.as_deref(),
            args.max_iterations,
            args.chat_id.as_deref(),
            args.continue_last,
            args.skills.clone().unwrap_or_default(),
        )
        .await?;
    } else {
        cli::run_single_prompt(
            &client,
            &agent_config,
            &app_config,
            args.model.as_deref(),
            args.max_iterations,
            &args,
        )
        .await?;
    }

    Ok(())
}
