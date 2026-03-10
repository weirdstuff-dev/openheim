use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use openheim::{
    api,
    cli::{self, Args},
    config::{build_http_client, init_config, load_config},
    rag::SkillsManager,
};

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.init {
        let path = init_config()?;
        println!("Config file created at {}", path.display());
        println!("Edit it to configure your LLM providers.");
        return Ok(());
    }

    let app_config = load_config()?;

    if args.list {
        print!("{}", app_config.list_models()?);
        return Ok(());
    }

    if args.list_skills {
        let mgr = SkillsManager::new()?;
        let skills = mgr.list_skills()?;
        if skills.is_empty() {
            println!("No skills found. Add .md files to ~/.openheim/skills/");
        } else {
            println!("Available skills:");
            for name in &skills {
                println!("  - {}", name);
            }
        }
        return Ok(());
    }

    let agent_config = app_config.resolve(None)?;
    let client = build_http_client(agent_config.timeout_secs);

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

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt::Subscriber::builder().with_env_filter(env_filter).init();

    let args = Args::parse();

    if let Err(e) = run(args).await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
