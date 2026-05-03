use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use openheim::{
    config::init_config,
    transport::{run, stdio},
};

#[derive(Parser, Debug)]
#[command(name = "openheim", about = "AI agent")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Serve as an ACP agent over stdio (for Zed, Claude Code, etc.)
    Acp,
    /// Run a single prompt headlessly and stream output to stdout
    Run {
        /// Prompt to send to the agent
        prompt: String,
        /// Model name override (must be configured in a provider)
        #[arg(long)]
        model: Option<String>,
    },
    /// Start WebSocket/ACP server
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value = "1217")]
        port: u16,
    },
    /// Initialize config file at ~/.openheim/config.toml
    Init,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt::Subscriber::builder().with_env_filter(env_filter).init();

    let cli = Cli::parse();

    match cli.command {
        None => {
            eprintln!("TUI not yet implemented (Phase 4)");
            std::process::exit(1);
        }
        Some(Command::Acp) => {
            if let Err(e) = stdio::run().await {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Some(Command::Run { prompt, model }) => {
            if let Err(e) = run::run_headless(prompt, model).await {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Some(Command::Serve { .. }) => {
            eprintln!("WebSocket server not yet implemented (Phase 3)");
            std::process::exit(1);
        }
        Some(Command::Init) => {
            match init_config() {
                Ok(path) => {
                    println!("Config file created at {}", path.display());
                    println!("Edit it to configure your LLM providers.");
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}
