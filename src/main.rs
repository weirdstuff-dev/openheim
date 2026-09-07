use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use openheim::{
    OpenheimClient,
    config::init_config,
    transport::{run, stdio, ws},
    tui,
};

#[derive(Parser, Debug)]
#[command(name = "openheim", about = "AI agent")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Comma-separated skills to activate for this session (e.g. --skills rust,nodejs)
    #[arg(
        long = "skills",
        value_name = "NAMES",
        value_delimiter = ',',
        global = false
    )]
    skills: Vec<String>,
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

/// Builds the client every subcommand below runs against — the one place
/// this binary assembles config load, resolve, `MemoryContext::new`, and
/// `AgentState::new`, so each transport just takes the finished
/// `OpenheimClient` instead of hand-rolling its own build path. `model`
/// overrides the config's default model (only `openheim run` uses it).
async fn build_client(model: Option<String>) -> openheim::Result<OpenheimClient> {
    let mut builder = OpenheimClient::builder();
    if let Some(model) = model {
        builder = builder.model(model);
    }
    builder.build().await
}

/// Prints the error and exits(1) — the uniform failure path every
/// subcommand below shares, whether the failure is building the client or
/// running the transport.
fn die(e: impl std::fmt::Display) -> ! {
    eprintln!("Error: {e}");
    std::process::exit(1);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .init();

    let cli = Cli::parse();

    match cli.command {
        None => {
            let client = build_client(None).await.unwrap_or_else(|e| die(e));
            if let Err(e) = tui::run(client, cli.skills).await {
                die(e);
            }
        }
        Some(Command::Acp) => {
            let client = build_client(None).await.unwrap_or_else(|e| die(e));
            if let Err(e) = stdio::run(client).await {
                die(e);
            }
        }
        Some(Command::Run { prompt, model }) => {
            let client = build_client(model).await.unwrap_or_else(|e| die(e));
            if let Err(e) = run::run_headless(client, prompt).await {
                die(e);
            }
        }
        Some(Command::Serve { host, port }) => {
            let client = build_client(None).await.unwrap_or_else(|e| die(e));
            if let Err(e) = ws::serve(client, host, port).await {
                die(e);
            }
        }
        Some(Command::Init) => match init_config() {
            Ok(path) => {
                println!("Config file created at {}", path.display());
                println!("Edit it to configure your LLM providers.");
            }
            Err(e) => die(e),
        },
    }

    Ok(())
}
