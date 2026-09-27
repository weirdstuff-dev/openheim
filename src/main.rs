use std::{fs::File, io::Write, sync::OnceLock};

use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use openheim::{
    OpenheimClient,
    config::{config_dir, init_config},
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

/// Builds the client every subcommand runs against; `model` overrides the
/// config's default model.
async fn build_client(model: Option<String>) -> openheim::Result<OpenheimClient> {
    let mut builder = OpenheimClient::builder();
    if let Some(model) = model {
        builder = builder.model(model);
    }
    builder.build().await
}

/// On SIGTERM and SIGHUP, and on SIGINT when `sigint` is set, kills the
/// shell commands the agent is running and exits: dying of the signal
/// would leave them running (see `tools::kill_running_commands`). `serve`
/// passes `false`, since it shuts down gracefully on Ctrl-C by itself.
#[cfg(unix)]
fn exit_on_signal(sigint: bool) {
    use tokio::signal::unix::{SignalKind, signal};

    let mut kinds = vec![SignalKind::terminate(), SignalKind::hangup()];
    if sigint {
        kinds.push(SignalKind::interrupt());
    }
    for kind in kinds {
        let Ok(mut signals) = signal(kind) else {
            continue;
        };
        tokio::spawn(async move {
            if signals.recv().await.is_some() {
                openheim::tools::kill_running_commands();
                std::process::exit(128 + kind.as_raw_value());
            }
        });
    }
}

#[cfg(not(unix))]
fn exit_on_signal(_sigint: bool) {}

/// Prints the error and exits with status 1.
fn die(e: impl std::fmt::Display) -> ! {
    eprintln!("Error: {e}");
    std::process::exit(1);
}

/// Set once the TUI owns the terminal: logs then go to this file (or
/// nowhere, if it couldn't be opened) instead of stderr.
static TUI_LOG: OnceLock<Option<File>> = OnceLock::new();

/// Where each log line goes: stderr, since stdout carries `openheim acp`'s
/// JSON-RPC stream and `openheim run`'s answer, until the TUI takes over.
fn log_writer() -> Box<dyn Write> {
    match TUI_LOG.get() {
        None => Box::new(std::io::stderr()),
        Some(Some(file)) => Box::new(file),
        Some(None) => Box::new(std::io::sink()),
    }
}

/// Opens `~/.openheim/openheim.log` for appending.
fn open_tui_log() -> Option<File> {
    let path = config_dir().ok()?.join("openheim.log");
    File::options().create(true).append(true).open(path).ok()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(log_writer)
        .init();

    let cli = Cli::parse();
    exit_on_signal(!matches!(cli.command, Some(Command::Serve { .. })));

    match cli.command {
        None => {
            let client = build_client(None).await.unwrap_or_else(|e| die(e));
            // Written to the TUI's screen, a log line would garble it.
            let _ = TUI_LOG.set(open_tui_log());
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
