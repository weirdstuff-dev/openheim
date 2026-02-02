# Openheim

Openheim is an open-source, lightweight LLM agent written in Rust that is built for the home, cloud, and enterprise.

## Features

- **Multiple Operation Modes**: CLI single-prompt, interactive conversation, and HTTP API server
- **Tool Execution**: Built-in tools for command execution, file reading, and file writing
- **Streaming Support**: Real-time WebSocket streaming for agent execution and filesystem operations
- **OpenAI Compatible**: Works with any OpenAI-compatible LLM API endpoint
- **Docker Ready**: Multi-stage Docker build with docker-compose for easy deployment
- **Cross-Platform**: Supports both Unix and Windows environments

## Quick Start

### Prerequisites

- Rust 1.85+ (uses 2024 edition)
- An API key for at least one LLM provider (OpenAI, Anthropic, or any OpenAI-compatible service)

### Installation

```bash
# Clone the repository
git clone https://github.com/weirdstuff-dev/openheim.git
cd openheim

# Build the project
cargo build --release
```

### Configuration

```bash
# Initialize the config file
cargo run -- --init

# Edit ~/.openheim/config.toml to add your provider(s) and API keys
```

### Basic Usage

```bash
# Run a single query
cargo run -- --query "List the files in the current directory"

# Start interactive mode
cargo run -- --agent-mode

# Start API server
cargo run -- --api-mode --port 8080

# Use a specific model
cargo run -- --query "Hello" --model gpt-4-turbo

# List configured providers and models
cargo run -- --list
```

## Operation Modes

### 1. Single Prompt Mode (Default)

Execute a one-off task and exit:

```bash
openheim --query "Create a hello world script" --max-iterations 5
```

### 2. Interactive Mode

REPL-style conversation with persistent history:

```bash
openheim --agent-mode
```

Commands: `exit`, `quit`, or `:q` to exit.

### 3. API Server Mode

Start an HTTP server with REST and WebSocket endpoints:

```bash
openheim --api-mode --host 0.0.0.0 --port 8080
```

## API Reference

### REST Endpoint

**POST /query**

Execute an agent task.

```json
// Request
{
  "prompt": "What files are in the workspace?",
  "max_iterations": 10
}

// Response
{
  "success": true,
  "result": "...",
  "iterations": 3
}
```

### WebSocket Endpoints

**WS /ws** - Agent Streaming

Connect and send:
```json
{"prompt": "Build a web server", "max_iterations": 10}
```

Receives real-time events for each iteration.

**WS /ws/fs** - Filesystem Operations

| Operation | Payload |
|-----------|---------|
| watch | `{"type": "watch", "path": "/workspace"}` |
| list | `{"type": "list", "path": "/workspace", "recursive": true}` |
| read | `{"type": "read", "path": "/workspace/file.txt"}` |
| write | `{"type": "write", "path": "/workspace/file.txt", "content": "..."}` |
| mkdir | `{"type": "mkdir", "path": "/workspace/newdir"}` |
| delete | `{"type": "delete", "path": "/workspace/file.txt"}` |
| rename | `{"type": "rename", "from": "/old.txt", "to": "/new.txt"}` |

## Configuration

Openheim uses a TOML configuration file at `~/.openheim/config.toml` to manage LLM providers and settings.

### Setup

```bash
# Initialize the default config file
openheim --init

# Edit the config to add your providers
vim ~/.openheim/config.toml
```

### Config File Format

```toml
# Default provider to use
default_provider = "openai"

# Maximum agent iterations (can be overridden with --max-iterations)
max_iterations = 10

# Provider configurations
[providers.openai]
api_base = "https://api.openai.com/v1"
default_model = "gpt-4"
models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
env_var = "OPENAI_API_KEY"   # reads API key from this env var

[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-3-5-sonnet"
models = ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"]
env_var = "ANTHROPIC_API_KEY"

# Local Ollama (no API key needed)
[providers.ollama]
api_base = "http://localhost:11434/v1"
default_model = "llama2"
models = ["llama2", "mistral", "codellama", "mixtral"]
```

Each provider supports:
- `api_base` - The API endpoint URL
- `default_model` - Model used when no `--model` flag is given
- `models` - List of available models for this provider
- `env_var` - Environment variable name for the API key (recommended)
- `api_key` - Inline API key (not recommended; prefer `env_var`)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (referenced via `env_var` in config) |
| `ANTHROPIC_API_KEY` | Anthropic API key (referenced via `env_var` in config) |
| `RUST_LOG` | Logging level (default: `info`) |

### CLI Arguments

```
OPTIONS:
    --query <PROMPT>          Execute a single prompt
    --agent-mode              Start interactive conversation mode
    --api-mode                Start HTTP/WebSocket server
    --host <HOST>             Server bind address [default: 0.0.0.0]
    --port <PORT>             Server port [default: 8080]
    --max-iterations <N>      Override max agent iterations from config
    --model <NAME>            Use a specific model (must be configured in a provider)
    --list                    List all configured providers and models
    --init                    Initialize config file at ~/.openheim/config.toml
```

## Available Tools

The agent has access to three built-in tools:

| Tool | Description |
|------|-------------|
| `execute_command` | Run shell commands (platform-aware) |
| `read_file` | Read file contents |
| `write_file` | Create or overwrite files |

## Docker Deployment

### Using Docker Compose

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d
```

### Manual Docker Build

```bash
# Build image
docker build -t openheim .

# Run container
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/workspace:/workspace \
  openheim --api-mode
```

The Docker setup includes:
- Multi-stage build for minimal image size
- Non-root user for security
- Entrypoint script that auto-initializes config on first run
- Persistent volume for config (`~/.openheim/config.toml`)
- Volume mount for persistent workspace
- Auto-restart policy

## Project Structure

```
openheim/
├── src/
│   ├── main.rs           # Entry point
│   ├── error.rs          # Error types
│   ├── config/
│   │   ├── mod.rs        # Config loading & initialization
│   │   ├── types.rs      # AppConfig, ProviderConfig, AgentConfig types
│   │   ├── resolve.rs    # Provider/model resolution logic
│   │   └── config.toml.default  # Default config template
│   ├── core/
│   │   ├── agent.rs      # Agent orchestration
│   │   ├── llm.rs        # LLM client abstraction
│   │   └── models.rs     # Data structures
│   ├── tools/
│   │   ├── mod.rs        # Tool definitions
│   │   └── executor/     # Tool execution engine
│   ├── api/
│   │   ├── mod.rs        # Server setup
│   │   ├── rest.rs       # REST endpoints
│   │   ├── ws.rs         # Agent WebSocket
│   │   └── ws_fs.rs      # Filesystem WebSocket
│   └── cli/
│       └── mod.rs        # CLI interface
├── workspace/            # Default working directory
├── Cargo.toml
├── Dockerfile
├── docker-compose.yml
└── docker-entrypoint.sh  # Docker config initialization
```

## Architecture

Openheim uses a modular architecture:

- **Core**: Agent loop, LLM client trait, and data models
- **Tools**: Extensible tool system with async/blocking execution
- **API**: Actix-web server with REST and WebSocket handlers
- **CLI**: Clap-based command-line interface

The agent runs an iterative loop:
1. Send prompt to LLM with available tools
2. If LLM requests tool execution, run tools
3. Feed tool results back to LLM
4. Repeat until LLM returns final response or max iterations reached

## Development

```bash
# Run with logging
RUST_LOG=debug cargo run -- --query "test"

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```

## License

This project is open source. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
