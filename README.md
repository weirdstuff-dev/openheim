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
- OpenAI API key (or compatible service)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/openheim.git
cd openheim

# Build the project
cargo build --release
```

### Basic Usage

```bash
# Run a single query
cargo run -- --query "List the files in the current directory"

# Start interactive mode
cargo run -- --agent-mode

# Start API server
cargo run -- --api-mode --port 8080
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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API authentication token | *required* |
| `OPENAI_API_BASE` | LLM API endpoint URL | OpenAI default |
| `OPENAI_MODEL` | Model identifier | glm-4.7 |
| `RUST_LOG` | Logging level | info |

### CLI Arguments

```
OPTIONS:
    --query <PROMPT>          Execute a single prompt
    --agent-mode           Start interactive conversation mode
    --api-mode                Start HTTP/WebSocket server
    --host <HOST>             Server bind address [default: 0.0.0.0]
    --port <PORT>             Server port [default: 8080]
    --max-iterations <N>      Agent iteration limit [default: 10]
    --api-base <URL>          LLM API endpoint
    --api-key <KEY>           LLM authentication token
    --model <NAME>            LLM model name
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
- Volume mount for persistent workspace
- Auto-restart policy

## Project Structure

```
openheim/
├── src/
│   ├── main.rs           # Entry point
│   ├── config.rs         # Configuration management
│   ├── error.rs          # Error types
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
└── docker-compose.yml
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
