# openheim

Openheim is an open-source LLM agent written in Rust. It connects to multiple LLM providers, executes tools on their behalf, persists conversations, and can be extended with user-defined skills. It runs as a CLI, an interactive REPL, or an HTTP/WebSocket server.

## Features

- **Multi-provider** -- OpenAI, Anthropic, Google Gemini, and any OpenAI-compatible endpoint (Ollama, vLLM, etc.)
- **Tool execution** -- Built-in tools for running shell commands, reading files, and writing files. The tool system is trait-based and extensible.
- **Conversation memory** -- Conversations are persisted to disk and can be resumed across sessions.
- **Skills** -- Drop markdown files into `~/.openheim/skills/` to give the agent custom instructions or domain knowledge.
- **Streaming** -- Real-time WebSocket streaming of agent iterations and a separate filesystem-operations WebSocket with file watching.
- **Retry with backoff** -- Transient LLM failures (429, 5xx, network errors) are retried automatically with exponential backoff.
- **Docker ready** -- Multi-stage Docker build with docker-compose for quick deployment.

## Getting Started

### Prerequisites

- Rust 1.85+
- An API key for at least one supported LLM provider

### Build

```bash
git clone https://github.com/weirdstuff-dev/openheim.git
cd openheim
cargo build --release
```

### Configure

```bash
# Create the default config file
cargo run -- --init

# Edit it to add your providers and API keys
vim ~/.openheim/config.toml
```

The config file uses TOML. Each provider entry specifies an API base URL, a default model, and either an `env_var` (recommended) or an inline `api_key`:

```toml
default_provider = "openai"
max_iterations = 10

[providers.openai]
api_base = "https://api.openai.com/v1"
default_model = "gpt-4"
models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
env_var = "OPENAI_API_KEY"

[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-3-5-sonnet-20241022"
models = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
env_var = "ANTHROPIC_API_KEY"

[providers.ollama]
api_base = "http://localhost:11434/v1"
default_model = "llama2"
models = ["llama2", "mistral", "codellama"]
```

### Run

```bash
# Single prompt
cargo run -- --query "List the files in the current directory"

# Interactive REPL
cargo run -- --agent-mode

# HTTP/WebSocket server
cargo run -- --api-mode --port 8080

# Override model
cargo run -- --query "Hello" --model gpt-4-turbo

# Resume your last conversation
cargo run -- --agent-mode --continue-last

# Load skills
cargo run -- --agent-mode --skills coding,debug

# List configured providers and models
cargo run -- --list
```

## How It Works

Openheim runs an iterative agent loop:

1. Send the conversation and available tools to the LLM.
2. If the LLM requests a tool call, execute it and feed the result back.
3. Repeat until the LLM returns a final response or the iteration limit is reached.

Conversations, including tool call history, are saved to `~/.openheim/history/` as JSON so they can be continued later.

## Skills

Skills are markdown files stored in `~/.openheim/skills/`. When loaded, their content is prepended to the system prompt. Use them to define personas, coding guidelines, or any recurring context you want the agent to have.

```bash
# List available skills
cargo run -- --list-skills
```

## API Server

Start the server with `--api-mode`. It exposes:

- **POST /query** -- Submit a prompt and receive the agent's result.
- **WS /ws** -- Stream agent events (iteration starts, tool calls, responses) in real time.
- **WS /ws/fs** -- Filesystem operations (list, read, write, watch, delete, rename) over a WebSocket.

## Docker

```bash
# Build and start
docker-compose up --build

# Or run manually
docker build -t openheim .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/workspace:/workspace \
  openheim --api-mode
```

## Project Layout

```
src/
  main.rs           Entry point and CLI argument parsing
  lib.rs            Public API surface
  error.rs          Error types
  config/           Config loading, provider/model resolution, LLM client factory
  core/
    agent.rs        Agent loop (sync and streaming variants)
    models.rs       Message, Tool, Choice, and related types
    llm/            LLM client trait and provider implementations
  tools/            Tool trait, registry, and built-in tools
  rag/              Conversation history, prompt builder, and skills manager
  api/              Actix-web server, REST endpoint, WebSocket handlers
  cli/              Interactive and single-prompt CLI modes
```

## Development

```bash
RUST_LOG=debug cargo run -- --query "test"
cargo test
cargo fmt --check
cargo clippy
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request.
