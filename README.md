<div align="center">

# <img src="hexagon-multiple.svg" width="36" height="36" alt="" valign="middle" /> openheim

[![openheim.io](https://openheim.io)]
</div>

**A fast, multi-provider LLM agent runtime built in Rust.**

Openheim runs an iterative agent loop — it calls your LLM, executes tools on its behalf, feeds results back, and repeats until the task is done. It works as a CLI, an interactive REPL, or a self-hosted HTTP/WebSocket server.

---

## Why Rust?

Openheim is built in Rust from the ground up:

- **Low memory** — runs in a fraction of the RAM a Python agent would need
- **Fast startup** — no interpreter warmup
- **True concurrency** — async Tokio runtime, multiple agents without threading headaches
- **Safe by default** — Rust's ownership model means fewer footguns in long-running agent processes

---

## Features

- **Multi-provider** — OpenAI, Anthropic Claude, Google Gemini, and any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, etc.)
- **Tool execution** — built-in shell, file read, and file write tools. Trait-based, so you can add your own.
- **Conversation memory** — conversations (including full tool call history) persist to disk and resume across sessions
- **Skills** — drop a markdown file into `~/.openheim/skills/` and it's prepended to the system prompt. Define personas, coding guidelines, or any recurring context.
- **Streaming** — real-time WebSocket streaming of agent iterations, tool calls, and responses
- **Filesystem WebSocket** — WS channel for file operations with live file watching
- **Retry with backoff** — transient failures (429s, 5xx, network errors) are retried automatically with exponential backoff
- **Docker ready** — multi-stage Dockerfile and docker-compose included

---

## Quickstart

### Prerequisites

- Rust 1.85+
- An API key for at least one supported provider

### Install

```bash
git clone https://github.com/weirdstuff-dev/openheim.git
cd openheim
cargo build --release
```

### Configure

```bash
# Generate the default config
cargo run -- --init

# Edit it
vim ~/.openheim/config.toml
```

Example config:

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
cargo run -- --api-mode --port 1217

# Resume your last conversation
cargo run -- --agent-mode --continue-last

# Load skills
cargo run -- --agent-mode --skills coding,debug

# Override model for a single run
cargo run -- --query "Hello" --model gpt-4-turbo

# List configured providers and models
cargo run -- --list
```

---

## How the agent loop works

```
User prompt
    │
    ▼
Send conversation + tools → LLM
    │
    ├─ Tool call requested? → Execute tool → feed result back → repeat
    │
    └─ Final response → done
```

Conversations are saved to `~/.openheim/history/` as JSON after every run and can be continued with `--continue-last`.

---

## Skills

Skills are markdown files in `~/.openheim/skills/`. When loaded, their content is injected into the system prompt before the conversation starts.

Use them to give the agent a persona, a set of coding standards, domain knowledge, or anything you'd otherwise paste into the system prompt every time.

```bash
# List available skills
cargo run -- --list-skills

# Run with specific skills loaded
cargo run -- --agent-mode --skills rust,debugging
```

---

## API Server

Start with `--api-mode`:

| Endpoint | Description |
|---|---|
| `POST /query` | Submit a prompt, get the agent's full result |
| `WS /ws` | Stream agent events in real time (iterations, tool calls, responses) |
| `WS /ws/fs` | Filesystem operations over WebSocket with file watching |

---

## Docker

```bash
# Build and start with docker-compose
docker-compose up --build

# Or run manually
docker build -t openheim .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/workspace:/workspace \
  openheim --api-mode
```

---

## Project structure

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

---

## Development

```bash
RUST_LOG=debug cargo run -- --query "test"
cargo test
cargo fmt --check
cargo clippy
```

---

## Contributing

Contributions are welcome.

---

## License

See [LICENSE](./LICENSE) for details.