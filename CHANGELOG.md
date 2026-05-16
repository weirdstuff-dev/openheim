# Changelog

## [0.1.0] - 2026-05-15

First public release of openheim — a fast, multi-provider LLM agent runtime written in Rust.

### What's included

- **Multi-provider support** — OpenAI, Anthropic, Gemini, and any OpenAI-compatible endpoint
- **MCP integration** — connect external tools via Model Context Protocol (stdio and HTTP transports)
- **ACP server** — expose the agent over the Agent Client Protocol with WebSocket streaming
- **Tool execution** — built-in filesystem, shell, and extensible tool framework
- **Conversation history** — persistent sessions with RAG context and skill injection
- **Interactive REPL** — terminal UI for local development
- **Headless / programmatic mode** — embed openheim as a library in your own Rust application

### Install

```bash
cargo install openheim
```

Or add as a library:

```toml
[dependencies]
openheim = "0.1.0"
```

See the [README](https://github.com/weirdstuff-dev/openheim/blob/main/README.md) for configuration and usage.
