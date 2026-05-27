# Changelog

## [0.1.1] - 2026-05-21

### Fixed

- **Tool call history gaps** — ACP sessions were silently dropping tool calls from stored history; all tool calls are now captured correctly.
- **Accurate status on replay** — Replayed tool calls now emit `InProgress` before resolving, matching the behaviour of live sessions.
- **Failed tool calls now surface as `Failed`** — Previously, tool failures were stored as plain text and replayed as `Completed`. The `is_error` flag is now persisted in `Message` and propagated through `StreamEvent::ToolResult` so both live and replayed paths emit `ToolCallStatus::Failed`.
- **Tool error logging** — Improved logging for tool call errors.

### Improved

- **LLM accuracy on failures** — `is_error` is forwarded to Anthropic's `tool_result` block, giving the model accurate signal when a tool has failed.
- Added `CHANGELOG.md`.
- Updated documentation for `is_error` and tool call history replay semantics.
- README updates.

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
