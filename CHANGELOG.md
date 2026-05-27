# Changelog

## [0.2.0] - 2026-05-27

### Added

- **ratatui TUI** — complete rewrite of the terminal UI on ratatui; interactive picker for sessions, models, and commands replacing static text lists.
- **Token streaming & thinking blocks** — real token-level streaming and extended thinking display across all LLM providers (Anthropic, OpenAI, Gemini, OpenAI-compatible).
- **Themes** — built-in theme support; `:theme` command to switch at runtime.
- **`:models` command** — interactive popup picker to switch provider/model without leaving the session.
- **`session/set_model` in ACP** — clients can now switch the model mid-session via the Agent Client Protocol; thinking convention is advertised in session metadata.
- **`TerminalGuard` RAII** — terminal state is reliably restored on panic or forced exit.

### Fixed

- **MCP child process leak** — MCP server subprocesses are now killed when the client is dropped.
- **SSE buffer reallocation** — fixed a bug where the SSE read buffer could be reallocated incorrectly, silently dropping streamed tokens.
- **JSON serialization errors propagated** — LLM serialization failures now surface as errors instead of silently producing empty output.
- **Session restore** — restoring a session from history now correctly sets it as the active session and clears stale state.
- **Model switch shows full pair** — switching models now displays the full `(provider, model)` pair in the UI.
- **Provider config on session load** — ACP session load now resolves the full provider configuration instead of a partial view.
- **Model-whitelist enforced on provider resolution** — `resolve_with_provider` now checks the model whitelist before accepting a model.
- **Unconfigured provider warning on restore** — a clear warning is emitted when a restored session references a provider that is not in the current config.
- **Message word wrap** — long messages now wrap correctly in the TUI viewport.
- **Tool command error handling** — tool execution errors are captured and reported rather than silently dropped.

### Improved

- **Config, MCP, and skills views** — each view was refactored into a cleaner layout with consistent popup helpers.
- **Info panels moved to prompt area** — contextual info panels now render next to the input prompt instead of overlapping the message list.
- **TUI internals** — extracted `handle_scroll_key`, `highlight_row`, `centered_popup`, `push_screen`, and `Screen::is_overlay()` helpers; eliminated duplicated scroll/picker rendering code.

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
