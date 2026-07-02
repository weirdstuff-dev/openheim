# Changelog

## [Unreleased]

### Added

- **Tool-call permission requests** — before executing a tool call, the ACP layer now sends `session/request_permission` and waits for the client's decision (allow/reject, once or always) instead of executing immediately. `AllowAlways`/`RejectAlways` decisions are remembered for the rest of the session. Non-ACP callers (TUI, library embedding, subagents) are unaffected — they use an always-allow gate.
- **`session/cancel` actually cancels a running turn** — previously a `session/prompt` handler ran to completion on the single-task ACP event loop, so `session/cancel` sent mid-turn had no effect until the turn finished (and any client round-trip mid-turn, like a permission request, would have deadlocked). Prompt turns now run in a spawned task; cancellation is checked between LLM iterations and before each tool call, and any pending permission request resolves to "cancelled" immediately.
- **Session modes** — `session/set_mode` now works, with two modes: `code` (full tool access, default) and `architect` (read-only — only `read_file` is offered to the LLM). Advertised via `session/new` and `session/load` responses.
- **Tool-call plan reporting** — the agent now emits `SessionUpdate::Plan` as tool calls are issued and complete, giving ACP clients a running view of in-progress/completed steps for the current turn.
- **Client-side filesystem delegation** — when the ACP client advertises `fs.readTextFile` / `fs.writeTextFile` support at `initialize`, `read_file`/`write_file` are delegated to `fs/read_text_file` / `fs/write_text_file` instead of local disk I/O, falling back to local I/O otherwise.
- **`GET /acp` WebSocket endpoint** — a second, minimal WebSocket endpoint alongside `/ws` that speaks bare ACP JSON-RPC with no envelope and no filesystem sidecar, for generic ACP-only clients. `/ws` is unchanged. See `docs/api.md` §3.4.

### Fixed

- **`openheim run` could hang or falsely deny every tool call** — the ACP server's fallback dispatch handler (`on_receive_dispatch`, used to reply "unsupported method" to unrecognized incoming requests) also intercepted *responses* to requests the agent itself sent, since both share the same generic `Dispatch` type. Once the agent started sending real requests to the client — `session/request_permission` (this release) and `fs/read_text_file`/`fs/write_text_file` — their responses were being converted into spurious errors instead of reaching the code awaiting them. This is now fixed: the fallback only handles genuinely unclaimed requests/notifications and explicitly declines responses, letting them route normally. Also added a `session/request_permission` handler to `openheim run`'s in-process ACP client (auto-allow, since it's a non-interactive one-shot CLI invocation) so headless runs behave as before.

### Breaking changes (library)

- `SandboxedExecutor::new` takes an additional `client_io: Arc<dyn ClientIo>` argument (use `Arc::new(NoClientIo)` for the previous local-disk-only behavior).
- `core::agent::run_agent_with_history` and `run_agent_streaming_with_history` take a `&TurnContext` in place of no equivalent parameter previously — bundles cancellation and permission-gate hooks (see `core::agent::TurnContext`, `core::permission`).
- `StreamEvent::ToolCall` and `StreamEvent::ToolResult` gained an `id` field; `StreamEvent` gained a `PlanUpdate` variant.

## [0.4.0] - 2026-06-11

### Added

- **Subagents** — drop a Markdown profile in `~/.openheim/agents/{name}.md` (optional `+++`-delimited TOML frontmatter for `description`, `model`, `provider`, `tools`, `max_iterations`, mirroring how skills work) and the agent gains a `delegate_task` tool for handing off self-contained work to it. Each delegation runs an isolated agent loop — its own message history, persona, and optionally its own model/provider and restricted tool set — sandboxed identically to the parent, and returns only the subagent's final answer. See `docs/subagents.md`.

### Fixed

- **Streaming requests are now retried** — `RetryClient` previously only retried non-streaming `send` calls, so the interactive (streaming) path got no retry on transient failures. Streaming calls are now retried with the same exponential backoff, but only while it is still safe — before the first chunk reaches the caller. Once tokens have been forwarded, a mid-stream failure is returned as-is rather than replayed.

### Breaking changes (library)

- Removed `config::resolve_client_and_config` (unused outside its own tests). Its "reuse the client unless the provider/model changed" logic is now exposed as `config::client_for_config(target, baseline, baseline_llm)`.

## [0.3.0] - 2026-06-01

### Added

- **System identity (`system.md`)** — `~/.openheim/system.md` defines the agent's base identity and is injected into the system prompt on every session. `openheim init` creates a default file. The prompt is now structured: identity block first, then skills, separated by clear section headers.
- **`default_skills` in config** — new `default_skills` array in `config.toml` auto-loads a set of skills into every session without passing `--skills` each time. Per-session skills are merged on top; duplicates are removed with defaults appearing first.
- **`default_skills` in `OpenheimBuilder`** — `.default_skills(vec![...])` builder method brings the same control to programmatic embeddings.
- **Work-directory sandbox** — new `work_dir` field in `config.toml` restricts `read_file` and `write_file` to a directory tree. When unset, the directory from which openheim is invoked is used. Symlinks are followed and canonicalized so they cannot be used to escape the boundary.
- **Shell access control** — new `allow_shell` boolean in `config.toml` (default `true`). When `false`, the `execute_command` tool is removed from the tool list entirely — the LLM never sees it and cannot request it.
- **Builder methods for security controls** — `OpenheimClient::builder()` gains `.work_dir(path)` and `.allow_shell(bool)`. Both override the corresponding config-file values.
- **Cross-compilation config** — `Cross.toml` added for building Linux targets from macOS.

### Fixed

- **MCP subprocess stderr suppression** — stderr from spawned MCP server processes no longer leaks into the terminal.
- **`run` command exits cleanly** — the process now exits after a headless `openheim run` prompt completes instead of hanging.
- **`merge_skills` deduplication** — skills within the `default_skills` list itself are now deduplicated, not just across the default/session boundary.
- **Whitespace preserved in system identity** — leading and trailing whitespace in `system.md` content is preserved when building the system prompt.
- **Accurate `init` error message** — `openheim init` now correctly reports whether `system.md` was created when the config already exists.

### Breaking changes (library)

- `AppConfig` gained two new public fields: `work_dir: Option<PathBuf>` and `allow_shell: bool`. Code constructing `AppConfig` via struct literal (rather than TOML or the builder) must now supply these fields. Both have serde defaults so TOML loading is unaffected.
- `SystemToolExecutor::build` takes an additional `allow_shell: bool` argument.

## [0.2.1] - 2026-05-28

### Added

- github release workflow

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
