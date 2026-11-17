# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.19.0] - 2026-11-17

### Added

- When the turn in progress alone overflows the context window, its older tool outputs are left out (oldest first) instead of the turn failing with `Error::ContextOverflow`. New `StreamEvent::ToolOutputsOmitted { omitted }`, shown as a notice in the TUI.

### Changed

- **Breaking:** the `code` and `architect` modes are renamed `execute` and `plan` (`AgentMode::Execute` / `AgentMode::Plan`, ACP `modeId`, TUI `:mode`). Saved sessions with the old names still load.
- **Breaking:** `delete_session` cancels a running turn first and fails with `Error::SessionBusy` if it doesn't end within 10 seconds, or `Error::SessionLocked` if another process holds the session.
- ACP model option values are `provider/model`, so a model listed under two providers can be picked from either. Bare model names are still accepted. See `docs/api.md`.
- API keys and MCP `headers` are allowed over `http://` to loopback hosts. Requests to loopback ignore environment proxies.
- `execute_command` includes stderr, under a `Stderr:` label, in a successful result.

### Fixed

- `execute_command` returns once the shell exits, even when the command left a process running in the background (`server &`).
- `write_file` and `edit_file` write atomically, keeping the file's permissions, owner, group and hard links.
- `openheim serve` shuts down gracefully on SIGTERM, as it does on SIGINT.
- A partial last line in a session's history log, left by a crash, no longer corrupts the next appended message.
- A session lock is taken over when its holder has died and its pid was reused, or when the lock comes from another PID namespace on the same hostname and has expired.

## [0.18.0] - 2026-11-10

### Added

- `effort` provider setting (`"default"`, `"low"`, `"medium"`, `"high"`), mapped to each API's reasoning parameter. Sessions can change it over ACP (`session/set_config_option`, id `effort`), in the TUI (`:effort`) and in code (`SessionBuilder::effort`, `SessionHandle::set_effort`, `OpenheimClient::set_session_effort`). Subagent profiles take `effort` too. See `docs/configuration.md`.
- Modes can be set in the TUI (`:mode`) and in code (`SessionBuilder::mode`, `SessionHandle::set_mode`, `OpenheimClient::set_session_mode`).
- A session's mode and effort are saved with it and restored on load (`ConversationMeta::mode` / `effort`, `LoadedSession::effort`). `SessionHandle::settings()` returns a live session's provider, model, mode and effort.
- `GET /api/config` shows each provider's `effort`. The TUI footer shows mode and effort when they aren't the defaults.

### Changed

- **Breaking:** Anthropic providers no longer think unless `effort` is set. `openheim init` writes `effort = "medium"` for its Anthropic provider.

### Removed

- **Breaking:** the `thinking` provider key and `ThinkingMode`, replaced by `effort` (`Effort`, `ProviderConfig::with_effort`, `AgentConfig::with_effort`). `AnthropicClient::new` no longer takes `thinking`; built-in clients have `with_effort(level)`.

## [0.17.1] - 2026-10-26

### Fixed

- `list_sessions` finds a session whatever spelling of its cwd is used (trailing slash, `..`, symlinks).
- Concurrent writes to one memory database wait for each other instead of failing with `SQLITE_BUSY`.

## [0.17.0] - 2026-10-20

### Added

- `replay_thinking` provider setting: sends signed thinking back as `thinking_blocks` within the current tool loop, for Claude behind LiteLLM. The OpenAI clients now keep the thinking signatures LiteLLM streams. Off by default. See `docs/configuration.md`.
- `Error::from_response(response)` builds an `Error::HttpError` from a non-2xx response, for custom `LlmClient`s.

### Changed

- **Breaking:** `Error::HttpError` has a `retry_after: Option<Duration>` field; `Error::retry_after()` returns it.
- LLM and embedding clients honour `Retry-After` on 429 and 5xx. A wait over 10 seconds isn't retried.
- HTTP 529 (Anthropic `overloaded_error`) is retried.
- Provider HTTP errors reach ACP clients as `internal_error` with `data: { kind: "http_error", status, retry_after_secs, body }`. See `docs/api.md`.

## [0.16.0] - 2026-10-16

### Added

- `always_allowed_tools` config key: listed tools run without a permission prompt. Also `AppConfig::with_always_allowed_tools` and `OpenheimBuilder::always_allowed_tools`; shown in `GET /api/config`.
- `--config <FILE>` on every subcommand, for running several agents with separate configs. See `docs/deployment.md`.
- `OpenheimClient::data_dir()` and `config::init_config_at(config_path)`.

### Changed

- **Breaking:** with a custom config path and no `data_dir`, data lives next to the config file instead of in `~/.openheim`. Add `.data_dir(openheim::config::config_dir()?)` to keep the old location.

## [0.15.0] - 2026-09-28

### Added

- Unknown config keys are logged as warnings, with their path (e.g. `providers.openai.api-key`).
- MCP tool calls time out after 10 minutes, or after `tool_timeout_secs` on the server's entry (`McpServerConfig::with_tool_timeout_secs`).
- `tools::kill_running_commands()`, for an embedder's own signal handler.
- `OpenheimClient::skills()`, `OpenheimClient::switch_model`, `OpenheimBuilder::app_config` and `OpenheimBuilder::long_term_memory` (feature `rag`). `OpenheimClient` is `Clone`.
- `LoadedSession::provider`. `ThinkingMode` is exported. `ModelsInfo` and `ProviderModels` implement `Deserialize`.

### Changed

- **Breaking:** public enums and data structs are `#[non_exhaustive]`. Matches need a `_ =>` arm (except on `Role`), and structs are built with constructors and `with_*` setters (`AppConfig::new`, `ProviderConfig::new`, `McpServerConfig::stdio` / `http`, `Message::new`, `Usage::new`, `TurnContext::new`, …). Adding fields or variants is no longer breaking.
- **Breaking:** `AgentConfig::with_max_iterations` takes `self` by value.
- **Breaking:** `acp::serve` takes an `OpenheimClient`.
- **Breaking:** each public item has one path: `openheim::core::{agent, llm, models}` (no longer re-exported at the root), `openheim::tools::{ToolCapabilities, ToolKindHint, ApprovalScope}`, and `openheim::{AgentMode, LoadedSession}`.
- **Breaking:** a `work_dir` from `config.toml` is resolved at startup, and `build()` fails with `ConfigError` if it isn't an existing directory.
- The `acp` feature follows `agent-client-protocol`'s major version (2.x).

### Removed

- **Breaking:** runtime internals are no longer public. Replacements:

  | Removed | Use instead |
  |---|---|
  | `AgentState`, `AgentState::new` | `OpenheimClient::builder()`, with `.app_config(config)` for a loaded and adjusted config |
  | `AgentState` fields (`memory.history`, `memory.skills`, `app_config.models_info()`, `executor.list_tools()`, `mcp_statuses`), `AgentState::switch_model` | `list_sessions`, `get_session`, `delete_session`, `skills`, `models`, `tools`, `mcp_servers`, `switch_model` on `OpenheimClient` |
  | `client.memory()`, `MemoryContext`, `HistoryManager`, `SkillsManager`, `SystemLoader`, `SessionLease`, `config::RuntimePaths` | the client methods above; `.data_dir()` / `.config_path()` on the builder |
  | `rag::{RememberTool, SearchMemoryTool, EditMemoryTool, ForgetTool}` | `OpenheimBuilder::long_term_memory` |
  | `tools::sandbox::{validate_path, validate_path_from, validate_entry}` | `TurnContext::resolve_path` |
  | `tools::{DelegateTool, DELEGATE_TOOL_NAME, ScopedExecutor}`, `config::client_for_config`, `config::save_theme_to_config_at`, `core::permission::{Approvals, approval_key}` | none needed |
  | `transport::ws::{FsRequest, FsResponse, FsRequestEnvelope, FsReply, FileEntry}` | the JSON shapes in `docs/api.md` |

### Fixed

- openheim works without a `system.md`, using the default identity.
- An OpenAI session no longer fails on every prompt after an empty reply.
- Gemini accepts MCP tool schemas (sent as `parametersJsonSchema`).
- Shell commands and their children are killed when `openheim run` or `openheim acp` gets SIGINT, SIGTERM or SIGHUP, or when their turn is dropped.
- An MCP server whose connection closed is reconnected on the next call.
- TUI: a prompt sent right after switching sessions or models goes to the new one.
- `/ws`: a slow `fs` request no longer holds up `agent` frames on the same connection.
- `openheim acp` and `openheim run` log to stderr instead of stdout, so logs don't break the protocol stream. The TUI logs to `~/.openheim/openheim.log`.
- TUI: resuming a session switches to that session's model.
- Containers sharing a data volume no longer take over each other's session leases.
- A session on a non-default model no longer slows down other sessions; HTTP clients are reused.

## [0.14.0] - 2026-09-27

### Added

- TUI: `Ctrl+C` cancels a running turn; press it twice to quit.
- `context_window` provider setting (also `OpenheimBuilder::context_window`, `AgentConfig::context_window`), used to leave older turns out before a request is sent.
- `read_file` takes `offset` and `limit`.
- `Error::ContextOverflow`, `Error::IncompleteResponse`, `StreamEvent::ContextTrimmed { dropped }` and `ContentBlock::RedactedThinking`.
- `ContentBlock::ToolUse::signature` and `ContentBlock::tool_use(id, name, arguments)`.
- `Message::transcript()`, which lists what a UI shows for a saved message.
- `PromptInput`, for prompts with images.
- `LongTermMemory::search_detailed`, and `VectorStore::unembedded_up_to`, `unembedded_count` and `search_keyword_unembedded`.

### Changed

- **Breaking:** one prompt method and one resume method on the library facade:

  | Before | Now |
  |---|---|
  | `session.prompt_events(text, cb)` | `session.prompt(text, cb)` |
  | `session.prompt_events_with_images(text, images, cb)` | `session.prompt(PromptInput::text(text).image(data, mime), cb)` |
  | `session.prompt(text, acp_cb)` | `session.prompt(text, session.acp_updates(acp_cb))` |
  | `client.load_session(id, cwd, acp_cb)` / `handle.restore(..)` | `client.resume_session(id, cwd)`, then `session.acp_replay(&loaded.messages, acp_cb)` |
  | `handle.resume(id, cwd)` | `client.resume_session(id, cwd)` |
  | `session.id` | `session.id()` |

- **Breaking:** `core::agent::run_agent` replaces `run_agent_with_history` and `run_agent_streaming_with_history`, and takes `&dyn LlmClient` / `&dyn ToolExecutor`.
- **Breaking:** `PermissionGate::check` takes a `&PermissionRequest`, which names the subagent asking. The TUI and ACP show it in the prompt.
- **Breaking:** `ClientIo::read_file` takes a `LineRange`.
- **Breaking:** tools work in the session's `cwd` when it's inside `work_dir`. `TurnContext` has a `cwd` field; custom tools should use `TurnContext::resolve_path`.
- **Breaking:** struct literals of `ProviderConfig` and `AgentConfig` need `context_window`, and `ContentBlock::ToolUse` needs `signature`. Exhaustive matches need arms for the new variants above.
- The default Gemini model is `gemini-3.8-flash`.

### Fixed

- Sessions that outgrow the context window keep working: the oldest turns are left out of the request.
- Cancelling a turn while tools run no longer breaks the session, and sessions already broken this way work again.
- A tool call cut off at `max_tokens` no longer breaks the session.
- Tool results are cut to 128 KiB.
- `new_session` fails with `Error::NotFound` for an unknown skill. A saved session's deleted skill is skipped with a warning.
- A stream that ends before the provider's end marker fails with `Error::IncompleteResponse`, retried if nothing was streamed yet.
- An unreadable session lease is taken over once it's 10 seconds old.
- Function calling works on Gemini 3.
- Anthropic replies with several thinking blocks are kept intact and in order.
- An Anthropic tool call with no arguments is saved as `{}`.
- MCP tool names are sanitized to what providers accept, up to 64 characters.
- `openai` providers send `max_completion_tokens`, so reasoning models accept `max_tokens`.
- Gemini tool calls get unique ids.
- A Gemini reply blocked for safety ends the turn as a refusal.
- `OpenAiClient::send` and `OpenAiCompatibleClient::send` stream, like the other clients.
- `read_file` streams large files and returns at most 100 KiB per call. `edit_file` refuses files over 10 MiB.
- Long-term memory keeps working when the embeddings provider fails: requests are retried, notes are saved without a vector, and search falls back to keywords.
- MCP servers connect concurrently, and one that takes over 60 seconds is reported as failed instead of blocking startup.
- "Allow Always" with `ApprovalScope::ExactArguments` compares arguments as JSON for any tool.
- `delegate_task` returns bad arguments as tool errors, and an inline subagent's `max_iterations` is capped at the session's.
- TUI: a multi-line paste goes into the input instead of sending its first line.
- TUI: a panic prints a readable message.
- TUI: saving a theme can no longer corrupt `config.toml`.

### Security

- `/ws` `fs` `delete` and `rename` refuse `work_dir` itself and act on a symlink rather than its target.
- `web_fetch` refuses every non-public address range, including CGNAT, multicast, reserved, and IPv4 embedded in NAT64 and 6to4 addresses.

## [0.13.0] - 2026-09-24

### Added

- `/ws` `fs` requests can carry an `id`, echoed in the reply.
- `kind` on `[providers.<name>]` (`"openai"`, `"anthropic"`, `"gemini"`, `"openai_compatible"`), so a provider can have any name. Inferred from the name when omitted. Exposed as `ProviderKind`.
- `Error::InvalidArgument`, `StopReason::notice()`, `HistoryManager::save_meta`, `ConversationMeta::fill_title_from` and `tools::args::parse` / `NonEmptyString`.

### Changed

- **Breaking:** `StopReason` gains `MaxTokens` and `Refusal`; `FinishReason` gains `Refusal` and `Paused`. ACP clients get the matching stop reasons.
- **Breaking:** ACP errors use specific JSON-RPC codes: `-32002` not found, `-32602` invalid params, `-32601` method not found.
- **Breaking:** `AppConfig::to_public_json` is replaced by `AppConfig::to_public`, which returns a `PublicConfig`.
- **Breaking:** resolved paths live in `RuntimePaths`. `AppConfig::config_path` is gone, `AgentState::new` takes a `RuntimePaths`, and `LongTermMemory::from_config` takes the data directory.
- **Breaking:** `AgentState::llm` and `AgentState::config` are private; use `AgentState::config()`.
- **Breaking:** `ProviderConfig::resolve_thinking` takes a `ProviderKind`.
- Built-in tools reject arguments of the wrong type instead of ignoring them.
- "Allow Always" and "Reject Always" are remembered by the runtime per session, for every permission gate. TUI approvals no longer carry across sessions.
- Turns append to the message log instead of rewriting it.
- Tools are offered in name order, so provider prompt caches keep hitting.
- `Debug` output of config types redacts secrets.
- Anthropic's default `max_tokens` is 16000.

### Fixed

- The agent loop ends the turn after a truncated or refused reply instead of calling the model again.
- A model switch is saved with the session.
- Subagents without their own `model` run on the session's current model.
- Providers with custom names get the right API client, via `kind`.
- Long subagent replies no longer time out; the agent loop always streams.
- A custom `LlmClient` that closes its chunk channel early no longer fails the reply.
- TUI: the permission prompt's highlight no longer resets to "Allow Once" on its own.
- TUI: Esc no longer gets stuck on a popup after a permission prompt.

### Security

- `GET /api/config` no longer leaks MCP `args` or credentials in URLs; the response is built from an allowlist of fields.

## [0.12.0] - 2026-09-23

### Changed

- **Breaking:** ACP model selection moved from `session/set_model` to `session/set_config_option` (category `model`).
- Dependencies updated, including `agent-client-protocol` 2.2.0 (`agent-client-protocol-tokio` folded in) and `rmcp` 3.4.

## [0.11.1] - 2026-09-19

### Fixed

- `execute_command` no longer flashes a console window on Windows.

## [0.11.0] - 2026-09-16

### Added

- `edit_memory` tool, to replace a long-term memory note's content by id.
- `headers` on `[mcp_servers.<name>]`, sent with every request to an HTTP MCP server. Refused over `http://`.

## [0.10.0] - 2026-09-09

### Added

- Long-term memory tools `remember`, `search_memory` and `forget` (`openheim::rag`, feature `rag`, on by default): SQLite FTS5 keyword search, with optional semantic search through an embeddings provider.
- `acp` feature, on by default through `cli` and `server`; the ACP dependencies are optional.
- `OpenheimClient::resume_session` and `SessionHandle::resume`, available without `acp`.
- `SessionHandle::prompt_events` and `prompt_events_with_images`, which report raw `StreamEvent`s.
- The `stdio`, `run`, `ws` and `tui` transports take a caller-built `OpenheimClient`, so custom tools and clients work with them.
- `data_dir` config key and `OpenheimBuilder::data_dir`.
- `thinking = "adaptive" | "off"` provider setting.
- `ToolHandler::capabilities()` (read-only, kind, approval scope); `architect` mode uses it.
- `Error::SessionBusy`, `Error::DatabaseError` and `Tool::function(name, description, parameters)`.

### Changed

- **Breaking:** the `rag` module is renamed `memory` and `RagContext` is `MemoryContext`. `openheim::rag` is now the retrieval module.
- **Breaking:** `AgentState` and `AgentMode` moved to `core::runtime`, and the `acp_` method prefixes are gone.
- **Breaking:** `ToolHandler::execute` takes `turn: &TurnContext`, which now carries `work_dir` and `client_io`. Each tool enforces its own sandboxing.
- **Breaking:** `list_sessions` returns `Vec<ConversationMeta>` and is always available.
- **Breaking:** `theme_color` moved to a `[tui]` table (`AppConfig::tui`).
- **Breaking:** `SessionHandle::restore` takes an `on_history` callback.
- **Breaking:** `OpenheimBuilder::build` validates programmatic providers and can fail.
- **Breaking:** `AnthropicClient::new` takes `thinking: bool`; `ProviderConfig` and `AgentConfig` have a `thinking` field.
- **Breaking:** `run_headless` no longer takes `model`; set it on the builder.
- **Breaking:** `approval_key` takes an `ApprovalScope`, and `register_builtins` takes `allow_shell`.
- `new_session` returns a `ConfigError` for a bad model instead of falling back to the default.
- `list_dir` and `search` default to `work_dir`.
- `openheim run` prints only the final response.
- `tokio`'s `full` feature is no longer required, so `default-features = false` builds pull in less.
- The OpenAI `stream_options` retry only happens when the 400 names `stream_options`.

### Removed

- **Breaking:** `list_all_sessions()` (use `list_sessions(None)`), `SandboxedExecutor`, `with_delegation` / `WithDelegate`, `AgentResult::steps`, `AgentStep`, `ToolExecutionResult`, `AgentState::allow_shell`, `AgentConfig::arc_with_max_iterations`, `AppConfig::mcp_servers_info`, `HistoryManager::get_last_conversation`, and `Deserialize` on `Choice`.
- Support for the pre-0.5.0 single-file conversation format.

### Fixed

- TUI: `:skills` reads the configured `data_dir`, and `:theme` writes to the config file that was loaded.
- TUI: restored sessions keep their thinking and image blocks.
- Doc tests compile with `--no-default-features`.

## [0.9.0] - 2026-09-01

### Added

- Built-in tools `web_fetch`, `list_dir`, `search` and `edit_file`. `web_fetch` is SSRF-hardened; the others are sandboxed to `work_dir`.
- Tool calls in one turn run concurrently, so batched subagents run in parallel.
- Each tool result is reported as soon as its call finishes.
- Context-size tracking: `AgentResult::context_usage`, `ConversationMeta::context_usage`, `SessionHandle::context_usage()`, and a counter in the TUI footer.

## [0.8.0] - 2026-08-26

### Added

- Cross-process session lease: a turn from a second process on the same session fails with `Error::SessionLocked`. A crashed holder's lease is taken over.
- OpenAI-compatible reasoning models (GLM, DeepSeek, …) keep their thinking in saved history.

### Changed

- New project logo.

### Fixed

- `session/load` is rejected while a prompt is in flight on that session.
- `session/load` reports the session's actual mode.
- `session/load` replays images in user messages.

## [0.7.0] - 2026-08-16

### Added

- Cargo features `cli` (default), `tui` and `server`. `default-features = false` drops 85 dependencies.

### Removed

- Unused dependencies `rustyline`, `colored` and `once_cell`.

### Fixed

- Multi-byte UTF-8 split across HTTP chunks is no longer corrupted.
- `timeout_secs` bounds the connect and the gap between reads, not the whole request, so long streams aren't cut off.
- `session/load` on a live session keeps its state. Idle sessions are evicted after 7 days, and the live-session map is capped at 512.

### Security

- Fixed a sandbox escape through paths like `x/../../outside`; paths are normalized before they're checked.
- "Allow Always" on a shell command covers only that exact command.
- `execute_command` runs in its own process group with a 120-second timeout, 64 KiB output caps, and cancellation.
- Gemini API keys are sent in a header, so they no longer appear in error messages and logs.
- Skill names are validated, closing a path traversal out of the skills directory. See `docs/skills.md`.

## [0.6.0] - 2026-07-17

### Added

- Inline subagents: `delegate_task` with a `system_prompt` instead of a profile name. See `docs/subagents.md`.
- `SessionHandle::prompt_with_images`.

## [0.5.0] - 2026-07-06

### Added

- Permission requests before tool calls (ACP `session/request_permission`), with Allow / Reject, Once / Always.
- TUI permission prompt.
- `SessionHandle::permission_gate`, `SessionHandle::client_io` and `SessionHandle::cancel()`.
- `OpenheimBuilder::tool`, for custom in-process tools.
- `session/cancel` stops a running turn, including an in-flight LLM request.
- Session modes `code` and `architect` (read-only), set with `session/set_mode`.
- `read_file` and `write_file` go through the ACP client's filesystem when it supports it.
- `GET /acp` WebSocket endpoint, for plain ACP clients.
- Image content in prompts. `ResourceLink` blocks become a text hint; audio and embedded resources are rejected.
- `StreamEvent::MessageAppended`, `HistoryManager::append_message` and `AppConfig::resolve_provider_default`.
- `Error::NotFound`, `Error::JoinError` and `StopReason`. Hitting `max_iterations` is reported to ACP clients as `MaxTurnRequests`.

### Changed

- **Breaking:** `Message` is built from content blocks (`content: Vec<ContentBlock>`). History from earlier versions doesn't load. `ToolCall`, `FunctionCall`, `ChatRequest` and `ChatResponse` are removed.
- **Breaking:** `Choice::finish_reason` is an `Option<FinishReason>`.
- **Breaking:** the agent loop functions and `ToolExecutor::execute` take a `&TurnContext` (moved to `core::turn`). `DelegateTool` is no longer a `ToolHandler`.
- **Breaking:** `SandboxedExecutor::new` takes `client_io`, and `AgentState::new` takes `custom_tools`.
- **Breaking:** `StreamEvent::ToolCall` and `ToolResult` have an `id`, `AgentResult` has `stop_reason`, `acp_prompt` returns a `StopReason`, and `AgentState::is_session_cancelled` is removed.
- **Breaking:** the `MODE_CODE` / `MODE_ARCHITECT` constants are replaced by `AgentMode`.
- **Breaking:** `get_session`, `list_all_sessions` and `delete_session` are `async`.
- **Breaking:** the crate root re-exports an explicit list. The `/ws` filesystem types moved to `transport::ws`; the unused `WsRequest`, `WsResponse`, `ClientEnvelope`, `ServerEnvelope` and `SystemEvent` are removed.
- Conversations are stored as a `{id}.json` meta file and an append-only `{id}.jsonl` message log.
- `/ws`: `watch` only turns on `fs_event` notifications, `fs` requests no longer need it, and relative paths resolve from `work_dir`.
- A second `session/prompt` on a session with a turn in flight is rejected.

### Fixed

- `openheim run` no longer hangs or denies every tool call.
- Anthropic extended thinking works, including on subagents, and thinking is sent back with tool results.
- A crash mid-turn no longer loses the turn or corrupts saved history.
- `session/load` resolves the saved provider through the validated config, falling back to the default with a warning.
- `openheim init` writes current default models.
- `GET /api/sessions/{id}` returns 404 only for a missing session.
- `GET /api/sessions` no longer blocks the async runtime.
- The TUI reports the real error when startup fails.

### Security

- The `/ws` filesystem sidecar is sandboxed to `work_dir`.
- Subagents go through the session's permission gate.
- "Allow Always" on a shell command only covers commands with the same program name.

## [0.4.0] - 2026-06-11

### Added

- Subagents: Markdown profiles in `~/.openheim/agents/`, run through the `delegate_task` tool. See `docs/subagents.md`.

### Removed

- **Breaking:** `config::resolve_client_and_config`; use `config::client_for_config`.

### Fixed

- Streaming requests are retried until the first chunk reaches the caller.

## [0.3.0] - 2026-06-01

### Added

- `~/.openheim/system.md` sets the agent's identity; `openheim init` creates it.
- `default_skills` config key and `OpenheimBuilder::default_skills`.
- `work_dir` config key, sandboxing `read_file` and `write_file`.
- `allow_shell` config key (default `true`) to remove `execute_command`.
- `OpenheimBuilder::work_dir` and `OpenheimBuilder::allow_shell`.
- `Cross.toml` for building Linux targets from macOS.

### Changed

- **Breaking:** `AppConfig` has `work_dir` and `allow_shell` fields, and `SystemToolExecutor::build` takes `allow_shell`.

### Fixed

- MCP server stderr no longer leaks into the terminal.
- `openheim run` exits after the prompt completes.
- Duplicate entries in `default_skills` are removed.
- Whitespace in `system.md` is preserved.
- `openheim init` reports correctly whether it created `system.md`.

## [0.2.1] - 2026-05-28

### Added

- GitHub release workflow.

## [0.2.0] - 2026-05-27

### Added

- Terminal UI rewritten on ratatui, with pickers for sessions, models and commands.
- Token streaming and thinking display for every provider.
- Themes and the `:theme` command.
- `:models` command to switch provider and model.
- ACP `session/set_model`.
- The terminal is restored on panic or forced exit.

### Changed

- Config, MCP and skills views have a cleaner layout, and info panels sit next to the input.

### Fixed

- MCP server processes are killed when the client is dropped.
- Streamed tokens are no longer dropped when the SSE buffer grows.
- LLM serialization errors are reported instead of producing empty output.
- Restoring a session makes it the active session.
- Switching models shows the provider and model.
- ACP session load resolves the full provider config.
- The model allowlist is enforced when resolving a provider.
- Restoring a session whose provider isn't configured shows a warning.
- Long messages wrap in the TUI.
- Tool execution errors are reported.

## [0.1.1] - 2026-05-21

### Added

- This changelog.

### Changed

- Tool errors are sent to Anthropic with `is_error`.

### Fixed

- ACP sessions no longer drop tool calls from saved history.
- Replayed tool calls report `InProgress` before their result, as live ones do.
- Failed tool calls are saved and replayed as `Failed`.

## [0.1.0] - 2026-05-15

First public release.

### Added

- OpenAI, Anthropic, Gemini and OpenAI-compatible providers.
- MCP tools over stdio and HTTP.
- ACP server with WebSocket streaming.
- Built-in filesystem and shell tools, and custom tools.
- Saved sessions with skills.
- Terminal UI.
- Library API for embedding openheim in a Rust application.

[Unreleased]: https://github.com/weirdstuff-dev/openheim/compare/v0.19.0...HEAD
[0.19.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.17.1...v0.18.0
[0.17.1]: https://github.com/weirdstuff-dev/openheim/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/weirdstuff-dev/openheim/compare/0.15.0...v0.16.0
[0.15.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.14.0...0.15.0
[0.14.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/weirdstuff-dev/openheim/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/weirdstuff-dev/openheim/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/weirdstuff-dev/openheim/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/weirdstuff-dev/openheim/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/weirdstuff-dev/openheim/releases/tag/v0.1.0
