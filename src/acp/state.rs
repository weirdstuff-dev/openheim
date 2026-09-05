//! [`AgentState`]: the process-wide, per-connection-shared state behind every
//! ACP entry point — session bookkeeping plus the `acp_*` methods `serve()`
//! dispatches into.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use agent_client_protocol::schema::{
    ContentBlock as AcpContentBlock, ContentChunk, ModelInfo, SessionInfo, SessionModelState,
    SessionUpdate, ToolCall as AcpToolCall, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::{AgentConfig, AppConfig, build_http_client, create_client},
    core::{
        agent::run_agent_streaming_with_history,
        client_io::ClientIo,
        models::{Message, Role, StopReason as CoreStopReason, StreamEvent},
        permission::PermissionGate,
        turn::TurnContext,
    },
    error::{Error, Result},
    llm::LlmClient,
    memory::{Conversation, MemoryContext},
    subagents::SubagentLoader,
    tools::{
        SandboxedExecutor, ScopedExecutor, SystemToolExecutor, ToolExecutor, ToolHandler,
        with_delegation,
    },
};

use super::{
    convert::convert_prompt_blocks,
    session::{
        MAX_LIVE_SESSIONS, SESSION_IDLE_EVICTION_AFTER, SessionState, evict_idle_sessions,
        insert_or_keep_live, prompt_in_flight,
    },
    util::{AgentMode, replay_history_messages, thinking_chunk, tool_kind_for},
};

type Sessions = Arc<RwLock<HashMap<String, SessionState>>>;

pub struct AgentState {
    pub llm: Arc<dyn LlmClient>,
    pub executor: Arc<dyn ToolExecutor>,
    pub config: AgentConfig,
    pub app_config: AppConfig,
    pub memory: MemoryContext,
    /// Long-term memory behind the `remember` / `search_memory` / `forget`
    /// tools (keyword-only unless `[memory]` names an embedding provider).
    #[cfg(feature = "rag")]
    pub long_term_memory: Arc<crate::rag::LongTermMemory>,
    pub mcp_statuses: Vec<crate::mcp::McpServerStatus>,
    /// Resolved work directory used as the sandbox boundary for every session.
    pub work_dir: PathBuf,
    /// Whether shell command execution is enabled for the LLM.
    pub allow_shell: bool,
    /// Visible to the rest of `acp` (e.g. [`super::permission::AcpPermissionGate`]
    /// reads remembered approvals directly) but not outside it.
    pub(super) sessions: Sessions,
}

impl AgentState {
    /// `custom_tools` are registered alongside the built-ins (`execute_command`,
    /// `read_file`, `write_file`) and any MCP-sourced tools, before the
    /// sandbox/delegation wrappers are applied — so custom tools are subject
    /// to the same `work_dir`/`allow_shell` boundary as everything else.
    pub async fn new(
        config: AgentConfig,
        app_config: AppConfig,
        memory: MemoryContext,
        custom_tools: Vec<Box<dyn ToolHandler>>,
    ) -> Result<Self> {
        let http_client = build_http_client(config.timeout_secs)?;
        let llm = create_client(&config, &http_client);
        let allow_shell = app_config.allow_shell;
        let work_dir = match app_config.work_dir.clone() {
            Some(wd) => wd,
            None => std::env::current_dir().map_err(|e| {
                crate::error::Error::ConfigError(format!(
                    "failed to determine current directory for work_dir: {e}"
                ))
            })?,
        };
        let (mut sys_executor, mcp_statuses) =
            SystemToolExecutor::build(&app_config.mcp_servers, allow_shell).await;
        for tool in custom_tools {
            sys_executor.register(tool);
        }
        #[cfg(feature = "rag")]
        let long_term_memory = Arc::new(crate::rag::LongTermMemory::from_config(&app_config)?);
        #[cfg(feature = "rag")]
        {
            let m = &long_term_memory;
            sys_executor.register(Box::new(crate::rag::RememberTool::new(m.clone())));
            sys_executor.register(Box::new(crate::rag::SearchMemoryTool::new(m.clone())));
            sys_executor.register(Box::new(crate::rag::ForgetTool::new(m.clone())));
        }
        let executor = Arc::new(sys_executor) as Arc<dyn ToolExecutor>;

        let profiles = SubagentLoader::new()?.load()?;
        let executor = with_delegation(
            executor,
            work_dir.clone(),
            allow_shell,
            profiles,
            llm.clone(),
            app_config.clone(),
            config.clone(),
        );

        Ok(Self {
            llm,
            executor,
            config,
            app_config,
            memory,
            #[cfg(feature = "rag")]
            long_term_memory,
            mcp_statuses,
            work_dir,
            allow_shell,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn acp_new_session(
        &self,
        model: Option<&str>,
        skills: Vec<String>,
        cwd: PathBuf,
    ) -> Result<String> {
        let chat_id = Uuid::new_v4();
        let session_key = chat_id.to_string();
        let config = model
            .and_then(|m| self.app_config.resolve(Some(m)).ok())
            .unwrap_or_else(|| self.config.clone());
        // No write lease taken here — merely creating/holding a session open
        // doesn't touch history, so it doesn't contend with other processes.
        // The cross-process write lease is acquired per-turn in `acp_prompt`.
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(
                session_key.clone(),
                SessionState {
                    chat_id,
                    config,
                    cwd,
                    skills,
                    cancel: CancellationToken::new(),
                    approved_tools: HashMap::new(),
                    mode: AgentMode::Code,
                    prompt_lock: Arc::new(Mutex::new(())),
                    last_active: Instant::now(),
                },
            );
            // Bound the map on every insert; a brand-new session has the
            // freshest `last_active`, so the sweep can only claim others.
            evict_idle_sessions(
                &mut sessions,
                Instant::now(),
                SESSION_IDLE_EVICTION_AFTER,
                MAX_LIVE_SESSIONS,
            );
        }
        Ok(session_key)
    }

    /// Cancels the currently active prompt turn for `session_id`, if any.
    /// No-op if the session doesn't exist or has no turn in flight.
    pub async fn cancel_session(&self, session_id: &str) {
        // Write lock: bumping `last_active` marks the session as recently
        // used so the eviction sweep can't claim an actively used session.
        if let Some(s) = self.sessions.write().await.get_mut(session_id) {
            s.last_active = Instant::now();
            s.cancel.cancel();
        }
    }

    /// Swaps a live session's [`AgentConfig`], returning its `(provider, model)`.
    /// Shared by the two public model-switch entry points below.
    async fn apply_session_config(
        &self,
        session_id: &str,
        new_config: AgentConfig,
    ) -> Result<(String, String)> {
        let provider_name = new_config.provider_name.clone();
        let model_name = new_config.model.clone();
        let mut sessions = self.sessions.write().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
        s.config = new_config;
        s.last_active = Instant::now();
        Ok((provider_name, model_name))
    }

    pub async fn acp_update_session_model(
        &self,
        session_id: &str,
        provider: &str,
        model: &str,
    ) -> Result<(String, String)> {
        let new_config = self.app_config.resolve_with_provider(provider, model)?;
        self.apply_session_config(session_id, new_config).await
    }

    pub async fn acp_set_session_model(
        &self,
        session_id: &str,
        model_id: &str,
    ) -> Result<(String, String)> {
        let new_config = self.app_config.resolve(Some(model_id))?;
        self.apply_session_config(session_id, new_config).await
    }

    pub async fn acp_set_session_mode(&self, session_id: &str, mode_id: &str) -> Result<()> {
        let mode = AgentMode::parse(mode_id)?;
        let mut sessions = self.sessions.write().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
        s.mode = mode;
        s.last_active = Instant::now();
        Ok(())
    }

    pub fn session_model_state(&self, current_model: &str) -> SessionModelState {
        let available_models = self
            .app_config
            .providers
            .iter()
            .flat_map(|(provider_name, p)| {
                p.models.iter().map(move |m| {
                    let mut meta = serde_json::Map::new();
                    meta.insert(
                        "provider".to_string(),
                        serde_json::Value::String(provider_name.clone()),
                    );
                    ModelInfo::new(m.clone(), m.clone()).meta(meta)
                })
            })
            .collect();
        SessionModelState::new(current_model.to_string(), available_models)
    }

    /// Persists `conv`'s full current state off the async runtime thread,
    /// logging (not propagating) any failure — history durability is
    /// best-effort and must never fail a turn that otherwise succeeded.
    /// `context` is folded into the warning log line to identify which of
    /// this method's call sites failed.
    async fn persist_conversation(&self, conv: &Conversation, context: &str) {
        let history = self.memory.history.clone();
        let conv = conv.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || history.save_conversation(&conv))
            .await
            .unwrap_or_else(|e| Err(Error::from(e)))
        {
            tracing::warn!("failed to {context}: {e}");
        }
    }

    /// Runs one prompt turn to completion and returns why it stopped, so the
    /// caller can map it to an ACP [`agent_client_protocol::schema::StopReason`]
    /// directly instead of having to reverse-engineer it (e.g. by polling
    /// session state for cancellation after the fact).
    pub async fn acp_prompt<F>(
        &self,
        session_id: &str,
        prompt: Vec<AcpContentBlock>,
        permission_gate: Arc<dyn PermissionGate>,
        client_io: Arc<dyn ClientIo>,
        mut on_update: F,
    ) -> Result<CoreStopReason>
    where
        F: FnMut(SessionUpdate) + Send,
    {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| Error::ParseError("invalid session id format".to_string()))?;

        let (llm, executor, config, chat_id, skills, cwd, cancel, _prompt_guard) = {
            // Write lock: each new prompt turn gets a fresh cancellation token,
            // since a token can only ever transition uncancelled -> cancelled
            // and must not leak a previous turn's cancellation into this one.
            let mut sessions = self.sessions.write().await;
            let s = sessions
                .get_mut(session_id)
                .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
            // Held until this function returns (success, error, or cancellation);
            // a second overlapping `session/prompt` on the same session would
            // otherwise race this one to reset `cancel` and to save history.
            // Must be acquired — and must fail fast on an overlapping call —
            // before the cross-process lease below: `SessionLease`'s Drop
            // can't tell "this guard's turn was legitimately accepted, then
            // later dropped" apart from "this guard was for a redundant,
            // rejected overlapping call", so if a rejected call had already
            // created its own lease guard, returning its error here would
            // drop *that* guard and delete the still-running accepted turn's
            // lockfile out from under it.
            let prompt_guard = s.try_acquire_prompt_lock(session_id)?;
            s.cancel = CancellationToken::new();
            s.last_active = Instant::now();
            let llm = crate::config::client_for_config(&s.config, &self.config, &self.llm)?;
            let base: Arc<dyn ToolExecutor> = if s.mode == AgentMode::Architect {
                Arc::new(ScopedExecutor::new(
                    self.executor.clone(),
                    vec![
                        "read_file".to_string(),
                        "list_dir".to_string(),
                        "search".to_string(),
                        "search_memory".to_string(),
                    ],
                ))
            } else {
                self.executor.clone()
            };
            let sandboxed = Arc::new(SandboxedExecutor::new(
                base,
                self.work_dir.clone(),
                self.allow_shell,
                client_io,
            )) as Arc<dyn ToolExecutor>;
            (
                llm,
                sandboxed,
                s.config.clone(),
                s.chat_id,
                s.skills.clone(),
                s.cwd.clone(),
                s.cancel.clone(),
                prompt_guard,
            )
        };

        // Cross-process write lease for this turn only (see `memory::lease`).
        // Held until this function returns — success, error, or cancellation
        // — via `_lease` staying in scope for the whole body, so an
        // overlapping `session/prompt` on this session from *another*
        // process is rejected immediately instead of racing history writes
        // or generating against a context that's about to go stale. Merely
        // loading/holding a session open never takes this lease — see
        // `SessionState::prompt_lock`'s doc comment — only an in-flight turn
        // does, in any process.
        let _lease = self.memory.history.acquire_lease(&uuid)?;

        let (mut conversation, prompt_builder) = self.memory.prepare(
            Some(chat_id),
            &skills,
            Some(config.model.clone()),
            Some(config.provider_name.clone()),
        )?;

        conversation.meta.cwd = Some(cwd);
        conversation.messages.push(Message {
            role: Role::User,
            content: convert_prompt_blocks(&prompt)?,
        });

        // Full checkpoint before the turn starts: durably records this
        // turn's new user message even if the turn crashes before producing
        // anything else, and — since `save_conversation` always rewrites the
        // message log from scratch — transparently upgrades a pre-split-
        // format conversation (see `memory::history::HistoryManager`'s doc
        // comment) so the `append_message` calls below have a `.jsonl` log
        // that already reflects everything up to this point to append onto.
        self.persist_conversation(&conversation, "persist conversation before turn start")
            .await;

        let history_for_append = self.memory.history.clone();
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
        };
        let run_result = run_agent_streaming_with_history(
            llm,
            executor,
            &config,
            &mut conversation.messages,
            Some(&prompt_builder),
            &turn,
            move |event| match event {
                // Blocking I/O called synchronously (not via `spawn_blocking`)
                // deliberately: appends must land in the log in the same
                // order messages are produced, and this closure already runs
                // strictly sequentially with the rest of the turn, so a
                // small, fast local-disk append here doesn't race anything —
                // spawning it would only risk two concurrent appends landing
                // out of order.
                StreamEvent::MessageAppended { message } => {
                    if let Err(e) = history_for_append.append_message(&chat_id, &message) {
                        tracing::warn!("failed to append message to history: {e}");
                    }
                }
                StreamEvent::LlmResponse { content } => {
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::from(content),
                    )));
                }
                StreamEvent::ThinkingContent { content } => {
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::Text(thinking_chunk(content)),
                    )));
                }
                StreamEvent::ToolCall {
                    id,
                    tool_name,
                    arguments,
                } => {
                    // Pending, not InProgress: the permission gate (invoked by the
                    // agent loop right after this event) hasn't authorized
                    // execution yet at this point.
                    let raw_input = serde_json::from_str(&arguments).ok();
                    on_update(SessionUpdate::ToolCall(
                        AcpToolCall::new(id, &*tool_name)
                            .kind(tool_kind_for(&tool_name))
                            .status(ToolCallStatus::Pending)
                            .raw_input(raw_input),
                    ));
                }
                StreamEvent::ToolResult {
                    id,
                    result,
                    is_error,
                    ..
                } => {
                    let status = if is_error {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    };
                    on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                        id,
                        ToolCallUpdateFields::new()
                            .status(status)
                            .raw_output(serde_json::Value::String(result)),
                    )));
                }
                _ => {}
            },
        )
        .await;

        // Folded into the conversation's context-size snapshot before the
        // final checkpoint below, so it's persisted even for a turn that
        // only partially completed (cancelled mid-turn still made LLM calls
        // worth accounting for). A turn with no successful calls leaves the
        // previous snapshot in place rather than clearing it.
        if let Ok(r) = &run_result
            && r.context_usage.is_some()
        {
            conversation.meta.context_usage = r.context_usage;
        }

        // Final full checkpoint: reconciles whatever `append_message` calls
        // landed above into one consistent, complete log, and is the only
        // save at all for a turn that produced no messages (cancelled or
        // errored before the first LLM response).
        self.persist_conversation(&conversation, "save conversation")
            .await;

        run_result.map(|r| r.stop_reason)
    }

    pub async fn acp_list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<SessionInfo>> {
        let history = self.memory.history.clone();
        let metas = tokio::task::spawn_blocking(move || history.list_conversations())
            .await
            .map_err(Error::from)??;
        Ok(metas
            .iter()
            .filter(|m| cwd.is_none_or(|filter| m.cwd.as_deref() == Some(filter)))
            .map(|m| {
                let path = m.cwd.clone().unwrap_or_else(|| PathBuf::from("/"));
                let mut info = SessionInfo::new(m.id.to_string(), path);
                if let Some(t) = &m.title {
                    info = info.title(t.clone());
                }
                info.updated_at(m.updated_at.to_rfc3339())
            })
            .collect())
    }

    pub async fn acp_load_session<F>(
        &self,
        session_id: &str,
        cwd: PathBuf,
        mut on_update: F,
    ) -> Result<AgentMode>
    where
        F: FnMut(SessionUpdate) + Send,
    {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| Error::ParseError("invalid session id format".to_string()))?;

        let history = self.memory.history.clone();
        let conversation = tokio::task::spawn_blocking(move || history.load_conversation(&uuid))
            .await
            .map_err(Error::from)??;

        let mut session_config = self.config.clone();
        if let Some(provider_name) = &conversation.meta.provider {
            // Same resolution (and validation) as every other config path;
            // a session whose saved provider/model no longer resolves —
            // removed from the config, model dropped from the allowlist —
            // falls back to the default provider rather than failing the load.
            let resolved = match &conversation.meta.model {
                Some(model) => self.app_config.resolve_with_provider(provider_name, model),
                None => self.app_config.resolve_provider_default(provider_name),
            };
            match resolved {
                Ok(config) => session_config = config,
                Err(e) => {
                    let warning = format!(
                        "[warning] Could not restore this session's provider '{}' ({e}). Falling back to the default provider '{}'.",
                        provider_name, session_config.provider_name
                    );
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::from(warning),
                    )));
                }
            }
        } else if let Some(model) = &conversation.meta.model {
            session_config.model = model.clone();
        }

        let mode = {
            let mut sessions = self.sessions.write().await;
            // A second connection attaching to an already-live session
            // must not replace its control state — a fresh `cancel` token
            // would orphan an in-flight turn, wiping `approved_tools` loses
            // remembered AllowAlways decisions, and a fresh `prompt_lock`
            // would let two turns overlap on one chat. The live entry (if
            // any) is also newer than the disk snapshot above. Note the
            // history replay below is a one-shot dump of what's on disk, not
            // a live subscription — it never sees chunks from a turn that's
            // still streaming, and this connection gets no further updates
            // for that turn (only the connection that called `session/prompt`
            // does). The in-flight check below rejects the load outright in
            // that case rather than silently handing back a stale picture.
            // No write lease is taken here — loading/attaching to a session
            // doesn't touch history by itself, so it never contends with
            // another process merely viewing (or even holding open) the same
            // session; only an in-flight `session/prompt` turn does.
            if !insert_or_keep_live(&mut sessions, session_id, || {
                Ok(SessionState {
                    chat_id: uuid,
                    config: session_config,
                    cwd,
                    skills: conversation.meta.skills.clone(),
                    cancel: CancellationToken::new(),
                    approved_tools: HashMap::new(),
                    mode: AgentMode::Code,
                    prompt_lock: Arc::new(Mutex::new(())),
                    last_active: Instant::now(),
                })
            })? {
                tracing::debug!("session {session_id} is already live; keeping live control state");
            }
            evict_idle_sessions(
                &mut sessions,
                Instant::now(),
                SESSION_IDLE_EVICTION_AFTER,
                MAX_LIVE_SESSIONS,
            );
            // The entry was just touched above (inserted or kept live), so it
            // survives the idle sweep.
            let live = sessions
                .get(session_id)
                .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
            // A turn in flight on this session streams its updates only to
            // the connection that called `session/prompt` (see comment
            // above); reject the load instead of handing this connection a
            // history snapshot that's already stale and will never catch up.
            if prompt_in_flight(live) {
                return Err(Error::Other(format!(
                    "a prompt is already in flight for session {session_id}; retry once it completes"
                )));
            }
            // Read back the mode so the response reflects whatever
            // `acp_prompt` is actually enforcing for it, not the
            // fresh-session default.
            live.mode
        };

        replay_history_messages(&conversation.messages, &mut on_update);

        Ok(mode)
    }
}

#[cfg(test)]
mod prompt_lease_ordering_tests {
    use std::collections::HashMap;

    use tempfile::tempdir;

    use crate::memory::history::HistoryManager;

    use super::*;

    fn sample_session_state(chat_id: Uuid) -> SessionState {
        SessionState {
            chat_id,
            config: AgentConfig::new(
                "mock".into(),
                "https://example.com".into(),
                "key".into(),
                "mock-model".into(),
                5,
            ),
            cwd: PathBuf::from("/tmp"),
            skills: vec![],
            cancel: CancellationToken::new(),
            approved_tools: HashMap::new(),
            mode: AgentMode::Code,
            prompt_lock: Arc::new(Mutex::new(())),
            last_active: Instant::now(),
        }
    }

    // Regression test for the ordering `acp_prompt` relies on: the
    // in-process `prompt_lock` must be acquired (and fail fast on an
    // overlapping call) *before* the cross-process `SessionLease` is
    // acquired. Getting this backwards let an overlapping, rejected
    // `session/prompt` call create its own lease guard and then drop it
    // (`SessionLease::drop` can't tell that apart from a legitimately
    // superseded one) — deleting the still-running accepted turn's lockfile
    // out from under it.
    #[test]
    fn overlapping_prompt_in_same_process_never_touches_the_accepted_turns_lease() {
        let dir = tempdir().unwrap();
        let history = HistoryManager::with_dir(dir.path().to_path_buf());
        let chat_id = Uuid::new_v4();
        let lock_path = dir.path().join(format!("{chat_id}.lock"));
        let state = sample_session_state(chat_id);

        // Turn A: accepted, in the same order `acp_prompt` now uses.
        let _prompt_guard_a = state.try_acquire_prompt_lock("s1").unwrap();
        let _lease_a = history.acquire_lease(&chat_id).unwrap();
        assert!(lock_path.exists());

        // Turn B: an overlapping `session/prompt` for the same session, same
        // process. Must be rejected via the prompt lock...
        let turn_b = state.try_acquire_prompt_lock("s1");
        assert!(turn_b.is_err());

        // ...which means it never got as far as calling `acquire_lease`, so
        // there's no second `SessionLease` guard to drop here. Turn A's
        // lease must still be on disk, untouched.
        assert!(
            lock_path.exists(),
            "an overlapping, rejected prompt must not delete the accepted turn's lease"
        );

        drop(_lease_a);
        assert!(
            !lock_path.exists(),
            "turn A's own lease still releases normally"
        );
    }
}
