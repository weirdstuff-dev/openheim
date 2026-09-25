//! [`AgentState`]: the process-wide, per-connection-shared state behind every
//! entry point — session bookkeeping plus the methods `acp::serve()` and the
//! other transports dispatch into.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::{AgentConfig, AppConfig, RuntimePaths, build_http_client, create_client},
    core::{
        agent::run_agent_streaming_with_history,
        client_io::ClientIo,
        models::{ContentBlock, Message, Role, StopReason as CoreStopReason, StreamEvent},
        permission::{Approvals, PermissionGate, RememberingGate},
        turn::TurnContext,
    },
    error::{Error, Result},
    llm::LlmClient,
    memory::{ConversationMeta, HistoryManager, MemoryContext},
    subagents::SubagentLoader,
    tools::{
        DelegateTool, OverlayExecutor, ScopedExecutor, SystemToolExecutor, ToolExecutor,
        ToolHandler,
    },
};

use super::{
    AgentMode,
    session::{
        MAX_LIVE_SESSIONS, SESSION_IDLE_EVICTION_AFTER, SessionState, evict_idle_sessions,
        insert_or_keep_live, prompt_in_flight,
    },
};

type Sessions = Arc<RwLock<HashMap<String, SessionState>>>;

pub struct AgentState {
    /// Client for `config`. Private, together with `config`: sessions reuse
    /// this client only while their config matches `config`
    /// (`client_for_config`), so the two must never be changed separately.
    llm: Arc<dyn LlmClient>,
    pub executor: Arc<dyn ToolExecutor>,
    /// The default model/provider new sessions start on (read it via
    /// [`Self::config`]).
    config: AgentConfig,
    pub app_config: AppConfig,
    pub memory: MemoryContext,
    /// Long-term memory behind the `remember` / `search_memory` /
    /// `edit_memory` / `forget` tools (keyword-only unless `[memory]` names
    /// an embedding provider).
    #[cfg(feature = "rag")]
    pub long_term_memory: Arc<crate::rag::LongTermMemory>,
    pub mcp_statuses: Vec<crate::mcp::McpServerStatus>,
    /// Resolved work directory used as the sandbox boundary for every session.
    pub work_dir: PathBuf,
    /// Resolved data directory and config file (read via [`Self::paths`]).
    paths: RuntimePaths,
    sessions: Sessions,
    /// The `delegate_task` registered in `executor`, bound to the startup
    /// model. `prompt` rebinds a copy to the session's live model each turn
    /// (see [`DelegateTool::for_session`]).
    delegate: DelegateTool,
}

impl AgentState {
    /// `custom_tools` are registered alongside the built-ins (`execute_command`,
    /// `read_file`, `write_file`, …) and any MCP-sourced tools. Every handler
    /// receives the turn's [`TurnContext`] — `work_dir`, cancel token, client
    /// I/O — so custom tools can enforce the same boundary the built-ins do.
    /// `paths` says where subagent profiles and (by default) the memory
    /// database live; `memory` should already be rooted at `paths.data_dir`.
    pub async fn new(
        config: AgentConfig,
        app_config: AppConfig,
        paths: RuntimePaths,
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
        let long_term_memory = Arc::new(crate::rag::LongTermMemory::from_config(
            &app_config,
            &paths.data_dir,
        )?);
        #[cfg(feature = "rag")]
        {
            let m = &long_term_memory;
            sys_executor.register(Box::new(crate::rag::RememberTool::new(m.clone())));
            sys_executor.register(Box::new(crate::rag::SearchMemoryTool::new(m.clone())));
            sys_executor.register(Box::new(crate::rag::EditMemoryTool::new(m.clone())));
            sys_executor.register(Box::new(crate::rag::ForgetTool::new(m.clone())));
        }

        // `delegate_task` is always exposed — even with no configured
        // profiles the orchestrator can define an ephemeral subagent inline.
        // It's built from a snapshot of the registry taken *before* it
        // registers itself, so subagents structurally never see
        // `delegate_task` and can't delegate recursively.
        let profiles = SubagentLoader::with_dir(paths.data_dir.join("agents")).load()?;
        let base: Arc<dyn ToolExecutor> = Arc::new(sys_executor.clone());
        let delegate = DelegateTool::new(
            base,
            profiles,
            llm.clone(),
            app_config.clone(),
            config.clone(),
        );
        sys_executor.register(Box::new(delegate.clone()));
        let executor = Arc::new(sys_executor) as Arc<dyn ToolExecutor>;

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
            paths,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            delegate,
        })
    }

    /// The default model/provider new sessions start on.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// The data directory and config file this client resolved at build time.
    pub fn paths(&self) -> &RuntimePaths {
        &self.paths
    }

    pub async fn new_session(
        &self,
        model: Option<&str>,
        skills: Vec<String>,
        cwd: PathBuf,
    ) -> Result<String> {
        let chat_id = Uuid::new_v4();
        let session_key = chat_id.to_string();
        let config = match model {
            Some(m) => self.app_config.resolve(Some(m))?,
            None => self.config.clone(),
        };
        // No write lease taken here — merely creating/holding a session open
        // doesn't touch history, so it doesn't contend with other processes.
        // The cross-process write lease is acquired per-turn in `Self::prompt`.
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
                    approved_tools: Approvals::default(),
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

    pub async fn switch_model(
        &self,
        session_id: &str,
        provider: &str,
        model: &str,
    ) -> Result<(String, String)> {
        let new_config = self.app_config.resolve_with_provider(provider, model)?;
        self.apply_session_config(session_id, new_config).await
    }

    pub async fn set_session_model(
        &self,
        session_id: &str,
        model_id: &str,
    ) -> Result<(String, String)> {
        let new_config = self.app_config.resolve(Some(model_id))?;
        self.apply_session_config(session_id, new_config).await
    }

    pub async fn set_session_mode(&self, session_id: &str, mode_id: &str) -> Result<()> {
        let mode = AgentMode::parse(mode_id)?;
        let mut sessions = self.sessions.write().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::NotFound(format!("session not found: {session_id}")))?;
        s.mode = mode;
        s.last_active = Instant::now();
        Ok(())
    }

    /// Runs the history write `write` off the async runtime thread, logging
    /// (not propagating) any failure — history durability is best-effort and
    /// must never fail a turn that otherwise succeeded. Returns whether it
    /// succeeded. `context` names the write in the warning log line.
    async fn persist(
        &self,
        context: &str,
        write: impl FnOnce(&HistoryManager) -> Result<()> + Send + 'static,
    ) -> bool {
        let history = self.memory.history.clone();
        match tokio::task::spawn_blocking(move || write(&history))
            .await
            .unwrap_or_else(|e| Err(Error::from(e)))
        {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("failed to {context}: {e}");
                false
            }
        }
    }

    /// Runs one prompt turn to completion and returns why it stopped, so the
    /// caller can map it to an ACP `agent_client_protocol::schema::StopReason`
    /// directly instead of having to reverse-engineer it (e.g. by polling
    /// session state for cancellation after the fact).
    ///
    /// `on_update` sees every [`StreamEvent`] the turn produces, including
    /// ones with no ACP wire equivalent (`IterationStart`, `Usage`,
    /// `Finished`, `MessageAppended`) — mapping onto ACP's `SessionUpdate` is
    /// the caller's concern (see `acp::util::stream_event_to_session_update`),
    /// not this runtime's.
    pub async fn prompt<F>(
        &self,
        session_id: &str,
        prompt: Vec<ContentBlock>,
        permission_gate: Arc<dyn PermissionGate>,
        client_io: Arc<dyn ClientIo>,
        mut on_update: F,
    ) -> Result<CoreStopReason>
    where
        F: FnMut(StreamEvent) + Send,
    {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| Error::InvalidArgument("invalid session id format".to_string()))?;

        let (llm, executor, config, chat_id, skills, cwd, cancel, approvals, _prompt_guard) = {
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
            let executor: Arc<dyn ToolExecutor> = if s.mode == AgentMode::Architect {
                // Every read-only tool is available in Architect mode — built-in
                // or custom — derived from each `ToolHandler`'s own declared
                // `capabilities()` instead of a hand-maintained name list.
                let read_only: Vec<String> = self
                    .executor
                    .list_tools()
                    .into_iter()
                    .map(|t| t.function.name)
                    .filter(|name| self.executor.capabilities(name).read_only)
                    .collect();
                Arc::new(ScopedExecutor::new(self.executor.clone(), read_only))
            } else {
                // Subagents without a model of their own fall back to this
                // session's model, not the one `delegate_task` was built
                // with at startup. (Architect mode needs no override:
                // `delegate_task` isn't read-only, so it's filtered out.)
                let delegate = self.delegate.for_session(llm.clone(), s.config.clone());
                Arc::new(OverlayExecutor::new(
                    self.executor.clone(),
                    Arc::new(delegate),
                ))
            };
            (
                llm,
                executor,
                s.config.clone(),
                s.chat_id,
                s.skills.clone(),
                s.cwd.clone(),
                s.cancel.clone(),
                s.approved_tools.clone(),
                prompt_guard,
            )
        };
        // The caller's gate only asks; remembering `*Always` answers for the
        // rest of the session happens here, the same for every front-end.
        let permission_gate: Arc<dyn PermissionGate> = Arc::new(RememberingGate::new(
            permission_gate,
            approvals,
            executor.clone(),
        ));

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

        // Loads the conversation, `system.md` and skills from disk, so it runs
        // off the async runtime like every other history read.
        let memory = self.memory.clone();
        let (model, provider) = (config.model.clone(), config.provider_name.clone());
        let (mut conversation, prompt_builder) = tokio::task::spawn_blocking(move || {
            memory.prepare(Some(chat_id), &skills, Some(model), Some(provider))
        })
        .await
        .map_err(Error::from)??;

        conversation.meta.cwd = Some(cwd);
        // `prepare` only applies model/provider when it creates the
        // conversation; an existing one comes back with whatever was saved
        // before. Overwrite with the session's live values so a mid-session
        // model switch is persisted below and survives a reload instead of
        // reverting to the original model.
        conversation.meta.model = Some(config.model.clone());
        conversation.meta.provider = Some(config.provider_name.clone());
        let user_message = Message {
            role: Role::User,
            content: prompt,
        };
        conversation.meta.fill_title_from(&user_message);
        conversation.messages.push(user_message.clone());

        // Record the turn's metadata and user message before the turn starts,
        // so both survive a crash mid-turn. Everything earlier is already in
        // the log, so this appends rather than rewriting it.
        let meta = conversation.meta.clone();
        let started_ok = self
            .persist("record turn start", move |history| {
                history.save_meta(&meta)?;
                history.append_message(&chat_id, &user_message)
            })
            .await;

        // Set once any write fails; from then on nothing more is appended and
        // the end of the turn rewrites the whole log instead. Appending past
        // a failure would leave a gap in the log (or bury a half-written line
        // mid-file), and `save_conversation` then refuses the rewrite, or
        // can't read the log at all, so the history could never recover.
        let write_failed = AtomicBool::new(!started_ok);
        let write_failed_flag = &write_failed;
        let history_for_append = self.memory.history.clone();
        // The work-directory boundary and client I/O hook reach every tool
        // through this context; there is no per-session executor wrapper.
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
            work_dir: &self.work_dir,
            client_io: &*client_io,
        };
        let run_result = run_agent_streaming_with_history(
            llm,
            executor,
            &config,
            &mut conversation.messages,
            Some(&prompt_builder),
            &turn,
            move |event| {
                // Blocking I/O called synchronously (not via `spawn_blocking`)
                // deliberately: appends must land in the log in the same
                // order messages are produced, and this closure already runs
                // strictly sequentially with the rest of the turn, so a
                // small, fast local-disk append here doesn't race anything —
                // spawning it would only risk two concurrent appends landing
                // out of order.
                if let StreamEvent::MessageAppended { message } = &event
                    && !write_failed_flag.load(Ordering::Relaxed)
                    && let Err(e) = history_for_append.append_message(&chat_id, message)
                {
                    tracing::warn!("failed to append message to history: {e}");
                    write_failed_flag.store(true, Ordering::Relaxed);
                }
                on_update(event);
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

        // Every message is normally in the log by now, so only the metadata
        // (context usage, `updated_at`) needs writing. If any write above
        // failed, rewrite the whole log from memory instead, which
        // `save_conversation` refuses to do if another process has written
        // to it meanwhile.
        if !write_failed.load(Ordering::Relaxed) {
            let meta = conversation.meta.clone();
            self.persist("save conversation metadata", move |history| {
                history.save_meta(&meta)
            })
            .await;
        } else {
            self.persist(
                "rewrite conversation after a failed write",
                move |history| history.save_conversation(&conversation),
            )
            .await;
        }

        run_result.map(|r| r.stop_reason)
    }

    /// Persisted session metadata (all or filtered by `cwd`); the ACP
    /// `SessionInfo` shape is the caller's concern.
    pub async fn list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<ConversationMeta>> {
        let history = self.memory.history.clone();
        let metas = tokio::task::spawn_blocking(move || history.list_conversations())
            .await
            .map_err(Error::from)??;
        Ok(metas
            .into_iter()
            .filter(|m| cwd.is_none_or(|filter| m.cwd.as_deref() == Some(filter)))
            .collect())
    }

    /// Loads a persisted session as the active session for `session_id`,
    /// returning the mode it's live under, its full message history (for the
    /// caller to replay in whatever form its transport needs), and a
    /// warning to surface if the session's saved provider no longer
    /// resolves.
    pub async fn load_session(&self, session_id: &str, cwd: PathBuf) -> Result<LoadedSession> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| Error::InvalidArgument("invalid session id format".to_string()))?;

        let history = self.memory.history.clone();
        let conversation = tokio::task::spawn_blocking(move || history.load_conversation(&uuid))
            .await
            .map_err(Error::from)??;

        let mut session_config = self.config.clone();
        let mut warning = None;
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
                    warning = Some(format!(
                        "[warning] Could not restore this session's provider '{}' ({e}). Falling back to the default provider '{}'.",
                        provider_name, session_config.provider_name
                    ));
                }
            }
        } else if let Some(model) = &conversation.meta.model {
            session_config.model = model.clone();
        }

        let (mode, model) = {
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
                    approved_tools: Approvals::default(),
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
                return Err(Error::SessionBusy {
                    session_id: session_id.to_string(),
                });
            }
            // Read back the mode and model so the response reflects whatever
            // is actually live for this session (e.g. a prior
            // `session/set_config_option`), not the fresh-session default or
            // the just-loaded disk snapshot.
            (live.mode, live.config.model.clone())
        };

        Ok(LoadedSession {
            mode,
            messages: conversation.messages,
            model,
            warning,
        })
    }
}

/// The result of [`AgentState::load_session`]: enough to both replay the
/// conversation in whatever wire form the caller needs (see
/// `acp::util::replay_history_messages` for the ACP shape) and reflect the
/// mode it's now live under.
pub struct LoadedSession {
    pub mode: AgentMode,
    pub messages: Vec<Message>,
    /// The session's active model after resolution (falls back to the
    /// default provider's model if the saved provider/model no longer
    /// resolves — see `warning`).
    pub model: String,
    /// Set if the session's saved provider/model no longer resolves and the
    /// load fell back to the default provider.
    pub warning: Option<String>,
}

#[cfg(test)]
mod prompt_lease_ordering_tests {
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
            approved_tools: Approvals::default(),
            mode: AgentMode::Code,
            prompt_lock: Arc::new(Mutex::new(())),
            last_active: Instant::now(),
        }
    }

    // The ordering `Self::prompt` relies on: the in-process `prompt_lock`
    // must be acquired (and fail fast on an overlapping call) *before* the
    // cross-process `SessionLease` is acquired. Getting this backwards lets
    // an overlapping, rejected `session/prompt` call create its own lease
    // guard and then drop it (`SessionLease::drop` can't tell that apart
    // from a legitimately superseded one) — deleting the still-running
    // accepted turn's lockfile out from under it.
    #[test]
    fn overlapping_prompt_in_same_process_never_touches_the_accepted_turns_lease() {
        let dir = tempdir().unwrap();
        let history = HistoryManager::with_dir(dir.path().to_path_buf());
        let chat_id = Uuid::new_v4();
        let lock_path = dir.path().join(format!("{chat_id}.lock"));
        let state = sample_session_state(chat_id);

        // Turn A: accepted, in the order `Self::prompt` uses.
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

#[cfg(test)]
mod new_session_tests {
    use tempfile::tempdir;

    use crate::config::ProviderConfig;

    use super::*;

    /// A minimal, network-free `AgentState`: one provider/model, empty MCP
    /// servers, everything rooted at a temp `data_dir`.
    async fn sample_state(dir: &std::path::Path) -> AgentState {
        let mut app_config = AppConfig::for_tests("mock");
        app_config.max_iterations = 5;
        app_config.work_dir = Some(dir.to_path_buf());
        app_config.providers.insert(
            "mock".to_string(),
            ProviderConfig::for_tests("https://example.com", &["mock-model", "other-model"]),
        );
        let paths = RuntimePaths {
            data_dir: dir.to_path_buf(),
            config_path: dir.join("config.toml"),
        };
        let agent_config = AgentConfig::new(
            "mock".into(),
            "https://example.com".into(),
            "key".into(),
            "mock-model".into(),
            5,
        );
        let memory = MemoryContext::new(vec![], dir).unwrap();
        AgentState::new(agent_config, app_config, paths, memory, vec![])
            .await
            .unwrap()
    }

    // Regression test for `new_session` swallowing a bad `model` override:
    // it used to fall back to the session default silently
    // (`resolve(model).ok().unwrap_or_else(default)`) instead of surfacing
    // the `ConfigError` `resolve` returns for an unknown model.
    #[tokio::test]
    async fn bad_model_override_is_an_error() {
        let dir = tempdir().unwrap();
        let state = sample_state(dir.path()).await;
        let result = state
            .new_session(Some("nope"), vec![], dir.path().to_path_buf())
            .await;
        assert!(result.is_err(), "{result:?}");
    }

    /// Always answers with a final text reply, so a turn completes in one
    /// LLM call without touching the network.
    struct EndTurnLlm;

    #[async_trait::async_trait]
    impl LlmClient for EndTurnLlm {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[crate::core::models::Tool],
        ) -> Result<crate::core::models::Choice> {
            Ok(crate::core::models::Choice {
                message: Message::assistant("ok"),
                finish_reason: Some(crate::core::models::FinishReason::Stop),
                usage: None,
            })
        }
    }

    /// Runs one "hi" turn on `session_id`, allowing every tool call.
    async fn run_turn(state: &AgentState, session_id: &str) {
        state
            .prompt(
                session_id,
                vec![ContentBlock::from("hi")],
                Arc::new(crate::core::permission::AllowAll),
                Arc::new(crate::core::client_io::NoClientIo),
                |_| {},
            )
            .await
            .unwrap();
    }

    /// A state answering every turn with `EndTurnLlm`, plus a fresh session.
    async fn state_with_session(dir: &std::path::Path) -> (AgentState, String) {
        std::fs::write(dir.join("system.md"), "You are a test agent.").unwrap();
        let mut state = sample_state(dir).await;
        state.llm = Arc::new(EndTurnLlm);
        let session_id = state
            .new_session(None, vec![], dir.to_path_buf())
            .await
            .unwrap();
        (state, session_id)
    }

    // A turn appends to the message log instead of rewriting it: a rewrite
    // replaces the file (temp file + rename), which gives it a new inode.
    #[cfg(unix)]
    #[tokio::test]
    async fn turn_appends_to_the_log_instead_of_rewriting_it() {
        use std::os::unix::fs::MetadataExt;

        let dir = tempdir().unwrap();
        let (state, session_id) = state_with_session(dir.path()).await;
        let log = dir
            .path()
            .join("history")
            .join(format!("{session_id}.jsonl"));

        run_turn(&state, &session_id).await;
        let inode = std::fs::metadata(&log).unwrap().ino();
        run_turn(&state, &session_id).await;
        assert_eq!(
            std::fs::metadata(&log).unwrap().ino(),
            inode,
            "the second turn must append, not rewrite the log"
        );

        let uuid = Uuid::parse_str(&session_id).unwrap();
        let saved = state.memory.history.load_conversation(&uuid).unwrap();
        let texts: Vec<_> = saved.messages.iter().filter_map(Message::text).collect();
        assert_eq!(texts, ["hi", "ok", "hi", "ok"]);
        assert_eq!(saved.meta.title.as_deref(), Some("hi"));
        assert_eq!(saved.meta.model.as_deref(), Some("mock-model"));
    }

    // If an append fails, the end of the turn rewrites the whole log from
    // memory, so the history still ends up complete.
    #[cfg(unix)]
    #[tokio::test]
    async fn failed_append_falls_back_to_a_full_rewrite() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let (state, session_id) = state_with_session(dir.path()).await;
        let log = dir
            .path()
            .join("history")
            .join(format!("{session_id}.jsonl"));

        run_turn(&state, &session_id).await;
        // Appends open the log for writing, so they now fail; the rewrite
        // replaces the file through its (still writable) directory.
        std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o444)).unwrap();
        run_turn(&state, &session_id).await;

        let uuid = Uuid::parse_str(&session_id).unwrap();
        let saved = state.memory.history.load_conversation(&uuid).unwrap();
        let texts: Vec<_> = saved.messages.iter().filter_map(Message::text).collect();
        assert_eq!(texts, ["hi", "ok", "hi", "ok"]);
    }

    /// Makes the message log writable again, then replies "ok": a write
    /// failure that clears up partway through a turn.
    #[cfg(unix)]
    struct RestoreLogPermissionsLlm(std::path::PathBuf);

    #[cfg(unix)]
    #[async_trait::async_trait]
    impl LlmClient for RestoreLogPermissionsLlm {
        async fn send(
            &self,
            messages: &[Message],
            tools: &[crate::core::models::Tool],
        ) -> Result<crate::core::models::Choice> {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o644)).unwrap();
            EndTurnLlm.send(messages, tools).await
        }
    }

    // Regression test (PR #61 review): after one failed write, later appends
    // kept going. Here the user message's append fails but the reply's would
    // succeed, which left a gap in the log that the end-of-turn rewrite then
    // refused to overwrite, losing the user message for good.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_transient_write_failure_still_ends_with_a_complete_log() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let (mut state, session_id) = state_with_session(dir.path()).await;
        let log = dir
            .path()
            .join("history")
            .join(format!("{session_id}.jsonl"));

        run_turn(&state, &session_id).await;
        std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o444)).unwrap();
        state.llm = Arc::new(RestoreLogPermissionsLlm(log.clone()));
        run_turn(&state, &session_id).await;

        let uuid = Uuid::parse_str(&session_id).unwrap();
        let saved = state.memory.history.load_conversation(&uuid).unwrap();
        let texts: Vec<_> = saved.messages.iter().filter_map(Message::text).collect();
        assert_eq!(texts, ["hi", "ok", "hi", "ok"]);
    }

    // Regression test: a model switched after the conversation's first turn
    // was never written to its meta file (`resolve_conversation` returns an
    // existing conversation as saved), so reloading the session brought
    // back the original model.
    #[tokio::test]
    async fn model_switch_is_persisted_to_conversation_meta() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("system.md"), "You are a test agent.").unwrap();
        let mut state = sample_state(dir.path()).await;
        state.llm = Arc::new(EndTurnLlm);

        let session_id = state
            .new_session(None, vec![], dir.path().to_path_buf())
            .await
            .unwrap();

        // First turn creates the conversation on disk under the default model.
        run_turn(&state, &session_id).await;

        state
            .set_session_model(&session_id, "other-model")
            .await
            .unwrap();
        // `client_for_config` reuses `state.llm` only when the session's
        // config matches `state.config`; line them up so the second turn
        // stays on the mock instead of building a real HTTP client.
        state.config = state.app_config.resolve(Some("other-model")).unwrap();
        run_turn(&state, &session_id).await;

        let uuid = Uuid::parse_str(&session_id).unwrap();
        let saved = state.memory.history.load_conversation(&uuid).unwrap();
        assert_eq!(saved.meta.model.as_deref(), Some("other-model"));
        assert_eq!(saved.meta.provider.as_deref(), Some("mock"));
    }

    /// Answers each call with the next scripted choice, whoever makes it —
    /// the orchestrator and any subagent share it when they share a client.
    struct ScriptedLlm(std::sync::Mutex<std::collections::VecDeque<crate::core::models::Choice>>);

    #[async_trait::async_trait]
    impl LlmClient for ScriptedLlm {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[crate::core::models::Tool],
        ) -> Result<crate::core::models::Choice> {
            self.0
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| Error::Other("script exhausted".into()))
        }
    }

    // Regression test: `delegate_task` captured the startup model/client, so
    // a subagent with no model of its own ignored the session's model switch.
    // Here the startup client is a real HTTP client (never reachable from
    // the test) and the session's client is the script, so the subagent's
    // answer only comes back if it ran on the session's client.
    #[tokio::test]
    async fn subagent_without_model_override_runs_on_the_sessions_model() {
        use crate::core::models::{Choice, FinishReason};

        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("system.md"), "You are a test agent.").unwrap();
        let mut state = sample_state(dir.path()).await;

        let delegate_call = Choice {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: crate::tools::DELEGATE_TOOL_NAME.into(),
                    arguments: r#"{"system_prompt":"You are a helper.","task":"say hi"}"#.into(),
                }],
            },
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
        };
        let text = |t: &str| Choice {
            message: Message::assistant(t),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };
        state.llm = Arc::new(ScriptedLlm(std::sync::Mutex::new(
            [delegate_call, text("hi from the subagent"), text("done")].into(),
        )));

        let session_id = state
            .new_session(None, vec![], dir.path().to_path_buf())
            .await
            .unwrap();
        state
            .set_session_model(&session_id, "other-model")
            .await
            .unwrap();
        // Same alignment as the test above: keeps the session's turn on
        // `state.llm` (the script) instead of building a real client.
        state.config = state.app_config.resolve(Some("other-model")).unwrap();

        let mut tool_results = Vec::new();
        state
            .prompt(
                &session_id,
                vec![ContentBlock::from("delegate something")],
                Arc::new(crate::core::permission::AllowAll),
                Arc::new(crate::core::client_io::NoClientIo),
                |event| {
                    if let StreamEvent::ToolResult { result, .. } = event {
                        tool_results.push(result);
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(tool_results, vec!["hi from the subagent".to_string()]);
    }
}
