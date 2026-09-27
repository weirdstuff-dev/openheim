use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "acp")]
use agent_client_protocol::schema::v1::SessionUpdate;
use uuid::Uuid;

#[cfg(feature = "acp")]
use crate::acp::util::{replay_history_messages, stream_event_to_session_update};
use crate::{
    config::{
        AgentConfig, AppConfig, McpServerConfig, ProviderConfig, RuntimePaths, TuiConfig,
        config_dir, config_path, load_config, load_config_from,
    },
    core::{
        client_io::{ClientIo, NoClientIo},
        models::{ContentBlock, StopReason, StreamEvent},
        permission::{AllowAll, PermissionGate},
        runtime::{AgentState, LoadedSession},
    },
    error::Result,
    mcp::McpServerStatus,
    memory::{Conversation, ConversationMeta, MemoryContext},
    tools::ToolHandler,
};

/// The main entry point for embedding openheim in your application.
///
/// Exposes all agent capabilities: sessions, history, RAG, MCP servers,
/// tools, and models. Cheap to clone; clones share one runtime.
#[derive(Clone)]
pub struct OpenheimClient {
    state: Arc<AgentState>,
}

impl OpenheimClient {
    /// Start building a client with programmatic config or a config file.
    pub fn builder() -> OpenheimBuilder {
        OpenheimBuilder::default()
    }

    /// Shorthand to start from a specific config file path.
    pub fn from_config(path: impl AsRef<Path>) -> OpenheimBuilder {
        OpenheimBuilder {
            config_path: Some(path.as_ref().to_path_buf()),
            ..Default::default()
        }
    }

    /// The runtime behind this client, for the transports (which hand it to
    /// `acp::serve`) and the TUI, so both build it through the builder.
    #[cfg(any(feature = "acp", feature = "tui"))]
    pub(crate) fn state(&self) -> &Arc<AgentState> {
        &self.state
    }

    // ── Sessions ──────────────────────────────────────────────────────────────

    /// Create a new session. Returns a builder to set model, skills, and cwd.
    pub fn new_session(&self) -> SessionBuilder<'_> {
        SessionBuilder {
            state: &self.state,
            model: None,
            skills: vec![],
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
        }
    }

    /// Saved sessions: all of them, or only those whose `cwd` matches.
    pub async fn list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<ConversationMeta>> {
        self.state.list_sessions(cwd).await
    }

    /// Loads a saved session and returns a live handle for it plus what was
    /// loaded (see [`LoadedSession`]; `SessionHandle::acp_replay` shows its
    /// messages as ACP updates). Like a new session, the handle starts with
    /// [`AllowAll`] and [`NoClientIo`].
    pub async fn resume_session(
        &self,
        session_id: &str,
        cwd: PathBuf,
    ) -> Result<(SessionHandle, LoadedSession)> {
        let loaded = self.state.load_session(session_id, cwd).await?;
        Ok((
            SessionHandle::new(session_id.to_string(), Arc::clone(&self.state)),
            loaded,
        ))
    }

    /// Fetch the full `Conversation` (messages + metadata) for a session id.
    pub async fn get_session(&self, session_id: &str) -> Result<Conversation> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| crate::error::Error::InvalidArgument("invalid session id".to_string()))?;
        let history = self.state.memory.history.clone();
        tokio::task::spawn_blocking(move || history.load_conversation(&uuid)).await?
    }

    /// Permanently delete a persisted session.
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| crate::error::Error::InvalidArgument("invalid session id".to_string()))?;
        let history = self.state.memory.history.clone();
        tokio::task::spawn_blocking(move || history.delete_conversation(&uuid)).await?
    }

    /// Switch a live session to another model mid-conversation; the same as
    /// [`SessionHandle::switch_model`], for callers that hold only the id.
    /// Returns `(provider_name, model_name)`.
    pub async fn switch_model(
        &self,
        session_id: &str,
        provider: &str,
        model: &str,
    ) -> Result<(String, String)> {
        self.state.switch_model(session_id, provider, model).await
    }

    // ── Memory ────────────────────────────────────────────────────────────────

    /// Names of the skills in the data directory's `skills/`, sorted.
    pub fn skills(&self) -> Result<Vec<String>> {
        self.state.memory.skills.list_skills()
    }

    /// The long-term memory behind the `remember` / `search_memory` /
    /// `edit_memory` / `forget` tools. Keyword-only unless `[memory]` names
    /// an embedding provider.
    #[cfg(feature = "rag")]
    pub fn long_term_memory(&self) -> &Arc<crate::rag::LongTermMemory> {
        &self.state.long_term_memory
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// All tool definitions available to the agent (built-in + MCP).
    pub fn tools(&self) -> Vec<crate::core::models::Tool> {
        self.state.executor.list_tools()
    }

    /// MCP server connection statuses.
    pub fn mcp_servers(&self) -> &[McpServerStatus] {
        &self.state.mcp_statuses
    }

    /// Available models per provider (no credentials).
    pub fn models(&self) -> crate::config::ModelsInfo {
        self.state.app_config.models_info()
    }
}

// ── SessionBuilder ────────────────────────────────────────────────────────────

/// Builder returned by `OpenheimClient::new_session()`.
pub struct SessionBuilder<'a> {
    state: &'a Arc<AgentState>,
    model: Option<String>,
    skills: Vec<String>,
    cwd: PathBuf,
}

impl<'a> SessionBuilder<'a> {
    /// Override the model for this session (must be listed in the config).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Skills to inject into the system prompt (names of `~/.openheim/skills/*.md` files).
    pub fn skills(mut self, skills: Vec<String>) -> Self {
        self.skills = skills;
        self
    }

    /// Working directory for this session: where its tools resolve relative
    /// paths and run commands when it's inside `work_dir` (otherwise they use
    /// `work_dir`). Also saved with the conversation for
    /// [`OpenheimClient::list_sessions`] filtering. Defaults to the process's
    /// current directory.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = cwd.into();
        self
    }

    /// Create the session and return a handle for prompting.
    pub async fn start(self) -> Result<SessionHandle> {
        let id = self
            .state
            .new_session(self.model.as_deref(), self.skills, self.cwd)
            .await?;
        Ok(SessionHandle::new(id, self.state.clone()))
    }
}

// ── SessionHandle ─────────────────────────────────────────────────────────────

/// A live session that can receive prompts.
pub struct SessionHandle {
    id: String,
    state: Arc<AgentState>,
    permission_gate: Arc<dyn PermissionGate>,
    client_io: Arc<dyn ClientIo>,
}

impl SessionHandle {
    fn new(id: String, state: Arc<AgentState>) -> Self {
        Self {
            id,
            state,
            permission_gate: Arc::new(AllowAll),
            client_io: Arc::new(NoClientIo),
        }
    }

    /// The [`PermissionGate`] asked before each of this session's tool calls.
    /// Defaults to [`AllowAll`]; an interactive embedder should set its own.
    pub fn permission_gate(mut self, gate: Arc<dyn PermissionGate>) -> Self {
        self.permission_gate = gate;
        self
    }

    /// Delegate `read_file`/`write_file` to the embedder's own I/O (e.g. an
    /// editor's unsaved buffers) before falling back to local disk. Defaults
    /// to [`NoClientIo`], which always uses local disk.
    pub fn client_io(mut self, io: Arc<dyn ClientIo>) -> Self {
        self.client_io = io;
        self
    }

    /// This session's id, for [`OpenheimClient::resume_session`] later.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Runs one turn on `input` (text, or text with images; see
    /// [`PromptInput`]) and returns why it stopped. `on_update` gets every
    /// [`StreamEvent`] of the turn: streamed text and thinking, tool calls
    /// and results, context usage, and the finish. To get ACP
    /// `SessionUpdate`s instead, pass `Self::acp_updates` (feature `acp`).
    pub async fn prompt(
        &self,
        input: impl Into<PromptInput>,
        on_update: impl FnMut(StreamEvent) + Send,
    ) -> Result<StopReason> {
        self.state
            .prompt(
                &self.id,
                input.into().blocks,
                self.permission_gate.clone(),
                self.client_io.clone(),
                on_update,
            )
            .await
    }

    /// Adapts an ACP `SessionUpdate` callback into one [`Self::prompt`]
    /// accepts: `session.prompt("hi", session.acp_updates(|update| …))`.
    /// Events with no ACP equivalent (`IterationStart`, `Usage`,
    /// `Finished`, `MessageAppended`) are dropped.
    #[cfg(feature = "acp")]
    pub fn acp_updates(
        &self,
        mut on_update: impl FnMut(SessionUpdate) + Send,
    ) -> impl FnMut(StreamEvent) + Send {
        let executor = self.state.executor.clone();
        move |event| {
            if let Some(update) = stream_event_to_session_update(event, executor.as_ref()) {
                on_update(update);
            }
        }
    }

    /// Replays `messages` (e.g. [`LoadedSession::messages`] from
    /// [`OpenheimClient::resume_session`]) as the ACP `SessionUpdate`s a live
    /// turn would have produced, thinking included (tagged via
    /// `_meta.kind`), so an ACP-speaking UI can show the conversation so far.
    #[cfg(feature = "acp")]
    pub fn acp_replay(
        &self,
        messages: &[crate::core::models::Message],
        mut on_update: impl FnMut(SessionUpdate),
    ) {
        replay_history_messages(messages, self.state.executor.as_ref(), &mut on_update);
    }

    /// This session's saved `ConversationMeta`, or `None` before its first
    /// turn (a new session isn't saved until then).
    async fn conversation_meta(&self) -> Result<Option<crate::memory::ConversationMeta>> {
        let uuid = Uuid::parse_str(&self.id)
            .map_err(|_| crate::error::Error::InvalidArgument("invalid session id".to_string()))?;
        let history = self.state.memory.history.clone();
        match tokio::task::spawn_blocking(move || history.load_conversation(&uuid)).await? {
            Ok(conversation) => Ok(Some(conversation.meta)),
            Err(crate::error::Error::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Snapshot of the most recent turn's context size — the last LLM
    /// call's usage, i.e. how full the context window is right now. `None`
    /// if no turn has completed yet.
    pub async fn context_usage(&self) -> Result<Option<crate::core::models::Usage>> {
        Ok(self
            .conversation_meta()
            .await?
            .and_then(|meta| meta.context_usage))
    }

    /// Cancels the turn currently in flight for this session, if any.
    /// No-op if no prompt is running.
    pub async fn cancel(&self) {
        self.state.cancel_session(&self.id).await;
    }

    /// Switch the model for this session mid-conversation.
    ///
    /// The model must be listed under a provider in the config. Returns
    /// `(provider_name, model_name)` on success; the next prompt will use
    /// the new model while preserving conversation history.
    pub async fn switch_model(&self, provider: &str, model: &str) -> Result<(String, String)> {
        self.state.switch_model(&self.id, provider, model).await
    }
}

// ── PromptInput ───────────────────────────────────────────────────────────────

/// What one [`SessionHandle::prompt`] sends. Plain text converts directly
/// (`session.prompt("hi", …)`); add images with the builder:
/// `PromptInput::text("what is this?").image(base64_data, "image/png")`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptInput {
    blocks: Vec<ContentBlock>,
}

impl PromptInput {
    /// A prompt with `text` (no block at all if it's empty, so an
    /// image-only prompt can start from `PromptInput::default()`).
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        let blocks = if text.is_empty() {
            Vec::new()
        } else {
            vec![ContentBlock::from(text)]
        };
        Self { blocks }
    }

    /// Adds an image: `data` is the raw base64 payload (e.g. of a `data:`
    /// URL), `mime_type` e.g. `"image/png"`. Images follow the text, in the
    /// order they're added.
    pub fn image(mut self, data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        self.blocks.push(ContentBlock::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        });
        self
    }
}

impl From<&str> for PromptInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<String> for PromptInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&String> for PromptInput {
    fn from(text: &String) -> Self {
        Self::text(text.as_str())
    }
}

/// Any content blocks as-is, for prompts the builder doesn't cover.
impl From<Vec<ContentBlock>> for PromptInput {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self { blocks }
    }
}

// ── OpenheimBuilder ───────────────────────────────────────────────────────────

/// Builder for `OpenheimClient`.
///
/// Supports three modes:
/// 1. **Programmatic** — set `.provider()`, `.api_key()`, or `.api_base()`
///    directly, building the whole config from scratch.
/// 2. **File-based** — call `OpenheimClient::from_config(path)` or leave
///    everything unset to load from `~/.openheim/config.toml`.
/// 3. **Given config** — pass a whole [`AppConfig`] with `.app_config()`,
///    e.g. one loaded with `load_config_from` and then adjusted. No file is
///    read.
///
/// `.model()` works in every mode: it picks a configured model in the
/// file-based and given-config modes, and is the provider's model in
/// programmatic mode.
///
/// MCP servers can be added in any mode with `.mcp_server()`.
#[derive(Default)]
pub struct OpenheimBuilder {
    // file-based path (None = ~/.openheim/config.toml)
    config_path: Option<PathBuf>,
    // Given-config mode: used instead of reading `config_path`.
    app_config: Option<AppConfig>,
    // Programmatic fields: setting any of these but `model` skips the
    // config file.
    provider: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    api_base: Option<String>,
    max_iterations: Option<usize>,
    timeout_secs: Option<u64>,
    max_tokens: Option<u32>,
    context_window: Option<u64>,
    mcp_servers: BTreeMap<String, McpServerConfig>,
    default_skills: Vec<String>,
    work_dir: Option<PathBuf>,
    allow_shell: Option<bool>,
    data_dir: Option<PathBuf>,
    tools: Vec<Box<dyn ToolHandler>>,
    #[cfg(feature = "rag")]
    long_term_memory: Option<crate::rag::LongTermMemory>,
}

impl OpenheimBuilder {
    /// Path to a config file (overrides `~/.openheim/config.toml`).
    pub fn config_path(mut self, path: impl AsRef<Path>) -> Self {
        self.config_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Use `config` as the whole configuration instead of reading a file.
    /// `config_path` still names the file that config writers (the TUI's
    /// `:theme`) update. Can't be combined with `.provider()`, `.api_key()`
    /// or `.api_base()`; set those on the config's provider entry instead.
    pub fn app_config(mut self, config: AppConfig) -> Self {
        self.app_config = Some(config);
        self
    }

    /// Provider name: `"openai"`, `"anthropic"`, `"gemini"`, or any custom name
    /// for OpenAI-compatible endpoints.
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// API key for the provider.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Model name (e.g. `"claude-opus-4-7"`, `"gpt-4o"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Override the provider API base URL (useful for proxies or local models).
    pub fn api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = Some(base.into());
        self
    }

    /// Maximum number of agent iterations before stopping.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Connect and idle-read timeout in seconds. This bounds the connect
    /// phase and the maximum gap between body reads — not the total request
    /// duration — so long streaming generations aren't cut off mid-stream.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Maximum output tokens for LLM responses.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// The model's context window in tokens. Requests estimated at over 90%
    /// of it leave out the oldest turns; see `context_window` in
    /// `docs/configuration.md`.
    pub fn context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Register an MCP server. Tools will be available as `{name}__{tool_name}`.
    pub fn mcp_server(mut self, name: impl Into<String>, config: McpServerConfig) -> Self {
        self.mcp_servers.insert(name.into(), config);
        self
    }

    /// Skills loaded automatically in every new session.
    pub fn default_skills(mut self, skills: Vec<String>) -> Self {
        self.default_skills = skills;
        self
    }

    /// Root directory the agent is allowed to read/write.
    /// Overrides `work_dir` from the config file. When not set, defaults to the
    /// directory from which the process was invoked.
    pub fn work_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.work_dir = Some(path.into());
        self
    }

    /// Whether to expose the `execute_command` shell tool to the LLM.
    /// Overrides `allow_shell` from the config file. Defaults to `false`.
    pub fn allow_shell(mut self, allow: bool) -> Self {
        self.allow_shell = Some(allow);
        self
    }

    /// Directory backing history, skills, `system.md`, subagent profiles,
    /// and (absent an explicit `[memory].db_path`) the memory database.
    /// Overrides `data_dir` from the config file. Defaults to `~/.openheim`.
    pub fn data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.data_dir = Some(path.into());
        self
    }

    /// Registers a custom tool (see [`crate::tools::ToolHandler`]) alongside
    /// the built-in and MCP tools. Call once per tool.
    pub fn tool(mut self, handler: Box<dyn ToolHandler>) -> Self {
        self.tools.push(handler);
        self
    }

    /// The long-term memory behind the `remember` / `search_memory` /
    /// `edit_memory` / `forget` tools, instead of the one the config's
    /// `[memory]` section describes. Use it for a custom
    /// [`EmbeddingClient`](crate::rag::EmbeddingClient):
    /// `LongTermMemory::new(VectorStore::open(path)?, Some(embedder), top_k)`.
    #[cfg(feature = "rag")]
    pub fn long_term_memory(mut self, memory: crate::rag::LongTermMemory) -> Self {
        self.long_term_memory = Some(memory);
        self
    }

    /// Build the client, connecting to MCP servers and initialising the agent state.
    pub async fn build(mut self) -> Result<OpenheimClient> {
        let programmatic =
            self.provider.is_some() || self.api_key.is_some() || self.api_base.is_some();
        if programmatic && self.app_config.is_some() {
            return Err(crate::error::Error::ConfigError(
                "app_config can't be combined with provider, api_key or api_base; \
                 set them on the config's provider entry"
                    .to_string(),
            ));
        }
        let (agent_config, mut app_config) = if programmatic {
            self.programmatic_config()?
        } else {
            let app_config = match (self.app_config.take(), &self.config_path) {
                (Some(config), _) => config,
                (None, Some(path)) => load_config_from(path)?,
                (None, None) => load_config()?,
            };
            let mut agent_config = app_config.resolve(self.model.as_deref())?;
            if let Some(n) = self.max_iterations {
                agent_config.max_iterations = n;
            }
            if let Some(s) = self.timeout_secs {
                agent_config.timeout_secs = s;
            }
            if let Some(t) = self.max_tokens {
                agent_config.max_tokens = Some(t);
            }
            if let Some(w) = self.context_window {
                agent_config.context_window = Some(w);
            }
            (agent_config, app_config)
        };

        // Merge any extra MCP servers from the builder
        for (name, cfg) in self.mcp_servers {
            app_config.mcp_servers.insert(name, cfg);
        }

        // Apply builder default_skills for the file-based path (programmatic path sets them directly)
        if !self.default_skills.is_empty() {
            app_config.default_skills = self.default_skills;
        }

        // The builder's `work_dir` wins over the config's. Either is resolved
        // once, here, so a bad one fails the build instead of the first tool
        // call.
        if let Some(wd) = self.work_dir.or_else(|| app_config.work_dir.take()) {
            app_config.work_dir = Some(resolve_work_dir(&wd)?);
        }
        if let Some(shell) = self.allow_shell {
            app_config.allow_shell = shell;
        }
        // Resolved once here; everything downstream reads `RuntimePaths`.
        let paths = RuntimePaths {
            data_dir: match self.data_dir.or_else(|| app_config.data_dir.clone()) {
                Some(dir) => dir,
                None => config_dir()?,
            },
            config_path: match self.config_path {
                Some(path) => path,
                None => config_path()?,
            },
        };

        let memory = MemoryContext::new(app_config.default_skills.clone(), &paths.data_dir)?;
        let state = AgentState::new(
            agent_config,
            app_config,
            paths,
            memory,
            self.tools,
            #[cfg(feature = "rag")]
            self.long_term_memory,
        )
        .await?;
        Ok(OpenheimClient {
            state: Arc::new(state),
        })
    }

    /// The config for programmatic mode (no config file): a single provider
    /// built from this builder's `provider`/`api_key`/`model`/`api_base`,
    /// with built-in defaults for whatever is unset.
    fn programmatic_config(&self) -> Result<(AgentConfig, AppConfig)> {
        let provider = self
            .provider
            .clone()
            .unwrap_or_else(|| "openai".to_string());
        let (default_api_base, default_model) = crate::config::builtin_provider_defaults(&provider);
        let api_base = self
            .api_base
            .clone()
            .unwrap_or_else(|| default_api_base.to_string());
        let model = self
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_string());
        let api_key = self.api_key.clone().unwrap_or_default();
        let max_iter = self.max_iterations.unwrap_or(10);
        let timeout = self
            .timeout_secs
            .unwrap_or_else(crate::config::default_timeout_secs);

        let mut providers = BTreeMap::new();
        providers.insert(
            provider.clone(),
            ProviderConfig {
                kind: None,
                api_base,
                default_model: model.clone(),
                models: vec![model],
                env_var: None,
                api_key: Some(api_key),
                timeout_secs: Some(timeout),
                max_tokens: self.max_tokens,
                context_window: self.context_window,
                thinking: None,
            },
        );

        let app_config = AppConfig {
            default_provider: provider.clone(),
            max_iterations: max_iter,
            tui: TuiConfig::default(),
            providers,
            mcp_servers: BTreeMap::new(),
            default_skills: self.default_skills.clone(),
            work_dir: None,
            allow_shell: false,
            memory: None,
            data_dir: None,
        };

        let agent_config = app_config.resolve_provider_default(&provider)?;
        Ok((agent_config, app_config))
    }
}

/// `work_dir` as the sandbox root: absolute (a relative path is taken from
/// the process's current directory), symlinks resolved, and an existing
/// directory, or a `ConfigError` saying why not.
fn resolve_work_dir(work_dir: &Path) -> Result<PathBuf> {
    let config_error = |why: String| {
        crate::error::Error::ConfigError(format!("work_dir '{}' {why}", work_dir.display()))
    };
    let absolute = if work_dir.is_absolute() {
        work_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| {
                config_error(format!(
                    "is relative, and the current directory is unknown: {e}"
                ))
            })?
            .join(work_dir)
    };
    let canonical = absolute
        .canonicalize()
        .map_err(|e| config_error(format!("is inaccessible: {e}")))?;
    if !canonical.is_dir() {
        return Err(config_error("is not a directory".to_string()));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build()` resolves `RuntimePaths` once: the builder's `data_dir` wins,
    /// and `config_path` falls back to the default location. Uses the
    /// programmatic path (`provider`/`api_key`/`model` set) so this stays
    /// mock-free — no config file, no MCP servers, no network.
    #[tokio::test]
    async fn build_resolves_runtime_paths() {
        let dir = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .provider("openai")
            .api_key("test-key")
            .model("gpt-4o")
            .data_dir(dir.path())
            .build()
            .await
            .unwrap();
        let paths = client.state.paths();
        assert_eq!(paths.data_dir, dir.path());
        assert_eq!(paths.config_path, config_path().unwrap());
        // The data dir is actually used, not just recorded.
        assert!(dir.path().join("history").is_dir());
    }

    /// `resume_session` works without the `acp` feature: it hands back the
    /// persisted `Message`s straight from `HistoryManager`. Writes a
    /// conversation directly via the client's `HistoryManager` (mock-free,
    /// no LLM call) and resumes it by id.
    #[tokio::test]
    async fn resume_session_loads_a_conversation_written_via_history_manager() {
        let dir = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .provider("openai")
            .api_key("test-key")
            .model("gpt-4o")
            .data_dir(dir.path())
            .build()
            .await
            .unwrap();

        let history = client.state.memory.history.clone();
        let mut conv = history
            .create_conversation(Some("gpt-4o".into()), Some("openai".into()), vec![])
            .unwrap();
        conv.messages.push(crate::core::models::Message::user("hi"));
        conv.messages
            .push(crate::core::models::Message::assistant("hello there"));
        history.save_conversation(&conv).unwrap();

        let (handle, loaded) = client
            .resume_session(&conv.meta.id.to_string(), dir.path().to_path_buf())
            .await
            .unwrap();

        assert_eq!(handle.id(), conv.meta.id.to_string());
        assert_eq!(loaded.messages, conv.messages);
        assert!(loaded.warning.is_none());
    }

    fn local_config() -> AppConfig {
        AppConfig::new("local").with_provider(
            "local",
            ProviderConfig::new("http://127.0.0.1:1/v1", "m1").with_models(["m1", "m2"]),
        )
    }

    /// A given `AppConfig` is used as-is: no file is read (the config path
    /// doesn't exist), and `.model()` picks from its providers.
    #[tokio::test]
    async fn build_uses_a_given_app_config() {
        let dir = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .app_config(local_config())
            .config_path(dir.path().join("missing.toml"))
            .model("m2")
            .data_dir(dir.path())
            .work_dir(dir.path())
            .build()
            .await
            .unwrap();

        let models = client.models();
        assert_eq!(models.default_provider, "local");
        assert_eq!(models.providers["local"].models, ["m1", "m2"]);
        assert_eq!(client.state.config().model, "m2");
    }

    async fn build_with_config_work_dir(
        work_dir: &Path,
        data_dir: &Path,
    ) -> Result<OpenheimClient> {
        OpenheimClient::builder()
            .app_config(local_config().with_work_dir(work_dir))
            .data_dir(data_dir)
            .build()
            .await
    }

    // A relative config `work_dir` is taken from the current directory and
    // stored absolute, so the sandbox doesn't move if the cwd does.
    #[tokio::test]
    async fn a_relative_config_work_dir_is_resolved_at_build() {
        let data = tempfile::tempdir().unwrap();
        let here = std::env::current_dir().unwrap();
        let work = tempfile::tempdir_in(&here).unwrap();
        let relative = work.path().strip_prefix(&here).unwrap();
        assert!(relative.is_relative());

        let client = build_with_config_work_dir(relative, data.path())
            .await
            .unwrap();

        assert_eq!(client.state.work_dir, work.path().canonicalize().unwrap());
    }

    #[tokio::test]
    async fn a_missing_config_work_dir_fails_the_build() {
        let data = tempfile::tempdir().unwrap();
        let missing = data.path().join("no-such-dir");

        let err = build_with_config_work_dir(&missing, data.path())
            .await
            .err()
            .unwrap();

        assert!(matches!(err, crate::error::Error::ConfigError(_)), "{err}");
        assert!(err.to_string().contains("no-such-dir"), "{err}");
    }

    #[tokio::test]
    async fn a_config_work_dir_that_is_a_file_fails_the_build() {
        let data = tempfile::tempdir().unwrap();
        let file = data.path().join("file.txt");
        std::fs::write(&file, "").unwrap();

        let err = build_with_config_work_dir(&file, data.path())
            .await
            .err()
            .unwrap();

        assert!(err.to_string().contains("not a directory"), "{err}");
    }

    // A symlinked config `work_dir` becomes its target, which is what the
    // file tools compare paths against.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_config_work_dir_is_resolved_to_its_target() {
        let data = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let link = data.path().join("link");
        std::os::unix::fs::symlink(target.path(), &link).unwrap();

        let client = build_with_config_work_dir(&link, data.path())
            .await
            .unwrap();

        assert_eq!(client.state.work_dir, target.path().canonicalize().unwrap());
        let public = client.state.app_config.to_public(&client.state.work_dir);
        assert_eq!(
            public.work_dir,
            target.path().canonicalize().unwrap().display().to_string()
        );
    }

    // Same check for a `work_dir` read from a config file.
    #[tokio::test]
    async fn a_bad_work_dir_in_the_config_file_fails_the_build() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
default_provider = "local"
work_dir = "/no/such/openheim/work/dir"

[providers.local]
api_base = "http://127.0.0.1:1/v1"
default_model = "m1"
models = ["m1"]
"#,
        )
        .unwrap();

        let err = OpenheimClient::from_config(&config)
            .data_dir(dir.path())
            .build()
            .await
            .err()
            .unwrap();

        assert!(
            err.to_string().contains("/no/such/openheim/work/dir"),
            "{err}"
        );
    }

    // The builder's `work_dir` replaces the config's, even a bad one.
    #[tokio::test]
    async fn the_builder_work_dir_overrides_the_configs() {
        let data = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .app_config(local_config().with_work_dir("/no/such/openheim/work/dir"))
            .work_dir(data.path())
            .data_dir(data.path())
            .build()
            .await
            .unwrap();

        assert_eq!(client.state.work_dir, data.path().canonicalize().unwrap());
    }

    #[tokio::test]
    async fn app_config_with_programmatic_fields_is_rejected() {
        let result = OpenheimClient::builder()
            .app_config(local_config())
            .api_key("k")
            .build()
            .await;
        assert!(matches!(result, Err(crate::error::Error::ConfigError(_))));
    }

    #[tokio::test]
    async fn skills_lists_the_skill_files() {
        let dir = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .app_config(local_config())
            .data_dir(dir.path())
            .work_dir(dir.path())
            .build()
            .await
            .unwrap();
        std::fs::write(dir.path().join("skills/rust.md"), "# Rust").unwrap();
        std::fs::write(dir.path().join("skills/go.md"), "# Go").unwrap();

        assert_eq!(client.skills().unwrap(), ["go", "rust"]);
    }

    /// A long-term memory given to the builder is the one the client (and
    /// its memory tools) use, not a fresh one under `data_dir`.
    #[cfg(feature = "rag")]
    #[tokio::test]
    async fn build_uses_a_given_long_term_memory() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let store = crate::rag::VectorStore::open(&elsewhere.path().join("m.db")).unwrap();
        let memory = crate::rag::LongTermMemory::new(store, None, 5);
        memory.remember("seeded before build").await.unwrap();

        let client = OpenheimClient::builder()
            .app_config(local_config())
            .data_dir(dir.path())
            .work_dir(dir.path())
            .long_term_memory(memory)
            .build()
            .await
            .unwrap();

        assert_eq!(client.long_term_memory().stats().await.unwrap().memories, 1);
        assert!(!dir.path().join("memory.db").exists());
    }

    /// `OpenheimClient::switch_model` reaches the same live session as the
    /// handle, from a clone of the client too.
    #[tokio::test]
    async fn client_switch_model_switches_a_live_session() {
        let dir = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .app_config(local_config())
            .data_dir(dir.path())
            .work_dir(dir.path())
            .build()
            .await
            .unwrap();
        let session = client.new_session().start().await.unwrap();

        let switched = client
            .clone()
            .switch_model(session.id(), "local", "m2")
            .await
            .unwrap();
        assert_eq!(switched, ("local".to_string(), "m2".to_string()));

        let err = client
            .switch_model("not-a-session", "local", "m2")
            .await
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::NotFound(_)), "{err}");
    }

    #[cfg(feature = "acp")]
    #[tokio::test]
    async fn acp_updates_maps_events_and_drops_ones_acp_has_no_room_for() {
        let dir = tempfile::tempdir().unwrap();
        let client = OpenheimClient::builder()
            .provider("openai")
            .api_key("test-key")
            .data_dir(dir.path())
            .build()
            .await
            .unwrap();
        let session = client.new_session().start().await.unwrap();

        let mut updates = Vec::new();
        let mut on_event = session.acp_updates(|update| updates.push(update));
        on_event(StreamEvent::LlmResponse {
            content: "hi".into(),
        });
        on_event(StreamEvent::Usage {
            usage: Default::default(),
        });
        drop(on_event);

        assert_eq!(updates.len(), 1);
        assert!(matches!(updates[0], SessionUpdate::AgentMessageChunk(_)));
    }

    #[test]
    fn prompt_input_puts_text_before_images_and_skips_empty_text() {
        let image = |data: &str| ContentBlock::Image {
            data: data.into(),
            mime_type: "image/png".into(),
        };
        assert_eq!(
            PromptInput::text("look")
                .image("a", "image/png")
                .image("b", "image/png"),
            PromptInput::from(vec![ContentBlock::from("look"), image("a"), image("b")])
        );
        assert_eq!(
            PromptInput::text("").image("a", "image/png"),
            PromptInput::from(vec![image("a")])
        );
        assert_eq!(PromptInput::from("hi"), PromptInput::text("hi"));
    }
}
