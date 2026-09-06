use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use agent_client_protocol::schema::{ContentBlock, ImageContent, SessionInfo, SessionUpdate};
use uuid::Uuid;

use crate::{
    config::{
        AgentConfig, AppConfig, McpServerConfig, ProviderConfig, load_config, load_config_from,
    },
    core::{
        client_io::{ClientIo, NoClientIo},
        permission::{AllowAll, PermissionGate},
        runtime::AgentState,
    },
    error::Result,
    mcp::McpServerStatus,
    memory::{Conversation, ConversationMeta, MemoryContext},
    tools::ToolHandler,
};

/// The main entry point for embedding openheim in your application.
///
/// Wraps an `AgentState` and exposes all agent capabilities:
/// sessions, history, RAG, MCP servers, tools, and models.
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

    /// The shared runtime handle behind this client. `pub(crate)` — for the
    /// transports (`transport::{run,stdio,ws}`), which need the raw
    /// `AgentState` to hand to `acp::serve`, so every entry point builds it
    /// the same way (config load, resolve, `MemoryContext::new`, custom
    /// tools) instead of each hand-rolling that sequence.
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

    /// List persisted sessions (all or filtered by cwd).
    pub async fn list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<SessionInfo>> {
        self.state.list_sessions(cwd).await
    }

    /// Load a persisted session into a live `SessionHandle`.
    ///
    /// `on_history` is called once for each message in the conversation history
    /// (as `SessionUpdate::UserMessageChunk` / `AgentMessageChunk`) so callers
    /// can replay the conversation in their UI.
    pub async fn load_session(
        &self,
        session_id: &str,
        cwd: PathBuf,
        on_history: impl FnMut(SessionUpdate) + Send,
    ) -> Result<SessionHandle> {
        self.state.load_session(session_id, cwd, on_history).await?;
        Ok(SessionHandle::new(
            session_id.to_string(),
            self.state.clone(),
        ))
    }

    /// Fetch the full `Conversation` (messages + metadata) for a session id.
    pub async fn get_session(&self, session_id: &str) -> Result<Conversation> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| crate::error::Error::ParseError("invalid session id".to_string()))?;
        let history = self.state.memory.history.clone();
        tokio::task::spawn_blocking(move || history.load_conversation(&uuid)).await?
    }

    /// List all conversation metadata without loading messages.
    pub async fn list_all_sessions(&self) -> Result<Vec<ConversationMeta>> {
        let history = self.state.memory.history.clone();
        tokio::task::spawn_blocking(move || history.list_conversations()).await?
    }

    /// Permanently delete a persisted session.
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| crate::error::Error::ParseError("invalid session id".to_string()))?;
        let history = self.state.memory.history.clone();
        tokio::task::spawn_blocking(move || history.delete_conversation(&uuid)).await?
    }

    // ── Memory ────────────────────────────────────────────────────────────────

    /// Direct access to the agent memory (history + skills managers).
    pub fn memory(&self) -> &MemoryContext {
        &self.state.memory
    }

    /// The long-term memory behind the `remember` / `search_memory` / `forget`
    /// tools. Keyword-only unless `[memory]` names an embedding provider.
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

    /// Working directory for this session (used for history filtering).
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
    pub id: String,
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

    /// Supply a permission gate consulted before every tool call this session
    /// makes (see [`PermissionGate`]). Defaults to [`AllowAll`] — the caller
    /// is trusted to have already consented to the run (e.g. `openheim run`,
    /// a one-shot library embedding). An interactive embedder (like the TUI)
    /// should set a real gate here instead of relying on the default.
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

    /// Send a prompt and stream ACP `SessionUpdate` events to `on_update`.
    ///
    /// The callback receives:
    /// - `SessionUpdate::AgentMessageChunk` — streaming text from the LLM
    /// - `SessionUpdate::ToolCall` — a tool the agent is about to invoke
    /// - `SessionUpdate::ToolCallUpdate` — result of the tool call
    pub async fn prompt(
        &self,
        text: &str,
        on_update: impl FnMut(SessionUpdate) + Send,
    ) -> Result<()> {
        self.prompt_with_images(text, Vec::new(), on_update).await
    }

    /// Send a prompt that mixes text with one or more images.
    ///
    /// Each image is `(base64_data, mime_type)` — e.g. the raw base64 payload
    /// of a `data:` URL and `"image/png"`. The text block (when non-empty)
    /// leads, followed by the images, matching the order a user composes them.
    /// Streams the same `SessionUpdate` events as [`prompt`].
    pub async fn prompt_with_images(
        &self,
        text: &str,
        images: Vec<(String, String)>,
        on_update: impl FnMut(SessionUpdate) + Send,
    ) -> Result<()> {
        let mut blocks: Vec<ContentBlock> = Vec::new();
        if !text.is_empty() {
            blocks.push(ContentBlock::from(text));
        }
        for (data, mime_type) in images {
            blocks.push(ContentBlock::Image(ImageContent::new(data, mime_type)));
        }
        self.state
            .prompt(
                &self.id,
                blocks,
                self.permission_gate.clone(),
                self.client_io.clone(),
                on_update,
            )
            .await
            .map(|_| ())
    }

    /// Loads this session's persisted `ConversationMeta` (the source
    /// [`Self::context_usage`] reads from), off the runtime thread since
    /// it's synchronous file I/O. `None` if the session hasn't been
    /// persisted yet — a brand-new session isn't written to disk until its
    /// first turn completes.
    async fn conversation_meta(&self) -> Result<Option<crate::memory::ConversationMeta>> {
        let uuid = Uuid::parse_str(&self.id)
            .map_err(|_| crate::error::Error::ParseError("invalid session id".to_string()))?;
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

    /// Restore a persisted session as the active session for this handle.
    ///
    /// Registers the conversation in the agent state so subsequent `prompt`
    /// calls continue from its history. Pass a no-op callback — the TUI
    /// already replays history visually before calling this. The returned
    /// handle inherits this handle's permission gate and client I/O.
    pub async fn restore(
        &self,
        session_id: &str,
        cwd: std::path::PathBuf,
    ) -> Result<SessionHandle> {
        self.state.load_session(session_id, cwd, |_| {}).await?;
        Ok(SessionHandle {
            id: session_id.to_string(),
            state: Arc::clone(&self.state),
            permission_gate: self.permission_gate.clone(),
            client_io: self.client_io.clone(),
        })
    }
}

// ── OpenheimBuilder ───────────────────────────────────────────────────────────

/// Builder for `OpenheimClient`.
///
/// Supports two modes:
/// 1. **Programmatic** — set `.provider()`, `.api_key()`, or `.api_base()`
///    directly, building the whole config from scratch.
/// 2. **File-based** — call `OpenheimClient::from_config(path)` or leave
///    everything unset to load from `~/.openheim/config.toml`.
///
/// `.model()` works in either mode: in file-based mode it overrides which
/// configured model gets resolved (same as the config file's model with a
/// different name picked); in programmatic mode it's the model of the
/// from-scratch provider.
///
/// MCP servers can be added in either mode with `.mcp_server()`.
#[derive(Default)]
pub struct OpenheimBuilder {
    // file-based path (None = ~/.openheim/config.toml)
    config_path: Option<PathBuf>,
    // programmatic fields — if any of these (besides `model`) are set we
    // skip the config file entirely; `model` alone is just a resolve()
    // override on top of file-based config.
    provider: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    api_base: Option<String>,
    max_iterations: Option<usize>,
    timeout_secs: Option<u64>,
    max_tokens: Option<u32>,
    mcp_servers: BTreeMap<String, McpServerConfig>,
    default_skills: Vec<String>,
    work_dir: Option<PathBuf>,
    allow_shell: Option<bool>,
    tools: Vec<Box<dyn ToolHandler>>,
}

impl OpenheimBuilder {
    /// Path to a config file (overrides `~/.openheim/config.toml`).
    pub fn config_path(mut self, path: impl AsRef<Path>) -> Self {
        self.config_path = Some(path.as_ref().to_path_buf());
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

    /// Register a custom tool (see [`crate::tools::ToolHandler`]). Registered
    /// alongside the built-ins and any MCP-sourced tools, and subject to the
    /// same `work_dir`/`allow_shell` sandbox boundary. Call multiple times to
    /// register more than one.
    pub fn tool(mut self, handler: Box<dyn ToolHandler>) -> Self {
        self.tools.push(handler);
        self
    }

    /// Build the client, connecting to MCP servers and initialising the agent state.
    pub async fn build(self) -> Result<OpenheimClient> {
        let (agent_config, mut app_config) =
            if self.provider.is_some() || self.api_key.is_some() || self.api_base.is_some() {
                build_programmatic(
                    self.provider,
                    self.api_key,
                    self.model,
                    self.api_base,
                    self.max_iterations,
                    self.timeout_secs,
                    self.max_tokens,
                    self.default_skills.clone(),
                )
            } else {
                let app_config = match self.config_path {
                    Some(ref path) => load_config_from(path)?,
                    None => load_config()?,
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

        if let Some(wd) = self.work_dir {
            let abs = if wd.is_absolute() {
                wd.clone()
            } else {
                std::env::current_dir()
                    .map_err(|e| {
                        crate::error::Error::ConfigError(format!(
                            "cannot resolve relative work_dir: {e}"
                        ))
                    })?
                    .join(&wd)
            };
            let canonical = abs.canonicalize().map_err(|e| {
                crate::error::Error::ConfigError(format!(
                    "work_dir '{}' is inaccessible: {e}",
                    wd.display()
                ))
            })?;
            app_config.work_dir = Some(canonical);
        }
        if let Some(shell) = self.allow_shell {
            app_config.allow_shell = shell;
        }

        let memory = MemoryContext::new(app_config.default_skills.clone())?;
        let state = Arc::new(AgentState::new(agent_config, app_config, memory, self.tools).await?);
        Ok(OpenheimClient { state })
    }
}

#[allow(clippy::too_many_arguments)]
fn build_programmatic(
    provider: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    api_base: Option<String>,
    max_iterations: Option<usize>,
    timeout_secs: Option<u64>,
    max_tokens: Option<u32>,
    default_skills: Vec<String>,
) -> (AgentConfig, AppConfig) {
    let provider = provider.unwrap_or_else(|| "openai".to_string());
    let (default_api_base, default_model) = crate::config::builtin_provider_defaults(&provider);
    let api_base = api_base.unwrap_or_else(|| default_api_base.to_string());
    let model = model.unwrap_or_else(|| default_model.to_string());
    let api_key = api_key.unwrap_or_default();
    let max_iter = max_iterations.unwrap_or(10);
    let timeout = timeout_secs.unwrap_or_else(crate::config::default_timeout_secs);

    let mut providers = BTreeMap::new();
    providers.insert(
        provider.clone(),
        ProviderConfig {
            api_base: api_base.clone(),
            default_model: model.clone(),
            models: vec![model.clone()],
            env_var: None,
            api_key: Some(api_key.clone()),
            timeout_secs: Some(timeout),
            max_tokens,
        },
    );

    let app_config = AppConfig {
        default_provider: provider.clone(),
        max_iterations: max_iter,
        theme_color: None,
        providers,
        mcp_servers: BTreeMap::new(),
        default_skills,
        work_dir: None,
        allow_shell: false,
        memory: None,
    };

    let agent_config = AgentConfig {
        provider_name: provider,
        api_base,
        api_key,
        model,
        max_iterations: max_iter,
        timeout_secs: timeout,
        max_tokens,
    };

    (agent_config, app_config)
}
