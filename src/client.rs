use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use agent_client_protocol::schema::{SessionInfo, SessionUpdate};
use uuid::Uuid;

use crate::{
    acp::AgentState,
    config::{
        AgentConfig, AppConfig, McpServerConfig, ProviderConfig, load_config, load_config_from,
    },
    error::Result,
    mcp::McpServerStatus,
    rag::{Conversation, ConversationMeta, RagContext},
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
        self.state.acp_list_sessions(cwd).await
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
        self.state
            .acp_load_session(session_id, cwd, on_history)
            .await?;
        Ok(SessionHandle {
            id: session_id.to_string(),
            state: self.state.clone(),
        })
    }

    /// Fetch the full `Conversation` (messages + metadata) for a session id.
    pub fn get_session(&self, session_id: &str) -> Result<Conversation> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| crate::error::Error::Other("invalid session id".to_string()))?;
        self.state.rag.history.load_conversation(&uuid)
    }

    /// List all conversation metadata without loading messages.
    pub fn list_all_sessions(&self) -> Result<Vec<ConversationMeta>> {
        self.state.rag.history.list_conversations()
    }

    /// Permanently delete a persisted session.
    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|_| crate::error::Error::Other("invalid session id".to_string()))?;
        self.state.rag.history.delete_conversation(&uuid)
    }

    // ── RAG ───────────────────────────────────────────────────────────────────

    /// Direct access to the RAG context (history + skills managers).
    pub fn rag(&self) -> &RagContext {
        &self.state.rag
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
            .acp_new_session(self.model.as_deref(), self.skills, self.cwd)
            .await?;
        Ok(SessionHandle {
            id,
            state: self.state.clone(),
        })
    }
}

// ── SessionHandle ─────────────────────────────────────────────────────────────

/// A live session that can receive prompts.
pub struct SessionHandle {
    pub id: String,
    state: Arc<AgentState>,
}

impl SessionHandle {
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
        self.state
            .acp_prompt(&self.id, text.to_string(), on_update)
            .await
    }

    /// Switch the model for this session mid-conversation.
    ///
    /// The model must be listed under a provider in the config. Returns
    /// `(provider_name, model_name)` on success; the next prompt will use
    /// the new model while preserving conversation history.
    pub async fn switch_model(&self, provider: &str, model: &str) -> Result<(String, String)> {
        self.state
            .acp_update_session_model(&self.id, provider, model)
            .await
    }

    /// Restore a persisted session as the active session for this handle.
    ///
    /// Registers the conversation in the agent state so subsequent `prompt`
    /// calls continue from its history. Pass a no-op callback — the TUI
    /// already replays history visually before calling this.
    pub async fn restore(
        &self,
        session_id: &str,
        cwd: std::path::PathBuf,
    ) -> Result<SessionHandle> {
        self.state.acp_load_session(session_id, cwd, |_| {}).await?;
        Ok(SessionHandle {
            id: session_id.to_string(),
            state: Arc::clone(&self.state),
        })
    }
}

// ── OpenheimBuilder ───────────────────────────────────────────────────────────

/// Builder for `OpenheimClient`.
///
/// Supports two modes:
/// 1. **Programmatic** — set `.provider()`, `.api_key()`, `.model()` directly.
/// 2. **File-based** — call `OpenheimClient::from_config(path)` or leave
///    everything unset to load from `~/.openheim/config.toml`.
///
/// MCP servers can be added in either mode with `.mcp_server()`.
#[derive(Default)]
pub struct OpenheimBuilder {
    // file-based path (None = ~/.openheim/config.toml)
    config_path: Option<PathBuf>,
    // programmatic fields — if any of these are set we skip the config file
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

    /// Request timeout in seconds.
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
    /// Overrides `allow_shell` from the config file. Defaults to `true`.
    pub fn allow_shell(mut self, allow: bool) -> Self {
        self.allow_shell = Some(allow);
        self
    }

    /// Build the client, connecting to MCP servers and initialising the agent state.
    pub async fn build(self) -> Result<OpenheimClient> {
        let (agent_config, mut app_config) = if self.provider.is_some()
            || self.api_key.is_some()
            || self.model.is_some()
            || self.api_base.is_some()
        {
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
            let mut agent_config = app_config.resolve(None)?;
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
            app_config.work_dir = Some(wd);
        }
        if let Some(shell) = self.allow_shell {
            app_config.allow_shell = shell;
        }

        let rag = RagContext::new(app_config.default_skills.clone())?;
        let state = Arc::new(AgentState::new(agent_config, app_config, rag).await?);
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
    let api_base = api_base.unwrap_or_else(|| default_api_base(&provider));
    let model = model.unwrap_or_else(|| default_model(&provider));
    let api_key = api_key.unwrap_or_default();
    let max_iter = max_iterations.unwrap_or(10);
    let timeout = timeout_secs.unwrap_or(120);

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
        allow_shell: true,
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

fn default_api_base(provider: &str) -> String {
    match provider {
        "anthropic" => "https://api.anthropic.com/v1".to_string(),
        "gemini" => "https://generativelanguage.googleapis.com/v1beta".to_string(),
        _ => "https://api.openai.com/v1".to_string(),
    }
}

fn default_model(provider: &str) -> String {
    match provider {
        "anthropic" => "claude-sonnet-4-6".to_string(),
        "gemini" => "gemini-2.0-flash".to_string(),
        _ => "gpt-4o".to_string(),
    }
}
