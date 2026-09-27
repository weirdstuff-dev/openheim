use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

/// Public model info for a single provider (no credentials).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderModels {
    pub default_model: String,
    pub models: Vec<String>,
}

impl ProviderModels {
    pub fn new(default_model: impl Into<String>, models: Vec<String>) -> Self {
        Self {
            default_model: default_model.into(),
            models,
        }
    }
}

/// JSON-safe summary of all configured providers and their models. Also
/// what `openheim serve`'s `GET /api/models` returns, so a remote client can
/// deserialize the response straight into it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelsInfo {
    pub default_provider: String,
    pub providers: BTreeMap<String, ProviderModels>,
}

impl ModelsInfo {
    pub fn new(
        default_provider: impl Into<String>,
        providers: BTreeMap<String, ProviderModels>,
    ) -> Self {
        Self {
            default_provider: default_provider.into(),
            providers,
        }
    }
}

/// Top-level configuration loaded from ~/.openheim/config.toml
///
/// To build one in code, start from [`Self::new`] and chain the `with_*`
/// setters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AppConfig {
    pub default_provider: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// The `[tui]` section: terminal UI display preferences.
    #[serde(default)]
    pub tui: TuiConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
    /// Skills loaded automatically in every new session (merged with --skills at runtime).
    #[serde(default)]
    pub default_skills: Vec<String>,
    /// Root directory the agent is allowed to read/write. Defaults to the
    /// directory from which openheim was invoked when not set.
    #[serde(default)]
    pub work_dir: Option<PathBuf>,
    /// Whether to expose the `execute_command` shell tool to the LLM.
    /// Defaults to `false`. Set to `true` to explicitly opt in to shell access.
    #[serde(default = "default_allow_shell")]
    pub allow_shell: bool,
    /// Long-term memory (`remember` / `search_memory` / `edit_memory` /
    /// `forget` tools).
    /// Optional: without it memory still works, keyword-only, in
    /// `~/.openheim/memory.db`. Set `embedding_provider` / `embedding_model`
    /// to make `search_memory` semantic.
    #[serde(default)]
    pub memory: Option<MemoryConfig>,
    /// Overrides where history, skills, `system.md`, subagent profiles, and
    /// (absent an explicit `memory.db_path`) the memory database live.
    /// `None` means "default to `~/.openheim`". This is only the setting as
    /// written; a builder's `data_dir` overrides it.
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
}

fn default_allow_shell() -> bool {
    false
}

/// Paths resolved once, at `OpenheimBuilder::build`, and used as-is from
/// then on (see `AgentState::paths`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePaths {
    /// Where history, skills, `system.md`, subagent profiles and (by default)
    /// the memory database live: the builder's or config's `data_dir`, else
    /// `~/.openheim`.
    pub data_dir: PathBuf,
    /// The config file this client was loaded from, or would write to for a
    /// programmatic config, so config writers like the TUI's `:theme` target
    /// the file actually in use.
    pub config_path: PathBuf,
}

/// The `[tui]` section: terminal UI display preferences. Every field is
/// optional; so is the section.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct TuiConfig {
    /// Named color theme (see `tui::render::theme_color` for the accepted
    /// names). Set via `:theme` in the running TUI, which persists it here.
    #[serde(default)]
    pub theme_color: Option<String>,
}

impl TuiConfig {
    pub fn with_theme_color(mut self, theme_color: impl Into<String>) -> Self {
        self.theme_color = Some(theme_color.into());
        self
    }
}

/// The `[memory]` section: where long-term memory lives, how many notes a
/// search returns, and (optionally) which embeddings endpoint makes search
/// semantic. Every field is optional; so is the section.
///
/// To build one in code, start from the default and chain the `with_*`
/// setters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryConfig {
    /// Name of a `[providers.<name>]` entry whose `api_base` / API key serve
    /// the embeddings endpoint. A Gemini-kind provider speaks Gemini's
    /// `embedContent` API; any other kind is treated as OpenAI-compatible
    /// `/embeddings` (OpenAI, Ollama, Together, …). Anthropic-kind providers
    /// are rejected: Anthropic has no embeddings API.
    /// Unset means keyword (FTS5) search only.
    #[serde(default)]
    pub embedding_provider: Option<String>,
    /// Embedding model name (e.g. `text-embedding-3-small`,
    /// `gemini-embedding-001`, `nomic-embed-text`). Required when
    /// `embedding_provider` is set.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// SQLite file holding notes and vectors. Defaults to
    /// `~/.openheim/memory.db`. Must be an absolute path (no `~` expansion).
    #[serde(default)]
    pub db_path: Option<PathBuf>,
    /// Default number of notes a `search_memory` call returns (default 5).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding_provider: None,
            embedding_model: None,
            db_path: None,
            top_k: default_top_k(),
        }
    }
}

impl MemoryConfig {
    /// Makes search semantic: embeddings from `model`, served by the
    /// `[providers.<provider>]` entry.
    pub fn with_embedding(mut self, provider: impl Into<String>, model: impl Into<String>) -> Self {
        self.embedding_provider = Some(provider.into());
        self.embedding_model = Some(model.into());
        self
    }

    pub fn with_db_path(mut self, db_path: impl Into<PathBuf>) -> Self {
        self.db_path = Some(db_path.into());
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

/// Resolved embeddings endpoint, assembled from a [`MemoryConfig`] plus the
/// provider entry it names (see `AppConfig::resolve_embedding`).
#[derive(Clone)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    pub provider_name: String,
    /// Wire protocol, i.e. which embeddings client gets built. Never
    /// [`ProviderKind::Anthropic`] (`resolve_embedding` rejects it).
    pub kind: ProviderKind,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub timeout_secs: u64,
}

/// How a secret field shows up in `Debug` output: that it's set, not its
/// value. The configs holding secrets implement `Debug` by hand with this so
/// a stray `{:?}` in a log line can't leak a key.
fn redacted_if_set(secret: &str) -> &'static str {
    if secret.is_empty() {
        ""
    } else {
        super::public::REDACTED
    }
}

impl std::fmt::Debug for EmbeddingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingConfig")
            .field("provider_name", &self.provider_name)
            .field("kind", &self.kind)
            .field("api_base", &super::public::scrub_url(&self.api_base))
            .field("api_key", &redacted_if_set(&self.api_key))
            .field("model", &self.model)
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

/// Configuration for a single MCP server connection.
/// The map key in `[mcp_servers.<name>]` is used as the server name and tool-name prefix.
///
/// To build one in code, start from [`Self::stdio`] or [`Self::http`] and
/// chain [`Self::with_env`] / [`Self::with_header`].
#[derive(Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpServerConfig {
    /// Binary to spawn for stdio transport (e.g. `"npx"`, `"uvx"`).
    pub command: Option<String>,
    /// Arguments passed to `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the spawned process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Base URL for Streamable HTTP transport (e.g. `"http://localhost:8080/mcp"`).
    pub url: Option<String>,
    /// Extra HTTP headers sent with every request to an HTTP server, e.g.
    /// `headers = { Authorization = "Bearer <token>" }`.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

impl McpServerConfig {
    /// A server spawned as `command args…`, spoken to over stdio.
    pub fn stdio<I, S>(command: impl Into<String>, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            command: Some(command.into()),
            args: args.into_iter().map(Into::into).collect(),
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
        }
    }

    /// A server reached over Streamable HTTP at `url`.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            command: None,
            args: Vec::new(),
            env: HashMap::new(),
            url: Some(url.into()),
            headers: HashMap::new(),
        }
    }

    /// Adds an environment variable for the spawned process (stdio).
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Adds an HTTP header sent with every request (HTTP).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }
}

/// `args` often carry tokens (`--api-key …`), and `env`/`headers` values
/// are credentials, so only their count or keys are shown.
impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn keys(map: &HashMap<String, String>) -> Vec<&String> {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            keys
        }
        f.debug_struct("McpServerConfig")
            .field("command", &self.command)
            .field("args", &format!("<{} redacted>", self.args.len()))
            .field("env", &keys(&self.env))
            .field("url", &self.url.as_deref().map(super::public::scrub_url))
            .field("headers", &keys(&self.headers))
            .finish()
    }
}

fn default_max_iterations() -> usize {
    10
}

impl AppConfig {
    /// A config with no providers, MCP servers or skills, and every optional
    /// setting at its default, as for an empty config file. Add the
    /// `default_provider` entry with [`Self::with_provider`].
    pub fn new(default_provider: impl Into<String>) -> Self {
        Self {
            default_provider: default_provider.into(),
            max_iterations: default_max_iterations(),
            tui: TuiConfig::default(),
            providers: BTreeMap::new(),
            mcp_servers: BTreeMap::new(),
            default_skills: vec![],
            work_dir: None,
            allow_shell: default_allow_shell(),
            memory: None,
            data_dir: None,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_tui(mut self, tui: TuiConfig) -> Self {
        self.tui = tui;
        self
    }

    /// Adds (or replaces) the `[providers.<name>]` entry.
    pub fn with_provider(mut self, name: impl Into<String>, provider: ProviderConfig) -> Self {
        self.providers.insert(name.into(), provider);
        self
    }

    /// Adds (or replaces) the `[mcp_servers.<name>]` entry.
    pub fn with_mcp_server(mut self, name: impl Into<String>, server: McpServerConfig) -> Self {
        self.mcp_servers.insert(name.into(), server);
        self
    }

    pub fn with_default_skills<I, S>(mut self, skills: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.default_skills = skills.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_work_dir(mut self, work_dir: impl Into<PathBuf>) -> Self {
        self.work_dir = Some(work_dir.into());
        self
    }

    pub fn with_allow_shell(mut self, allow_shell: bool) -> Self {
        self.allow_shell = allow_shell;
        self
    }

    pub fn with_memory(mut self, memory: MemoryConfig) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_data_dir(mut self, data_dir: impl Into<PathBuf>) -> Self {
        self.data_dir = Some(data_dir.into());
        self
    }

    pub fn models_info(&self) -> ModelsInfo {
        ModelsInfo {
            default_provider: self.default_provider.clone(),
            providers: self
                .providers
                .iter()
                .map(|(name, p)| {
                    (
                        name.clone(),
                        ProviderModels {
                            default_model: p.default_model.clone(),
                            models: p.models.clone(),
                        },
                    )
                })
                .collect(),
        }
    }
}

/// Extended-thinking mode for a provider. Only [`AnthropicClient`](crate::core::llm::AnthropicClient)
/// consults this today; other providers ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ThinkingMode {
    /// Request Anthropic's adaptive extended thinking.
    Adaptive,
    /// Never request extended thinking.
    Off,
}

/// Which wire protocol a provider speaks, i.e. which client talks to it.
/// Set per provider with `kind = "…"`; when omitted it is inferred from the
/// provider's name (see [`Self::infer_from_name`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProviderKind {
    /// OpenAI's Chat Completions API.
    #[serde(rename = "openai")]
    OpenAi,
    /// Anthropic's Messages API.
    #[serde(rename = "anthropic")]
    Anthropic,
    /// Google's Gemini `generateContent` API.
    #[serde(rename = "gemini")]
    Gemini,
    /// Any endpoint speaking OpenAI's Chat Completions format (Ollama,
    /// OpenRouter, Together, vLLM, …).
    #[default]
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

impl ProviderKind {
    /// The kind a provider gets when its config sets none: the built-in
    /// names map to their own kind, anything else is OpenAI-compatible.
    pub fn infer_from_name(provider_name: &str) -> Self {
        match provider_name {
            "openai" => ProviderKind::OpenAi,
            "anthropic" => ProviderKind::Anthropic,
            "gemini" => ProviderKind::Gemini,
            _ => ProviderKind::OpenAiCompatible,
        }
    }
}

/// Per-provider configuration
///
/// To build one in code, start from [`Self::new`] and chain the `with_*`
/// setters.
#[derive(Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderConfig {
    /// Wire protocol for this provider. Optional: inferred from the
    /// provider's name when unset (see [`ProviderKind::infer_from_name`]),
    /// so it's only needed when the name isn't `openai`/`anthropic`/`gemini`
    /// but the API is, e.g. `[providers.claude-work] kind = "anthropic"`.
    #[serde(default)]
    pub kind: Option<ProviderKind>,
    pub api_base: String,
    pub default_model: String,
    pub models: Vec<String>,
    /// Name of the environment variable holding the API key (e.g. "OPENAI_API_KEY")
    pub env_var: Option<String>,
    /// Inline API key (not recommended - prefer env_var)
    pub api_key: Option<String>,
    /// Request timeout in seconds (default: 120)
    pub timeout_secs: Option<u64>,
    /// Maximum output tokens for LLM responses
    pub max_tokens: Option<u32>,
    /// The models' context window in tokens. When set, requests estimated
    /// at over 90% of it leave out the oldest turns before they're sent.
    /// Unset, older turns are left out only after the provider rejects a
    /// request as too long.
    #[serde(default)]
    pub context_window: Option<u64>,
    /// Extended thinking (`"adaptive"` or `"off"`); see
    /// [`Self::resolve_thinking`] for the default. Applies to every model of
    /// the entry, so set `"off"` if one of them lacks adaptive thinking.
    #[serde(default)]
    pub thinking: Option<ThinkingMode>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("api_base", &super::public::scrub_url(&self.api_base))
            .field("default_model", &self.default_model)
            .field("models", &self.models)
            .field("env_var", &self.env_var)
            .field("api_key", &self.api_key.as_deref().map(redacted_if_set))
            .field("timeout_secs", &self.timeout_secs)
            .field("max_tokens", &self.max_tokens)
            .field("context_window", &self.context_window)
            .field("thinking", &self.thinking)
            .finish()
    }
}

impl ProviderConfig {
    /// An entry serving only `default_model`, with no API key and every
    /// optional setting unset.
    pub fn new(api_base: impl Into<String>, default_model: impl Into<String>) -> Self {
        let default_model = default_model.into();
        Self {
            kind: None,
            api_base: api_base.into(),
            models: vec![default_model.clone()],
            default_model,
            env_var: None,
            api_key: None,
            timeout_secs: None,
            max_tokens: None,
            context_window: None,
            thinking: None,
        }
    }

    pub fn with_kind(mut self, kind: ProviderKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Replaces the list of models offered for this provider.
    pub fn with_models<I, S>(mut self, models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.models = models.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_env_var(mut self, env_var: impl Into<String>) -> Self {
        self.env_var = Some(env_var.into());
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingMode) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Request timeout for this provider, falling back to the crate-wide
    /// default when `timeout_secs` is not set.
    pub fn resolve_timeout_secs(&self) -> u64 {
        self.timeout_secs.unwrap_or_else(default_timeout_secs)
    }

    /// Resolve the API key: try env_var first, then inline api_key, then empty string (for keyless providers like Ollama)
    pub fn resolve_api_key(&self) -> String {
        if let Some(env_var) = &self.env_var
            && let Ok(key) = std::env::var(env_var)
            && !key.trim().is_empty()
        {
            return key;
        }
        self.api_key.clone().unwrap_or_default()
    }

    /// This provider's [`ProviderKind`]: the configured `kind`, or the one
    /// inferred from `provider_name` (the key it's registered under).
    pub fn resolve_kind(&self, provider_name: &str) -> ProviderKind {
        self.kind
            .unwrap_or_else(|| ProviderKind::infer_from_name(provider_name))
    }

    /// Whether extended thinking should be requested, given this provider's
    /// `thinking` setting and its resolved `kind`. Unset defaults to `true`
    /// only for [`ProviderKind::Anthropic`] — the only client that reads this.
    pub fn resolve_thinking(&self, kind: ProviderKind) -> bool {
        match self.thinking {
            Some(ThinkingMode::Adaptive) => true,
            Some(ThinkingMode::Off) => false,
            None => kind == ProviderKind::Anthropic,
        }
    }
}

/// Runtime configuration passed to agent/LLM code
///
/// To build one in code, start from [`Self::new`] and chain the `with_*`
/// setters.
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "AgentConfigWire")]
#[non_exhaustive]
pub struct AgentConfig {
    /// The `[providers.<name>]` key this config was resolved from; used for
    /// display, persistence, and model-switch lookups — never to pick a
    /// client (that's `kind`).
    pub provider_name: String,
    /// Wire protocol, i.e. which client [`crate::config::create_client`] builds.
    pub kind: ProviderKind,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub max_iterations: usize,
    pub timeout_secs: u64,
    /// Maximum output tokens for LLM responses (provider-specific defaults if not set)
    pub max_tokens: Option<u32>,
    /// The model's context window in tokens; see
    /// [`ProviderConfig::context_window`].
    pub context_window: Option<u64>,
    /// Whether to request extended thinking (`AnthropicClient` only); see
    /// [`ProviderConfig::resolve_thinking`].
    pub thinking: bool,
}

/// `AgentConfig`'s deserialization shape. A missing `kind` is inferred from
/// `provider_name`, as in config resolution.
#[derive(Deserialize)]
struct AgentConfigWire {
    provider_name: String,
    #[serde(default)]
    kind: Option<ProviderKind>,
    api_base: String,
    api_key: String,
    model: String,
    max_iterations: usize,
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,
    max_tokens: Option<u32>,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    thinking: bool,
}

impl From<AgentConfigWire> for AgentConfig {
    fn from(wire: AgentConfigWire) -> Self {
        Self {
            kind: wire
                .kind
                .unwrap_or_else(|| ProviderKind::infer_from_name(&wire.provider_name)),
            provider_name: wire.provider_name,
            api_base: wire.api_base,
            api_key: wire.api_key,
            model: wire.model,
            max_iterations: wire.max_iterations,
            timeout_secs: wire.timeout_secs,
            max_tokens: wire.max_tokens,
            context_window: wire.context_window,
            thinking: wire.thinking,
        }
    }
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("provider_name", &self.provider_name)
            .field("kind", &self.kind)
            .field("api_base", &super::public::scrub_url(&self.api_base))
            .field("api_key", &redacted_if_set(&self.api_key))
            .field("model", &self.model)
            .field("max_iterations", &self.max_iterations)
            .field("timeout_secs", &self.timeout_secs)
            .field("max_tokens", &self.max_tokens)
            .field("context_window", &self.context_window)
            .field("thinking", &self.thinking)
            .finish()
    }
}

/// The default request timeout, used wherever none is configured.
pub(crate) fn default_timeout_secs() -> u64 {
    120
}

impl AgentConfig {
    /// `kind` is inferred from `provider_name` (see
    /// [`ProviderKind::infer_from_name`]); override it with
    /// [`Self::with_kind`]. `thinking` is on for an Anthropic `kind`.
    pub fn new(
        provider_name: String,
        api_base: String,
        api_key: String,
        model: String,
        max_iterations: usize,
    ) -> Self {
        let kind = ProviderKind::infer_from_name(&provider_name);
        Self {
            thinking: kind == ProviderKind::Anthropic,
            kind,
            provider_name,
            api_base,
            api_key,
            model,
            max_iterations,
            timeout_secs: default_timeout_secs(),
            max_tokens: None,
            context_window: None,
        }
    }

    /// Sets the wire protocol. Leaves `thinking` as it was.
    pub fn with_kind(mut self, kind: ProviderKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider_name: String::new(),
            kind: ProviderKind::default(),
            api_base: String::new(),
            api_key: String::new(),
            model: String::new(),
            max_iterations: 10,
            timeout_secs: default_timeout_secs(),
            max_tokens: None,
            context_window: None,
            thinking: false,
        }
    }
}

#[cfg(test)]
impl ProviderConfig {
    /// A provider entry serving `models` (the first is the default) with an
    /// inline API key and everything else unset.
    pub(crate) fn for_tests(api_base: &str, models: &[&str]) -> Self {
        Self::new(api_base, models[0])
            .with_models(models.iter().copied())
            .with_api_key("key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_provider(env_var: Option<&str>, api_key: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            env_var: env_var.map(String::from),
            api_key: api_key.map(String::from),
            ..ProviderConfig::for_tests("https://api.example.com", &["model-1"])
        }
    }

    #[test]
    fn resolve_api_key_from_env_var() {
        let var_name = "OPENHEIM_TEST_KEY_ENV";
        unsafe {
            std::env::set_var(var_name, "secret-from-env");
        }
        let provider = sample_provider(Some(var_name), Some("inline-key"));
        assert_eq!(provider.resolve_api_key(), "secret-from-env");
        unsafe {
            std::env::remove_var(var_name);
        }
    }

    #[test]
    fn resolve_api_key_falls_back_to_inline() {
        let var_name = "OPENHEIM_TEST_KEY_MISSING";
        unsafe {
            std::env::remove_var(var_name);
        }
        let provider = sample_provider(Some(var_name), Some("inline-key"));
        assert_eq!(provider.resolve_api_key(), "inline-key");
    }

    #[test]
    fn resolve_api_key_returns_empty_when_none() {
        let var_name = "OPENHEIM_TEST_KEY_NONE";
        unsafe {
            std::env::remove_var(var_name);
        }
        let provider = sample_provider(Some(var_name), None);
        assert_eq!(provider.resolve_api_key(), "");
    }

    #[test]
    fn resolve_api_key_no_env_var_configured() {
        let provider = sample_provider(None, Some("inline-only"));
        assert_eq!(provider.resolve_api_key(), "inline-only");
    }

    #[test]
    fn resolve_api_key_empty_env_var_falls_back() {
        let var_name = "OPENHEIM_TEST_KEY_EMPTY";
        unsafe {
            std::env::set_var(var_name, "  ");
        }
        let provider = sample_provider(Some(var_name), Some("fallback"));
        assert_eq!(provider.resolve_api_key(), "fallback");
        unsafe {
            std::env::remove_var(var_name);
        }
    }

    #[test]
    fn agent_config_new_sets_defaults() {
        let cfg = AgentConfig::new(
            "openai".into(),
            "https://api.openai.com".into(),
            "key".into(),
            "gpt-4".into(),
            5,
        );
        assert_eq!(cfg.provider_name, "openai");
        assert_eq!(cfg.max_iterations, 5);
        assert_eq!(cfg.timeout_secs, 120);
        assert!(cfg.max_tokens.is_none());
    }

    #[test]
    fn with_max_iterations_keeps_other_fields() {
        let cfg = AgentConfig::new("p".into(), "b".into(), "k".into(), "m".into(), 5);
        let updated = cfg.with_max_iterations(20);
        assert_eq!(updated.max_iterations, 20);
        assert_eq!(updated.provider_name, "p");
    }

    /// `AppConfig::new` gives what a config file with only
    /// `default_provider` parses to.
    #[test]
    fn app_config_new_matches_minimal_file() {
        let parsed: AppConfig = toml::from_str(r#"default_provider = "openai""#).unwrap();
        let built = AppConfig::new("openai");
        assert_eq!(
            serde_json::to_value(&built).unwrap(),
            serde_json::to_value(&parsed).unwrap()
        );
    }

    #[test]
    fn app_config_setters_fill_sections() {
        let cfg = AppConfig::new("local")
            .with_provider(
                "local",
                ProviderConfig::new("http://localhost:11434/v1", "llama3")
                    .with_models(["llama3", "qwen3"])
                    .with_kind(ProviderKind::OpenAiCompatible)
                    .with_context_window(8192),
            )
            .with_mcp_server("fs", McpServerConfig::stdio("npx", ["-y", "server-fs"]))
            .with_memory(MemoryConfig::default().with_top_k(3))
            .with_allow_shell(true);
        let local = &cfg.providers["local"];
        assert_eq!(local.default_model, "llama3");
        assert_eq!(local.models, ["llama3", "qwen3"]);
        assert_eq!(local.context_window, Some(8192));
        assert_eq!(cfg.mcp_servers["fs"].args, ["-y", "server-fs"]);
        assert_eq!(cfg.memory.as_ref().unwrap().top_k, 3);
        assert!(cfg.allow_shell);
    }

    #[test]
    fn provider_config_new_offers_its_default_model() {
        let p = ProviderConfig::new("https://api.example.com", "m1");
        assert_eq!(p.models, ["m1"]);
        assert!(p.api_key.is_none() && p.kind.is_none());
    }

    #[test]
    fn mcp_server_config_constructors_pick_the_transport() {
        let stdio = McpServerConfig::stdio("uvx", ["tool"]).with_env("TOKEN", "t");
        assert_eq!(stdio.command.as_deref(), Some("uvx"));
        assert!(stdio.url.is_none());
        assert_eq!(stdio.env["TOKEN"], "t");

        let http =
            McpServerConfig::http("https://x.example/mcp").with_header("Authorization", "Bearer k");
        assert!(http.command.is_none() && http.args.is_empty());
        assert_eq!(http.url.as_deref(), Some("https://x.example/mcp"));
        assert_eq!(http.headers["Authorization"], "Bearer k");
    }

    #[test]
    fn agent_config_default_has_correct_values() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.max_iterations, 10);
        assert_eq!(cfg.timeout_secs, 120);
        assert!(cfg.provider_name.is_empty());
    }

    #[test]
    fn app_config_deserializes_with_default_max_iterations() {
        let toml_str = r#"
            default_provider = "openai"
            [providers]
        "#;
        let cfg: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_iterations, 10);
        assert!(cfg.memory.is_none());
    }

    #[test]
    fn memory_section_deserializes_with_defaults() {
        let toml_str = r#"
            default_provider = "openai"
            [memory]
            embedding_provider = "openai"
            embedding_model = "text-embedding-3-small"
        "#;
        let cfg: AppConfig = toml::from_str(toml_str).unwrap();
        let memory = cfg.memory.unwrap();
        assert_eq!(memory.embedding_provider.as_deref(), Some("openai"));
        assert_eq!(
            memory.embedding_model.as_deref(),
            Some("text-embedding-3-small")
        );
        assert_eq!(memory.top_k, 5);
        assert!(memory.db_path.is_none());

        let bare: AppConfig = toml::from_str("default_provider = \"openai\"\n[memory]\n").unwrap();
        let memory = bare.memory.unwrap();
        assert!(memory.embedding_provider.is_none());
        assert_eq!(memory.top_k, 5);
    }

    // `AgentConfig` data without a `kind` field infers it from the provider
    // name rather than defaulting to OpenAI-compatible.
    #[test]
    fn agent_config_without_kind_infers_it_from_the_provider_name() {
        let json = |provider: &str| {
            serde_json::json!({
                "provider_name": provider,
                "api_base": "https://example.com",
                "api_key": "k",
                "model": "m",
                "max_iterations": 5,
                "max_tokens": null,
            })
        };
        let anthropic: AgentConfig = serde_json::from_value(json("anthropic")).unwrap();
        assert_eq!(anthropic.kind, ProviderKind::Anthropic);
        let custom: AgentConfig = serde_json::from_value(json("ollama")).unwrap();
        assert_eq!(custom.kind, ProviderKind::OpenAiCompatible);

        // An explicit kind still wins, and a round trip preserves it.
        let mut explicit = json("claude-work");
        explicit["kind"] = serde_json::json!("anthropic");
        let config: AgentConfig = serde_json::from_value(explicit).unwrap();
        assert_eq!(config.kind, ProviderKind::Anthropic);
        let round_trip: AgentConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(round_trip.kind, ProviderKind::Anthropic);
        assert_eq!(round_trip.timeout_secs, default_timeout_secs());
    }

    #[test]
    fn debug_output_never_contains_secrets() {
        let app: AppConfig = toml::from_str(
            r#"
            default_provider = "openai"
            [providers.openai]
            api_base = "https://user:pass-secret@api.example.com/v1?key=query-secret"
            default_model = "m"
            models = ["m"]
            api_key = "sk-key-secret"

            [mcp_servers.demo]
            command = "npx"
            args = ["--token", "arg-secret"]
            env = { TOKEN = "env-secret" }

            [mcp_servers.remote]
            url = "https://mcp.example.com/?token=url-secret"
            headers = { Authorization = "Bearer header-secret" }
            "#,
        )
        .unwrap();
        let agent = app.resolve(None).unwrap();
        let embedding = EmbeddingConfig {
            provider_name: "openai".into(),
            kind: ProviderKind::OpenAi,
            api_base: "https://api.example.com".into(),
            api_key: "sk-embed-secret".into(),
            model: "m".into(),
            timeout_secs: 10,
        };

        let debug = format!("{app:?}\n{agent:?}\n{embedding:?}");
        for secret in [
            "pass-secret",
            "query-secret",
            "sk-key-secret",
            "arg-secret",
            "env-secret",
            "url-secret",
            "header-secret",
            "sk-embed-secret",
        ] {
            assert!(!debug.contains(secret), "{secret} leaked:\n{debug}");
        }
        // Still useful for debugging: the non-secret parts are there.
        assert!(debug.contains("api.example.com"), "{debug}");
        assert!(debug.contains("TOKEN"), "{debug}");
        assert!(debug.contains("<redacted>"), "{debug}");
    }
}
