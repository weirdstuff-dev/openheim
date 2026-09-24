use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

/// Public model info for a single provider (no credentials).
#[derive(Debug, Clone, Serialize)]
pub struct ProviderModels {
    pub default_model: String,
    pub models: Vec<String>,
}

/// JSON-safe summary of all configured providers and their models.
#[derive(Debug, Clone, Serialize)]
pub struct ModelsInfo {
    pub default_provider: String,
    pub providers: BTreeMap<String, ProviderModels>,
}

/// Top-level configuration loaded from ~/.openheim/config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// written; the directory actually in use is [`RuntimePaths::data_dir`].
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
}

fn default_allow_shell() -> bool {
    false
}

/// Paths a running client resolved once, at `OpenheimBuilder::build`, and
/// everything downstream uses as-is (see `AgentState::paths`). Kept out of
/// [`AppConfig`], which is the config file's shape, where they could only be
/// `Option`s that "are always set by the time anyone reads them".
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
pub struct TuiConfig {
    /// Named color theme (see `tui::render::theme_color` for the accepted
    /// names). Set via `:theme` in the running TUI, which persists it here.
    #[serde(default)]
    pub theme_color: Option<String>,
}

/// The `[memory]` section: where long-term memory lives, how many notes a
/// search returns, and (optionally) which embeddings endpoint makes search
/// semantic. Every field is optional; so is the section.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Resolved embeddings endpoint, assembled from a [`MemoryConfig`] plus the
/// provider entry it names (see `AppConfig::resolve_embedding`).
#[derive(Clone)]
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
#[derive(Clone, Serialize, Deserialize)]
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
    /// `headers = { Authorization = "Bearer <token>" }`. The stdio equivalent
    /// of this is `env` — same inline-table shape, applied to what an HTTP
    /// server actually consumes (headers, not process env vars).
    #[serde(default)]
    pub headers: HashMap<String, String>,
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
    /// Kept so configs written before `kind` existed behave as they did.
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
#[derive(Clone, Serialize, Deserialize)]
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
    /// Extended thinking (`"adaptive"` or `"off"`). Defaults to `adaptive`
    /// for an Anthropic-kind provider, `off` for everything else — see
    /// [`Self::resolve_thinking`]. Set explicitly to `"off"` for an Anthropic
    /// model that doesn't support adaptive thinking (e.g. `claude-haiku-4-5`,
    /// `claude-3-7-sonnet`), since a single `[providers.<name>]` entry has no
    /// per-model granularity.
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
            .field("thinking", &self.thinking)
            .finish()
    }
}

impl ProviderConfig {
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
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "AgentConfigWire")]
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
    /// Whether to request extended thinking (`AnthropicClient` only); see
    /// [`ProviderConfig::resolve_thinking`].
    pub thinking: bool,
}

/// `AgentConfig`'s deserialization shape. `kind` is optional so data written
/// before it existed resolves it from `provider_name`, the same inference
/// config resolution uses, instead of silently becoming OpenAI-compatible.
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
            .field("thinking", &self.thinking)
            .finish()
    }
}

/// The one source of truth for the request-timeout default; every path that
/// needs a timeout when none is configured goes through this (or through
/// [`ProviderConfig::resolve_timeout_secs`], which wraps it).
pub(crate) fn default_timeout_secs() -> u64 {
    120
}

impl AgentConfig {
    /// `kind` is inferred from `provider_name` (see
    /// [`ProviderKind::infer_from_name`]); set the field afterwards to
    /// override it.
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
        }
    }

    pub fn with_max_iterations(&self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self.clone()
        }
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
            thinking: false,
        }
    }
}

/// Shared test fixtures, so a new field on these structs is added here once
/// instead of in every test module that builds one by hand.
#[cfg(test)]
impl AppConfig {
    /// A config with no providers, MCP servers or skills, and every optional
    /// setting unset. Set fields on the result for what a test needs.
    pub(crate) fn for_tests(default_provider: &str) -> Self {
        Self {
            default_provider: default_provider.to_string(),
            max_iterations: default_max_iterations(),
            tui: TuiConfig::default(),
            providers: BTreeMap::new(),
            mcp_servers: BTreeMap::new(),
            default_skills: vec![],
            work_dir: None,
            allow_shell: false,
            memory: None,
            data_dir: None,
        }
    }
}

#[cfg(test)]
impl ProviderConfig {
    /// A provider entry serving `models` (the first is the default) with an
    /// inline API key and everything else unset.
    pub(crate) fn for_tests(api_base: &str, models: &[&str]) -> Self {
        Self {
            kind: None,
            api_base: api_base.to_string(),
            default_model: models[0].to_string(),
            models: models.iter().map(|m| m.to_string()).collect(),
            env_var: None,
            api_key: Some("key".to_string()),
            timeout_secs: None,
            max_tokens: None,
            thinking: None,
        }
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
    fn with_max_iterations_clones_with_new_value() {
        let cfg = AgentConfig::new("p".into(), "b".into(), "k".into(), "m".into(), 5);
        let updated = cfg.with_max_iterations(20);
        assert_eq!(updated.max_iterations, 20);
        assert_eq!(updated.provider_name, "p");
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

    // Regression test (PR #61 review): `AgentConfig` data from before `kind`
    // existed deserialized as OpenAI-compatible whatever the provider.
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
