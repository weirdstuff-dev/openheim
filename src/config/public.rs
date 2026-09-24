//! The configuration view served to clients (`GET /api/config`).
//!
//! Built field by field from [`AppConfig`] rather than by serializing it and
//! stripping known secrets: only what is listed here is ever exposed, so a
//! field added to `AppConfig` later stays private until someone adds it here
//! on purpose.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use super::types::{AppConfig, McpServerConfig, ProviderConfig, ProviderKind};

/// Placeholder for a value that exists but isn't shown.
const REDACTED: &str = "<redacted>";

/// Public view of [`AppConfig`]. See the module docs for why it's an
/// allow-list.
#[derive(Debug, Clone, Serialize)]
pub struct PublicConfig {
    pub default_provider: String,
    pub max_iterations: usize,
    /// The resolved sandbox root, not the raw (often unset) config value.
    pub work_dir: String,
    pub allow_shell: bool,
    pub tui: PublicTuiConfig,
    pub providers: BTreeMap<String, PublicProviderConfig>,
    pub mcp_servers: BTreeMap<String, PublicMcpServerConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<PublicMemoryConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicTuiConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme_color: Option<String>,
}

/// A provider entry without its API key. `api_base` has any credentials or
/// query string removed (see [`scrub_url`]).
#[derive(Debug, Clone, Serialize)]
pub struct PublicProviderConfig {
    /// Resolved, so it's present even when the config infers it from the name.
    pub kind: ProviderKind,
    pub api_base: String,
    pub default_model: String,
    pub models: Vec<String>,
    /// The *name* of the environment variable holding the key, never its value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_var: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// An MCP server entry. `args` is left out entirely (command lines routinely
/// carry tokens), `env`/`headers` keep their keys but not their values, and
/// `url` is scrubbed like a provider's `api_base`.
#[derive(Debug, Clone, Serialize)]
pub struct PublicMcpServerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub env: BTreeMap<String, String>,
    pub headers: BTreeMap<String, String>,
}

/// The `[memory]` section without `db_path` (a local filesystem path).
#[derive(Debug, Clone, Serialize)]
pub struct PublicMemoryConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    pub top_k: usize,
}

impl AppConfig {
    /// The client-facing view of this config; `work_dir` is the resolved
    /// sandbox root (`AgentState::work_dir`).
    pub fn to_public(&self, work_dir: &Path) -> PublicConfig {
        PublicConfig {
            default_provider: self.default_provider.clone(),
            max_iterations: self.max_iterations,
            work_dir: work_dir.display().to_string(),
            allow_shell: self.allow_shell,
            tui: PublicTuiConfig {
                theme_color: self.tui.theme_color.clone(),
            },
            providers: self
                .providers
                .iter()
                .map(|(name, p)| (name.clone(), public_provider(name, p)))
                .collect(),
            mcp_servers: self
                .mcp_servers
                .iter()
                .map(|(name, s)| (name.clone(), public_mcp_server(s)))
                .collect(),
            memory: self.memory.as_ref().map(|m| PublicMemoryConfig {
                embedding_provider: m.embedding_provider.clone(),
                embedding_model: m.embedding_model.clone(),
                top_k: m.top_k,
            }),
        }
    }
}

fn public_provider(name: &str, p: &ProviderConfig) -> PublicProviderConfig {
    PublicProviderConfig {
        kind: p.resolve_kind(name),
        api_base: scrub_url(&p.api_base),
        default_model: p.default_model.clone(),
        models: p.models.clone(),
        env_var: p.env_var.clone(),
        timeout_secs: p.timeout_secs,
        max_tokens: p.max_tokens,
    }
}

fn public_mcp_server(s: &McpServerConfig) -> PublicMcpServerConfig {
    let keys_only = |map: &std::collections::HashMap<String, String>| {
        map.keys()
            .map(|k| (k.clone(), REDACTED.to_string()))
            .collect()
    };
    PublicMcpServerConfig {
        command: s.command.clone(),
        url: s.url.as_deref().map(scrub_url),
        env: keys_only(&s.env),
        headers: keys_only(&s.headers),
    }
}

/// `url` without userinfo, query string, or fragment, the parts where
/// credentials end up (`https://user:pass@host`, `?api_key=…`). Anything that
/// doesn't parse as a URL is replaced outright rather than guessed at.
fn scrub_url(url: &str) -> String {
    let Ok(mut parsed) = reqwest::Url::parse(url) else {
        return REDACTED.to_string();
    };
    // Both only fail for URLs that can't have userinfo at all, which then
    // has nothing to remove.
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    parsed.set_query(None);
    parsed.set_fragment(None);
    parsed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AppConfig {
        toml::from_str(
            r#"
            default_provider = "openai"
            data_dir = "/Users/alice/.openheim"
            allow_shell = true
            [tui]
            theme_color = "green"

            [providers.openai]
            api_base = "https://api.openai.com/v1"
            default_model = "gpt-4"
            models = ["gpt-4"]
            api_key = "sk-super-secret"
            env_var = "OPENAI_API_KEY"

            [providers.proxy]
            api_base = "https://user:hunter2@proxy.example.com/v1?key=also-secret"
            default_model = "m"
            models = ["m"]

            [mcp_servers.demo]
            command = "npx"
            args = ["-y", "server", "--token", "cli-secret"]
            env = { API_TOKEN = "env-secret" }

            [mcp_servers.remote]
            url = "https://mcp.example.com/mcp?token=url-secret"
            headers = { Authorization = "Bearer header-secret" }

            [memory]
            embedding_provider = "openai"
            embedding_model = "text-embedding-3-small"
            db_path = "/Users/alice/.openheim/memory.db"
            top_k = 7
            "#,
        )
        .unwrap()
    }

    #[test]
    fn no_secret_or_local_path_reaches_the_public_view() {
        let json = serde_json::to_string(&sample().to_public(Path::new("/work/dir"))).unwrap();
        for secret in [
            "sk-super-secret",
            "hunter2",
            "also-secret",
            "cli-secret",
            "env-secret",
            "url-secret",
            "header-secret",
            "/Users/alice",
        ] {
            assert!(!json.contains(secret), "{secret} leaked: {json}");
        }
    }

    #[test]
    fn public_view_keeps_what_clients_use() {
        let val = serde_json::to_value(sample().to_public(Path::new("/work/dir"))).unwrap();

        assert_eq!(val["work_dir"], "/work/dir");
        assert_eq!(val["allow_shell"], true);
        assert_eq!(val["tui"]["theme_color"], "green");

        let openai = &val["providers"]["openai"];
        assert_eq!(openai["kind"], "openai");
        assert_eq!(openai["api_base"], "https://api.openai.com/v1");
        assert_eq!(openai["env_var"], "OPENAI_API_KEY");
        assert!(openai.get("api_key").is_none());
        assert_eq!(val["providers"]["proxy"]["kind"], "openai_compatible");
        assert_eq!(
            val["providers"]["proxy"]["api_base"],
            "https://proxy.example.com/v1"
        );

        let demo = &val["mcp_servers"]["demo"];
        assert_eq!(demo["command"], "npx");
        assert!(demo.get("args").is_none());
        assert_eq!(demo["env"]["API_TOKEN"], REDACTED);
        let remote = &val["mcp_servers"]["remote"];
        assert_eq!(remote["url"], "https://mcp.example.com/mcp");
        assert_eq!(remote["headers"]["Authorization"], REDACTED);

        assert_eq!(val["memory"]["embedding_model"], "text-embedding-3-small");
        assert_eq!(val["memory"]["top_k"], 7);
        assert!(val["memory"].get("db_path").is_none());
        assert!(val.get("data_dir").is_none());
    }

    #[test]
    fn unparseable_urls_are_replaced_not_guessed_at() {
        assert_eq!(scrub_url("not a url with secret"), REDACTED);
    }
}
