mod client;
mod resolve;
mod types;

pub use client::{build_http_client, client_for_config, create_client};
pub(crate) use types::default_timeout_secs;
pub use types::{
    AgentConfig, AppConfig, EmbeddingConfig, McpServerConfig, MemoryConfig, ModelsInfo,
    ProviderConfig, ProviderModels,
};

use std::path::PathBuf;

use crate::error::{Error, Result};

const DEFAULT_CONFIG: &str = include_str!("config.toml.default");

/// `(provider name, default api_base, default model)` for openheim's
/// first-class providers. This is the one place these values are written —
/// `client::OpenheimBuilder`'s programmatic path (no config file) reads it
/// directly via [`builtin_provider_defaults`], and `config.toml.default`'s
/// `[providers.*]` sections are hand-kept in sync with it (enforced by
/// `config_toml_default_matches_builtin_provider_defaults` below) rather than
/// each independently guessing at "the current default model".
const BUILTIN_PROVIDER_DEFAULTS: &[(&str, &str, &str)] = &[
    ("openai", "https://api.openai.com/v1", "gpt-4o"),
    (
        "anthropic",
        "https://api.anthropic.com/v1",
        "claude-sonnet-4-6",
    ),
    (
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-2.0-flash",
    ),
];

/// Looks up `(api_base, default_model)` for a built-in provider name.
/// Anything not in [`BUILTIN_PROVIDER_DEFAULTS`] (e.g. a fully custom
/// OpenAI-compatible endpoint) falls back to the `openai` entry, since that's
/// the wire format `OpenAiCompatibleClient` speaks.
pub(crate) fn builtin_provider_defaults(provider: &str) -> (&'static str, &'static str) {
    BUILTIN_PROVIDER_DEFAULTS
        .iter()
        .find(|(name, ..)| *name == provider)
        .map(|(_, api_base, model)| (*api_base, *model))
        .unwrap_or_else(|| {
            let (_, api_base, model) = BUILTIN_PROVIDER_DEFAULTS[0];
            (api_base, model)
        })
}

pub fn config_dir() -> Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| Error::config("Could not determine home directory"))?;
    Ok(home.join(".openheim"))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

const DEFAULT_SYSTEM_MD: &str = "You are Openheim, a multipurpose, multiprovider LLM agent.";

/// Initialize the config file at ~/.openheim/config.toml with the default template.
/// Also writes ~/.openheim/system.md if it does not already exist.
/// Returns the path of the config file written.
///
/// Errors if `config.toml` already exists. `system.md` is written regardless —
/// so existing users who already have a config can still run `openheim init` to
/// get their `system.md` created.
pub fn init_config() -> Result<PathBuf> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;

    // Always write system.md first so existing users who re-run `init` get it
    // even though config.toml already exists and will cause an early return below.
    let system_path = dir.join("system.md");
    let system_written = !system_path.exists();
    if system_written {
        std::fs::write(&system_path, DEFAULT_SYSTEM_MD)?;
    }

    let config_path = dir.join("config.toml");
    if config_path.exists() {
        let system_note = if system_written {
            format!("system.md has been created at {}.", system_path.display())
        } else {
            format!("system.md is available at {}.", system_path.display())
        };
        return Err(Error::config(format!(
            "Config file already exists at {}. {}",
            config_path.display(),
            system_note
        )));
    }
    std::fs::write(&config_path, DEFAULT_CONFIG)?;

    Ok(config_path)
}

/// Load AppConfig from a specific path
pub fn load_config_from(path: impl AsRef<std::path::Path>) -> Result<AppConfig> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(Error::config(format!(
            "Config file not found at {}",
            path.display()
        )));
    }
    let contents = std::fs::read_to_string(path)?;
    let config: AppConfig = toml::from_str(&contents)?;
    Ok(config)
}

/// Load AppConfig from ~/.openheim/config.toml
pub fn load_config() -> Result<AppConfig> {
    let path = config_path()?;
    if !path.exists() {
        return Err(Error::config(format!(
            "Config file not found at {}. Run `openheim init` to create one.",
            path.display()
        )));
    }
    let contents = std::fs::read_to_string(&path)?;
    let config: AppConfig = toml::from_str(&contents)?;
    Ok(config)
}

/// Sets or updates the top-level `theme_color` key in the config file at
/// `path`, leaving every other line untouched. Deliberately not a full TOML
/// round-trip (no `toml_edit` dependency pulled in for one field): it only
/// ever touches a line that already looks like `theme_color = "..."`, or —
/// if there isn't one yet — inserts a new one right after `max_iterations`/
/// `default_provider` (matching the shipped template's field order), so it
/// can't corrupt a `[table]` or an unrelated key.
///
/// `name` is interpolated into a TOML basic string rather than run through a
/// full TOML encoder, so quotes, backslashes, and newlines are rejected
/// outright instead of being escaped — a caller passing one of those through
/// (this is `pub`, so an embedder could pass anything) can't break out of
/// the string and inject arbitrary lines into the config file.
pub fn save_theme_to_config_at(path: &std::path::Path, name: &str) -> Result<()> {
    if name.contains(['"', '\\', '\n', '\r']) {
        return Err(Error::config(format!(
            "invalid theme name {name:?}: quotes, backslashes, and newlines are not allowed"
        )));
    }
    let contents = std::fs::read_to_string(path)?;
    let new_line = format!("theme_color = \"{name}\"");
    let has_theme = contents
        .lines()
        .any(|l| l.trim_start().starts_with("theme_color"));
    let updated: String = if has_theme {
        contents
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("theme_color") {
                    new_line.clone()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let mut lines: Vec<String> = contents.lines().map(String::from).collect();
        let insert_pos = lines
            .iter()
            .rposition(|l| {
                l.trim_start().starts_with("max_iterations")
                    || l.trim_start().starts_with("default_provider")
            })
            .map(|i| i + 1)
            .unwrap_or(0);
        lines.insert(insert_pos, new_line);
        lines.join("\n")
    };
    let trailing = if contents.ends_with('\n') { "\n" } else { "" };
    std::fs::write(path, format!("{updated}{trailing}"))?;
    Ok(())
}

/// [`save_theme_to_config_at`] against `~/.openheim/config.toml`.
pub fn save_theme_to_config(name: &str) -> Result<()> {
    save_theme_to_config_at(&config_path()?, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_provider_defaults_known_providers() {
        assert_eq!(
            builtin_provider_defaults("openai"),
            ("https://api.openai.com/v1", "gpt-4o")
        );
        assert_eq!(
            builtin_provider_defaults("anthropic"),
            ("https://api.anthropic.com/v1", "claude-sonnet-4-6")
        );
        assert_eq!(
            builtin_provider_defaults("gemini"),
            (
                "https://generativelanguage.googleapis.com/v1beta",
                "gemini-2.0-flash"
            )
        );
    }

    #[test]
    fn builtin_provider_defaults_unknown_provider_falls_back_to_openai() {
        assert_eq!(
            builtin_provider_defaults("some-custom-endpoint"),
            builtin_provider_defaults("openai")
        );
    }

    /// `config.toml.default`'s `[providers.openai]`/`[providers.anthropic]`
    /// sections are the ones actually shipped active-by-default (Gemini is
    /// commented out, so it isn't valid TOML data to parse here) — this is
    /// the regression test for the drift `BUILTIN_PROVIDER_DEFAULTS`'s doc
    /// comment describes. If this fails, either the template or the
    /// constant is stale; update whichever one is wrong.
    #[test]
    fn config_toml_default_matches_builtin_provider_defaults() {
        let config: AppConfig = toml::from_str(DEFAULT_CONFIG).unwrap();

        for provider in ["openai", "anthropic"] {
            let (expected_api_base, expected_model) = builtin_provider_defaults(provider);
            let shipped = &config.providers[provider];
            assert_eq!(shipped.api_base, expected_api_base, "provider: {provider}");
            assert_eq!(
                shipped.default_model, expected_model,
                "provider: {provider}"
            );
            assert!(
                shipped.models.contains(&expected_model.to_string()),
                "provider {provider}'s default_model must be in its own models list"
            );
        }

        // Gemini is commented out in the shipped template (no API key means
        // it won't validate), so it can't be parsed as live TOML data — a
        // plain substring check on the commented block is the best available
        // guard against the same drift.
        let (gemini_api_base, gemini_model) = builtin_provider_defaults("gemini");
        assert!(DEFAULT_CONFIG.contains(gemini_api_base));
        assert!(DEFAULT_CONFIG.contains(gemini_model));
    }

    #[test]
    fn save_theme_to_config_at_inserts_a_new_line_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"openai\"\nmax_iterations = 10\n\n[providers.openai]\n",
        )
        .unwrap();

        save_theme_to_config_at(&path, "blue").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            contents,
            "default_provider = \"openai\"\nmax_iterations = 10\ntheme_color = \"blue\"\n\n[providers.openai]\n"
        );
    }

    #[test]
    fn save_theme_to_config_at_replaces_an_existing_line_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"openai\"\ntheme_color = \"red\"\nmax_iterations = 10\n",
        )
        .unwrap();

        save_theme_to_config_at(&path, "green").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            contents,
            "default_provider = \"openai\"\ntheme_color = \"green\"\nmax_iterations = 10\n"
        );
    }

    #[test]
    fn save_theme_to_config_at_preserves_missing_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "default_provider = \"openai\"").unwrap();

        save_theme_to_config_at(&path, "blue").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            contents,
            "default_provider = \"openai\"\ntheme_color = \"blue\""
        );
    }

    #[test]
    fn save_theme_to_config_at_rejects_names_that_would_break_out_of_the_toml_string() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "default_provider = \"openai\"\n").unwrap();

        for bad_name in [
            "blue\"\ninjected = true",
            "blue\\",
            "blue\nred",
            "blue\r\nred",
        ] {
            assert!(
                save_theme_to_config_at(&path, bad_name).is_err(),
                "expected {bad_name:?} to be rejected"
            );
        }

        // Rejected names must not have modified the file.
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "default_provider = \"openai\"\n");
    }
}
