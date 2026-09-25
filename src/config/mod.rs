mod client;
mod public;
mod resolve;
mod types;

pub use client::{build_http_client, client_for_config, create_client};
pub use public::{
    PublicConfig, PublicMcpServerConfig, PublicMemoryConfig, PublicProviderConfig, PublicTuiConfig,
};
pub(crate) use types::default_timeout_secs;
pub use types::{
    AgentConfig, AppConfig, EmbeddingConfig, McpServerConfig, MemoryConfig, ModelsInfo,
    ProviderConfig, ProviderKind, ProviderModels, RuntimePaths, TuiConfig,
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
        "gemini-3.8-flash",
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
    let contents = std::fs::read_to_string(path).map_err(|e| {
        if e.kind() != std::io::ErrorKind::NotFound {
            return e.into();
        }
        // `openheim init` only writes the default path, so only suggest it
        // for that one.
        let hint = if config_path().is_ok_and(|default| default == path) {
            " Run `openheim init` to create one."
        } else {
            ""
        };
        Error::config(format!(
            "Config file not found at {}.{hint}",
            path.display()
        ))
    })?;
    let config: AppConfig = toml::from_str(&contents)?;
    Ok(config)
}

/// Load AppConfig from ~/.openheim/config.toml
pub fn load_config() -> Result<AppConfig> {
    load_config_from(config_path()?)
}

/// Sets or updates the `theme_color` key inside the `[tui]` table in the
/// config file at `path`, leaving every other line untouched. Deliberately
/// not a full TOML round-trip (no `toml_edit` dependency pulled in for one
/// field): it only ever touches a line that already looks like
/// `theme_color = "..."` within an existing `[tui]` section (bounded by that
/// section's header and the next table header or EOF), or — if there is
/// no `[tui]` section yet — appends a new one at the end of the file.
///
/// Lines alone can't always tell a table header apart: `["a"]` on its own
/// line is also the last element of a multi-line array, and a multi-line
/// string can hold anything. So the edited file is parsed before it's
/// written, and must equal the original with only `tui.theme_color` set;
/// if it doesn't, the file is left alone and an error is returned.
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
    let mut expected: toml::Table = toml::from_str(&contents)?;
    match expected
        .entry("tui")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
    {
        toml::Value::Table(tui) => {
            tui.insert("theme_color".into(), toml::Value::String(name.into()));
        }
        _ => return Err(Error::config("`tui` in the config file is not a table")),
    }

    let new_line = format!("theme_color = \"{name}\"");
    let mut lines: Vec<String> = contents.lines().map(String::from).collect();

    let tui_header = lines.iter().position(|l| is_tui_header(l));
    match tui_header {
        Some(header_idx) => {
            // The section runs until the next table header or EOF.
            let section_end = lines[header_idx + 1..]
                .iter()
                .position(|l| table_header(l).is_some())
                .map(|i| header_idx + 1 + i)
                .unwrap_or(lines.len());
            let existing_theme =
                (header_idx + 1..section_end).find(|&i| is_theme_color_line(&lines[i]));
            match existing_theme {
                Some(i) => lines[i] = new_line,
                None => lines.insert(section_end, new_line),
            }
        }
        None => {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push("[tui]".to_string());
            lines.push(new_line);
        }
    }
    let trailing = if contents.ends_with('\n') { "\n" } else { "" };
    let edited = format!("{}{trailing}", lines.join("\n"));

    if toml::from_str::<toml::Table>(&edited).ok() != Some(expected) {
        return Err(Error::config(format!(
            "couldn't safely set theme_color in {}; set `theme_color = \"{name}\"` under \
             [tui] by hand",
            path.display()
        )));
    }
    std::fs::write(path, edited)?;
    Ok(())
}

/// The name inside `line` if it's a table header: `[name]` or `[[name]]`,
/// optionally followed by whitespace and a `# comment`. A line that merely
/// starts with `[` (say, `[1, 2],` inside a multi-line array) isn't one.
fn table_header(line: &str) -> Option<&str> {
    let line = line.trim();
    let (open, close) = if line.starts_with("[[") {
        ("[[", "]]")
    } else {
        ("[", "]")
    };
    let rest = line.strip_prefix(open)?;
    let end = rest.find(close)?;
    let after = rest[end + close.len()..].trim_start();
    (after.is_empty() || after.starts_with('#')).then(|| rest[..end].trim())
}

/// Whether `line` is the `[tui]` table header (not `[[tui]]`), allowing for
/// spaces inside the brackets and a trailing `# comment`, so such a header
/// doesn't cause a second `[tui]` to be appended.
fn is_tui_header(line: &str) -> bool {
    !line.trim_start().starts_with("[[") && table_header(line) == Some("tui")
}

/// Whether `line` assigns the `theme_color` key, as opposed to a
/// similarly-prefixed but distinct key like `theme_color_backup`. Requires
/// the next non-whitespace character after `theme_color` to be `=`.
fn is_theme_color_line(line: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix("theme_color") else {
        return false;
    };
    rest.trim_start().starts_with('=')
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
                "gemini-3.8-flash"
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
    /// commented out, so it isn't valid TOML data to parse here). Pins the
    /// shipped template against `BUILTIN_PROVIDER_DEFAULTS` so the two
    /// can't drift; if this fails, either the template or the constant is
    /// stale — update whichever one is wrong.
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
    fn save_theme_to_config_at_appends_a_new_tui_section_when_absent() {
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
            "default_provider = \"openai\"\nmax_iterations = 10\n\n[providers.openai]\n\n[tui]\ntheme_color = \"blue\"\n"
        );
    }

    #[test]
    fn save_theme_to_config_at_inserts_into_an_existing_empty_tui_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"openai\"\n\n[tui]\n[providers.openai]\n",
        )
        .unwrap();

        save_theme_to_config_at(&path, "blue").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            contents,
            "default_provider = \"openai\"\n\n[tui]\ntheme_color = \"blue\"\n[providers.openai]\n"
        );
    }

    #[test]
    fn save_theme_to_config_at_replaces_an_existing_line_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"openai\"\n\n[tui]\ntheme_color = \"red\"\n\n[providers.openai]\n",
        )
        .unwrap();

        save_theme_to_config_at(&path, "green").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            contents,
            "default_provider = \"openai\"\n\n[tui]\ntheme_color = \"green\"\n\n[providers.openai]\n"
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
            "default_provider = \"openai\"\n\n[tui]\ntheme_color = \"blue\""
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

    #[test]
    fn save_theme_to_config_at_recognizes_a_commented_tui_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"openai\"\n\n[tui] # theme settings\n[providers.openai]\n",
        )
        .unwrap();

        save_theme_to_config_at(&path, "blue").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        // Inserted into the existing (commented) [tui] section rather than
        // appending a second, duplicate [tui] header at the end.
        assert_eq!(
            contents,
            "default_provider = \"openai\"\n\n[tui] # theme settings\ntheme_color = \"blue\"\n[providers.openai]\n"
        );
    }

    #[test]
    fn save_theme_to_config_at_does_not_end_the_section_at_a_nested_array_line() {
        // `  [1, 2],` starts with `[` but isn't a table header; ending the
        // section there would insert the key inside the array.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let before = "[tui]\npalette = [\n  [1, 2],\n  [3, 4],\n]\ntheme_color = \"red\"\n\n[providers.openai]\n";
        std::fs::write(&path, before).unwrap();

        save_theme_to_config_at(&path, "green").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, before.replace("\"red\"", "\"green\""));
    }

    #[test]
    fn save_theme_to_config_at_leaves_the_file_alone_when_the_edit_would_not_parse() {
        // `  ["a"]` is also a valid (quoted-key) table header, so the line
        // edit lands inside the array; the re-parse must catch that.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let before = "[tui]\nnames = [\n  \"x\",\n  [\"a\"]\n]\n";
        std::fs::write(&path, before).unwrap();

        let err = save_theme_to_config_at(&path, "blue").unwrap_err();

        assert!(err.to_string().contains("by hand"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    }

    #[test]
    fn save_theme_to_config_at_recognizes_a_spaced_tui_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[ tui ]\ntheme_color = \"red\"\n").unwrap();

        save_theme_to_config_at(&path, "blue").unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "[ tui ]\ntheme_color = \"blue\"\n"
        );
    }

    #[test]
    fn table_header_accepts_only_real_headers() {
        assert_eq!(table_header("[tui]"), Some("tui"));
        assert_eq!(table_header("  [[agents]] # list"), Some("agents"));
        assert_eq!(table_header("[providers.openai]"), Some("providers.openai"));
        assert_eq!(table_header("[1, 2],"), None);
        assert_eq!(table_header("key = [1]"), None);
        assert!(!is_tui_header("[[tui]]"));
    }

    #[test]
    fn load_config_from_a_missing_custom_path_does_not_suggest_init() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.toml");

        let err = load_config_from(&path).unwrap_err().to_string();

        assert!(err.contains("Config file not found at"), "{err}");
        assert!(!err.contains("openheim init"), "{err}");
    }

    #[test]
    fn save_theme_to_config_at_does_not_touch_a_similarly_prefixed_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"openai\"\n\n[tui]\ntheme_color_backup = \"red\"\n[providers.openai]\n",
        )
        .unwrap();

        save_theme_to_config_at(&path, "blue").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        // theme_color_backup is left alone; the new theme_color key is
        // appended at the end of the [tui] section instead of overwriting it.
        assert_eq!(
            contents,
            "default_provider = \"openai\"\n\n[tui]\ntheme_color_backup = \"red\"\ntheme_color = \"blue\"\n[providers.openai]\n"
        );
    }
}
