mod client;
mod resolve;
mod types;

pub use client::{build_http_client, create_client, resolve_client_and_config};
pub use types::{
    AgentConfig, AppConfig, McpServerConfig, ModelsInfo, ProviderConfig, ProviderModels,
};

use std::path::PathBuf;

use crate::error::{Error, Result};

const DEFAULT_CONFIG: &str = include_str!("config.toml.default");

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
    if !system_path.exists() {
        std::fs::write(&system_path, DEFAULT_SYSTEM_MD)?;
    }

    let config_path = dir.join("config.toml");
    if config_path.exists() {
        return Err(Error::config(format!(
            "Config file already exists at {}. system.md has been created at {}.",
            config_path.display(),
            system_path.display()
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
