mod client;
mod resolve;
mod types;

pub use client::resolve_client_and_config;
pub use types::{AgentConfig, AppConfig, ProviderConfig};

use std::path::PathBuf;

const DEFAULT_CONFIG: &str = include_str!("config.toml.default");

pub fn config_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
    Ok(home.join(".openheim"))
}

pub fn config_path() -> anyhow::Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

/// Initialize the config file at ~/.openheim/config.toml with the default template.
/// Returns the path written to.
pub fn init_config() -> anyhow::Result<PathBuf> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("config.toml");
    if path.exists() {
        anyhow::bail!("Config file already exists at {}", path.display());
    }
    std::fs::write(&path, DEFAULT_CONFIG)?;
    Ok(path)
}

/// Load AppConfig from ~/.openheim/config.toml
pub fn load_config() -> anyhow::Result<AppConfig> {
    let path = config_path()?;
    if !path.exists() {
        anyhow::bail!(
            "Config file not found at {}. Run `openheim --init` to create one.",
            path.display()
        );
    }
    let contents = std::fs::read_to_string(&path)?;
    let config: AppConfig = toml::from_str(&contents)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", path.display(), e))?;
    Ok(config)
}
