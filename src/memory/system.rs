use crate::error::Result;
use std::path::PathBuf;

/// The identity used when there is no `system.md`, and the one
/// `openheim init` writes into a new one.
pub(crate) const DEFAULT_SYSTEM_MD: &str =
    "You are Openheim, a multipurpose, multiprovider LLM agent.";

/// Loads the system identity from `~/.openheim/system.md`.
#[derive(Clone)]
pub struct SystemLoader {
    path: PathBuf,
}

impl SystemLoader {
    /// Creates a `SystemLoader` pointed at `{dir}/system.md`, e.g. for an
    /// injected `AppConfig::data_dir` or in tests.
    pub fn with_dir(dir: PathBuf) -> Self {
        Self {
            path: dir.join("system.md"),
        }
    }

    /// Returns the contents of `system.md`, or [`DEFAULT_SYSTEM_MD`] if the
    /// file doesn't exist. Fails only if it exists but can't be read.
    pub fn load(&self) -> Result<String> {
        match std::fs::read_to_string(&self.path) {
            Ok(content) => Ok(content),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DEFAULT_SYSTEM_MD.to_string()),
            Err(e) => Err(e.into()),
        }
    }
}
