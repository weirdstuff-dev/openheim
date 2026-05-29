use crate::config::config_dir;
use crate::error::{Error, Result};
use std::path::PathBuf;

/// Loads the system identity from `~/.openheim/system.md`.
///
/// The file must exist — run `openheim init` to create it. Returns an error
/// if the file is absent, mirroring the behaviour of a missing `config.toml`.
#[derive(Clone)]
pub struct SystemLoader {
    path: PathBuf,
}

impl SystemLoader {
    /// Creates a `SystemLoader` pointed at `~/.openheim/system.md`.
    pub fn new() -> Result<Self> {
        Ok(Self {
            path: config_dir()?.join("system.md"),
        })
    }

    /// Returns the contents of `system.md`.
    ///
    /// Returns an error if the file does not exist.
    pub fn load(&self) -> Result<String> {
        if !self.path.exists() {
            return Err(Error::config(format!(
                "system.md not found at {}. Run `openheim init` to create one.",
                self.path.display()
            )));
        }
        Ok(std::fs::read_to_string(&self.path)?)
    }
}
