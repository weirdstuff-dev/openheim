use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Request error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("TOML parse error: {0}")]
    TomlError(#[from] toml::de::Error),

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn config(msg: impl fmt::Display) -> Self {
        Error::ConfigError(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
