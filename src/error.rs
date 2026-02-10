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

    /// Returns true for transient errors that may succeed on retry (429, 5xx, network errors).
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::ApiError(msg) => {
                msg.contains("status 429")
                    || msg.contains("status 500")
                    || msg.contains("status 502")
                    || msg.contains("status 503")
                    || msg.contains("status 504")
            }
            Error::ReqwestError(_) => true,
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
