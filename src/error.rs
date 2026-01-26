use std::fmt;

#[derive(Debug)]
pub enum Error {
    ApiError(String),
    ToolExecutionError(String),
    ParseError(String),
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    ReqwestError(reqwest::Error),
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ApiError(msg) => write!(f, "API error: {}", msg),
            Error::ToolExecutionError(msg) => write!(f, "Tool execution error: {}", msg),
            Error::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Error::IoError(e) => write!(f, "IO error: {}", e),
            Error::JsonError(e) => write!(f, "JSON error: {}", e),
            Error::ReqwestError(e) => write!(f, "Request error: {}", e),
            Error::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::JsonError(err)
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::ReqwestError(err)
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
