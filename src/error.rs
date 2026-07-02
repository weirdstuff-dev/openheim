use std::fmt;

/// All errors that openheim can produce.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Non-HTTP error returned by an LLM provider (e.g. auth rejection in the response body).
    #[error("API error: {0}")]
    ApiError(String),

    /// HTTP error response from a provider (status code is preserved for retry logic).
    #[error("HTTP {status}: {body}")]
    HttpError { status: u16, body: String },

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    /// A requested resource (session, conversation, skill, …) doesn't exist.
    /// No fixed prefix (unlike the other typed variants) so existing
    /// call-site messages keep their exact wording — some are echoed
    /// verbatim in documented API responses (see `docs/api.md`).
    #[error("{0}")]
    NotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Request error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("TOML parse error: {0}")]
    TomlError(#[from] toml::de::Error),

    #[error("Task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Convenience constructor for [`Error::ConfigError`].
    pub fn config(msg: impl fmt::Display) -> Self {
        Error::ConfigError(msg.to_string())
    }

    /// Returns true for transient errors that may succeed on retry (429, 5xx, network errors).
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::HttpError { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 504),
            Error::ReqwestError(e) => e.is_timeout() || e.is_connect(),
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_helper_creates_config_error() {
        let err = Error::config("missing field");
        assert!(matches!(err, Error::ConfigError(_)));
        assert_eq!(err.to_string(), "Config error: missing field");
    }

    fn http_err(status: u16) -> Error {
        Error::HttpError {
            status,
            body: "error".into(),
        }
    }

    #[test]
    fn is_retryable_for_429() {
        assert!(http_err(429).is_retryable());
    }

    #[test]
    fn is_retryable_for_5xx() {
        for code in [500u16, 502, 503, 504] {
            assert!(
                http_err(code).is_retryable(),
                "expected retryable for status {}",
                code
            );
        }
    }

    #[test]
    fn is_not_retryable_for_400() {
        assert!(!http_err(400).is_retryable());
        assert!(!http_err(401).is_retryable());
        assert!(!http_err(404).is_retryable());
    }

    #[test]
    fn is_not_retryable_for_non_http_errors() {
        assert!(!Error::ApiError("something".into()).is_retryable());
        assert!(!Error::ParseError("bad json".into()).is_retryable());
        assert!(!Error::ConfigError("missing".into()).is_retryable());
        assert!(!Error::ToolExecutionError("failed".into()).is_retryable());
        assert!(!Error::NotFound("missing".into()).is_retryable());
        assert!(!Error::Other("something".into()).is_retryable());
    }

    #[tokio::test]
    async fn reqwest_connection_error_is_retryable() {
        let err = reqwest::Client::new()
            .get("http://127.0.0.1:1")
            .send()
            .await
            .unwrap_err();
        assert!(Error::ReqwestError(err).is_retryable());
    }
}
