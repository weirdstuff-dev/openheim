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

    /// A provider's streamed reply ended before the provider said it was
    /// finished (typically a dropped connection). Retryable.
    #[error("Incomplete response: {0}")]
    IncompleteResponse(String),

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    /// Something failed to parse while openheim was doing its own work (a
    /// provider reply, a subagent profile, the model's tool-call JSON). Not
    /// for rejecting what a caller sent; that's [`Error::InvalidArgument`].
    #[error("Parse error: {0}")]
    ParseError(String),

    /// A caller passed something invalid: a malformed session id, an
    /// unknown mode, unsupported prompt content. Transports report it as the
    /// caller's mistake (ACP `invalid_params`), unlike [`Error::ParseError`].
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    /// A requested resource (session, conversation, skill, …) doesn't exist.
    /// Shown without a prefix: some messages appear verbatim in documented
    /// API responses (see `docs/api.md`).
    #[error("{0}")]
    NotFound(String),

    /// A turn couldn't start because another live process holds the
    /// session's write lease (see `memory::lease`). Reading history never
    /// hits this. The GUI matches on it to offer a read-only fallback.
    #[error("session {session_id} is active in another process (pid {pid} on {host})")]
    SessionLocked {
        session_id: String,
        pid: u32,
        host: String,
    },

    /// A `session/prompt` or `session/load` arrived while a turn was running
    /// on this session in this process. Retry once the turn completes.
    /// ([`Error::SessionLocked`] is the cross-process case.)
    #[error("a prompt is already in flight for session {session_id}; retry once it completes")]
    SessionBusy { session_id: String },

    /// `save_conversation` refused to rewrite a message log that another
    /// process has written to since the conversation was loaded. Reload it
    /// and retry.
    #[error("conversation {session_id} was modified by another process since it was loaded")]
    HistoryDiverged { session_id: String },

    /// SQLite / sqlite-vec failure in the long-term memory store (feature `rag`).
    #[error("Database error: {0}")]
    DatabaseError(String),

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

    /// Returns true for transient errors that may succeed on retry (429, 5xx,
    /// network errors, a reply cut off mid-stream).
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::HttpError { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 504),
            Error::ReqwestError(e) => e.is_timeout() || e.is_connect(),
            Error::IncompleteResponse(_) => true,
            _ => false,
        }
    }
}

#[cfg(feature = "rag")]
impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::DatabaseError(e.to_string())
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
    fn is_retryable_for_an_incomplete_response() {
        assert!(Error::IncompleteResponse("cut off".into()).is_retryable());
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
