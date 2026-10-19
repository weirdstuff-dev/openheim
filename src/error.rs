use std::fmt;
use std::time::Duration;

/// The longest `Retry-After` openheim waits out before retrying. A longer one
/// makes the error non-retryable, so the caller hears about it at once
/// instead of after a silent wait.
pub(crate) const MAX_RETRY_AFTER: Duration = Duration::from_secs(10);

/// All errors that openheim can produce.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Non-HTTP error returned by an LLM provider (e.g. auth rejection in the response body).
    #[error("API error: {0}")]
    ApiError(String),

    /// HTTP error response from a provider. `retry_after` is the response's
    /// `Retry-After` header, if it had a valid one.
    #[error("HTTP {status}: {body}")]
    HttpError {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
    },

    /// The provider rejected a request as too long for the model's context
    /// window. The agent loop answers it by leaving out older turns, so a
    /// caller only sees it once even the current turn alone doesn't fit.
    #[error("Context window exceeded: {0}")]
    ContextOverflow(String),

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

    /// [`Error::HttpError`] for a non-2xx `response`: its status, its body
    /// and its `Retry-After` header.
    pub async fn from_response(response: reqwest::Response) -> Self {
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| parse_retry_after(value, chrono::Utc::now()));
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<failed to read error body>".into());
        Error::HttpError {
            status,
            body,
            retry_after,
        }
    }

    /// Returns true for transient errors that may succeed on retry (429, 5xx
    /// including Anthropic's 529 "overloaded", network errors, a reply cut
    /// off mid-stream). An HTTP error whose
    /// `Retry-After` is longer than openheim waits out isn't.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::HttpError {
                status,
                retry_after,
                ..
            } => {
                matches!(status, 429 | 500 | 502 | 503 | 504 | 529)
                    && retry_after.is_none_or(|wait| wait <= MAX_RETRY_AFTER)
            }
            Error::ReqwestError(e) => e.is_timeout() || e.is_connect(),
            Error::IncompleteResponse(_) => true,
            _ => false,
        }
    }

    /// How long the provider asked to wait before retrying (its
    /// `Retry-After`), if it said.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Error::HttpError { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// A `Retry-After` value as a wait from `now`: either delay-seconds or an
/// HTTP date (a date already past is no wait). `None` if it's neither.
fn parse_retry_after(value: &str, now: chrono::DateTime<chrono::Utc>) -> Option<Duration> {
    let value = value.trim();
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    Some(
        (at.with_timezone(&chrono::Utc) - now)
            .to_std()
            .unwrap_or_default(),
    )
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
            retry_after: None,
        }
    }

    fn http_err_retry_after(status: u16, secs: u64) -> Error {
        Error::HttpError {
            status,
            body: "error".into(),
            retry_after: Some(Duration::from_secs(secs)),
        }
    }

    #[test]
    fn is_retryable_for_429() {
        assert!(http_err(429).is_retryable());
    }

    #[test]
    fn is_retryable_for_a_short_retry_after() {
        assert!(http_err_retry_after(429, 0).is_retryable());
        assert!(http_err_retry_after(429, MAX_RETRY_AFTER.as_secs()).is_retryable());
        assert!(http_err_retry_after(503, 2).is_retryable());
    }

    // A limit that lifts in hours is reported at once, not after the backoff.
    #[test]
    fn is_not_retryable_for_a_long_retry_after() {
        assert!(!http_err_retry_after(429, MAX_RETRY_AFTER.as_secs() + 1).is_retryable());
        assert!(!http_err_retry_after(503, 18_000).is_retryable());
    }

    #[test]
    fn a_retry_after_on_a_non_transient_status_is_still_not_retryable() {
        assert!(!http_err_retry_after(403, 1).is_retryable());
    }

    #[test]
    fn retry_after_parses_seconds_and_http_dates() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-10-18T15:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(
            parse_retry_after("120", now),
            Some(Duration::from_secs(120))
        );
        assert_eq!(parse_retry_after(" 5 ", now), Some(Duration::from_secs(5)));
        assert_eq!(
            parse_retry_after("Sun, 18 Oct 2026 16:00:00 GMT", now),
            Some(Duration::from_secs(3600))
        );
        assert_eq!(
            parse_retry_after("Sun, 18 Oct 2026 14:00:00 GMT", now),
            Some(Duration::ZERO)
        );
        assert_eq!(parse_retry_after("soon", now), None);
        assert_eq!(parse_retry_after("-5", now), None);
    }

    fn response(status: u16, retry_after: Option<&str>, body: &str) -> reqwest::Response {
        let mut builder = http::Response::builder().status(status);
        if let Some(value) = retry_after {
            builder = builder.header("Retry-After", value);
        }
        reqwest::Response::from(builder.body(body.to_string()).unwrap())
    }

    #[tokio::test]
    async fn from_response_keeps_status_body_and_retry_after() {
        let err = Error::from_response(response(429, Some("18000"), "{\"error\":{}}")).await;
        assert!(matches!(
            err,
            Error::HttpError { status: 429, ref body, retry_after: Some(wait) }
                if body == "{\"error\":{}}" && wait == Duration::from_secs(18_000)
        ));
    }

    #[tokio::test]
    async fn from_response_ignores_a_missing_or_invalid_retry_after() {
        for header in [None, Some("later")] {
            let err = Error::from_response(response(503, header, "down")).await;
            assert_eq!(err.retry_after(), None, "{header:?}");
        }
    }

    #[test]
    fn retry_after_is_only_set_for_http_errors() {
        assert_eq!(
            http_err_retry_after(429, 3).retry_after(),
            Some(Duration::from_secs(3))
        );
        assert_eq!(http_err(429).retry_after(), None);
        assert_eq!(Error::Other("x".into()).retry_after(), None);
    }

    #[test]
    fn is_retryable_for_5xx() {
        for code in [500u16, 502, 503, 504, 529] {
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
    fn is_not_retryable_for_a_context_overflow() {
        assert!(!Error::ContextOverflow("too long".into()).is_retryable());
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
