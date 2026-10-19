use std::fmt;

/// All errors that openheim can produce.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Non-HTTP error returned by an LLM provider (e.g. auth rejection in the response body).
    #[error("API error: {0}")]
    ApiError(String),

    /// HTTP error response from a provider (status code is preserved for retry logic).
    #[error("HTTP {status}: {body}")]
    HttpError { status: u16, body: String },

    /// The provider rejected a request as too long for the model's context
    /// window. The agent loop answers it by leaving out older turns, so a
    /// caller only sees it once even the current turn alone doesn't fit.
    #[error("Context window exceeded: {0}")]
    ContextOverflow(String),

    /// A backend rejected the request because of the user's plan or account
    /// (a usage limit, an invalid token). Never retried: waiting a few
    /// seconds doesn't lift it.
    #[error("{}", .0.message)]
    ProviderRejection(Box<ProviderRejection>),

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

/// What a [`ProviderRejection`] is about, from the error body's `type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RejectionKind {
    /// `openheim_limit`: the plan doesn't cover the request, or one of its
    /// allowances is used up.
    Limit,
    /// `openheim_auth`: the token is invalid, expired or revoked.
    Auth,
}

impl RejectionKind {
    /// The `type` tag this kind has in an error body.
    pub fn as_str(self) -> &'static str {
        match self {
            RejectionKind::Limit => "openheim_limit",
            RejectionKind::Auth => "openheim_auth",
        }
    }

    fn from_type(tag: &str) -> Option<Self> {
        match tag {
            "openheim_limit" => Some(RejectionKind::Limit),
            "openheim_auth" => Some(RejectionKind::Auth),
            _ => None,
        }
    }
}

/// A plan or account rejection, parsed from an error body of the form
/// `{"error": {"type": "openheim_limit" | "openheim_auth", "code": ..., "message": ..., ...}}`.
///
/// `code` is kept as sent, so codes this version doesn't know still parse.
/// The fields that depend on the code (`plan`, `limit`, `upgrade`, `action`,
/// …) are in `details`, which holds the whole `error` object unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRejection {
    /// HTTP status of the response.
    pub status: u16,
    pub kind: RejectionKind,
    /// E.g. `window_budget_exceeded` or `token_expired`; empty if the body
    /// had none.
    pub code: String,
    /// The backend's English message, or `code` if it sent none.
    pub message: String,
    /// The body's `error` object, every field included.
    pub details: serde_json::Map<String, serde_json::Value>,
}

impl ProviderRejection {
    /// The rejection a non-2xx response `body` describes, or `None` if it
    /// isn't one (any other JSON, or not JSON at all).
    pub fn from_body(status: u16, body: &str) -> Option<Self> {
        let serde_json::Value::Object(mut root) = serde_json::from_str(body).ok()? else {
            return None;
        };
        let serde_json::Value::Object(details) = root.remove("error")? else {
            return None;
        };
        let kind = RejectionKind::from_type(details.get("type")?.as_str()?)?;
        let field = |name: &str| {
            details
                .get(name)
                .and_then(|v| v.as_str())
                .map(str::to_owned)
        };
        let code = field("code").unwrap_or_default();
        let message = field("message").unwrap_or_else(|| code.clone());
        Some(Self {
            status,
            kind,
            code,
            message,
            details,
        })
    }
}

impl From<ProviderRejection> for Error {
    fn from(rejection: ProviderRejection) -> Self {
        Error::ProviderRejection(Box::new(rejection))
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
    fn is_not_retryable_for_a_context_overflow() {
        assert!(!Error::ContextOverflow("too long".into()).is_retryable());
    }

    fn limit_body(code: &str) -> String {
        serde_json::json!({
            "error": {
                "type": "openheim_limit",
                "code": code,
                "message": "You've used this 5-hour Lite allowance.",
                "plan": "lite",
                "limit": { "window": "5h", "resets_at": "2026-10-18T16:00:00Z" },
                "upgrade": { "plan": "pro", "plan_name": "Pro", "helps": true },
                "action": "upgrade",
            }
        })
        .to_string()
    }

    #[test]
    fn every_limit_code_parses() {
        for (code, status) in [
            ("subscription_required", 402),
            ("model_not_in_plan", 403),
            ("weekly_budget_exceeded", 402),
            ("window_budget_exceeded", 429),
            ("rate_limited", 429),
            ("some_future_code", 429),
        ] {
            let rejection = ProviderRejection::from_body(status, &limit_body(code)).unwrap();
            assert_eq!(rejection.status, status);
            assert_eq!(rejection.kind, RejectionKind::Limit);
            assert_eq!(rejection.code, code);
            assert_eq!(rejection.message, "You've used this 5-hour Lite allowance.");
            assert_eq!(rejection.details["upgrade"]["plan_name"], "Pro");
            assert_eq!(rejection.details["limit"]["window"], "5h");
        }
    }

    #[test]
    fn every_auth_code_parses() {
        for code in ["token_invalid", "token_expired", "token_revoked"] {
            let body = format!(
                r#"{{"error":{{"type":"openheim_auth","code":"{code}","message":"Sign in again."}}}}"#
            );
            let rejection = ProviderRejection::from_body(401, &body).unwrap();
            assert_eq!(rejection.kind, RejectionKind::Auth);
            assert_eq!(rejection.code, code);
            assert_eq!(rejection.message, "Sign in again.");
        }
    }

    #[test]
    fn a_rejection_without_a_message_shows_its_code() {
        let body = r#"{"error":{"type":"openheim_limit","code":"rate_limited"}}"#;
        let rejection = ProviderRejection::from_body(429, body).unwrap();
        assert_eq!(rejection.message, "rate_limited");
        assert_eq!(Error::from(rejection).to_string(), "rate_limited");
    }

    #[test]
    fn other_error_bodies_are_not_rejections() {
        for (status, body) in [
            (429, "rate limited"),
            (
                429,
                r#"{"error":{"message":"Rate limit reached","type":"requests"}}"#,
            ),
            (
                503,
                r#"{"error":{"type":"account_syncing","message":"try again"}}"#,
            ),
            (
                404,
                r#"{"error":{"type":"model_not_found","message":"no such model"}}"#,
            ),
            (
                400,
                r#"{"type":"error","error":{"type":"invalid_request_error"}}"#,
            ),
            (429, r#"{"error":"openheim_limit"}"#),
            (429, r#"[{"error":{"type":"openheim_limit"}}]"#),
        ] {
            assert!(
                ProviderRejection::from_body(status, body).is_none(),
                "{body}"
            );
        }
    }

    #[test]
    fn is_not_retryable_for_a_provider_rejection() {
        for code in ["rate_limited", "window_budget_exceeded"] {
            let rejection = ProviderRejection::from_body(429, &limit_body(code)).unwrap();
            assert!(!Error::from(rejection).is_retryable());
        }
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
