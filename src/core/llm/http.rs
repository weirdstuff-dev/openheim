//! The POST-and-check-status step every provider client shares.

use reqwest::{Client as ReqwestClient, Response};
use serde::Serialize;

use crate::error::{Error, ProviderRejection, Result};

/// Lowercased phrases in a 400 body that mean the request was too long for
/// the model's context window:
/// - OpenAI (and Groq, DeepSeek, vLLM, OpenRouter): error code
///   `context_length_exceeded`, "This model's maximum context length is N
///   tokens", "reduce the length of the messages".
/// - Anthropic: "prompt is too long: N tokens > M maximum".
/// - Gemini: "The input token count (N) exceeds the maximum number of tokens
///   allowed (M)".
/// - Mistral: "... too large for model with N maximum context length".
/// - llama.cpp: "the request exceeds the available context size".
const OVERFLOW_MARKERS: &[&str] = &[
    "context_length_exceeded",
    "maximum context length",
    "reduce the length of the messages",
    "prompt is too long",
    "exceeds the maximum number of tokens",
    "exceeds the available context size",
    "exceeds the context window",
];

/// Whether an error response says the request didn't fit the context window.
/// A 413 always does: the request body itself was too large (Anthropic's
/// `request_too_large`, or a proxy's limit), and a shorter one may pass.
fn is_context_overflow(status: u16, body: &str) -> bool {
    if status == 413 {
        return true;
    }
    if status != 400 {
        return false;
    }
    let body = body.to_lowercase();
    OVERFLOW_MARKERS.iter().any(|marker| body.contains(marker))
}

/// The error for a non-2xx response: `Error::ContextOverflow` when it says
/// the request was too long, `Error::ProviderRejection` when it's a plan or
/// account rejection, and `Error::HttpError { status, body }` otherwise.
fn response_error(status: u16, body: String) -> Error {
    if is_context_overflow(status, &body) {
        return Error::ContextOverflow(body);
    }
    if let Some(rejection) = ProviderRejection::from_body(status, &body) {
        return rejection.into();
    }
    Error::HttpError { status, body }
}

/// POSTs `body` as JSON to `url` with `headers`, turning a non-2xx response
/// into an error (see [`response_error`]).
pub(super) async fn post_json(
    client: &ReqwestClient,
    url: &str,
    headers: &[(&str, &str)],
    body: &impl Serialize,
) -> Result<Response> {
    let mut request = client.post(url).json(body);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }

    let response = request.send().await.map_err(Error::ReqwestError)?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<failed to read error body>".into());
        return Err(response_error(status, body));
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_overflow_is_recognised() {
        let body = r#"{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens. Please reduce the length of the messages.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"#;
        assert!(is_context_overflow(400, body));
    }

    #[test]
    fn anthropic_overflow_is_recognised() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208310 tokens > 200000 maximum"}}"#;
        assert!(is_context_overflow(400, body));
        let too_large = r#"{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."}}"#;
        assert!(is_context_overflow(413, too_large));
    }

    #[test]
    fn gemini_overflow_is_recognised() {
        let body = r#"{"error":{"code":400,"message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}"#;
        assert!(is_context_overflow(400, body));
    }

    #[test]
    fn openai_compatible_overflows_are_recognised() {
        for body in [
            // vLLM / OpenRouter
            "This endpoint's maximum context length is 32768 tokens. However, you requested about 40000 tokens.",
            // Mistral
            "Prompt contains 40000 tokens and 0 draft tokens, too large for model with 32768 maximum context length",
            // llama.cpp
            r#"{"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error"}}"#,
        ] {
            assert!(is_context_overflow(400, body), "{body}");
        }
    }

    #[test]
    fn other_errors_are_not_overflows() {
        assert!(!is_context_overflow(
            400,
            r#"{"error":{"message":"Invalid 'tools[0].name'"}}"#
        ));
        assert!(!is_context_overflow(401, "prompt is too long"));
        assert!(!is_context_overflow(429, "rate limited"));
        assert!(!is_context_overflow(500, "maximum context length"));
    }

    #[test]
    fn a_plan_limit_body_is_a_provider_rejection() {
        let body = r#"{"error":{"type":"openheim_limit","code":"window_budget_exceeded","message":"You've used this 5-hour Lite allowance.","plan":"lite","action":"wait"}}"#;
        match response_error(429, body.into()) {
            Error::ProviderRejection(rejection) => {
                assert_eq!(rejection.status, 429);
                assert_eq!(rejection.code, "window_budget_exceeded");
            }
            other => panic!("expected a provider rejection, got {other:?}"),
        }
    }

    #[test]
    fn an_overflow_is_still_an_overflow() {
        let body = r#"{"error":{"message":"This model's maximum context length is 128000 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}"#;
        assert!(matches!(
            response_error(400, body.into()),
            Error::ContextOverflow(_)
        ));
    }

    #[test]
    fn any_other_body_stays_an_http_error() {
        let body = r#"{"error":{"type":"account_syncing","message":"try again"}}"#;
        assert!(matches!(
            response_error(503, body.into()),
            Error::HttpError { status: 503, body: b } if b == body
        ));
    }
}
