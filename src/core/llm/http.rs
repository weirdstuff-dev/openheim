//! The POST-and-check-status step every provider client shares.

use reqwest::{Client as ReqwestClient, Response};
use serde::Serialize;

use crate::error::{Error, Result};

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

/// POSTs `body` as JSON to `url` with `headers`, turning a non-2xx response
/// into `Error::ContextOverflow` when it says the request was too long, and
/// `Error::HttpError` otherwise (see [`Error::from_response`]).
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
        return Err(match Error::from_response(response).await {
            Error::HttpError { status, body, .. } if is_context_overflow(status, &body) => {
                Error::ContextOverflow(body)
            }
            e => e,
        });
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real error bodies from each provider and common compatible backends.
    #[test]
    fn overflows_are_recognised_by_provider_message() {
        let overflows = [
            (
                400,
                r#"{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens. Please reduce the length of the messages.","code":"context_length_exceeded"}}"#,
            ),
            (
                400,
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208310 tokens > 200000 maximum"}}"#,
            ),
            (
                413,
                r#"{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."}}"#,
            ),
            (
                400,
                r#"{"error":{"code":400,"message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}"#,
            ),
            (
                400,
                "This endpoint's maximum context length is 32768 tokens. However, you requested about 40000 tokens.",
            ),
            (
                400,
                "Prompt contains 40000 tokens and 0 draft tokens, too large for model with 32768 maximum context length",
            ),
            (
                400,
                r#"{"error":{"code":400,"message":"the request exceeds the available context size, try increasing it"}}"#,
            ),
        ];
        for (status, body) in overflows {
            assert!(is_context_overflow(status, body), "{body}");
        }
        let others = [
            (400, r#"{"error":{"message":"Invalid 'tools[0].name'"}}"#),
            (401, "prompt is too long"),
            (429, "rate limited"),
            (500, "maximum context length"),
        ];
        for (status, body) in others {
            assert!(!is_context_overflow(status, body), "{status} {body}");
        }
    }
}
