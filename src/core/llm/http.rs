//! Shared HTTP plumbing for the provider clients: every one of them POSTs a
//! JSON body and needs the same status-check → `Error::HttpError { status,
//! body }` handling. Centralized here so that check (and the error body
//! read it requires) exists once instead of once per client.

use reqwest::{Client as ReqwestClient, Response};
use serde::Serialize;

use crate::error::{Error, Result};

/// POSTs `body` as JSON to `url` with `headers` applied on top of it (`.json`
/// already sets `Content-Type`), and turns a non-2xx response into
/// `Error::HttpError { status, body }` — reading the error body here, once,
/// instead of at every call site.
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
        return Err(Error::HttpError { status, body });
    }

    Ok(response)
}
