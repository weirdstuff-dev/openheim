//! The POST-and-check-status step every provider client shares.

use reqwest::{Client as ReqwestClient, Response};
use serde::Serialize;

use crate::error::{Error, Result};

/// POSTs `body` as JSON to `url` with `headers`, turning a non-2xx response
/// into `Error::HttpError { status, body }`.
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
