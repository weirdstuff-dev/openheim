//! Built-in tool: `web_fetch` — fetches a URL and returns its content as text.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::redirect::Policy;
use serde_json::json;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::{parse_args, require_str};

/// Wall-clock limit for the whole request (connect + headers + body).
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// Response body cap. Past this the body is truncated with a marker, so a
/// huge page can't balloon memory or the LLM's context.
const MAX_BODY_BYTES: usize = 256 * 1024;

/// Fetches `url` over HTTP(S) and returns its content as plain text: HTML is
/// stripped of markup, other text-like content types (plain text, JSON, XML)
/// are returned verbatim.
///
/// Single source of truth for the `web_fetch` behaviour.
///
/// Hardening applied to every fetch, since the URL is LLM-chosen input:
/// - **Scheme allowlist** — only `http`/`https`; no `file://`, `data:`, etc.
/// - **SSRF guard** — the host is resolved and every returned address is
///   checked against loopback/private/link-local/documentation ranges
///   (including the `169.254.169.254` cloud metadata address) before the
///   request is made, so the agent can't be steered into the internal
///   network. The checked address is then pinned for the actual connection
///   (`ClientBuilder::resolve`), so a second, attacker-influenced DNS lookup
///   inside the HTTP client (DNS rebinding) can't reach a different address
///   than the one that was validated.
/// - **No automatic redirects** — a 3xx response is reported back with its
///   `Location` header instead of being followed blindly, so a public URL
///   can't silently redirect the client into the internal network. The
///   caller can fetch the target URL itself if it wants to follow it.
/// - **Timeout** — the whole request is bounded by [`FETCH_TIMEOUT`].
/// - **Body cap** — capped at [`MAX_BODY_BYTES`], with a truncation marker.
/// - **Content-type allowlist** — only text-like responses are accepted;
///   binary payloads (images, archives, executables, ...) are rejected
///   rather than dumped into the model's context as noise.
pub(crate) async fn fetch_url(url: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| Error::ToolExecutionError(format!("Invalid URL '{url}': {e}")))?;

    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(Error::ToolExecutionError(format!(
                "Unsupported URL scheme '{other}': only http and https are allowed"
            )));
        }
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| Error::ToolExecutionError(format!("URL '{url}' has no host")))?;
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| Error::ToolExecutionError(format!("URL '{url}' has no resolvable port")))?;

    let pinned = resolve_and_check(host, port).await?;

    let mut builder = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .redirect(Policy::none());
    if let Some(addr) = pinned {
        builder = builder.resolve(host, addr);
    }
    let client = builder
        .build()
        .map_err(|e| Error::ToolExecutionError(format!("Failed to build HTTP client: {e}")))?;

    let response = client
        .get(parsed)
        .send()
        .await
        .map_err(|e| Error::ToolExecutionError(format!("Request to {url} failed: {e}")))?;

    let status = response.status();
    if status.is_redirection() {
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("(missing Location header)");
        return Err(Error::ToolExecutionError(format!(
            "{url} redirected ({status}) to {location}; fetch that URL directly if you want to follow it"
        )));
    }
    if !status.is_success() {
        return Err(Error::ToolExecutionError(format!(
            "{url} returned HTTP {status}"
        )));
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_html = content_type.contains("text/html") || content_type.contains("application/xhtml");
    let is_text_like = is_html
        || content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml");
    if !content_type.is_empty() && !is_text_like {
        return Err(Error::ToolExecutionError(format!(
            "{url} has unsupported content type '{content_type}'; web_fetch only supports text-like content"
        )));
    }

    let (body, truncated) = read_capped_body(response, MAX_BODY_BYTES).await?;
    let text = String::from_utf8_lossy(&body).into_owned();
    let mut result = if is_html { html_to_text(&text) } else { text };
    if truncated {
        result.push_str(&format!(
            "\n[content truncated at {MAX_BODY_BYTES} bytes]\n"
        ));
    }
    Ok(result)
}

/// Reads `response`'s body up to `cap` bytes, returning whether it was
/// truncated. Streamed rather than buffered whole, so an oversized response
/// stops as soon as the cap is hit instead of being downloaded in full.
async fn read_capped_body(response: reqwest::Response, cap: usize) -> Result<(Vec<u8>, bool)> {
    let mut stream = response.bytes_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|e| Error::ToolExecutionError(format!("Failed reading response body: {e}")))?;
        let remaining = cap - buf.len();
        let take = remaining.min(chunk.len());
        buf.extend_from_slice(&chunk[..take]);
        if buf.len() >= cap {
            return Ok((buf, true));
        }
    }
    Ok((buf, false))
}

/// Resolves `host` (if it's a hostname) and checks every candidate address
/// against [`is_disallowed_ip`], returning the first address so the caller
/// can pin the connection to exactly what was checked. Returns `None` for a
/// literal IP host (nothing to pin — there's only the one address, and it's
/// already been checked).
async fn resolve_and_check(host: &str, port: u16) -> Result<Option<SocketAddr>> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_disallowed_ip(ip) {
            return Err(Error::ToolExecutionError(format!(
                "Refusing to fetch {host}: address is not publicly routable"
            )));
        }
        return Ok(None);
    }

    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| Error::ToolExecutionError(format!("Failed to resolve host '{host}': {e}")))?
        .collect();
    if addrs.is_empty() {
        return Err(Error::ToolExecutionError(format!(
            "Host '{host}' did not resolve to any address"
        )));
    }
    for addr in &addrs {
        if is_disallowed_ip(addr.ip()) {
            return Err(Error::ToolExecutionError(format!(
                "Refusing to fetch {host}: resolves to non-public address {}",
                addr.ip()
            )));
        }
    }
    Ok(Some(addrs[0]))
}

/// True for loopback, private, link-local, unspecified, and other
/// non-publicly-routable addresses — including the `169.254.169.254`
/// cloud-metadata address (covered by the IPv4 link-local check) and IPv6
/// unique-local (`fc00::/7`) and link-local (`fe80::/10`) ranges.
fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
        }
        IpAddr::V6(v6) => {
            // IPv4-mapped (`::ffff:a.b.c.d`) and IPv4-compatible
            // (`::a.b.c.d`) addresses embed an IPv4 address that the native
            // IPv6 checks below wouldn't catch on their own — a request to
            // `::ffff:169.254.169.254` would sail straight past them despite
            // being cloud metadata. Run those embedded addresses back through
            // the IPv4 checks too. `to_ipv4_mapped()` covers the first form;
            // `ipv4_compatible` covers the second, deprecated (RFC 4291) one
            // that `to_ipv4_mapped()` doesn't — between them, `::` and `::1`
            // are still matched too (as 0.0.0.0 and 0.0.0.1, neither of which
            // is IPv4-loopback), same as the old `to_ipv4()` did, so this is
            // additive, not a replacement for the native checks below.
            let embedded_v4 = v6.to_ipv4_mapped().or_else(|| ipv4_compatible(&v6));
            let embeds_disallowed_ipv4 =
                embedded_v4.is_some_and(|v4| is_disallowed_ip(IpAddr::V4(v4)));
            embeds_disallowed_ipv4
                || v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique local, fc00::/7
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local, fe80::/10
        }
    }
}

/// The deprecated (RFC 4291) "IPv4-compatible" IPv6 form `::a.b.c.d`: the
/// first 96 bits are zero and the last 32 embed the IPv4 address directly —
/// distinct from the still-current "IPv4-mapped" form `::ffff:a.b.c.d`
/// that [`Ipv6Addr::to_ipv4_mapped`] already recognizes.
fn ipv4_compatible(v6: &Ipv6Addr) -> Option<Ipv4Addr> {
    let segments = v6.segments();
    if segments[0..6] == [0, 0, 0, 0, 0, 0] {
        let octets = v6.octets();
        Some(Ipv4Addr::new(
            octets[12], octets[13], octets[14], octets[15],
        ))
    } else {
        None
    }
}

/// Strips markup from an HTML document into roughly readable text:
/// `<script>`/`<style>` bodies are dropped entirely, block-level tags become
/// line breaks, remaining tags are removed, a handful of common entities are
/// decoded, and runs of blank lines are collapsed.
///
/// This is not a full HTML parser (no dependency pulled in for it) — it's
/// good enough to keep the model from paying for raw markup tokens, not for
/// extracting structured data.
fn html_to_text(html: &str) -> String {
    let without_scripts = strip_element(html, "script");
    let without_scripts_and_styles = strip_element(&without_scripts, "style");

    let mut out = String::with_capacity(without_scripts_and_styles.len());
    let mut chars = without_scripts_and_styles.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '<' {
            out.push(c);
            continue;
        }
        let mut tag = String::new();
        for c in chars.by_ref() {
            if c == '>' {
                break;
            }
            tag.push(c);
        }
        let tag_lower = tag.to_ascii_lowercase();
        // Leading '/' (closing tags) is stripped first, so "p" alone covers
        // both <p> and </p>.
        let name = tag_lower.trim_start_matches('/');
        let is_block = matches!(
            name.split(|c: char| c.is_whitespace()).next(),
            Some("br" | "p" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
        );
        if is_block {
            out.push('\n');
        }
    }

    collapse_blank_lines(&decode_entities(&out))
}

/// Removes every `<tag ...>...</tag>` span (case-insensitive) from `html`,
/// including the tags themselves. An unclosed opening tag drops everything
/// to the end of the document rather than leaving its body in the output.
fn strip_element(html: &str, tag: &str) -> String {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    while let Some(start_rel) = lower[pos..].find(&open) {
        let start = pos + start_rel;
        out.push_str(&html[pos..start]);
        match lower[start..].find(&close) {
            Some(end_rel) => pos = start + end_rel + close.len(),
            None => return out,
        }
    }
    out.push_str(&html[pos..]);
    out
}

/// Decodes the small set of HTML entities that show up in ordinary body
/// text; anything else is left as-is.
fn decode_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// Trims each line and collapses runs of blank lines to a single one, so
/// deeply nested markup doesn't turn into pages of empty lines.
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_blank = true; // swallow leading blank lines too
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            if !last_was_blank {
                out.push('\n');
            }
            last_was_blank = true;
        } else {
            out.push_str(line);
            out.push('\n');
            last_was_blank = false;
        }
    }
    out.trim_end().to_string()
}

/// Fetches a URL over HTTP(S) and returns its content as plain text.
///
/// HTML responses are stripped of markup; other text-like content types
/// (plain text, JSON, XML) are returned verbatim. Only public, non-redirect,
/// text-like responses under 256 KiB are supported — see [`fetch_url`] for
/// the full list of guards applied.
pub struct WebFetchTool;

#[async_trait]
impl ToolHandler for WebFetchTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "web_fetch",
            "Fetch a web page or other text-like resource (HTML, plain text, JSON, XML) from a public http(s) URL and return its content as plain text. HTML is stripped of markup. Requests time out after 20 seconds, redirects are not followed automatically (the redirect target is reported instead), and content is truncated at 256 KiB. Only publicly-routable addresses can be fetched.",
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The http:// or https:// URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args = parse_args(args)?;
        let url = require_str(&args, "url")?;
        // The request already has its own timeout; racing it against the
        // turn's cancel token additionally lets `session/cancel` drop a
        // fetch that's still in flight.
        tokio::select! {
            _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError(
                "web_fetch cancelled".to_string(),
            )),
            result = fetch_url(url) => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_support::TurnHarness;

    #[test]
    fn definition_has_correct_name() {
        let tool = WebFetchTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "web_fetch");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = WebFetchTool.execute("not json", &harness.turn()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_url() {
        let harness = TurnHarness::new();
        let result = WebFetchTool
            .execute(r#"{"other": "value"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("url"));
    }

    #[tokio::test]
    async fn rejects_non_http_scheme() {
        let err = fetch_url("file:///etc/passwd")
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("scheme"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn rejects_loopback_host() {
        let err = fetch_url("http://127.0.0.1/")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not publicly routable"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_localhost_hostname() {
        let err = fetch_url("http://localhost/")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("non-public address"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_cloud_metadata_address() {
        let err = fetch_url("http://169.254.169.254/latest/meta-data/")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not publicly routable"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_private_ip_range() {
        let err = fetch_url("http://10.0.0.1/").await.unwrap_err().to_string();
        assert!(
            err.contains("not publicly routable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn html_to_text_strips_tags_and_scripts() {
        let html = "<html><head><script>evil()</script><style>.x{}</style></head>\
                     <body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>";
        let text = html_to_text(html);
        assert!(!text.contains("evil()"));
        assert!(!text.contains(".x{}"));
        assert!(text.contains("Title"));
        assert!(text.contains("Hello world."));
    }

    #[test]
    fn html_to_text_decodes_entities() {
        let text = html_to_text("<p>Tom &amp; Jerry &lt;3&gt;</p>");
        assert_eq!(text, "Tom & Jerry <3>");
    }

    #[test]
    fn is_disallowed_ip_flags_expected_ranges() {
        assert!(is_disallowed_ip("127.0.0.1".parse().unwrap()));
        assert!(is_disallowed_ip("10.1.2.3".parse().unwrap()));
        assert!(is_disallowed_ip("172.16.0.1".parse().unwrap()));
        assert!(is_disallowed_ip("192.168.1.1".parse().unwrap()));
        assert!(is_disallowed_ip("169.254.169.254".parse().unwrap()));
        assert!(is_disallowed_ip("::1".parse().unwrap()));
        assert!(is_disallowed_ip("fc00::1".parse().unwrap()));
        assert!(is_disallowed_ip("fe80::1".parse().unwrap()));
        assert!(!is_disallowed_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_disallowed_ip("1.1.1.1".parse().unwrap()));
    }

    #[test]
    fn is_disallowed_ip_flags_ipv4_embedded_in_ipv6() {
        // IPv4-mapped (`::ffff:a.b.c.d`) must be checked against the same
        // IPv4 ranges as the bare address.
        assert!(is_disallowed_ip("::ffff:127.0.0.1".parse().unwrap()));
        assert!(is_disallowed_ip("::ffff:169.254.169.254".parse().unwrap()));
        assert!(is_disallowed_ip("::ffff:10.1.2.3".parse().unwrap()));
        assert!(is_disallowed_ip("::ffff:172.16.0.1".parse().unwrap()));
        assert!(is_disallowed_ip("::ffff:192.168.1.1".parse().unwrap()));
        // IPv4-compatible (`::a.b.c.d`, deprecated form) likewise.
        assert!(is_disallowed_ip("::127.0.0.1".parse().unwrap()));
        assert!(is_disallowed_ip("::169.254.169.254".parse().unwrap()));
        // A publicly routable address embedded either way stays allowed.
        assert!(!is_disallowed_ip("::ffff:8.8.8.8".parse().unwrap()));
        assert!(!is_disallowed_ip("::8.8.8.8".parse().unwrap()));
    }
}
