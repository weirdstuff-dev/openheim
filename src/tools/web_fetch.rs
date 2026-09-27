//! Built-in tool: `web_fetch` — fetches a URL and returns its content as text.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::redirect::Policy;
use serde::Deserialize;
use serde_json::json;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::parse;
use super::capabilities::{ToolCapabilities, ToolKindHint};

/// Wall-clock limit for the whole request (connect + headers + body).
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// Response body cap; past it the body is truncated with a marker.
const MAX_BODY_BYTES: usize = 256 * 1024;

/// Fetches `url` over HTTP(S) and returns its content as plain text: HTML is
/// stripped of markup, other text-like types (plain text, JSON, XML) are
/// returned verbatim. The URL is model-chosen, so every fetch is guarded:
///
/// - **Scheme allowlist** — only `http`/`https`.
/// - **SSRF guard** — every address the host resolves to is checked against
///   non-public ranges (cloud metadata included), and the checked address is
///   pinned for the connection, so DNS rebinding can't swap it.
/// - **No automatic redirects** — a 3xx is reported with its `Location`
///   instead of followed.
/// - **Timeout** ([`FETCH_TIMEOUT`]) and **body cap** ([`MAX_BODY_BYTES`]).
/// - **Content-type allowlist** — binary responses are rejected.
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
/// truncated. Stops downloading at the cap.
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

/// Resolves `host` (as `Url::host_str` gives it, so an IPv6 literal is in
/// brackets) and checks every address against [`is_disallowed_ip`],
/// returning the first to pin the connection to. `None` for a literal IP
/// (checked, nothing to pin).
async fn resolve_and_check(host: &str, port: u16) -> Result<Option<SocketAddr>> {
    let literal = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = literal.parse::<IpAddr>() {
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

/// True for every address that isn't publicly routable unicast, IPv4 or
/// IPv6: loopback, private, shared (CGNAT), link-local (which covers the
/// `169.254.169.254` cloud metadata address), unspecified, documentation,
/// benchmarking, multicast, reserved, and tunnels that could lead to any of
/// those. The ranges are written out because `Ipv4Addr::is_global` and
/// friends are unstable.
fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, c, _] = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_documentation()
                || v4.is_multicast() // 224.0.0.0/4
                || a == 0 // "this network", 0.0.0.0/8
                || (a == 100 && (b & 0xc0) == 64) // shared address space (CGNAT, Tailscale), 100.64.0.0/10
                || (a == 192 && b == 0 && c == 0) // IETF protocol assignments, 192.0.0.0/24
                || (a == 198 && (b & 0xfe) == 18) // benchmarking, 198.18.0.0/15
                || a >= 240 // reserved and broadcast, 240.0.0.0/4
        }
        IpAddr::V6(v6) => {
            // Addresses that carry an IPv4 address reach it (or are
            // translated to it), so it gets the IPv4 checks:
            // `::ffff:169.254.169.254` and `64:ff9b::a9fe:a9fe` are both
            // cloud metadata.
            if let Some(v4) = embedded_ipv4(&v6) {
                return is_disallowed_ip(IpAddr::V4(v4));
            }
            let s = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast() // ff00::/8
                || (s[0] & 0xfe00) == 0xfc00 // unique local, fc00::/7
                || (s[0] & 0xffc0) == 0xfe80 // link-local, fe80::/10
                || (s[0] & 0xffc0) == 0xfec0 // site-local (deprecated), fec0::/10
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation, 2001:db8::/32
                || (s[0] == 0x2001 && s[1] == 0) // Teredo, 2001::/32: tunnels to any IPv4 address
                || (s[0] == 0x64 && s[1] == 0xff9b && s[2] == 1) // local-use NAT64, 64:ff9b:1::/48
                || (s[0] == 0x100 && s[1..4] == [0, 0, 0]) // discard-only, 100::/64
        }
    }
}

/// The IPv4 address an IPv6 address carries, if it's one of the forms that
/// reach it: IPv4-mapped (`::ffff:a.b.c.d`), IPv4-compatible (`::a.b.c.d`,
/// deprecated, which [`Ipv6Addr::to_ipv4_mapped`] doesn't recognise),
/// NAT64 (`64:ff9b::a.b.c.d`, RFC 6052), or 6to4 (`2002:AABB:CCDD::`, the
/// address in bits 16–48).
fn embedded_ipv4(v6: &Ipv6Addr) -> Option<Ipv4Addr> {
    let s = v6.segments();
    let o = v6.octets();
    if let Some(v4) = v6.to_ipv4_mapped() {
        Some(v4)
    } else if s[0..6] == [0, 0, 0, 0, 0, 0] || s[0..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        Some(Ipv4Addr::new(o[12], o[13], o[14], o[15]))
    } else if s[0] == 0x2002 {
        Some(Ipv4Addr::new(o[2], o[3], o[4], o[5]))
    } else {
        None
    }
}

/// Strips markup from an HTML document into roughly readable text:
/// `<script>`/`<style>` bodies are dropped, block-level tags become line
/// breaks, other tags are removed, common entities are decoded, and blank
/// runs are collapsed. Not a real HTML parser; enough to spare the model the
/// markup.
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

/// Fetches a public URL and returns its content as plain text; see
/// [`fetch_url`] for the guards.
pub struct WebFetchTool;

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
}

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
        let args: WebFetchArgs = parse(args)?;
        // The request already has its own timeout; racing it against the
        // turn's cancel token additionally lets `session/cancel` drop a
        // fetch that's still in flight.
        tokio::select! {
            _ = turn.cancel.cancelled() => Err(Error::ToolExecutionError(
                "web_fetch cancelled".to_string(),
            )),
            result = fetch_url(&args.url) => result,
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            kind: ToolKindHint::Fetch,
            ..Default::default()
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

    #[test]
    fn is_disallowed_ip_flags_every_non_public_ipv4_range() {
        for (ip, range) in [
            ("0.0.0.0", "this network"),
            ("0.1.2.3", "this network"),
            ("100.64.0.1", "shared / CGNAT"),
            ("100.100.100.100", "shared / Tailscale"),
            ("100.127.255.254", "shared / CGNAT"),
            ("192.0.0.170", "IETF protocol assignments"),
            ("198.18.0.1", "benchmarking"),
            ("198.19.255.254", "benchmarking"),
            ("224.0.0.1", "multicast"),
            ("239.255.255.250", "multicast"),
            ("240.0.0.1", "reserved"),
            ("255.255.255.255", "broadcast"),
            ("192.0.2.1", "documentation"),
        ] {
            assert!(is_disallowed_ip(ip.parse().unwrap()), "{ip} ({range})");
        }
        // The public neighbours of those ranges stay allowed.
        for ip in [
            "1.0.0.1",
            "100.63.255.255",
            "100.128.0.1",
            "192.0.1.1",
            "198.17.255.255",
            "198.20.0.1",
            "223.255.255.254",
        ] {
            assert!(!is_disallowed_ip(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn is_disallowed_ip_flags_every_non_public_ipv6_range() {
        for (ip, range) in [
            ("ff02::1", "multicast"),
            ("ff0e::1", "multicast"),
            ("2001:db8::1", "documentation"),
            ("2001::1", "Teredo"),
            ("2001:0:4136:e378:8000:63bf:3fff:fdd2", "Teredo"),
            ("64:ff9b:1::a", "local-use NAT64"),
            ("fec0::1", "site-local"),
            ("100::1", "discard-only"),
        ] {
            assert!(is_disallowed_ip(ip.parse().unwrap()), "{ip} ({range})");
        }
        for ip in [
            "2606:4700:4700::1111",
            "2001:4860:4860::8888",
            "2001:db9::1",
        ] {
            assert!(!is_disallowed_ip(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn is_disallowed_ip_checks_the_ipv4_in_nat64_and_6to4() {
        for ip in [
            // 169.254.169.254, cloud metadata.
            "64:ff9b::a9fe:a9fe",
            "64:ff9b::169.254.169.254",
            "2002:a9fe:a9fe::",
            "2002:a9fe:a9fe:1::1",
            // 127.0.0.1 and 10.0.0.1.
            "64:ff9b::7f00:1",
            "2002:a00:1::",
            // 100.64.0.1, shared.
            "64:ff9b::6440:1",
        ] {
            assert!(is_disallowed_ip(ip.parse().unwrap()), "{ip}");
        }
        // 8.8.8.8 behind either stays allowed.
        assert!(!is_disallowed_ip("64:ff9b::808:808".parse().unwrap()));
        assert!(!is_disallowed_ip("2002:808:808::".parse().unwrap()));
    }

    #[tokio::test]
    async fn bracketed_ipv6_literals_are_checked_as_addresses() {
        for host in ["[::1]", "[64:ff9b::a9fe:a9fe]", "[2002:a9fe:a9fe::]"] {
            let err = resolve_and_check(host, 80).await.unwrap_err();
            assert!(
                err.to_string().contains("not publicly routable"),
                "{host}: {err}"
            );
        }
        // A public literal is allowed, with nothing to pin.
        assert!(
            resolve_and_check("[2606:4700:4700::1111]", 443)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn args_struct_matches_schema() {
        crate::tools::args::assert_args_match_schema::<WebFetchArgs>(
            &WebFetchTool,
            serde_json::json!({"url": "https://example.com"}),
        );
    }
}
