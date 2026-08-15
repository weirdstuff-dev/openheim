//! Minimal incremental Server-Sent Events decoder shared by the streaming
//! provider clients.
//!
//! Each provider's streaming endpoint frames its response as SSE: UTF-8 chunks
//! arrive over the wire, split into `\n`-terminated lines, and the payloads we
//! care about are carried on `data:` lines. This decoder owns the cross-chunk
//! line buffering and `data:` extraction so the three providers
//! ([`super::anthropic`], [`super::gemini`], [`super::openai`]) don't each
//! re-implement the same framing state machine (and drift apart in the details).
//!
//! Interpreting each payload — JSON shape, the `[DONE]` sentinel, etc. — stays
//! with the caller, since that part genuinely differs per provider.

/// Accumulates raw byte chunks and yields complete SSE `data:` payloads.
pub(crate) struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Appends a raw byte chunk (as received from the HTTP body) to the buffer.
    ///
    /// Bytes are buffered undecoded: a multi-byte UTF-8 sequence can straddle
    /// a chunk boundary, and decoding each chunk in isolation would corrupt it
    /// into U+FFFD replacement characters. Decoding happens per complete line
    /// in [`SseDecoder::next_payload`].
    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pops the next complete `data:` payload, or `None` if no full line is
    /// buffered yet. Blank lines, comment lines (`:` prefix), and non-`data`
    /// fields (`event:`, `id:`, …) are skipped. The returned payload is trimmed.
    pub(crate) fn next_payload(&mut self) -> Option<String> {
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches('\r');

            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if let Some(data) = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
            {
                return Some(data.trim().to_string());
            }
            // A non-data field line (event:/id:/retry:) — nothing to surface.
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yields_payloads_split_across_chunks() {
        let mut dec = SseDecoder::new();
        dec.feed(b"data: hel");
        assert_eq!(dec.next_payload(), None);
        dec.feed(b"lo\ndata: world\n");
        assert_eq!(dec.next_payload().as_deref(), Some("hello"));
        assert_eq!(dec.next_payload().as_deref(), Some("world"));
        assert_eq!(dec.next_payload(), None);
    }

    #[test]
    fn skips_blank_and_comment_lines() {
        let mut dec = SseDecoder::new();
        dec.feed(b"\n: keep-alive\nevent: ping\ndata: payload\n");
        assert_eq!(dec.next_payload().as_deref(), Some("payload"));
        assert_eq!(dec.next_payload(), None);
    }

    #[test]
    fn handles_crlf_and_missing_space_after_colon() {
        let mut dec = SseDecoder::new();
        dec.feed(b"data:no-space\r\ndata: [DONE]\r\n");
        assert_eq!(dec.next_payload().as_deref(), Some("no-space"));
        assert_eq!(dec.next_payload().as_deref(), Some("[DONE]"));
    }

    #[test]
    fn preserves_multibyte_utf8_split_across_chunk_boundary() {
        // "日" (U+65E5) is 0xE6 0x97 0xA5 and "語" (U+8A9E) is 0xE8 0xAA 0x9E;
        // the chunk split lands inside the first sequence. Per-chunk lossy
        // decoding would turn each orphaned fragment into U+FFFD.
        let mut dec = SseDecoder::new();
        dec.feed(b"data: a\xE6\x97");
        assert_eq!(dec.next_payload(), None);
        dec.feed(b"\xA5\xE8\xAA\x9E\n");
        assert_eq!(dec.next_payload().as_deref(), Some("a日語"));
    }

    #[test]
    fn preserves_4byte_utf8_split_across_chunks() {
        // "🚀" (U+1F680) is 0xF0 0x9F 0x9A 0x80; split after the second byte,
        // with the trailing newline arriving in yet another chunk.
        let mut dec = SseDecoder::new();
        dec.feed(b"data: \xF0\x9F");
        dec.feed(b"\x9A\x80");
        dec.feed(b"\n");
        assert_eq!(dec.next_payload().as_deref(), Some("🚀"));
    }
}
