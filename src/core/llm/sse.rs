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
//! with each provider's [`StreamParser`]; [`read_stream`] drives one over a
//! response body.

use tokio::sync::mpsc::UnboundedSender;

use super::LlmChunk;
use crate::core::models::Choice;
use crate::error::{Error, Result};

/// A provider's interpretation of its own SSE stream: one payload at a time,
/// then the finished reply.
pub(crate) trait StreamParser {
    /// Handles one `data:` payload, sending any text or thinking it carries
    /// to `chunk_tx`. Returns `true` once the provider has said the stream is
    /// over, so reading stops there.
    fn payload(&mut self, data: &str, chunk_tx: &UnboundedSender<LlmChunk>) -> Result<bool>;

    /// The complete reply, or [`Error::IncompleteResponse`] if the stream
    /// ended before the provider said the reply was finished. A connection
    /// that closes early looks like a normal end of body, so without this
    /// check a cut-off reply would be taken as complete.
    fn finish(self) -> Result<Choice>;
}

/// Reads `response`'s SSE body through `parser` until the parser sees the
/// end of the stream or the body ends, then returns `parser.finish()`.
pub(crate) async fn read_stream<P: StreamParser>(
    mut response: reqwest::Response,
    mut parser: P,
    chunk_tx: &UnboundedSender<LlmChunk>,
) -> Result<Choice> {
    let mut decoder = SseDecoder::new();
    loop {
        let bytes = response.chunk().await.map_err(Error::ReqwestError)?;
        match &bytes {
            Some(bytes) => decoder.feed(bytes),
            // End of body: terminate a last line that arrived without its
            // newline, so its payload still counts.
            None => decoder.feed(b"\n"),
        }
        while let Some(data) = decoder.next_payload() {
            if parser.payload(&data, chunk_tx)? {
                return parser.finish();
            }
        }
        if bytes.is_none() {
            return parser.finish();
        }
    }
}

/// Parses one payload as `T`, logging (not failing on) one that doesn't
/// parse. Skipping it keeps an unexpected event type from failing the whole
/// reply, but a skipped payload may have carried part of the reply, so it's
/// worth a warning with enough of the payload to see what it was.
pub(crate) fn parse_payload<T: serde::de::DeserializeOwned>(
    provider: &str,
    data: &str,
) -> Option<T> {
    match serde_json::from_str(data) {
        Ok(value) => Some(value),
        Err(e) => {
            let shown: String = data.chars().take(200).collect();
            tracing::warn!("{provider}: skipping unparseable stream payload ({e}): {shown}");
            None
        }
    }
}

/// Accumulates raw byte chunks and yields complete SSE `data:` payloads.
struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Appends a raw byte chunk (as received from the HTTP body) to the buffer.
    ///
    /// Bytes are buffered undecoded: a multi-byte UTF-8 sequence can straddle
    /// a chunk boundary, and decoding each chunk in isolation would corrupt it
    /// into U+FFFD replacement characters. Decoding happens per complete line
    /// in [`SseDecoder::next_payload`].
    fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pops the next complete `data:` payload, or `None` if no full line is
    /// buffered yet. Blank lines, comment lines (`:` prefix), and non-`data`
    /// fields (`event:`, `id:`, …) are skipped. The returned payload is trimmed.
    fn next_payload(&mut self) -> Option<String> {
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
    use crate::core::models::Message;

    /// Records every payload; says the stream is over at `"end"`, and only
    /// then counts as complete.
    #[derive(Default)]
    struct Recorder {
        payloads: Vec<String>,
        ended: bool,
    }

    impl StreamParser for Recorder {
        fn payload(&mut self, data: &str, _: &UnboundedSender<LlmChunk>) -> Result<bool> {
            self.payloads.push(data.to_string());
            self.ended = data == "end";
            Ok(self.ended)
        }

        fn finish(self) -> Result<Choice> {
            if !self.ended {
                return Err(Error::IncompleteResponse(self.payloads.join(",")));
            }
            Ok(Choice {
                message: Message::assistant(self.payloads.join(",")),
                finish_reason: None,
                usage: None,
            })
        }
    }

    async fn read(body: &'static str) -> Result<Choice> {
        let response = reqwest::Response::from(http::Response::new(body));
        let (chunk_tx, _chunk_rx) = tokio::sync::mpsc::unbounded_channel();
        read_stream(response, Recorder::default(), &chunk_tx).await
    }

    #[tokio::test]
    async fn read_stream_stops_at_the_end_signal() {
        let choice = read("data: a\n\ndata: end\n\ndata: ignored\n\n")
            .await
            .unwrap();
        assert_eq!(choice.message.text().as_deref(), Some("a,end"));
    }

    #[tokio::test]
    async fn read_stream_keeps_a_last_line_without_its_newline() {
        let choice = read("data: a\n\ndata: end").await.unwrap();
        assert_eq!(choice.message.text().as_deref(), Some("a,end"));
    }

    #[tokio::test]
    async fn read_stream_reports_a_body_that_ends_early() {
        let err = read("data: a\n\n").await.unwrap_err();
        assert!(matches!(err, Error::IncompleteResponse(_)), "{err}");
    }

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
