use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::time::sleep;

use super::{LlmChunk, LlmClient};
use crate::core::models::{Choice, Message, Tool};
use crate::error::Result;

const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 1000;

/// A wrapper around any `LlmClient` that retries transient failures with exponential backoff.
pub struct RetryClient {
    inner: Arc<dyn LlmClient>,
}

impl RetryClient {
    pub fn new(inner: Arc<dyn LlmClient>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl LlmClient for RetryClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        for attempt in 0..=MAX_RETRIES {
            match self.inner.send(messages, tools).await {
                Ok(choice) => return Ok(choice),
                Err(e) if attempt < MAX_RETRIES && e.is_retryable() => {
                    let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt));
                    tracing::warn!(
                        "LLM request failed (attempt {}/{}): {}. Retrying in {:?}...",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        e,
                        backoff
                    );
                    sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("loop always returns via Ok or Err arm")
    }

    /// Retries a failed streaming request **only while it is still safe to do
    /// so** — i.e. before any chunk has reached the caller. Once the first token
    /// has been forwarded a mid-stream failure can't be replayed without
    /// duplicating output, so it is returned as-is.
    ///
    /// Each attempt streams through a private channel; a forwarder relays chunks
    /// to the caller's `chunk_tx` and records whether anything was emitted.
    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        for attempt in 0..=MAX_RETRIES {
            let (inner_tx, mut inner_rx) = mpsc::unbounded_channel::<LlmChunk>();
            let emitted = Arc::new(AtomicBool::new(false));

            let forward_tx = chunk_tx.clone();
            let forward_emitted = emitted.clone();
            let forwarder = tokio::spawn(async move {
                while let Some(chunk) = inner_rx.recv().await {
                    forward_emitted.store(true, Ordering::SeqCst);
                    if forward_tx.send(chunk).is_err() {
                        break;
                    }
                }
            });

            let result = self.inner.send_streaming(messages, tools, inner_tx).await;
            // `inner_tx` is dropped when the call returns, so the forwarder drains
            // any remaining buffered chunks and then exits.
            let _ = forwarder.await;

            match result {
                Ok(choice) => return Ok(choice),
                Err(e)
                    if attempt < MAX_RETRIES
                        && e.is_retryable()
                        && !emitted.load(Ordering::SeqCst) =>
                {
                    let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt));
                    tracing::warn!(
                        "LLM stream failed before first chunk (attempt {}/{}): {}. Retrying in {:?}...",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        e,
                        backoff
                    );
                    sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("loop always returns via Ok or Err arm")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::models::FinishReason;
    use crate::error::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ok_choice(content: &str) -> Choice {
        Choice {
            message: Message::assistant(content),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        }
    }

    /// Mock that always succeeds
    struct AlwaysOk;

    #[async_trait]
    impl LlmClient for AlwaysOk {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> crate::error::Result<Choice> {
            Ok(ok_choice("success"))
        }
    }

    /// Mock that always fails with a non-retryable error
    struct AlwaysFailNonRetryable;

    #[async_trait]
    impl LlmClient for AlwaysFailNonRetryable {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> crate::error::Result<Choice> {
            Err(Error::HttpError {
                status: 400,
                body: "bad request".into(),
            })
        }
    }

    /// Mock that fails N times with a retryable error, then succeeds
    struct FailThenSucceed {
        remaining_failures: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for FailThenSucceed {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> crate::error::Result<Choice> {
            let remaining = self.remaining_failures.fetch_sub(1, Ordering::SeqCst);
            if remaining > 0 {
                Err(Error::HttpError {
                    status: 429,
                    body: "rate limited".into(),
                })
            } else {
                Ok(ok_choice("recovered"))
            }
        }
    }

    /// Mock that always fails with a retryable error (to test max retries)
    struct AlwaysFailRetryable {
        call_count: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for AlwaysFailRetryable {
        async fn send(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> crate::error::Result<Choice> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(Error::HttpError {
                status: 503,
                body: "service unavailable".into(),
            })
        }
    }

    #[tokio::test]
    async fn success_on_first_attempt() {
        let client = RetryClient::new(Arc::new(AlwaysOk));
        let result = client.send(&[], &[]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().message.text().as_deref(), Some("success"));
    }

    #[tokio::test]
    async fn non_retryable_error_returned_immediately() {
        let client = RetryClient::new(Arc::new(AlwaysFailNonRetryable));
        let result = client.send(&[], &[]).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::HttpError { status: 400, .. }
        ));
    }

    #[tokio::test]
    async fn success_after_transient_failure() {
        let inner = Arc::new(FailThenSucceed {
            remaining_failures: AtomicUsize::new(2),
        });
        let client = RetryClient::new(inner);
        let result = client.send(&[], &[]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().message.text().as_deref(), Some("recovered"));
    }

    #[tokio::test]
    async fn retryable_error_exhausts_retries() {
        let inner = Arc::new(AlwaysFailRetryable {
            call_count: AtomicUsize::new(0),
        });
        let client = RetryClient::new(inner.clone());
        let result = client.send(&[], &[]).await;
        assert!(result.is_err());
        // Should have been called 1 (initial) + 3 (retries) = 4 times
        assert_eq!(inner.call_count.load(Ordering::SeqCst), 4);
    }

    /// Streaming mock that fails (without emitting) the first `failures` times,
    /// then succeeds. Records how many times `send_streaming` was entered.
    struct StreamFailThenSucceed {
        remaining_failures: AtomicUsize,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for StreamFailThenSucceed {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            unimplemented!("streaming test only uses send_streaming")
        }

        async fn send_streaming(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
            chunk_tx: mpsc::UnboundedSender<LlmChunk>,
        ) -> Result<Choice> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.remaining_failures.fetch_sub(1, Ordering::SeqCst) > 0 {
                // Fail before emitting anything — safe to retry.
                Err(Error::HttpError {
                    status: 503,
                    body: "service unavailable".into(),
                })
            } else {
                let _ = chunk_tx.send(LlmChunk::Text("hello".into()));
                Ok(ok_choice("hello"))
            }
        }
    }

    /// Streaming mock that emits a chunk and *then* fails with a retryable error,
    /// to verify mid-stream failures are not retried.
    struct StreamEmitThenFail {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for StreamEmitThenFail {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            unimplemented!("streaming test only uses send_streaming")
        }

        async fn send_streaming(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
            chunk_tx: mpsc::UnboundedSender<LlmChunk>,
        ) -> Result<Choice> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _ = chunk_tx.send(LlmChunk::Text("partial".into()));
            Err(Error::HttpError {
                status: 503,
                body: "dropped mid-stream".into(),
            })
        }
    }

    #[tokio::test]
    async fn streaming_retries_when_failure_precedes_first_chunk() {
        let inner = Arc::new(StreamFailThenSucceed {
            remaining_failures: AtomicUsize::new(2),
            calls: AtomicUsize::new(0),
        });
        let client = RetryClient::new(inner.clone());
        let (tx, mut rx) = mpsc::unbounded_channel();

        let result = client.send_streaming(&[], &[], tx).await;

        assert!(result.is_ok());
        assert_eq!(inner.calls.load(Ordering::SeqCst), 3); // 2 failures + 1 success
        // The caller only sees the chunk from the successful attempt.
        assert!(matches!(rx.recv().await, Some(LlmChunk::Text(t)) if t == "hello"));
    }

    #[tokio::test]
    async fn streaming_does_not_retry_after_a_chunk_was_emitted() {
        let inner = Arc::new(StreamEmitThenFail {
            calls: AtomicUsize::new(0),
        });
        let client = RetryClient::new(inner.clone());
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = client.send_streaming(&[], &[], tx).await;

        assert!(result.is_err());
        // A single attempt: the emitted chunk makes the failure non-retryable.
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }
}
