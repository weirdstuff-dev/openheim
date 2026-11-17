use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::time::sleep;

use super::{LlmChunk, LlmClient};
use crate::core::models::{Choice, Message, Tool};
use crate::error::{Error, Result};

const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 1000;

/// How long to wait before retrying after `e` on attempt `attempt` (from 0):
/// the provider's `Retry-After` if it sent one, else exponential backoff.
fn retry_delay(e: &Error, attempt: u32) -> Duration {
    e.retry_after()
        .unwrap_or_else(|| Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt)))
}

/// A wrapper around any `LlmClient` that retries transient failures with
/// exponential backoff, or after the provider's `Retry-After`.
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
                    let backoff = retry_delay(&e, attempt);
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

    /// Retries a failed streaming request only before any chunk has reached
    /// the caller; after that a retry would duplicate output, so the error is
    /// returned. Each attempt streams through a forwarder that records
    /// whether anything was passed on.
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
                    let backoff = retry_delay(&e, attempt);
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
