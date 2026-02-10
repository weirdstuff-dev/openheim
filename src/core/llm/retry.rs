use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;

use super::LlmClient;
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
        let mut last_err = None;

        for attempt in 0..=MAX_RETRIES {
            match self.inner.send(messages, tools).await {
                Ok(choice) => return Ok(choice),
                Err(e) => {
                    if attempt < MAX_RETRIES && e.is_retryable() {
                        let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt));
                        tracing::warn!(
                            "LLM request failed (attempt {}/{}): {}. Retrying in {:?}...",
                            attempt + 1,
                            MAX_RETRIES + 1,
                            e,
                            backoff
                        );
                        sleep(backoff).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_err.unwrap())
    }
}
