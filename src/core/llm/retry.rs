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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::models::Role;
    use crate::error::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ok_choice(content: &str) -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content: Some(content.into()),
                tool_calls: None,
                tool_call_id: None,
                tool_name: None,
            },
            finish_reason: Some("stop".into()),
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
        assert_eq!(result.unwrap().message.content.as_deref(), Some("success"));
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
        assert_eq!(
            result.unwrap().message.content.as_deref(),
            Some("recovered")
        );
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
}
