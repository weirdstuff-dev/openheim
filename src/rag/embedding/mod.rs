//! Text-embedding providers behind one [`EmbeddingClient`] trait.
//!
//! Two wire formats cover every provider openheim can talk to: the OpenAI
//! `/embeddings` shape ([`OpenAiEmbeddingClient`], also spoken by Ollama,
//! Together, and most self-hosted gateways) and Gemini's
//! `batchEmbedContents` ([`GeminiEmbeddingClient`]). Anthropic has no
//! embeddings API, which `AppConfig::resolve_embedding` rejects up front.

mod gemini;
mod openai;

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client as ReqwestClient;

use crate::config::{EmbeddingConfig, ProviderKind};
use crate::error::{Error, Result};

pub use gemini::GeminiEmbeddingClient;
pub use openai::OpenAiEmbeddingClient;

/// Largest number of inputs per request, well under every provider's limit
/// (OpenAI 2048, Gemini 100) so a batch can't exceed its token limit.
pub(crate) const MAX_BATCH: usize = 64;

/// Turns text into fixed-size float vectors.
///
/// Implement this to plug in a custom embeddings backend (a local model, an
/// enterprise gateway, …) and hand it to [`crate::rag::LongTermMemory::new`].
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// The model identifier, recorded in the store so a model switch is
    /// detected and triggers a full re-index instead of mixing vector spaces.
    fn model(&self) -> &str;

    /// Embeds every input, returning one vector per input in the same order.
    /// Implementations must return exactly `inputs.len()` vectors.
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Retries of one embeddings request after a transient failure (see
/// [`Error::is_retryable`]).
const MAX_RETRIES: u32 = 2;

/// Wait before the first retry; doubled for each one after.
const INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_millis(500);

/// `client.embed(inputs)`, retried up to [`MAX_RETRIES`] times on a
/// transient failure (rate limit, 5xx, network).
async fn embed_with_retry(
    client: &dyn EmbeddingClient,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let mut attempt = 0;
    loop {
        match client.embed(inputs).await {
            Err(e) if attempt < MAX_RETRIES && e.is_retryable() => {
                let backoff = INITIAL_BACKOFF * 2u32.pow(attempt);
                tracing::warn!(error = %e, ?backoff, "embeddings request failed; retrying");
                tokio::time::sleep(backoff).await;
                attempt += 1;
            }
            result => return result,
        }
    }
}

/// Embeds `inputs` in batches of at most [`MAX_BATCH`], preserving order.
/// Each batch is retried on a transient failure.
pub(crate) async fn embed_batched(
    client: &dyn EmbeddingClient,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(inputs.len());
    for batch in inputs.chunks(MAX_BATCH) {
        let vectors = embed_with_retry(client, batch).await?;
        if vectors.len() != batch.len() {
            return Err(Error::ApiError(format!(
                "embeddings provider returned {} vectors for {} inputs",
                vectors.len(),
                batch.len()
            )));
        }
        out.extend(vectors);
    }
    Ok(out)
}

/// Picks the wire format for `config.kind`: Gemini speaks Gemini's API,
/// everything else is OpenAI-compatible (`resolve_embedding` already
/// rejected Anthropic, which has no embeddings API).
pub fn create_embedding_client(
    config: &EmbeddingConfig,
    http_client: &ReqwestClient,
) -> Arc<dyn EmbeddingClient> {
    match config.kind {
        ProviderKind::Gemini => Arc::new(GeminiEmbeddingClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
        )),
        _ => Arc::new(OpenAiEmbeddingClient::new(
            http_client.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.model.clone(),
        )),
    }
}

/// Reads a non-2xx response into [`Error::HttpError`] so retry logic and
/// callers see the same shape the chat clients produce.
pub(crate) async fn check_status(response: reqwest::Response) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    Err(Error::HttpError {
        status: status.as_u16(),
        body,
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Deterministic, provider-free embedder for tests: a bag-of-words hash
    /// into `dims` buckets, L2-normalised. Similar texts share buckets, so
    /// nearest-neighbour assertions behave sensibly.
    pub(crate) struct HashEmbedder {
        pub dims: usize,
        pub model: String,
    }

    impl HashEmbedder {
        pub(crate) fn new(dims: usize) -> Self {
            Self {
                dims,
                model: "hash-test".to_string(),
            }
        }

        pub(crate) fn vector(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0f32; self.dims];
            for word in text.split(|c: char| !c.is_alphanumeric()) {
                if word.is_empty() {
                    continue;
                }
                let mut h: u64 = 0xcbf29ce484222325;
                for b in word.to_lowercase().bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x100000001b3);
                }
                v[(h % self.dims as u64) as usize] += 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            } else {
                v[0] = 1.0;
            }
            v
        }
    }

    #[async_trait]
    impl EmbeddingClient for HashEmbedder {
        fn model(&self) -> &str {
            &self.model
        }

        async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(inputs.iter().map(|t| self.vector(t)).collect())
        }
    }

    /// A [`HashEmbedder`] that fails with HTTP `status`: its first
    /// `failures` requests, and every request containing an input with
    /// `poison` in it. Counts requests and the inputs it embedded.
    pub(crate) struct FlakyEmbedder {
        pub(crate) inner: HashEmbedder,
        pub(crate) status: u16,
        pub(crate) failures: std::sync::atomic::AtomicUsize,
        pub(crate) poison: Option<&'static str>,
        pub(crate) requests: std::sync::atomic::AtomicUsize,
        pub(crate) embedded: std::sync::atomic::AtomicUsize,
    }

    impl FlakyEmbedder {
        /// Fails every request with `status`.
        pub(crate) fn down(status: u16) -> Self {
            Self::failing(status, usize::MAX)
        }

        /// Fails the first `failures` requests with `status`.
        pub(crate) fn failing(status: u16, failures: usize) -> Self {
            Self {
                inner: HashEmbedder::new(16),
                status,
                failures: failures.into(),
                poison: None,
                requests: 0.into(),
                embedded: 0.into(),
            }
        }

        /// Fails only requests with an input containing `poison`, with a
        /// status that isn't retried.
        pub(crate) fn poisoned(poison: &'static str) -> Self {
            Self {
                poison: Some(poison),
                ..Self::failing(400, 0)
            }
        }

        pub(crate) fn requests(&self) -> usize {
            self.requests.load(std::sync::atomic::Ordering::SeqCst)
        }

        pub(crate) fn embedded(&self) -> usize {
            self.embedded.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingClient for FlakyEmbedder {
        fn model(&self) -> &str {
            self.inner.model()
        }

        async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            use std::sync::atomic::Ordering;
            self.requests.fetch_add(1, Ordering::SeqCst);
            let failing = self
                .failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok();
            let poisoned = self
                .poison
                .is_some_and(|p| inputs.iter().any(|input| input.contains(p)));
            if failing || poisoned {
                return Err(Error::HttpError {
                    status: self.status,
                    body: "embeddings unavailable".into(),
                });
            }
            self.embedded.fetch_add(inputs.len(), Ordering::SeqCst);
            self.inner.embed(inputs).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::HashEmbedder;
    use super::*;

    #[tokio::test]
    async fn embed_batched_preserves_order_across_batches() {
        let client = HashEmbedder::new(16);
        let inputs: Vec<String> = (0..(MAX_BATCH * 2 + 3))
            .map(|i| format!("word{i}"))
            .collect();
        let vectors = embed_batched(&client, &inputs).await.unwrap();
        assert_eq!(vectors.len(), inputs.len());
        for (i, text) in inputs.iter().enumerate() {
            assert_eq!(vectors[i], client.vector(text));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn transient_failures_are_retried() {
        let client = super::test_support::FlakyEmbedder::failing(429, 2);
        let vectors = embed_batched(&client, &["a".to_string()]).await.unwrap();
        assert_eq!(vectors.len(), 1);
        assert_eq!(client.requests(), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn retries_are_bounded() {
        let client = super::test_support::FlakyEmbedder::down(503);
        assert!(embed_batched(&client, &["a".to_string()]).await.is_err());
        assert_eq!(client.requests(), 1 + MAX_RETRIES as usize);
    }

    #[tokio::test]
    async fn permanent_failures_are_not_retried() {
        let client = super::test_support::FlakyEmbedder::down(401);
        assert!(embed_batched(&client, &["a".to_string()]).await.is_err());
        assert_eq!(client.requests(), 1);
    }

    #[tokio::test]
    async fn embed_batched_empty_input_is_empty() {
        let client = HashEmbedder::new(8);
        assert!(embed_batched(&client, &[]).await.unwrap().is_empty());
    }

    struct ShortChanger;

    #[async_trait]
    impl EmbeddingClient for ShortChanger {
        fn model(&self) -> &str {
            "short"
        }
        async fn embed(&self, _inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn embed_batched_rejects_wrong_vector_count() {
        let err = embed_batched(&ShortChanger, &["a".to_string()])
            .await
            .unwrap_err();
        assert!(matches!(err, Error::ApiError(_)));
    }

    #[test]
    fn create_embedding_client_picks_by_kind() {
        let http = reqwest::Client::new();
        let mut cfg = EmbeddingConfig {
            provider_name: "gemini".into(),
            kind: ProviderKind::Gemini,
            api_base: "https://example".into(),
            api_key: "k".into(),
            model: "m".into(),
            timeout_secs: 10,
        };
        assert_eq!(create_embedding_client(&cfg, &http).model(), "m");
        cfg.kind = ProviderKind::OpenAiCompatible;
        assert_eq!(create_embedding_client(&cfg, &http).model(), "m");
    }
}
