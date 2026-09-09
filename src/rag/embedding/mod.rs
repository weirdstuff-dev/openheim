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

use crate::config::EmbeddingConfig;
use crate::error::{Error, Result};

pub use gemini::GeminiEmbeddingClient;
pub use openai::OpenAiEmbeddingClient;

/// Largest number of inputs sent in one HTTP request. OpenAI accepts up to
/// 2048, Gemini 100; 64 keeps individual requests small enough that one
/// oversized chunk batch can't blow a provider's per-request token limit.
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

/// Embeds `inputs` in batches of at most [`MAX_BATCH`], preserving order.
pub(crate) async fn embed_batched(
    client: &dyn EmbeddingClient,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(inputs.len());
    for batch in inputs.chunks(MAX_BATCH) {
        let vectors = client.embed(batch).await?;
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

/// Picks the wire format for `config.provider_name`: `"gemini"` speaks
/// Gemini's API, everything else is OpenAI-compatible.
pub fn create_embedding_client(
    config: &EmbeddingConfig,
    http_client: &ReqwestClient,
) -> Arc<dyn EmbeddingClient> {
    match config.provider_name.as_str() {
        "gemini" => Arc::new(GeminiEmbeddingClient::new(
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
    fn create_embedding_client_picks_by_provider_name() {
        let http = reqwest::Client::new();
        let mut cfg = EmbeddingConfig {
            provider_name: "gemini".into(),
            api_base: "https://example".into(),
            api_key: "k".into(),
            model: "m".into(),
            timeout_secs: 10,
        };
        assert_eq!(create_embedding_client(&cfg, &http).model(), "m");
        cfg.provider_name = "ollama".into();
        assert_eq!(create_embedding_client(&cfg, &http).model(), "m");
    }
}
