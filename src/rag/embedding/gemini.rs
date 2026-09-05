//! Gemini `POST {api_base}/models/{model}:batchEmbedContents`.

use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::{EmbeddingClient, check_status};

/// Client for Google's Gemini embedding API (`gemini-embedding-001`,
/// `text-embedding-004`, …).
#[derive(Clone)]
pub struct GeminiEmbeddingClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
}

impl GeminiEmbeddingClient {
    pub fn new(client: ReqwestClient, api_base: String, api_key: String, model: String) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
        }
    }

    /// Gemini wants `models/<name>` in request bodies; accept either form
    /// in config.
    fn qualified_model(&self) -> String {
        if self.model.starts_with("models/") {
            self.model.clone()
        } else {
            format!("models/{}", self.model)
        }
    }
}

#[derive(Serialize)]
struct BatchRequest {
    requests: Vec<EmbedRequest>,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: Content,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct BatchResponse {
    embeddings: Vec<Embedding>,
}

#[derive(Deserialize)]
struct Embedding {
    values: Vec<f32>,
}

#[async_trait]
impl EmbeddingClient for GeminiEmbeddingClient {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        let model = self.qualified_model();
        let url = format!(
            "{}/{}:batchEmbedContents",
            self.api_base.trim_end_matches('/'),
            model
        );
        let body = BatchRequest {
            requests: inputs
                .iter()
                .map(|text| EmbedRequest {
                    model: model.clone(),
                    content: Content {
                        parts: vec![Part { text: text.clone() }],
                    },
                })
                .collect(),
        };
        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;
        let response = check_status(response).await?;
        let parsed: BatchResponse = response
            .json()
            .await
            .map_err(|e| Error::ParseError(format!("invalid Gemini embeddings response: {e}")))?;
        Ok(parsed.embeddings.into_iter().map(|e| e.values).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_model_adds_prefix_once() {
        let http = reqwest::Client::new();
        let bare = GeminiEmbeddingClient::new(http.clone(), "b".into(), "k".into(), "gem".into());
        assert_eq!(bare.qualified_model(), "models/gem");
        let full = GeminiEmbeddingClient::new(http, "b".into(), "k".into(), "models/gem".into());
        assert_eq!(full.qualified_model(), "models/gem");
    }

    #[test]
    fn response_parses_values() {
        let body = r#"{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3]}]}"#;
        let parsed: BatchResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.embeddings.len(), 2);
        assert_eq!(parsed.embeddings[1].values, vec![0.3]);
    }
}
