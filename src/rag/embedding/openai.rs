//! OpenAI-compatible `POST {api_base}/embeddings`.

use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::{EmbeddingClient, check_status};

/// Client for the OpenAI embeddings wire format — OpenAI itself, Ollama's
/// `/v1/embeddings`, Together, and any other compatible gateway.
#[derive(Clone)]
pub struct OpenAiEmbeddingClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
}

impl OpenAiEmbeddingClient {
    pub fn new(client: ReqwestClient, api_base: String, api_key: String, model: String) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
        }
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingClient for OpenAiEmbeddingClient {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        let url = format!("{}/embeddings", self.api_base.trim_end_matches('/'));
        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&EmbeddingRequest {
                model: &self.model,
                input: inputs,
            });
        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
        }
        let response = check_status(request.send().await?).await?;
        let mut parsed: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::ParseError(format!("invalid embeddings response: {e}")))?;
        // Providers document `data` as ordered, but `index` is authoritative.
        parsed.data.sort_by_key(|d| d.index);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_parses_and_is_reordered_by_index() {
        let body = r#"{"object":"list","data":[
            {"object":"embedding","index":1,"embedding":[0.5,0.5]},
            {"object":"embedding","index":0,"embedding":[1.0,0.0]}
        ],"model":"m","usage":{"prompt_tokens":2,"total_tokens":2}}"#;
        let mut parsed: EmbeddingResponse = serde_json::from_str(body).unwrap();
        parsed.data.sort_by_key(|d| d.index);
        assert_eq!(parsed.data[0].embedding, vec![1.0, 0.0]);
        assert_eq!(parsed.data[1].embedding, vec![0.5, 0.5]);
    }

    #[test]
    fn request_serialises_model_and_input_array() {
        let inputs = vec!["a".to_string(), "b".to_string()];
        let req = EmbeddingRequest {
            model: "text-embedding-3-small",
            input: &inputs,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "text-embedding-3-small");
        assert_eq!(json["input"], serde_json::json!(["a", "b"]));
    }
}
