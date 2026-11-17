use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedSender};

use crate::core::models::{Choice, ContentBlock, Effort, FinishReason, Message, Role, Tool, Usage};
use crate::error::{Error, Result};

use super::sse::{StreamParser, parse_payload, read_stream};
use super::{LlmChunk, LlmClient};

#[derive(Clone)]
pub struct GeminiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
    /// See [`Self::with_effort`].
    effort: Effort,
}

impl GeminiClient {
    pub fn new(
        client: ReqwestClient,
        api_base: String,
        api_key: String,
        model: String,
        max_tokens: Option<u32>,
    ) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
            max_tokens,
            effort: Effort::Default,
        }
    }

    /// Sends `effort` as `generationConfig.thinkingConfig.thinkingLevel`.
    /// [`Effort::Default`] (the default) leaves `thinkingConfig` out,
    /// so the model's own default applies. Models that predate thinking
    /// levels (Gemini 2.5 and earlier) reject the field, as do models that
    /// don't offer the level asked for.
    pub fn with_effort(self, effort: Effort) -> Self {
        Self { effort, ..self }
    }
}

// --- Gemini request types ---

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiToolDeclaration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    thinking_level: &'static str,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct GeminiContent {
    #[serde(default)]
    role: String,
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
    /// Gemini's encrypted reasoning for this part. On a `functionCall` part
    /// it has to be sent back unchanged: Gemini 3 rejects a request whose
    /// current-turn function calls lack it (400). With parallel calls only
    /// the first one carries it.
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
}

/// The documented stand-in `thoughtSignature` for a function call Gemini
/// didn't produce (history from another provider or model). Gemini 3 skips
/// signature validation for it instead of rejecting the request.
const SKIP_SIGNATURE_VALIDATION: &str = "skip_thought_signature_validator";

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    /// Gemini 3.x ids every call and expects the matching
    /// `functionResponse.id` back; calls without one get a generated id (see
    /// `tool_use_block`), which is sent on both sides too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    args: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    /// The `functionCall.id` this answers. Gemini 3.x requires it; without
    /// it `generateContent` replies with an empty `STOP`.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    response: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolDeclaration {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    /// The tool's parameters as JSON Schema. Not `parameters`, which takes
    /// Gemini's own OpenAPI-subset `Schema` and rejects the keys MCP servers'
    /// schemas commonly carry (`$schema`, `additionalProperties`, …), failing
    /// every request that lists the tool.
    parameters_json_schema: Value,
}

/// `schema` without its top-level `$schema` key: it only names the JSON
/// Schema dialect, and Gemini has no use for it.
fn without_dialect(mut schema: Value) -> Value {
    if let Some(object) = schema.as_object_mut() {
        object.remove("$schema");
    }
    schema
}

// --- Gemini response types ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    cached_content_token_count: u64,
}

impl From<GeminiUsageMetadata> for Usage {
    fn from(u: GeminiUsageMetadata) -> Self {
        Usage {
            // Gemini's `promptTokenCount` already includes cached tokens; see
            // the same normalization in the OpenAI client.
            input_tokens: u
                .prompt_token_count
                .saturating_sub(u.cached_content_token_count),
            output_tokens: u.candidates_token_count,
            cache_creation_tokens: 0,
            cache_read_tokens: u.cached_content_token_count,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    /// Absent on a candidate that only reports its finish, e.g. one blocked
    /// for safety.
    #[serde(default)]
    content: GeminiContent,
    finish_reason: Option<String>,
}

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Vec<GeminiContent> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::Assistant => {
                let mut parts = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } if !text.is_empty() => {
                            parts.push(GeminiPart {
                                text: Some(text.clone()),
                                ..Default::default()
                            });
                        }
                        ContentBlock::ToolUse {
                            id,
                            name,
                            arguments,
                            signature,
                        } => {
                            parts.push(GeminiPart {
                                function_call: Some(GeminiFunctionCall {
                                    id: Some(id.clone()),
                                    name: name.clone(),
                                    args: super::tool_input(name, arguments),
                                }),
                                thought_signature: signature.clone(),
                                ..Default::default()
                            });
                        }
                        _ => {}
                    }
                }
                sign_foreign_function_calls(&mut parts);
                if !parts.is_empty() {
                    result.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            Role::Tool => {
                // Tool results become functionResponse parts in a user turn.
                let Some(tr) = msg.tool_result_block() else {
                    continue;
                };
                let part = GeminiPart {
                    function_response: Some(GeminiFunctionResponse {
                        id: Some(tr.tool_call_id),
                        name: tr.tool_name,
                        response: serde_json::json!({ "result": tr.content }),
                    }),
                    ..Default::default()
                };
                // Merge into last user/function-response content if possible
                if let Some(last) = result.last_mut() {
                    let last: &mut GeminiContent = last;
                    if last.role == "user" {
                        last.parts.push(part);
                        continue;
                    }
                }
                result.push(GeminiContent {
                    role: "user".to_string(),
                    parts: vec![part],
                });
            }
            Role::User => {
                let mut parts = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(GeminiPart {
                                text: Some(text.clone()),
                                ..Default::default()
                            });
                        }
                        ContentBlock::Image { data, mime_type } => {
                            parts.push(GeminiPart {
                                inline_data: Some(GeminiInlineData {
                                    mime_type: mime_type.clone(),
                                    data: data.clone(),
                                }),
                                ..Default::default()
                            });
                        }
                        _ => {}
                    }
                }
                if parts.is_empty() {
                    parts.push(GeminiPart {
                        text: Some(String::new()),
                        ..Default::default()
                    });
                }
                result.push(GeminiContent {
                    role: "user".to_string(),
                    parts,
                });
            }
            Role::System => {
                // extracted into system_instruction field of GeminiRequest
            }
        }
    }

    result
}

fn convert_tools(tools: &[Tool]) -> Vec<GeminiToolDeclaration> {
    if tools.is_empty() {
        return Vec::new();
    }

    let declarations = tools
        .iter()
        .map(|t| GeminiFunctionDeclaration {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters_json_schema: without_dialect(t.function.parameters.clone()),
        })
        .collect();

    vec![GeminiToolDeclaration {
        function_declarations: declarations,
    }]
}

/// Maps Gemini's `finishReason` vocabulary onto the provider-agnostic
/// [`FinishReason`]; anything without a known equivalent passes through,
/// lowercased, as [`FinishReason::Other`].
fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::MaxTokens,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            FinishReason::Refusal
        }
        other => FinishReason::Other(other.to_lowercase()),
    }
}

/// Gives the first function call of a model turn the stand-in signature when
/// none of the turn's calls has one, i.e. when Gemini didn't produce them.
/// Calls Gemini did produce are left exactly as received: with parallel calls
/// only the first carries a signature, and the rest must stay without.
fn sign_foreign_function_calls(parts: &mut [GeminiPart]) {
    let mut calls = parts.iter_mut().filter(|p| p.function_call.is_some());
    let Some(first) = calls.next() else {
        return;
    };
    if first.thought_signature.is_none() && calls.all(|p| p.thought_signature.is_none()) {
        first.thought_signature = Some(SKIP_SIGNATURE_VALIDATION.to_string());
    }
}

/// A streamed `functionCall` part as a `ToolUse` block, keeping Gemini's id
/// when it sent one (generating a unique one otherwise) and the part's
/// `thoughtSignature`.
fn tool_use_block(call: GeminiFunctionCall, signature: Option<String>) -> Result<ContentBlock> {
    Ok(ContentBlock::ToolUse {
        id: call.id.unwrap_or_else(super::new_tool_call_id),
        arguments: serde_json::to_string(&call.args)?,
        name: call.name,
        signature,
    })
}

fn gemini_system_instruction(messages: &[Message]) -> Option<GeminiContent> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .filter_map(|m| m.text())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart {
                text: Some(parts.join("\n\n")),
                ..Default::default()
            }],
        })
    }
}

impl GeminiClient {
    fn build_request(&self, messages: &[Message], tools: &[Tool]) -> GeminiRequest {
        let thinking_config = self
            .effort
            .api_value()
            .map(|thinking_level| GeminiThinkingConfig { thinking_level });
        let generation_config = (self.max_tokens.is_some() || thinking_config.is_some()).then_some(
            GeminiGenerationConfig {
                max_output_tokens: self.max_tokens,
                thinking_config,
            },
        );
        GeminiRequest {
            contents: convert_messages(messages),
            tools: convert_tools(tools),
            generation_config,
            system_instruction: gemini_system_instruction(messages),
        }
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        super::send_discarding_chunks(self, messages, tools).await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let request = self.build_request(messages, tools);

        // The key goes in a header, not the query: reqwest puts the full URL
        // in its error messages, which end up in logs.
        let endpoint = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.api_base.trim_end_matches('/'),
            self.model
        );

        let response = super::http::post_json(
            &self.client,
            &endpoint,
            &[("x-goog-api-key", self.api_key.as_str())],
            &request,
        )
        .await?;

        read_stream(response, GeminiStream::default(), &chunk_tx).await
    }
}

/// One streamed reply, assembled chunk by chunk.
#[derive(Default)]
struct GeminiStream {
    text: String,
    tool_uses: Vec<ContentBlock>,
    /// Gemini has no end-of-stream event; the chunk carrying the finish
    /// reason is the last one, so a stream without one was cut off.
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl StreamParser for GeminiStream {
    fn payload(&mut self, data: &str, chunk_tx: &UnboundedSender<LlmChunk>) -> Result<bool> {
        let Some(event) = parse_payload::<GeminiResponse>("gemini", data) else {
            return Ok(false);
        };

        // Each chunk's `usageMetadata` is cumulative, not a delta, so the
        // last one seen before the stream ends is the true total.
        if let Some(u) = event.usage_metadata {
            self.usage = Some(Usage::from(u));
        }

        let Some(candidate) = event.candidates.into_iter().next() else {
            return Ok(false);
        };

        if let Some(fr) = candidate.finish_reason {
            self.finish_reason = Some(map_finish_reason(&fr));
        }

        for part in candidate.content.parts {
            if let Some(text) = part.text
                && !text.is_empty()
            {
                self.text.push_str(&text);
                let _ = chunk_tx.send(LlmChunk::Text(text));
            }
            if let Some(fc) = part.function_call {
                self.tool_uses
                    .push(tool_use_block(fc, part.thought_signature.clone())?);
            }
        }
        Ok(false)
    }

    fn finish(self) -> Result<Choice> {
        if self.finish_reason.is_none() {
            return Err(Error::IncompleteResponse(
                "Gemini stream ended before a finishReason".to_string(),
            ));
        }

        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        content.extend(self.tool_uses);

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content,
            },
            finish_reason: self.finish_reason,
            usage: self.usage,
        })
    }
}

/// Streams and histories the black-box suite's fake provider doesn't
/// produce.
#[cfg(test)]
mod tests {
    use super::*;

    fn id_of(block: &ContentBlock) -> &str {
        match block {
            ContentBlock::ToolUse { id, .. } => id,
            other => panic!("expected a tool use, got {other:?}"),
        }
    }

    // Calls get ids unique across replies; ids repeated per reply would make
    // ACP clients merge different calls into one.
    #[test]
    fn calls_without_an_id_get_unique_ones() {
        let call = || {
            let json = r#"{"name":"read_file","args":{"path":"a.txt"}}"#;
            tool_use_block(serde_json::from_str(json).unwrap(), None).unwrap()
        };
        let (first, second) = (call(), call());
        assert_ne!(id_of(&first), id_of(&second));
        assert!(id_of(&first).starts_with("call_"));
        assert_eq!(
            first,
            ContentBlock::tool_use(id_of(&first), "read_file", r#"{"path":"a.txt"}"#)
        );
    }

    // Gemini 3 rejects a request whose function call lacks its
    // `thoughtSignature`; calls from another provider get the stand-in, on
    // the first call only, and their results are merged into one turn.
    #[test]
    fn calls_from_another_provider_get_the_stand_in_signature() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::tool_use("toolu_1", "read_file", "{}"),
                    ContentBlock::tool_use("toolu_2", "read_file", "{}"),
                ],
            },
            Message::tool_result("toolu_1", "read_file", "a", false),
            Message::tool_result("toolu_2", "read_file", "b", false),
        ];
        let request = serde_json::to_value(convert_messages(&messages)).unwrap();

        let (model, results) = (&request[0], &request[1]);
        assert_eq!(request.as_array().unwrap().len(), 2);
        assert_eq!(
            model["parts"][0]["thoughtSignature"],
            SKIP_SIGNATURE_VALIDATION
        );
        assert!(model["parts"][1].get("thoughtSignature").is_none());
        assert_eq!(model["parts"][0]["functionCall"]["id"], "toolu_1");
        assert_eq!(results["role"], "user");
        assert_eq!(results["parts"][1]["functionResponse"]["id"], "toolu_2");
    }

    #[test]
    fn finish_without_content_still_ends_the_stream() {
        let (chunk_tx, _chunk_rx) = mpsc::unbounded_channel();
        let mut stream = GeminiStream::default();
        let payload = r#"{"candidates":[{"finishReason":"SAFETY"}]}"#;
        stream.payload(payload, &chunk_tx).unwrap();
        let choice = stream.finish().unwrap();
        assert_eq!(choice.finish_reason, Some(FinishReason::Refusal));
        assert!(choice.message.content.is_empty());
    }
}
