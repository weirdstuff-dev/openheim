use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedSender};

use crate::core::models::{Choice, ContentBlock, FinishReason, Message, Role, Tool, Usage};
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
        }
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
    max_output_tokens: u32,
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
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: Value,
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

fn convert_messages(messages: &[Message]) -> Result<Vec<GeminiContent>> {
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
                            let args: Value = serde_json::from_str(arguments).map_err(|e| {
                                Error::ParseError(format!(
                                    "invalid JSON in tool call arguments for '{name}': {e}"
                                ))
                            })?;
                            parts.push(GeminiPart {
                                function_call: Some(GeminiFunctionCall {
                                    id: Some(id.clone()),
                                    name: name.clone(),
                                    args,
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

    Ok(result)
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
            parameters: t.function.parameters.clone(),
        })
        .collect();

    vec![GeminiToolDeclaration {
        function_declarations: declarations,
    }]
}

/// Maps Gemini's `finishReason` vocabulary onto the provider-agnostic
/// [`FinishReason`]; anything without a known equivalent passes through as
/// [`FinishReason::Other`] (lowercased, matching this function's prior
/// string-based behavior).
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
    fn build_request(&self, messages: &[Message], tools: &[Tool]) -> Result<GeminiRequest> {
        Ok(GeminiRequest {
            contents: convert_messages(messages)?,
            tools: convert_tools(tools),
            generation_config: self.max_tokens.map(|t| GeminiGenerationConfig {
                max_output_tokens: t,
            }),
            system_instruction: gemini_system_instruction(messages),
        })
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    /// Implemented in terms of [`Self::send_streaming`] with a discarded
    /// channel — same rationale as `AnthropicClient::send`: one request-
    /// building and response-parsing path instead of two that could drift
    /// (Gemini's streaming and non-streaming responses carry the same
    /// information).
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();
        // Dropped immediately, before any chunk is sent: an unbounded
        // channel with a live receiver buffers every chunk in memory until
        // something calls `recv()`, and nothing here ever will. Dropping it
        // up front makes `chunk_tx.send()` fail fast (already ignored below
        // and in `send_streaming`) instead of accumulating the whole
        // response in the channel for the life of the request.
        drop(chunk_rx);
        self.send_streaming(messages, tools, chunk_tx).await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let request = self.build_request(messages, tools)?;

        // `alt=sse` is in the URL, not `.query()`, to keep the request
        // builder entirely inside `post_json`; the key stays in a header,
        // not a query param, since reqwest embeds the full URL (query
        // included) in transport error strings, which would leak it into
        // logs on any timeout/connect failure.
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_messages_user() {
        let messages = vec![Message::user("hello")];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn convert_messages_user_with_image() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "what is this?".into(),
                },
                ContentBlock::Image {
                    data: "base64data".into(),
                    mime_type: "image/png".into(),
                },
            ],
        }];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].parts.len(), 2);
        let inline = result[0].parts[1].inline_data.as_ref().unwrap();
        assert_eq!(inline.mime_type, "image/png");
        assert_eq!(inline.data, "base64data");
    }

    #[test]
    fn convert_messages_assistant_becomes_model() {
        let messages = vec![Message::assistant("response")];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "model");
        assert_eq!(result[0].parts[0].text.as_deref(), Some("response"));
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(
                "call_1",
                "read_file",
                r#"{"path":"a.txt"}"#,
            )],
        }];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "model");
        assert!(result[0].parts[0].function_call.is_some());
        assert_eq!(
            result[0].parts[0].function_call.as_ref().unwrap().name,
            "read_file"
        );
    }

    #[test]
    fn convert_messages_invalid_tool_arguments_returns_error() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(
                "call_1",
                "read_file",
                "not valid json",
            )],
        }];
        assert!(convert_messages(&messages).is_err());
    }

    #[test]
    fn convert_messages_tool_result_as_function_response() {
        let messages = vec![Message::tool_result(
            "call_1",
            "read_file",
            "file content",
            false,
        )];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        let fr = result[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "read_file");
    }

    #[test]
    fn convert_messages_merges_tool_results_into_user() {
        let messages = vec![
            Message::tool_result("call_1", "read_file", "a", false),
            Message::tool_result("call_2", "write_file", "b", false),
        ];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].parts.len(), 2);
    }

    fn streamed_call(json: &str) -> ContentBlock {
        tool_use_block(serde_json::from_str(json).unwrap(), None).unwrap()
    }

    fn id_of(block: &ContentBlock) -> &str {
        match block {
            ContentBlock::ToolUse { id, .. } => id,
            other => panic!("expected a tool use, got {other:?}"),
        }
    }

    // Regression test: calls were numbered per reply (`call_0`, `call_1`, …),
    // so every reply's first call had the same id and ACP clients merged
    // them into one.
    #[test]
    fn calls_without_an_id_get_unique_ones() {
        let call = r#"{"name":"read_file","args":{"path":"a.txt"}}"#;
        let (first, second) = (streamed_call(call), streamed_call(call));
        assert_ne!(id_of(&first), id_of(&second));
        assert!(id_of(&first).starts_with("call_"));
        assert_eq!(
            first,
            ContentBlock::tool_use(id_of(&first), "read_file", r#"{"path":"a.txt"}"#)
        );
    }

    /// The model turn and tool-result turn `convert_messages` sends for
    /// `calls` and one result per call, as JSON.
    fn sent_turns(calls: Vec<ContentBlock>) -> (Value, Value) {
        let results: Vec<Message> = calls
            .iter()
            .map(|call| match call {
                ContentBlock::ToolUse { id, name, .. } => {
                    Message::tool_result(id.as_str(), name.as_str(), "ok", false)
                }
                other => panic!("expected a tool use, got {other:?}"),
            })
            .collect();
        let mut messages = vec![Message {
            role: Role::Assistant,
            content: calls,
        }];
        messages.extend(results);
        let request = convert_messages(&messages).unwrap();
        (
            serde_json::to_value(&request[0]).unwrap(),
            serde_json::to_value(&request[1]).unwrap(),
        )
    }

    // Regression test: signatures were dropped, so on Gemini 3 every request
    // after a function call failed with "Function call … is missing a
    // thought_signature" (400).
    #[test]
    fn signatures_and_ids_from_gemini_are_sent_back_as_received() {
        // Parallel calls: only the first carries a signature.
        let reply = r#"{"candidates":[{"content":{"role":"model","parts":[
            {"functionCall":{"id":"fc_1","name":"read_file","args":{"path":"a"}},"thoughtSignature":"sig_1"},
            {"functionCall":{"id":"fc_2","name":"read_file","args":{"path":"b"}}}
        ]},"finishReason":"STOP"}]}"#
            .replace('\n', "");
        let calls = parse_stream(&[&reply]).unwrap().message.content;

        let (model, results) = sent_turns(calls);
        assert_eq!(
            model["parts"],
            json!([
                {"functionCall": {"id": "fc_1", "name": "read_file", "args": {"path": "a"}}, "thoughtSignature": "sig_1"},
                {"functionCall": {"id": "fc_2", "name": "read_file", "args": {"path": "b"}}},
            ])
        );
        // Gemini 3.x matches each response to its call by id.
        assert_eq!(results["parts"][0]["functionResponse"]["id"], "fc_1");
        assert_eq!(results["parts"][1]["functionResponse"]["id"], "fc_2");
    }

    // Calls Gemini didn't produce (another provider's, or from before
    // signatures were kept) carry none; the first gets the stand-in so
    // Gemini 3 doesn't reject the request.
    #[test]
    fn unsigned_calls_get_the_stand_in_signature_on_the_first_only() {
        let (model, results) = sent_turns(vec![
            ContentBlock::tool_use("toolu_1", "read_file", "{}"),
            ContentBlock::tool_use("toolu_2", "read_file", "{}"),
        ]);
        assert_eq!(
            model["parts"][0]["thoughtSignature"],
            SKIP_SIGNATURE_VALIDATION
        );
        assert!(model["parts"][1].get("thoughtSignature").is_none());
        assert_eq!(model["parts"][0]["functionCall"]["id"], "toolu_1");
        assert_eq!(results["parts"][0]["functionResponse"]["id"], "toolu_1");
    }

    #[test]
    fn convert_tools_wraps_in_declaration() {
        let tools = vec![Tool::function(
            "test_tool",
            "A test tool",
            json!({"type": "object"}),
        )];
        let result = convert_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function_declarations.len(), 1);
        assert_eq!(result[0].function_declarations[0].name, "test_tool");
    }

    #[test]
    fn convert_tools_empty_returns_empty() {
        let result = convert_tools(&[]);
        assert!(result.is_empty());
    }

    /// Feeds `payloads` through a fresh `GeminiStream` as if they were the
    /// whole response body.
    fn parse_stream(payloads: &[&str]) -> Result<Choice> {
        let (chunk_tx, _chunk_rx) = mpsc::unbounded_channel();
        let mut stream = GeminiStream::default();
        for payload in payloads {
            if stream.payload(payload, &chunk_tx)? {
                break;
            }
        }
        stream.finish()
    }

    const TOOL_REPLY: &[&str] = &[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Reading."}]}}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc_1","name":"read_file","args":{"path":"a.txt"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}"#,
    ];

    #[test]
    fn complete_stream_becomes_a_choice() {
        let choice = parse_stream(TOOL_REPLY).unwrap();
        assert_eq!(
            choice.message.content,
            [
                ContentBlock::from("Reading."),
                ContentBlock::tool_use("fc_1", "read_file", r#"{"path":"a.txt"}"#),
            ]
        );
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
        assert_eq!(choice.usage.unwrap().output_tokens, 5);
    }

    // Regression test: a connection that closed early looked like a normal
    // end of body, so a cut-off reply (here: missing its tool call) was
    // taken as a complete one.
    #[test]
    fn stream_cut_off_before_the_finish_reason_is_an_incomplete_response() {
        let err = parse_stream(&TOOL_REPLY[..1]).unwrap_err();
        assert!(matches!(err, Error::IncompleteResponse(_)), "{err}");
    }

    // A blocked reply's last chunk has a finish reason but no content; it
    // must still parse, or the refusal would look like a cut-off stream.
    #[test]
    fn finish_without_content_still_ends_the_stream() {
        let choice = parse_stream(&[r#"{"candidates":[{"finishReason":"SAFETY"}]}"#]).unwrap();
        assert_eq!(choice.finish_reason, Some(FinishReason::Refusal));
        assert!(choice.message.content.is_empty());
    }

    #[test]
    fn map_finish_reason_translates_known_values() {
        assert_eq!(map_finish_reason("STOP"), FinishReason::Stop);
        assert_eq!(map_finish_reason("MAX_TOKENS"), FinishReason::MaxTokens);
        assert_eq!(map_finish_reason("SAFETY"), FinishReason::Refusal);
        assert_eq!(map_finish_reason("RECITATION"), FinishReason::Refusal);
    }

    #[test]
    fn map_finish_reason_lowercases_unknown_values() {
        assert_eq!(
            map_finish_reason("MALFORMED_FUNCTION_CALL"),
            FinishReason::Other("malformed_function_call".to_string())
        );
    }
}
