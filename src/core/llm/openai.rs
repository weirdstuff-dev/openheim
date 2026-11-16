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
pub struct OpenAiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
    /// Which request field carries `max_tokens`; see [`TokenLimitField`].
    token_limit_field: TokenLimitField,
    /// Whether signed thinking goes back as `thinking_blocks`; see
    /// [`OpenAiClient::with_replay_thinking`].
    replay_thinking: bool,
    /// See [`OpenAiClient::with_effort`].
    effort: Effort,
}

/// OpenAI replaced `max_tokens` with `max_completion_tokens`, and its
/// reasoning models reject the old name. OpenAI-compatible backends mostly
/// still only know `max_tokens`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum TokenLimitField {
    MaxCompletionTokens,
    MaxTokens,
}

impl OpenAiClient {
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
            token_limit_field: TokenLimitField::MaxCompletionTokens,
            replay_thinking: false,
            effort: Effort::Default,
        }
    }

    /// Sends `effort` as `reasoning_effort`. [`Effort::Default`]
    /// (the default) leaves the field out, so the model's own default
    /// applies; models without reasoning reject the field, so they need it.
    pub fn with_effort(self, effort: Effort) -> Self {
        Self { effort, ..self }
    }

    /// With `replay_thinking`, assistant replies of the current tool loop
    /// (those since the last user message) carry their signed and redacted
    /// thinking as `thinking_blocks`, the way a LiteLLM proxy in front of
    /// Anthropic streams it. Anthropic needs it back to keep thinking through
    /// tool calls. Unsigned thinking is never sent. Off by default: strict
    /// OpenAI-compatible backends may reject the field.
    pub fn with_replay_thinking(self, replay_thinking: bool) -> Self {
        Self {
            replay_thinking,
            ..self
        }
    }

    /// This client sending its output limit as `field` instead.
    pub(super) fn with_token_limit_field(self, field: TokenLimitField) -> Self {
        Self {
            token_limit_field: field,
            ..self
        }
    }
}

// --- OpenAI request types ---

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    /// At most one of these two is set; see [`TokenLimitField`].
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiRequestToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// An assistant reply's signed thinking, when it's replayed (see
    /// [`OpenAiClient::with_replay_thinking`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_blocks: Option<Vec<OpenAiThinkingBlock>>,
}

/// A thinking block in LiteLLM's format: streamed in `delta.thinking_blocks`
/// and taken back on an assistant message. While a block streams, its text
/// arrives without a signature. The signature comes in a final entry,
/// usually with the block's whole text, which older LiteLLM versions leave
/// empty. Redacted thinking arrives whole.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiThinkingBlock {
    Thinking {
        #[serde(default, deserialize_with = "null_as_empty")]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    /// A block type this client doesn't know; ignored.
    #[serde(other, skip_serializing)]
    Other,
}

/// A string field that may be `null`, read as empty.
fn null_as_empty<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

/// OpenAI accepts `content` as either a plain string or an array of typed
/// parts. Text-only messages stay a plain string to remain maximally
/// compatible with picky OpenAI-compatible backends; the array form is only
/// used once an image is actually present (see `convert_messages`).
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Debug, Serialize)]
struct OpenAiImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct OpenAiRequestToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAiRequestFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAiRequestFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunctionDef,
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionDef {
    name: String,
    description: String,
    parameters: Value,
}

// --- OpenAI streaming response types ---
//
// One SSE `data:` payload; `choices[0].delta` carries the piece of the
// message that arrived in this chunk.

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

impl From<OpenAiUsage> for Usage {
    fn from(u: OpenAiUsage) -> Self {
        let cached = u
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        Usage {
            // OpenAI's `prompt_tokens` already includes any cached tokens;
            // subtract them so `input_tokens` means "newly processed", same
            // as the Anthropic client.
            input_tokens: u.prompt_tokens.saturating_sub(cached),
            output_tokens: u.completion_tokens,
            cache_creation_tokens: 0,
            cache_read_tokens: cached,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    /// Only on the final chunk when `stream_options.include_usage` is set,
    /// alongside an empty `choices`.
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiStreamChoice {
    #[serde(default)]
    delta: OpenAiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiStreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// Extended-thinking output, as returned by reasoning models behind
    /// OpenAI-compatible APIs (GLM, DeepSeek, …). Absent on non-reasoning
    /// models and plain OpenAI itself.
    #[serde(default)]
    reasoning_content: Option<String>,
    /// LiteLLM's Anthropic thinking, signatures included; see
    /// [`OpenAiThinkingBlock`]. Kept as raw JSON and parsed entry by entry
    /// (see `take_thinking_blocks`), so an odd entry can't fail the chunk's
    /// parse and lose its text or tool-call arguments.
    #[serde(default)]
    thinking_blocks: Option<Value>,
    /// LiteLLM copies `thinking_blocks` in here too. Raw JSON for the same
    /// reason.
    #[serde(default)]
    provider_specific_fields: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiStreamToolCallDelta>>,
}

impl OpenAiStreamDelta {
    /// `thinking_blocks`, or the copy in `provider_specific_fields` when the
    /// delta has no top-level one. An entry that doesn't parse is skipped,
    /// keeping the others.
    fn take_thinking_blocks(&mut self) -> Vec<OpenAiThinkingBlock> {
        let raw = self.thinking_blocks.take().or_else(|| {
            self.provider_specific_fields
                .as_mut()
                .and_then(|fields| fields.get_mut("thinking_blocks"))
                .map(Value::take)
        });
        let Some(Value::Array(entries)) = raw else {
            return Vec::new();
        };
        entries
            .into_iter()
            .filter_map(|entry| serde_json::from_value(entry).ok())
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiStreamFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiStreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// One streamed tool call, assembled from its deltas.
#[derive(Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    args: String,
}

impl ToolCallAcc {
    /// The finished call as a `ToolUse` block, or `None` if no name ever
    /// arrived. Some OpenAI-compatible backends leave out the id; those
    /// calls get a generated one. A call that streamed no arguments gets
    /// `{}`, so it stays valid JSON for the next request.
    fn into_tool_use(self) -> Option<ContentBlock> {
        if self.name.is_empty() {
            return None;
        }
        let id = if self.id.is_empty() {
            super::new_tool_call_id()
        } else {
            self.id
        };
        let args = if self.args.trim().is_empty() {
            "{}".to_string()
        } else {
            self.args
        };
        Some(ContentBlock::tool_use(id, self.name, args))
    }
}

/// Maps OpenAI's `finish_reason` vocabulary onto the provider-agnostic
/// [`FinishReason`]; anything without a known equivalent (e.g. the legacy
/// `function_call`) passes through as [`FinishReason::Other`].
fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::MaxTokens,
        "content_filter" => FinishReason::Refusal,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Whether a failed request should be retried once without `stream_options`:
/// only a 400 whose body names it, so an unrelated 400 isn't paid for twice.
fn should_retry_without_stream_options(status: u16, body: &str) -> bool {
    status == 400 && body.to_lowercase().contains("stream_options")
}

// --- Conversions ---

/// `messages` in OpenAI's format. With `replay_thinking`, the assistant
/// replies after the last user message (the current tool loop) carry their
/// signed thinking; see [`OpenAiClient::with_replay_thinking`]. Older turns
/// never do: Anthropic doesn't need them, and they'd grow every request.
fn convert_messages(messages: &[Message], replay_thinking: bool) -> Vec<OpenAiMessage> {
    let mut result = Vec::new();
    let tool_loop_start = if replay_thinking {
        messages
            .iter()
            .rposition(|m| m.role == Role::User)
            .map_or(0, |i| i + 1)
    } else {
        messages.len()
    };

    for (i, msg) in messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                if let Some(text) = msg.text() {
                    result.push(OpenAiMessage {
                        role: "system".to_string(),
                        content: Some(OpenAiContent::Text(text)),
                        tool_calls: None,
                        tool_call_id: None,
                        thinking_blocks: None,
                    });
                }
            }
            Role::User => {
                let mut parts = Vec::new();
                let mut has_image = false;
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(OpenAiContentPart::Text { text: text.clone() });
                        }
                        ContentBlock::Image { data, mime_type } => {
                            has_image = true;
                            parts.push(OpenAiContentPart::ImageUrl {
                                image_url: OpenAiImageUrl {
                                    url: format!("data:{mime_type};base64,{data}"),
                                },
                            });
                        }
                        _ => {}
                    }
                }
                let content = if has_image {
                    OpenAiContent::Parts(parts)
                } else {
                    let text: String = parts
                        .into_iter()
                        .map(|p| match p {
                            OpenAiContentPart::Text { text } => text,
                            OpenAiContentPart::ImageUrl { .. } => unreachable!(),
                        })
                        .collect();
                    OpenAiContent::Text(text)
                };
                result.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                    thinking_blocks: None,
                });
            }
            Role::Assistant => {
                let text = msg.text();
                let calls = msg.tool_calls();
                // A stored reply with neither (empty, or reasoning only) is
                // left out: OpenAI rejects an assistant message without
                // content or tool calls, which would fail every later request.
                if text.is_none() && calls.is_empty() {
                    continue;
                }
                let tool_calls = if calls.is_empty() {
                    None
                } else {
                    Some(
                        calls
                            .into_iter()
                            .map(|tc| OpenAiRequestToolCall {
                                id: tc.id,
                                call_type: "function".to_string(),
                                function: OpenAiRequestFunctionCall {
                                    name: tc.name,
                                    arguments: tc.arguments,
                                },
                            })
                            .collect(),
                    )
                };
                result.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: text.map(OpenAiContent::Text),
                    tool_calls,
                    tool_call_id: None,
                    thinking_blocks: if i >= tool_loop_start {
                        signed_thinking(msg)
                    } else {
                        None
                    },
                });
            }
            Role::Tool => {
                if let Some(tr) = msg.tool_result_block() {
                    result.push(OpenAiMessage {
                        role: "tool".to_string(),
                        content: Some(OpenAiContent::Text(tr.content)),
                        tool_calls: None,
                        tool_call_id: Some(tr.tool_call_id),
                        thinking_blocks: None,
                    });
                }
            }
        }
    }

    result
}

/// `msg`'s signed and redacted thinking blocks in order, or `None` if it has
/// none. Unsigned thinking is left out: only a signature makes thinking
/// replayable, and some providers (DeepSeek) reject thinking sent back.
fn signed_thinking(msg: &Message) -> Option<Vec<OpenAiThinkingBlock>> {
    let blocks: Vec<_> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking {
                thinking,
                signature: Some(signature),
            } => Some(OpenAiThinkingBlock::Thinking {
                thinking: thinking.clone(),
                signature: Some(signature.clone()),
            }),
            ContentBlock::RedactedThinking { data } => {
                Some(OpenAiThinkingBlock::RedactedThinking { data: data.clone() })
            }
            _ => None,
        })
        .collect();
    (!blocks.is_empty()).then_some(blocks)
}

/// An assistant message's content blocks: the signed and redacted
/// `thinking` blocks, then `reasoning` that no signature closed as an
/// unsigned `Thinking`, then `Text`, then tool uses (the stream doesn't say
/// how they interleave). Empty strings produce no block.
fn assemble_content(
    thinking: Vec<ContentBlock>,
    reasoning: String,
    text: String,
    tool_uses: Vec<ContentBlock>,
) -> Vec<ContentBlock> {
    let mut content = thinking;
    if !reasoning.is_empty() {
        content.push(ContentBlock::Thinking {
            thinking: reasoning,
            signature: None,
        });
    }
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
    content.extend(tool_uses);
    content
}

fn convert_tools(tools: &[Tool]) -> Vec<OpenAiTool> {
    tools
        .iter()
        .map(|t| OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionDef {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters: t.function.parameters.clone(),
            },
        })
        .collect()
}

impl OpenAiClient {
    fn request(&self, messages: &[Message], tools: &[Tool]) -> OpenAiRequest {
        let (max_completion_tokens, max_tokens) = match self.token_limit_field {
            TokenLimitField::MaxCompletionTokens => (self.max_tokens, None),
            TokenLimitField::MaxTokens => (None, self.max_tokens),
        };
        OpenAiRequest {
            model: self.model.clone(),
            messages: convert_messages(messages, self.replay_thinking),
            tools: convert_tools(tools),
            max_completion_tokens,
            max_tokens,
            reasoning_effort: self.effort.api_value(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.api_base.trim_end_matches('/'))
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        super::send_discarding_chunks(self, messages, tools).await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let request = self.request(messages, tools);

        let mut body =
            serde_json::to_value(&request).map_err(|e| Error::ParseError(e.to_string()))?;
        body["stream"] = serde_json::Value::Bool(true);
        // Asks for a final chunk carrying `usage`. A few strict compatible
        // backends reject the field with a 400; that case is retried without
        // it below.
        body["stream_options"] = serde_json::json!({ "include_usage": true });

        let endpoint = self.endpoint();
        let auth = self.auth_header();
        let headers = [("Authorization", auth.as_str())];

        let response = match super::http::post_json(&self.client, &endpoint, &headers, &body).await
        {
            Ok(response) => response,
            Err(Error::HttpError {
                status,
                body: err_body,
                ..
            }) if should_retry_without_stream_options(status, &err_body) => {
                let mut retry_body = body.clone();
                if let Some(obj) = retry_body.as_object_mut() {
                    obj.remove("stream_options");
                }
                super::http::post_json(&self.client, &endpoint, &headers, &retry_body).await?
            }
            Err(e) => return Err(e),
        };

        read_stream(response, OpenAiStream::default(), &chunk_tx).await
    }
}

impl OpenAiStream {
    /// Records one entry of a delta's `thinking_blocks`. Text without a
    /// signature already arrived as `reasoning_content`, so it's skipped. A
    /// signature closes a block: its text is the entry's, or, from LiteLLM
    /// versions that leave it empty, the reasoning streamed since the last
    /// one.
    fn thinking_block(&mut self, block: OpenAiThinkingBlock) {
        match block {
            OpenAiThinkingBlock::Thinking {
                thinking,
                signature: Some(signature),
            } if !signature.is_empty() => {
                let streamed = std::mem::take(&mut self.reasoning);
                self.thinking.push(ContentBlock::Thinking {
                    thinking: if thinking.is_empty() {
                        streamed
                    } else {
                        thinking
                    },
                    signature: Some(signature),
                });
            }
            OpenAiThinkingBlock::RedactedThinking { data } if !data.is_empty() => {
                self.thinking.push(ContentBlock::RedactedThinking { data });
            }
            _ => {}
        }
    }
}

/// One streamed reply, assembled chunk by chunk.
#[derive(Default)]
struct OpenAiStream {
    text: String,
    /// Thinking blocks a signature closed, and redacted ones, in the order
    /// they arrived (only from `thinking_blocks`).
    thinking: Vec<ContentBlock>,
    /// Reasoning text since the last signed block, kept so the final message
    /// persists it (the live chunks are UI-only). Without `thinking_blocks`,
    /// that's all of it.
    reasoning: String,
    tool_calls: Vec<ToolCallAcc>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    /// Set by `[DONE]`.
    done: bool,
}

impl StreamParser for OpenAiStream {
    fn payload(&mut self, data: &str, chunk_tx: &UnboundedSender<LlmChunk>) -> Result<bool> {
        if data == "[DONE]" {
            self.done = true;
            return Ok(true);
        }
        let Some(event) = parse_payload::<OpenAiStreamChunk>("openai", data) else {
            return Ok(false);
        };

        if let Some(u) = event.usage {
            self.usage = Some(Usage::from(u));
        }

        let Some(mut choice) = event.choices.into_iter().next() else {
            return Ok(false);
        };

        if let Some(fr) = choice.finish_reason {
            self.finish_reason = Some(map_finish_reason(&fr));
        }

        if let Some(reasoning) = choice.delta.reasoning_content.take()
            && !reasoning.is_empty()
        {
            self.reasoning.push_str(&reasoning);
            let _ = chunk_tx.send(LlmChunk::Thinking(reasoning));
        }

        for block in choice.delta.take_thinking_blocks() {
            self.thinking_block(block);
        }

        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            self.text.push_str(&content);
            let _ = chunk_tx.send(LlmChunk::Text(content));
        }

        for tc in choice.delta.tool_calls.unwrap_or_default() {
            let idx = tc.index;
            if self.tool_calls.len() <= idx {
                self.tool_calls.resize_with(idx + 1, ToolCallAcc::default);
            }
            let acc = &mut self.tool_calls[idx];
            if let Some(id) = tc.id {
                acc.id = id;
            }
            if let Some(function) = tc.function {
                if let Some(name) = function.name {
                    acc.name.push_str(&name);
                }
                if let Some(args) = function.arguments {
                    acc.args.push_str(&args);
                }
            }
        }
        Ok(false)
    }

    /// Complete once `[DONE]` or a `finish_reason` arrived: some
    /// OpenAI-compatible backends end the stream without `[DONE]`, but every
    /// one sends the finish reason on the last choice chunk.
    fn finish(self) -> Result<Choice> {
        if !self.done && self.finish_reason.is_none() {
            return Err(Error::IncompleteResponse(
                "OpenAI stream ended before a finish_reason or [DONE]".to_string(),
            ));
        }

        let tool_uses = self
            .tool_calls
            .into_iter()
            .filter_map(ToolCallAcc::into_tool_use)
            .collect();
        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content: assemble_content(self.thinking, self.reasoning, self.text, tool_uses),
            },
            finish_reason: self.finish_reason,
            usage: self.usage,
        })
    }
}

/// Streams as LiteLLM and other OpenAI-compatible proxies send them. Plain
/// OpenAI streams are covered by the black-box suite.
#[cfg(test)]
mod tests {
    use super::*;

    fn acc(id: &str, name: &str) -> ToolCallAcc {
        ToolCallAcc {
            id: id.into(),
            name: name.into(),
            args: "{}".into(),
        }
    }

    // A call streamed without an id gets a unique one, not one numbered by
    // its position in the reply; one without a name is dropped.
    #[test]
    fn streamed_calls_without_an_id_or_name() {
        let id = |block: Option<ContentBlock>| match block {
            Some(ContentBlock::ToolUse { id, .. }) => id,
            other => panic!("expected a tool use, got {other:?}"),
        };
        let first = id(acc("", "read_file").into_tool_use());
        let second = id(acc("", "read_file").into_tool_use());
        assert_ne!(first, second);
        assert!(first.starts_with("call_"));
        assert_eq!(id(acc("call_x", "read_file").into_tool_use()), "call_x");
        assert!(acc("call_x", "").into_tool_use().is_none());
    }

    /// Feeds `payloads` through a fresh `OpenAiStream` as if they were the
    /// whole response body.
    fn parse_stream(payloads: &[&str]) -> Choice {
        let (chunk_tx, _chunk_rx) = mpsc::unbounded_channel();
        let mut stream = OpenAiStream::default();
        for payload in payloads {
            if stream.payload(payload, &chunk_tx).unwrap() {
                break;
            }
        }
        stream.finish().unwrap()
    }

    fn signed(thinking: &str, signature: &str) -> ContentBlock {
        ContentBlock::Thinking {
            thinking: thinking.into(),
            signature: Some(signature.into()),
        }
    }

    #[test]
    fn a_signature_without_text_signs_the_streamed_reasoning() {
        let choice = parse_stream(&[
            r#"{"choices":[{"delta":{"reasoning_content":"Let me","thinking_blocks":[{"type":"thinking","thinking":"Let me","signature":null}]}}]}"#,
            r#"{"choices":[{"delta":{"thinking_blocks":[{"type":"thinking","thinking":"","signature":"EqQBCkgI"}]}}]}"#,
            r#"{"choices":[{"delta":{"content":"Done."},"finish_reason":"stop"}]}"#,
        ]);
        assert_eq!(
            choice.message.content,
            [signed("Let me", "EqQBCkgI"), ContentBlock::from("Done.")]
        );
    }

    #[test]
    fn redacted_and_several_signed_blocks_keep_their_order() {
        let choice = parse_stream(&[
            r#"{"choices":[{"delta":{"thinking_blocks":[{"type":"redacted_thinking","data":"EmwKAhgB"}]}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"one","thinking_blocks":[{"type":"thinking","thinking":"one"}]}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"","thinking_blocks":[{"type":"thinking","thinking":"one","signature":"sig-1"}]}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"two","thinking_blocks":[{"type":"thinking","thinking":"two"}]}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"","thinking_blocks":[{"type":"thinking","thinking":"two","signature":"sig-2"}]}}]}"#,
            r#"{"choices":[{"delta":{"content":"Done."},"finish_reason":"stop"}]}"#,
        ]);
        assert_eq!(
            choice.message.content,
            [
                ContentBlock::RedactedThinking {
                    data: "EmwKAhgB".into(),
                },
                signed("one", "sig-1"),
                signed("two", "sig-2"),
                ContentBlock::from("Done."),
            ]
        );
    }

    #[test]
    fn thinking_blocks_only_in_provider_specific_fields_are_read() {
        let choice = parse_stream(&[
            r#"{"choices":[{"delta":{"reasoning_content":"hm","provider_specific_fields":{"citation":null,"thinking_blocks":[{"type":"thinking","thinking":"hm","signature":"sig"}]}}}]}"#,
            r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        ]);
        assert_eq!(
            choice.message.content,
            [signed("hm", "sig"), ContentBlock::from("ok")]
        );
    }

    #[test]
    fn odd_or_malformed_thinking_entries_dont_drop_the_chunk() {
        let choice = parse_stream(&[
            r#"{"choices":[{"delta":{"content":"hi","thinking_blocks":[{"type":"summary","text":"x"}]}}]}"#,
            r#"{"choices":[{"delta":{"content":"!","provider_specific_fields":"odd"},"finish_reason":"stop"}]}"#,
        ]);
        assert_eq!(choice.message.content, [ContentBlock::from("hi!")]);

        let choice = parse_stream(&[
            r#"{"choices":[{"delta":{"reasoning_content":"plan"}}]}"#,
            r#"{"choices":[{"delta":{"thinking_blocks":[{"type":"thinking","thinking":null,"signature":"sig"},{"type":"thinking","thinking":"x","signature":7},{"type":"redacted_thinking","data":["bad"]},{"type":"redacted_thinking","data":"enc"}],"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ]);
        assert_eq!(
            choice.message.content,
            [
                signed("plan", "sig"),
                ContentBlock::RedactedThinking { data: "enc".into() },
                ContentBlock::tool_use("call_1", "read_file", "{}"),
            ]
        );
    }
}
