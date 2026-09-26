use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedSender};

use crate::core::models::{Choice, ContentBlock, FinishReason, Message, Role, Tool, Usage};
use crate::error::{Error, Result};

use super::sse::{StreamParser, parse_payload, read_stream};
use super::{LlmChunk, LlmClient};

const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Used when the provider sets no `max_tokens`. Adaptive thinking is on by
/// default and spends from this same budget, so a low cap truncates replies
/// (or leaves no room for any text at all). ~16k is Anthropic's recommended
/// default for non-streaming requests; the agent loop always streams, but
/// `send` is still public, so the default stays safe for callers of it.
const DEFAULT_MAX_TOKENS: u32 = 16_000;

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Clone)]
pub struct AnthropicClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: u32,
    /// Whether to request extended thinking. Resolved by the caller from
    /// `[providers.<name>].thinking` (see
    /// [`crate::config::ProviderConfig::resolve_thinking`]); the client
    /// makes no model-name-based guess of its own.
    thinking: bool,
}

impl AnthropicClient {
    pub fn new(
        client: ReqwestClient,
        api_base: String,
        api_key: String,
        model: String,
        max_tokens: Option<u32>,
        thinking: bool,
    ) -> Self {
        Self {
            client,
            api_base,
            api_key,
            model,
            max_tokens: max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            thinking,
        }
    }
}

// --- Anthropic request types ---

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct AnthropicThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: &'static str,
    /// Request thinking text back instead of the empty-string default, so it can
    /// be streamed to the user and replayed on the next turn.
    display: &'static str,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

// --- Anthropic streaming event types ---
//
// Unlike OpenAI/Gemini's single repeated envelope shape, each Anthropic SSE
// event's fields depend on its `type` — an externally-tagged enum on `type`
// models that directly. `#[serde(other)]` on `Other` absorbs any event type
// this client doesn't special-case (`ping`, and any future addition) instead
// of failing the whole stream.

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicStreamMessage },
    /// `index` is the block's position in the reply; its deltas carry the
    /// same index.
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: AnthropicStreamContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: AnthropicStreamDelta,
    },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDeltaFields,
        #[serde(default)]
        usage: Option<AnthropicDeltaUsage>,
    },
    /// The last event of a complete reply.
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: AnthropicStreamError },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    #[serde(default)]
    usage: AnthropicStartUsage,
}

#[derive(Debug, Deserialize, Default)]
struct AnthropicStartUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

/// A block as `content_block_start` opens it. Text, thinking and tool input
/// arrive afterwards as deltas; a redacted block arrives whole.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamContentBlock {
    // Struct variants so the start event's own (empty) `text`/`thinking`
    // fields are ignored rather than rejected.
    #[serde(rename = "text")]
    Text {},
    #[serde(rename = "thinking")]
    Thinking {},
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    /// Server-side tool blocks and anything newer: not kept.
    #[serde(other)]
    Other,
}

/// One content block of a streamed reply, filled in by its deltas.
enum StreamBlock {
    Text(String),
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking(String),
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    Other,
}

impl StreamBlock {
    fn opened(block: AnthropicStreamContentBlock) -> Self {
        match block {
            AnthropicStreamContentBlock::Text {} => StreamBlock::Text(String::new()),
            AnthropicStreamContentBlock::Thinking {} => StreamBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            AnthropicStreamContentBlock::RedactedThinking { data } => {
                StreamBlock::RedactedThinking(data)
            }
            AnthropicStreamContentBlock::ToolUse { id, name } => StreamBlock::ToolUse {
                id,
                name,
                json: String::new(),
            },
            AnthropicStreamContentBlock::Other => StreamBlock::Other,
        }
    }

    /// Applies one delta, forwarding text and thinking to `chunk_tx`. A delta
    /// that doesn't fit the block's type is ignored.
    fn apply(&mut self, delta: AnthropicStreamDelta, chunk_tx: &UnboundedSender<LlmChunk>) {
        match (self, delta) {
            (StreamBlock::Text(text), AnthropicStreamDelta::Text { text: more }) => {
                text.push_str(&more);
                let _ = chunk_tx.send(LlmChunk::Text(more));
            }
            (
                StreamBlock::Thinking { thinking, .. },
                AnthropicStreamDelta::Thinking { thinking: more },
            ) => {
                thinking.push_str(&more);
                let _ = chunk_tx.send(LlmChunk::Thinking(more));
            }
            (
                StreamBlock::Thinking { signature, .. },
                AnthropicStreamDelta::Signature { signature: more },
            ) => {
                signature.push_str(&more);
            }
            (
                StreamBlock::ToolUse { json, .. },
                AnthropicStreamDelta::InputJson { partial_json },
            ) => {
                json.push_str(&partial_json);
            }
            _ => {}
        }
    }

    /// The finished block, or `None` for one with nothing worth keeping.
    fn into_content(self) -> Option<ContentBlock> {
        match self {
            StreamBlock::Text(text) => (!text.is_empty()).then_some(ContentBlock::Text { text }),
            // Kept even without visible text: the signature alone still has
            // to be replayed.
            StreamBlock::Thinking {
                thinking,
                signature,
            } => (!thinking.is_empty() || !signature.is_empty()).then(|| ContentBlock::Thinking {
                thinking,
                signature: (!signature.is_empty()).then_some(signature),
            }),
            StreamBlock::RedactedThinking(data) => Some(ContentBlock::RedactedThinking { data }),
            // A call without arguments can stream no input at all; `{}` keeps
            // it valid JSON for the next request.
            StreamBlock::ToolUse { id, name, json } => {
                let arguments = if json.trim().is_empty() {
                    "{}".to_string()
                } else {
                    json
                };
                Some(ContentBlock::tool_use(id, name, arguments))
            }
            StreamBlock::Other => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDeltaFields {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDeltaUsage {
    /// Cumulative, not a per-event delta — see the `message_delta` handling
    /// in `send_streaming` for why the last value seen wins.
    #[serde(default)]
    output_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamError {
    #[serde(default)]
    message: Option<String>,
}

/// One streamed reply, assembled event by event.
#[derive(Default)]
struct AnthropicStream {
    /// The reply's content blocks by index, so they keep the order the model
    /// produced them in: interleaved thinking can put several thinking
    /// blocks between text and tool calls, each with its own signature.
    blocks: std::collections::BTreeMap<usize, StreamBlock>,
    stop_reason: Option<String>,
    usage: Usage,
    /// Set by `message_stop`; a stream that ends without it was cut off.
    complete: bool,
}

impl StreamParser for AnthropicStream {
    fn payload(&mut self, data: &str, chunk_tx: &UnboundedSender<LlmChunk>) -> Result<bool> {
        if data == "[DONE]" || data.is_empty() {
            return Ok(false);
        }
        let Some(event) = parse_payload::<AnthropicStreamEvent>("anthropic", data) else {
            return Ok(false);
        };

        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                let u = message.usage;
                self.usage.input_tokens = u.input_tokens;
                self.usage.output_tokens = u.output_tokens;
                self.usage.cache_creation_tokens = u.cache_creation_input_tokens;
                self.usage.cache_read_tokens = u.cache_read_input_tokens;
            }
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                self.blocks
                    .insert(index, StreamBlock::opened(content_block));
            }
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = self.blocks.get_mut(&index) {
                    block.apply(delta, chunk_tx);
                }
            }
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    self.stop_reason = Some(reason);
                }
                // Anthropic reports `output_tokens` cumulatively on each
                // `message_delta`, not as a per-event delta — the last value
                // seen before the stream ends is the true total.
                if let Some(ot) = usage.and_then(|u| u.output_tokens) {
                    self.usage.output_tokens = ot;
                }
            }
            AnthropicStreamEvent::MessageStop => {
                self.complete = true;
                return Ok(true);
            }
            AnthropicStreamEvent::Error { error } => {
                let msg = error
                    .message
                    .unwrap_or_else(|| "unknown streaming error".to_string());
                return Err(Error::ApiError(format!("Anthropic streaming error: {msg}")));
            }
            AnthropicStreamEvent::Other => {}
        }
        Ok(false)
    }

    fn finish(self) -> Result<Choice> {
        if !self.complete {
            return Err(Error::IncompleteResponse(
                "Anthropic stream ended before message_stop".to_string(),
            ));
        }

        let content = self
            .blocks
            .into_values()
            .filter_map(StreamBlock::into_content)
            .collect();

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content,
            },
            finish_reason: map_stop_reason(self.stop_reason.as_deref()),
            usage: Some(self.usage),
        })
    }
}

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::Tool => {
                // Tool results must be sent as user messages with tool_result content blocks
                let Some(tr) = msg.tool_result_block() else {
                    continue;
                };
                let block = AnthropicContentBlock::ToolResult {
                    tool_use_id: tr.tool_call_id,
                    content: tr.content,
                    is_error: tr.is_error,
                };
                // Merge into the last user message if it exists, otherwise create new
                if let Some(last) = result.last_mut() {
                    let last: &mut AnthropicMessage = last;
                    if last.role == "user" {
                        last.content.push(block);
                        continue;
                    }
                }
                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![block],
                });
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                // Thinking blocks (redacted ones too) are replayed verbatim and
                // in their original position, or the API rejects the next
                // request. Unsigned thinking came from another provider and
                // can't be replayed to Anthropic, so it's dropped.
                for block in &msg.content {
                    match block {
                        ContentBlock::Thinking {
                            thinking,
                            signature: Some(signature),
                        } => {
                            blocks.push(AnthropicContentBlock::Thinking {
                                thinking: thinking.clone(),
                                signature: signature.clone(),
                            });
                        }
                        ContentBlock::RedactedThinking { data } => {
                            blocks.push(AnthropicContentBlock::RedactedThinking {
                                data: data.clone(),
                            });
                        }
                        ContentBlock::Text { text } if !text.is_empty() => {
                            blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                        }
                        ContentBlock::ToolUse {
                            id,
                            name,
                            arguments,
                            ..
                        } => {
                            blocks.push(AnthropicContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: super::tool_input(name, arguments),
                            });
                        }
                        _ => {}
                    }
                }
                if !blocks.is_empty() {
                    result.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: blocks,
                    });
                }
            }
            Role::User => {
                let mut blocks = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                        }
                        ContentBlock::Image { data, mime_type } => {
                            blocks.push(AnthropicContentBlock::Image {
                                source: AnthropicImageSource {
                                    source_type: "base64",
                                    media_type: mime_type.clone(),
                                    data: data.clone(),
                                },
                            });
                        }
                        _ => {}
                    }
                }
                if blocks.is_empty() {
                    blocks.push(AnthropicContentBlock::Text {
                        text: String::new(),
                    });
                }
                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: blocks,
                });
            }
            Role::System => {
                // extracted into the top-level system field of AnthropicRequest
            }
        }
    }

    result
}

fn convert_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t.function.parameters.clone(),
        })
        .collect()
}

/// Maps Anthropic's `stop_reason` vocabulary onto the provider-agnostic
/// [`FinishReason`]; anything without a known equivalent passes through as
/// [`FinishReason::Other`].
fn map_stop_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("tool_use") => Some(FinishReason::ToolCalls),
        Some("end_turn") => Some(FinishReason::Stop),
        Some("max_tokens" | "model_context_window_exceeded") => Some(FinishReason::MaxTokens),
        Some("refusal") => Some(FinishReason::Refusal),
        Some("pause_turn") => Some(FinishReason::Paused),
        other => other.map(|s| FinishReason::Other(s.to_string())),
    }
}

fn extract_system(messages: &[Message]) -> Option<String> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .filter_map(|m| m.text())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// Returns Anthropic's adaptive-thinking request config when `enabled`.
///
/// Adaptive thinking (`type: "adaptive"`) is the form current Claude models
/// accept; the fixed-budget form (`type: "enabled", budget_tokens: N`)
/// returns a 400 on Opus 4.7/4.8, Sonnet 5, and Fable 5, and is deprecated
/// on Opus 4.6 / Sonnet 4.6. Models predating adaptive thinking
/// (Sonnet 3.7 and earlier Claude 4 releases) only support the fixed-budget
/// form and reject adaptive thinking outright — set `thinking = "off"` on
/// the provider entry for those (see [`crate::config::ProviderConfig::resolve_thinking`]).
/// Adaptive thinking also enables interleaved thinking automatically, so no
/// `anthropic-beta` header is needed here.
fn thinking_config(enabled: bool) -> Option<AnthropicThinkingConfig> {
    enabled.then_some(AnthropicThinkingConfig {
        thinking_type: "adaptive",
        display: "summarized",
    })
}

impl AnthropicClient {
    fn build_request(&self, messages: &[Message], tools: &[Tool]) -> AnthropicRequest {
        AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            stream: true,
            system: extract_system(messages),
            messages: convert_messages(messages),
            tools: convert_tools(tools),
            thinking: thinking_config(self.thinking),
        }
    }

    /// POSTs `request` and returns the response body stream, having already
    /// checked the status and turned a non-2xx response into an `Err` with
    /// its body attached.
    async fn post(&self, request: &AnthropicRequest) -> Result<reqwest::Response> {
        let endpoint = format!("{}/messages", self.api_base.trim_end_matches('/'));

        super::http::post_json(
            &self.client,
            &endpoint,
            &[
                ("x-api-key", self.api_key.as_str()),
                ("anthropic-version", ANTHROPIC_VERSION),
            ],
            request,
        )
        .await
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
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
        let response = self.post(&request).await?;

        read_stream(response, AnthropicStream::default(), &chunk_tx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_messages_user_message() {
        let messages = vec![Message::user("hello")];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert!(
            matches!(&result[0].content[0], AnthropicContentBlock::Text { text } if text == "hello")
        );
    }

    #[test]
    fn convert_messages_user_message_with_image() {
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
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content.len(), 2);
        assert!(matches!(
            &result[0].content[1],
            AnthropicContentBlock::Image { source }
                if source.media_type == "image/png" && source.data == "base64data"
        ));
    }

    #[test]
    fn convert_messages_system_is_excluded() {
        let messages = vec![Message {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "system prompt".into(),
            }],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "thinking".into(),
                },
                ContentBlock::tool_use("call_1", "read_file", r#"{"path":"a.txt"}"#),
            ],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        assert_eq!(result[0].content.len(), 2); // text + tool_use
    }

    #[test]
    fn convert_messages_replays_thinking_block_first() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "reasoning about the file".into(),
                    signature: Some("sig123".into()),
                },
                ContentBlock::Text {
                    text: "here's my answer".into(),
                },
                ContentBlock::tool_use("call_1", "read_file", r#"{"path":"a.txt"}"#),
            ],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content.len(), 3); // thinking + text + tool_use
        assert!(matches!(
            &result[0].content[0],
            AnthropicContentBlock::Thinking { thinking, signature }
                if thinking == "reasoning about the file" && signature == "sig123"
        ));
    }

    // A stored call with malformed arguments (e.g. cut off at the output
    // limit) must not fail every later request in the conversation.
    #[test]
    fn convert_messages_sends_malformed_tool_arguments_as_an_empty_object() {
        for arguments in [r#"{"path":"a.t"#, "[1]", ""] {
            let messages = vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::tool_use("call_1", "read_file", arguments)],
            }];
            let result = convert_messages(&messages);
            assert!(
                matches!(
                    &result[0].content[0],
                    AnthropicContentBlock::ToolUse { input, .. } if *input == json!({})
                ),
                "{arguments:?}"
            );
        }
    }

    #[test]
    fn convert_messages_merges_consecutive_tool_results() {
        let messages = vec![
            Message::tool_result("call_1", "read_file", "content1", false),
            Message::tool_result("call_2", "write_file", "content2", false),
        ];
        let result = convert_messages(&messages);
        // Both tool results should merge into a single user message
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content.len(), 2);
    }

    #[test]
    fn convert_tools_maps_definitions() {
        let tools = vec![Tool::function(
            "read_file",
            "Read a file",
            json!({"type": "object"}),
        )];
        let result = convert_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "read_file");
        assert_eq!(result[0].description, "Read a file");
    }

    #[test]
    fn convert_tools_empty_input() {
        let result = convert_tools(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn map_stop_reason_translates_known_values() {
        assert_eq!(map_stop_reason(Some("end_turn")), Some(FinishReason::Stop));
        assert_eq!(
            map_stop_reason(Some("tool_use")),
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(
            map_stop_reason(Some("max_tokens")),
            Some(FinishReason::MaxTokens)
        );
        assert_eq!(
            map_stop_reason(Some("model_context_window_exceeded")),
            Some(FinishReason::MaxTokens)
        );
        assert_eq!(
            map_stop_reason(Some("refusal")),
            Some(FinishReason::Refusal)
        );
        assert_eq!(
            map_stop_reason(Some("pause_turn")),
            Some(FinishReason::Paused)
        );
    }

    #[test]
    fn map_stop_reason_passes_through_unknown_values() {
        assert_eq!(
            map_stop_reason(Some("stop_sequence")),
            Some(FinishReason::Other("stop_sequence".to_string()))
        );
    }

    #[test]
    fn map_stop_reason_none_stays_none() {
        assert!(map_stop_reason(None).is_none());
    }

    #[test]
    fn thinking_config_enabled_when_requested() {
        let config = thinking_config(true);
        assert!(config.is_some());
        assert_eq!(config.unwrap().thinking_type, "adaptive");
    }

    #[test]
    fn thinking_config_disabled_when_not_requested() {
        assert!(thinking_config(false).is_none());
    }

    fn client_with_thinking(thinking: bool) -> AnthropicClient {
        AnthropicClient::new(
            ReqwestClient::new(),
            "https://api.anthropic.com/v1".into(),
            "test-key".into(),
            "claude-sonnet-5".into(),
            None,
            thinking,
        )
    }

    #[test]
    fn build_request_always_streams_and_respects_thinking() {
        // `build_request` is the single request source for both `send` and
        // `send_streaming`; both must stream and honor the thinking config.
        let request = client_with_thinking(true).build_request(&[Message::user("hi")], &[]);
        assert!(request.stream);
        assert!(request.thinking.is_some());

        let request = client_with_thinking(false).build_request(&[Message::user("hi")], &[]);
        assert!(request.thinking.is_none());
    }

    #[test]
    fn stream_event_deserializes_message_start_usage() {
        let event: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":12,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":1}}}"#,
        )
        .unwrap();
        let AnthropicStreamEvent::MessageStart { message } = event else {
            panic!("expected MessageStart");
        };
        assert_eq!(message.usage.input_tokens, 12);
        assert_eq!(message.usage.cache_creation_input_tokens, 3);
        assert_eq!(message.usage.cache_read_input_tokens, 1);
    }

    #[test]
    fn stream_event_deserializes_content_block_start_tool_use() {
        let event: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_1","name":"read_file","input":{}}}"#,
        )
        .unwrap();
        let AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } = event
        else {
            panic!("expected ContentBlockStart");
        };
        assert_eq!(index, 2);
        let AnthropicStreamContentBlock::ToolUse { id, name } = content_block else {
            panic!("expected ToolUse content block");
        };
        assert_eq!(id, "call_1");
        assert_eq!(name, "read_file");
    }

    #[test]
    fn stream_event_deserializes_every_content_block_start() {
        let block = |json: &str| {
            let event: AnthropicStreamEvent = serde_json::from_str(&format!(
                r#"{{"type":"content_block_start","index":0,"content_block":{json}}}"#
            ))
            .unwrap();
            let AnthropicStreamEvent::ContentBlockStart { content_block, .. } = event else {
                panic!("expected ContentBlockStart for {json}");
            };
            content_block
        };
        assert!(matches!(
            block(r#"{"type":"text","text":""}"#),
            AnthropicStreamContentBlock::Text {}
        ));
        assert!(matches!(
            block(r#"{"type":"thinking","thinking":"","signature":""}"#),
            AnthropicStreamContentBlock::Thinking {}
        ));
        assert!(matches!(
            block(r#"{"type":"redacted_thinking","data":"enc"}"#),
            AnthropicStreamContentBlock::RedactedThinking { data } if data == "enc"
        ));
        assert!(matches!(
            block(r#"{"type":"server_tool_use","id":"s1","name":"web_search","input":{}}"#),
            AnthropicStreamContentBlock::Other
        ));
    }

    #[test]
    fn stream_event_deserializes_every_delta_variant() {
        let cases = [
            (r#"{"type":"text_delta","text":"hi"}"#, "text"),
            (
                r#"{"type":"thinking_delta","thinking":"pondering"}"#,
                "thinking",
            ),
            (
                r#"{"type":"signature_delta","signature":"sig"}"#,
                "signature",
            ),
            (
                r#"{"type":"input_json_delta","partial_json":"{\"a\""}"#,
                "input_json",
            ),
        ];
        for (json, label) in cases {
            let event: AnthropicStreamEvent = serde_json::from_str(&format!(
                r#"{{"type":"content_block_delta","index":0,"delta":{json}}}"#
            ))
            .unwrap();
            let AnthropicStreamEvent::ContentBlockDelta { delta, .. } = event else {
                panic!("expected ContentBlockDelta for {label}");
            };
            assert!(
                !matches!(delta, AnthropicStreamDelta::Other),
                "expected a typed delta for {label}, got Other"
            );
        }
    }

    #[test]
    fn stream_event_deserializes_message_delta_with_cumulative_usage() {
        let event: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}"#,
        )
        .unwrap();
        let AnthropicStreamEvent::MessageDelta { delta, usage } = event else {
            panic!("expected MessageDelta");
        };
        assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(usage.unwrap().output_tokens, Some(42));
    }

    #[test]
    fn stream_event_deserializes_error() {
        let event: AnthropicStreamEvent =
            serde_json::from_str(r#"{"type":"error","error":{"message":"overloaded"}}"#).unwrap();
        let AnthropicStreamEvent::Error { error } = event else {
            panic!("expected Error");
        };
        assert_eq!(error.message.as_deref(), Some("overloaded"));
    }

    #[test]
    fn stream_event_unknown_type_falls_back_to_other() {
        // `ping` (and anything future) must not fail deserialization, or one
        // unrecognized event would fail the whole stream.
        for json in [r#"{"type":"ping"}"#, r#"{"type":"future_event"}"#] {
            let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
            assert!(matches!(event, AnthropicStreamEvent::Other));
        }
    }

    /// Feeds `payloads` through a fresh `AnthropicStream` as if they were
    /// the whole response body.
    fn parse_stream(payloads: &[&str]) -> Result<Choice> {
        let (chunk_tx, _chunk_rx) = mpsc::unbounded_channel();
        let mut stream = AnthropicStream::default();
        for payload in payloads {
            if stream.payload(payload, &chunk_tx)? {
                break;
            }
        }
        stream.finish()
    }

    const TOOL_REPLY: &[&str] = &[
        r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Reading."}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file"}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a.txt\"}"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}"#,
        r#"{"type":"message_stop"}"#,
    ];

    #[test]
    fn complete_stream_becomes_a_choice() {
        let choice = parse_stream(TOOL_REPLY).unwrap();
        assert_eq!(
            choice.message.content,
            [
                ContentBlock::from("Reading."),
                ContentBlock::tool_use("toolu_1", "read_file", r#"{"path":"a.txt"}"#),
            ]
        );
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(choice.usage.unwrap().output_tokens, 20);
    }

    // A connection that closes early is an incomplete reply (here: missing
    // its tool call), not a complete one that ends the turn.
    #[test]
    fn stream_cut_off_before_message_stop_is_an_incomplete_response() {
        for cut in [3, TOOL_REPLY.len() - 1] {
            let err = parse_stream(&TOOL_REPLY[..cut]).unwrap_err();
            assert!(matches!(err, Error::IncompleteResponse(_)), "{err}");
            assert!(err.is_retryable());
        }
    }

    // Each thinking block keeps its own signature, redacted thinking is
    // kept, and blocks stay in stored order: Anthropic rejects a turn sent
    // back any other way.
    #[test]
    fn interleaved_blocks_keep_their_order_and_own_signatures() {
        let choice = parse_stream(&[
            r#"{"type":"message_start","message":{"usage":{}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_a"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me look."}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"redacted_thinking","data":"enc"}}"#,
            r#"{"type":"content_block_stop","index":2}"#,
            r#"{"type":"content_block_start","index":3,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"{"type":"content_block_delta","index":3,"delta":{"type":"thinking_delta","thinking":"second"}}"#,
            r#"{"type":"content_block_delta","index":3,"delta":{"type":"signature_delta","signature":"sig_b"}}"#,
            r#"{"type":"content_block_stop","index":3}"#,
            r#"{"type":"content_block_start","index":4,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#,
            r#"{"type":"content_block_delta","index":4,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#,
            r#"{"type":"content_block_stop","index":4}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"{"type":"message_stop"}"#,
        ])
        .unwrap();

        let thinking = |text: &str, sig: &str| ContentBlock::Thinking {
            thinking: text.into(),
            signature: Some(sig.into()),
        };
        assert_eq!(
            choice.message.content,
            [
                thinking("first", "sig_a"),
                ContentBlock::from("Let me look."),
                ContentBlock::RedactedThinking { data: "enc".into() },
                thinking("second", "sig_b"),
                ContentBlock::tool_use("toolu_1", "read_file", "{}"),
            ]
        );

        // And it goes back out block for block.
        let sent = convert_messages(&[choice.message]);
        assert_eq!(
            serde_json::to_value(&sent[0].content).unwrap(),
            json!([
                {"type": "thinking", "thinking": "first", "signature": "sig_a"},
                {"type": "text", "text": "Let me look."},
                {"type": "redacted_thinking", "data": "enc"},
                {"type": "thinking", "thinking": "second", "signature": "sig_b"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}},
            ])
        );
    }

    #[test]
    fn tool_call_without_streamed_input_gets_empty_object_arguments() {
        let choice = parse_stream(&[
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"list_dir","input":{}}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_stop"}"#,
        ])
        .unwrap();
        assert_eq!(
            choice.message.content,
            [ContentBlock::tool_use("toolu_1", "list_dir", "{}")]
        );
    }

    #[test]
    fn unparseable_payload_is_skipped_not_fatal() {
        let mut payloads = TOOL_REPLY.to_vec();
        payloads.insert(1, "{not json");
        assert!(parse_stream(&payloads).is_ok());
    }
}
