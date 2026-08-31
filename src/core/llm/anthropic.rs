use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::core::models::{Choice, ContentBlock, FinishReason, Message, Role, Tool, Usage};
use crate::error::{Error, Result};

use super::sse::SseDecoder;
use super::{LlmChunk, LlmClient};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

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
}

impl AnthropicClient {
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
            max_tokens: max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
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

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Result<Vec<AnthropicMessage>> {
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
                // Thinking blocks must lead the assistant turn's content and be
                // replayed verbatim, or the API rejects the next request.
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
                        ContentBlock::Text { text } if !text.is_empty() => {
                            blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                        }
                        ContentBlock::ToolUse {
                            id,
                            name,
                            arguments,
                        } => {
                            let input: Value = serde_json::from_str(arguments).map_err(|e| {
                                Error::ParseError(format!(
                                    "invalid JSON in tool call arguments for '{name}': {e}"
                                ))
                            })?;
                            blocks.push(AnthropicContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input,
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

    Ok(result)
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
        Some("max_tokens") => Some(FinishReason::MaxTokens),
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

/// Returns a thinking config for models that support adaptive thinking.
///
/// Adaptive thinking (`type: "adaptive"`) replaced the old fixed-budget form
/// (`type: "enabled", budget_tokens: N`) starting with Claude 4.6; the old
/// form now returns a 400 on Opus 4.7/4.8, Sonnet 5, and Fable 5, and is
/// deprecated on Opus 4.6 / Sonnet 4.6. Models predating adaptive thinking
/// (Sonnet 3.7 and earlier Claude 4 releases) only support the fixed-budget
/// form; rather than special-case each one, thinking is left disabled for
/// them. Adaptive thinking also enables interleaved thinking automatically,
/// so no `anthropic-beta` header is needed here.
fn thinking_config(model: &str) -> Option<AnthropicThinkingConfig> {
    let supported = [
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    ]
    .iter()
    .any(|m| model.contains(m));
    if !supported {
        return None;
    }
    Some(AnthropicThinkingConfig {
        thinking_type: "adaptive",
        display: "summarized",
    })
}

impl AnthropicClient {
    fn build_request(&self, messages: &[Message], tools: &[Tool]) -> Result<AnthropicRequest> {
        Ok(AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            stream: true,
            system: extract_system(messages),
            messages: convert_messages(messages)?,
            tools: convert_tools(tools),
            thinking: thinking_config(&self.model),
        })
    }

    /// POSTs `request` and returns the response body stream, having already
    /// checked the status and turned a non-2xx response into an `Err` with
    /// its body attached.
    async fn post(&self, request: &AnthropicRequest) -> Result<reqwest::Response> {
        let endpoint = format!("{}/messages", self.api_base.trim_end_matches('/'));

        let response = self
            .client
            .post(&endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(Error::ReqwestError)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read error body>".into());
            return Err(Error::HttpError { status, body });
        }

        Ok(response)
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    /// Implemented in terms of [`Self::send_streaming`] with a discarded
    /// channel: Anthropic's streaming and non-streaming responses carry the
    /// same information, so there is no reason to maintain a second
    /// request/response code path (and the JSON non-streaming path used to
    /// silently skip `thinking_config`, leaving thinking enabled only for
    /// streaming callers).
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        let (chunk_tx, _chunk_rx) = mpsc::unbounded_channel();
        self.send_streaming(messages, tools, chunk_tx).await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        let request = self.build_request(messages, tools)?;
        let response = self.post(&request).await?;

        // Parse SSE stream.
        let mut decoder = SseDecoder::new();
        let mut current_block_type: Option<String> = None;
        let mut current_tool_id: Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_json = String::new();
        let mut text_content = String::new();
        let mut thinking_content = String::new();
        let mut thinking_signature = String::new();
        let mut tool_calls: Vec<ContentBlock> = Vec::new();
        let mut stop_reason: Option<String> = None;
        let mut usage = Usage::default();

        let mut response = response;
        while let Some(chunk) = response.chunk().await.map_err(Error::ReqwestError)? {
            decoder.feed(&chunk);

            while let Some(data) = decoder.next_payload() {
                if data == "[DONE]" || data.is_empty() {
                    continue;
                }

                let event: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                match event["type"].as_str().unwrap_or("") {
                    "message_start" => {
                        let u = &event["message"]["usage"];
                        usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                        usage.output_tokens = u["output_tokens"].as_u64().unwrap_or(0);
                        usage.cache_creation_tokens =
                            u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                        usage.cache_read_tokens =
                            u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                    }
                    "content_block_start" => {
                        let block_type = event["content_block"]["type"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        if block_type == "tool_use" {
                            current_tool_id =
                                event["content_block"]["id"].as_str().map(String::from);
                            current_tool_name =
                                event["content_block"]["name"].as_str().map(String::from);
                            current_tool_json.clear();
                        }
                        current_block_type = Some(block_type);
                    }
                    "content_block_delta" => {
                        let delta = &event["delta"];
                        match delta["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                if let Some(text) = delta["text"].as_str() {
                                    text_content.push_str(text);
                                    let _ = chunk_tx.send(LlmChunk::Text(text.to_string()));
                                }
                            }
                            "thinking_delta" => {
                                if let Some(thinking) = delta["thinking"].as_str() {
                                    thinking_content.push_str(thinking);
                                    let _ = chunk_tx.send(LlmChunk::Thinking(thinking.to_string()));
                                }
                            }
                            "signature_delta" => {
                                if let Some(signature) = delta["signature"].as_str() {
                                    thinking_signature.push_str(signature);
                                }
                            }
                            "input_json_delta" => {
                                if let Some(partial) = delta["partial_json"].as_str() {
                                    current_tool_json.push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        if current_block_type.as_deref() == Some("tool_use") {
                            if let (Some(id), Some(name)) =
                                (current_tool_id.take(), current_tool_name.take())
                            {
                                tool_calls.push(ContentBlock::ToolUse {
                                    id,
                                    name,
                                    arguments: current_tool_json.clone(),
                                });
                            }
                            current_tool_json.clear();
                        }
                        current_block_type = None;
                    }
                    "message_delta" => {
                        if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                            stop_reason = Some(reason.to_string());
                        }
                        // Anthropic reports `output_tokens` cumulatively on each
                        // `message_delta`, not as a per-event delta — the last
                        // value seen before the stream ends is the true total.
                        if let Some(ot) = event["usage"]["output_tokens"].as_u64() {
                            usage.output_tokens = ot;
                        }
                    }
                    "error" => {
                        let msg = event["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown streaming error")
                            .to_string();
                        return Err(Error::ApiError(format!("Anthropic streaming error: {msg}")));
                    }
                    _ => {}
                }
            }
        }

        let finish_reason = map_stop_reason(stop_reason.as_deref());

        let mut content = Vec::new();
        if !thinking_content.is_empty() {
            content.push(ContentBlock::Thinking {
                thinking: thinking_content,
                signature: if thinking_signature.is_empty() {
                    None
                } else {
                    Some(thinking_signature)
                },
            });
        }
        if !text_content.is_empty() {
            content.push(ContentBlock::Text { text: text_content });
        }
        content.extend(tool_calls);

        Ok(Choice {
            message: Message {
                role: Role::Assistant,
                content,
            },
            finish_reason,
            usage: Some(usage),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_messages_user_message() {
        let messages = vec![Message::user("hello")];
        let result = convert_messages(&messages).unwrap();
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
        let result = convert_messages(&messages).unwrap();
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
        let result = convert_messages(&messages).unwrap();
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
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            ],
        }];
        let result = convert_messages(&messages).unwrap();
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
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            ],
        }];
        let result = convert_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content.len(), 3); // thinking + text + tool_use
        assert!(matches!(
            &result[0].content[0],
            AnthropicContentBlock::Thinking { thinking, signature }
                if thinking == "reasoning about the file" && signature == "sig123"
        ));
    }

    #[test]
    fn convert_messages_invalid_tool_arguments_returns_error() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "not valid json".into(),
            }],
        }];
        assert!(convert_messages(&messages).is_err());
    }

    #[test]
    fn convert_messages_merges_consecutive_tool_results() {
        let messages = vec![
            Message::tool_result("call_1", "read_file", "content1", false),
            Message::tool_result("call_2", "write_file", "content2", false),
        ];
        let result = convert_messages(&messages).unwrap();
        // Both tool results should merge into a single user message
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content.len(), 2);
    }

    #[test]
    fn convert_tools_maps_definitions() {
        let tools = vec![Tool {
            tool_type: "function".into(),
            function: crate::core::models::FunctionDefinition {
                name: "read_file".into(),
                description: "Read a file".into(),
                parameters: json!({"type": "object"}),
            },
        }];
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
    }

    #[test]
    fn map_stop_reason_passes_through_unknown_values() {
        assert_eq!(
            map_stop_reason(Some("refusal")),
            Some(FinishReason::Other("refusal".to_string()))
        );
    }

    #[test]
    fn map_stop_reason_none_stays_none() {
        assert!(map_stop_reason(None).is_none());
    }

    #[test]
    fn thinking_config_enabled_for_adaptive_capable_models() {
        for model in [
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
        ] {
            let config = thinking_config(model);
            assert!(config.is_some(), "expected thinking enabled for {model}");
            assert_eq!(config.unwrap().thinking_type, "adaptive");
        }
    }

    #[test]
    fn thinking_config_disabled_for_unsupported_models() {
        for model in [
            "claude-haiku-4-5",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-5-20251101",
        ] {
            assert!(
                thinking_config(model).is_none(),
                "expected thinking disabled for {model}"
            );
        }
    }

    #[test]
    fn build_request_always_streams_and_enables_thinking_when_supported() {
        // Regression test for the non-streaming `send()` bug: it used to build
        // its own request with `stream: false, thinking: None`, so thinking
        // silently never worked outside of `send_streaming`. `build_request`
        // is now the single source for both, so this holds for both callers.
        let client = AnthropicClient::new(
            ReqwestClient::new(),
            "https://api.anthropic.com/v1".into(),
            "test-key".into(),
            "claude-sonnet-5".into(),
            None,
        );
        let request = client.build_request(&[Message::user("hi")], &[]).unwrap();
        assert!(request.stream);
        assert!(request.thinking.is_some());
    }
}
