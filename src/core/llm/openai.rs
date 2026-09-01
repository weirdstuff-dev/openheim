use async_trait::async_trait;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::core::models::{Choice, ContentBlock, FinishReason, Message, Role, Tool, Usage};
use crate::error::{Error, Result};

use super::sse::SseDecoder;
use super::{LlmChunk, LlmClient};

#[derive(Clone)]
pub struct OpenAiClient {
    client: ReqwestClient,
    api_base: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
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

// --- OpenAI response types ---

#[derive(Debug, Deserialize)]
struct OpenAiResponseEnvelope {
    choices: Vec<OpenAiResponseChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

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

#[derive(Debug, Deserialize)]
struct OpenAiResponseChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Extended-thinking output, as returned by reasoning models behind
    /// OpenAI-compatible APIs (GLM, DeepSeek, …). Absent on non-reasoning
    /// models and plain OpenAI itself.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseToolCall {
    id: String,
    function: OpenAiResponseFunctionCall,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseFunctionCall {
    name: String,
    arguments: String,
}

/// Maps OpenAI's `finish_reason` vocabulary onto the provider-agnostic
/// [`FinishReason`]; anything without a known equivalent (e.g.
/// `content_filter`, the legacy `function_call`) passes through as
/// [`FinishReason::Other`].
fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::MaxTokens,
        other => FinishReason::Other(other.to_string()),
    }
}

// --- Conversions ---

fn convert_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                if let Some(text) = msg.text() {
                    result.push(OpenAiMessage {
                        role: "system".to_string(),
                        content: Some(OpenAiContent::Text(text)),
                        tool_calls: None,
                        tool_call_id: None,
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
                });
            }
            Role::Assistant => {
                let text = msg.text();
                let calls = msg.tool_calls();
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
                });
            }
            Role::Tool => {
                if let Some(tr) = msg.tool_result_block() {
                    result.push(OpenAiMessage {
                        role: "tool".to_string(),
                        content: Some(OpenAiContent::Text(tr.content)),
                        tool_calls: None,
                        tool_call_id: Some(tr.tool_call_id),
                    });
                }
            }
        }
    }

    result
}

/// Assembles an assistant message's content blocks in canonical order:
/// `Thinking` (when the provider returned reasoning) first, then `Text`,
/// then tool uses — the same ordering the Anthropic client produces, which
/// the history persistence and ACP replay layers rely on. Empty strings
/// produce no block, so an all-empty response yields empty content.
fn assemble_content(
    reasoning: Option<String>,
    text: Option<String>,
    tool_uses: Vec<ContentBlock>,
) -> Vec<ContentBlock> {
    let mut content = Vec::new();
    if let Some(thinking) = reasoning
        && !thinking.is_empty()
    {
        content.push(ContentBlock::Thinking {
            thinking,
            // The OpenAI-compatible `reasoning_content` field carries no
            // replayable signature — that's Anthropic-specific.
            signature: None,
        });
    }
    if let Some(text) = text
        && !text.is_empty()
    {
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

pub(super) async fn send_openai_style(
    client: &ReqwestClient,
    api_base: &str,
    api_key: &str,
    model: &str,
    max_tokens: Option<u32>,
    messages: &[Message],
    tools: &[Tool],
) -> Result<Choice> {
    let request = OpenAiRequest {
        model: model.to_string(),
        messages: convert_messages(messages),
        tools: convert_tools(tools),
        max_tokens,
    };

    let endpoint = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&request)
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

    let envelope: OpenAiResponseEnvelope = response.json().await.map_err(Error::ReqwestError)?;
    let usage = envelope.usage.map(Usage::from);

    let choice = envelope
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::ApiError("No response from LLM".to_string()))?;

    let tool_uses = choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| ContentBlock::ToolUse {
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
        })
        .collect();
    let content = assemble_content(
        choice.message.reasoning_content,
        choice.message.content,
        tool_uses,
    );

    Ok(Choice {
        message: Message {
            role: Role::Assistant,
            content,
        },
        finish_reason: choice.finish_reason.as_deref().map(map_finish_reason),
        usage,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn send_openai_style_streaming(
    client: &ReqwestClient,
    api_base: &str,
    api_key: &str,
    model: &str,
    max_tokens: Option<u32>,
    messages: &[Message],
    tools: &[Tool],
    chunk_tx: mpsc::UnboundedSender<LlmChunk>,
) -> Result<Choice> {
    let request = OpenAiRequest {
        model: model.to_string(),
        messages: convert_messages(messages),
        tools: convert_tools(tools),
        max_tokens,
    };

    let mut body = serde_json::to_value(&request).map_err(|e| Error::ParseError(e.to_string()))?;
    body["stream"] = serde_json::Value::Bool(true);
    // Asks OpenAI (and most OpenAI-compatible backends, which pass unknown
    // fields through) to emit one extra chunk at the end of the stream
    // carrying the same `usage` object the non-streaming response has.
    // Backends that don't understand it just ignore the field — but a few
    // OpenAI-compatible endpoints validate the request body strictly and
    // reject the unknown field with a 400, so that specific case is retried
    // once without it below rather than failing the whole call.
    body["stream_options"] = serde_json::json!({ "include_usage": true });

    let endpoint = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let mut response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(Error::ReqwestError)?;

    if response.status().as_u16() == 400 {
        let mut retry_body = body.clone();
        if let Some(obj) = retry_body.as_object_mut() {
            obj.remove("stream_options");
        }
        let retry_response = client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&retry_body)
            .send()
            .await
            .map_err(Error::ReqwestError)?;
        if retry_response.status().is_success() {
            response = retry_response;
        }
    }

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<failed to read error body>".into());
        return Err(Error::HttpError { status, body });
    }

    struct ToolCallAcc {
        id: String,
        name: String,
        args: String,
    }

    let mut text_buf = String::new();
    // Accumulated alongside `text_buf` so the final message persists the
    // reasoning (the live chunks are UI-only); without this, thinking
    // displayed during a turn vanished on reload.
    let mut reasoning_buf = String::new();
    let mut tool_acc: Vec<ToolCallAcc> = Vec::new();
    let mut finish_reason: Option<FinishReason> = None;
    let mut usage: Option<Usage> = None;
    let mut decoder = SseDecoder::new();
    let mut done = false;

    while !done {
        let Some(bytes) = response.chunk().await.map_err(Error::ReqwestError)? else {
            break;
        };
        decoder.feed(&bytes);

        while let Some(data) = decoder.next_payload() {
            if data == "[DONE]" {
                done = true;
                break;
            }

            let Ok(event) = serde_json::from_str::<serde_json::Value>(&data) else {
                continue;
            };

            // The `include_usage` final chunk carries `usage` alongside an
            // empty `choices` array, so this is checked independently of the
            // choice fields below rather than folded into that branch.
            if !event["usage"].is_null()
                && let Ok(u) = serde_json::from_value::<OpenAiUsage>(event["usage"].clone())
            {
                usage = Some(Usage::from(u));
            }

            let choice = &event["choices"][0];

            if let Some(fr) = choice["finish_reason"].as_str() {
                finish_reason = Some(map_finish_reason(fr));
            }

            let delta = &choice["delta"];

            if let Some(reasoning) = delta["reasoning_content"].as_str()
                && !reasoning.is_empty()
            {
                reasoning_buf.push_str(reasoning);
                let _ = chunk_tx.send(LlmChunk::Thinking(reasoning.to_string()));
            }

            if let Some(content) = delta["content"].as_str()
                && !content.is_empty()
            {
                text_buf.push_str(content);
                let _ = chunk_tx.send(LlmChunk::Text(content.to_string()));
            }

            if let Some(tcs) = delta["tool_calls"].as_array() {
                for tc in tcs {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    while tool_acc.len() <= idx {
                        tool_acc.push(ToolCallAcc {
                            id: String::new(),
                            name: String::new(),
                            args: String::new(),
                        });
                    }
                    if let Some(id) = tc["id"].as_str() {
                        tool_acc[idx].id = id.to_string();
                    }
                    if let Some(name) = tc["function"]["name"].as_str() {
                        tool_acc[idx].name.push_str(name);
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        tool_acc[idx].args.push_str(args);
                    }
                }
            }
        }
    }

    let tool_uses = tool_acc
        .into_iter()
        .enumerate()
        .filter(|(_, tc)| !tc.name.is_empty())
        .map(|(i, tc)| ContentBlock::ToolUse {
            id: if tc.id.is_empty() {
                format!("call_{i}")
            } else {
                tc.id
            },
            name: tc.name,
            arguments: tc.args,
        })
        .collect();

    Ok(Choice {
        message: Message {
            role: Role::Assistant,
            content: assemble_content(Some(reasoning_buf), Some(text_buf), tool_uses),
        },
        finish_reason,
        usage,
    })
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn send(&self, messages: &[Message], tools: &[Tool]) -> Result<Choice> {
        send_openai_style(
            &self.client,
            &self.api_base,
            &self.api_key,
            &self.model,
            self.max_tokens,
            messages,
            tools,
        )
        .await
    }

    async fn send_streaming(
        &self,
        messages: &[Message],
        tools: &[Tool],
        chunk_tx: mpsc::UnboundedSender<LlmChunk>,
    ) -> Result<Choice> {
        send_openai_style_streaming(
            &self.client,
            &self.api_base,
            &self.api_key,
            &self.model,
            self.max_tokens,
            messages,
            tools,
            chunk_tx,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_messages_text_only_user_stays_plain_string() {
        let messages = vec![Message::user("hello")];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert!(matches!(&result[0].content, Some(OpenAiContent::Text(t)) if t == "hello"));
    }

    #[test]
    fn convert_messages_user_with_image_becomes_parts() {
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
        let Some(OpenAiContent::Parts(parts)) = &result[0].content else {
            panic!("expected parts");
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], OpenAiContentPart::Text { text } if text == "what is this?"));
        assert!(matches!(
            &parts[1],
            OpenAiContentPart::ImageUrl { image_url }
                if image_url.url == "data:image/png;base64,base64data"
        ));
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.txt"}"#.into(),
            }],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        let tool_calls = result[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    // `convert_messages` is the single conversion point shared by both
    // `send_openai_style` (non-streaming) and `send_openai_style_streaming`
    // (see their bodies above), so this covers the continuation request
    // built by either path.
    //
    // Thinking is persisted to history and replayed through ACP (see
    // `assemble_content`) so the *user* sees it again, but it is
    // deliberately never sent back to the provider: reasoning traces from
    // models like DeepSeek/GLM can be large, and replaying every prior
    // turn's thinking on every continuation request would balloon context
    // usage turn over turn. Only the tool calls survive onto the
    // continuation request.
    #[test]
    fn convert_messages_assistant_never_replays_thinking_back_to_the_provider() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "let me check the file".into(),
                    signature: None,
                },
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            ],
        }];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert!(result[0].tool_calls.is_some());
        // `OpenAiMessage` has no `reasoning_content` field at all, so the
        // thinking has nowhere to go — this also serializes to confirm it
        // never appears in the JSON sent over the wire.
        let json = serde_json::to_value(&result[0]).unwrap();
        assert!(json.get("reasoning_content").is_none());
    }

    #[test]
    fn convert_messages_tool_result_uses_tool_role() {
        let messages = vec![Message::tool_result(
            "call_1",
            "read_file",
            "file content",
            false,
        )];
        let result = convert_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "tool");
        assert_eq!(result[0].tool_call_id.as_deref(), Some("call_1"));
        assert!(matches!(&result[0].content, Some(OpenAiContent::Text(t)) if t == "file content"));
    }

    #[test]
    fn openai_request_skips_empty_tools_and_max_tokens() {
        let req = OpenAiRequest {
            model: "gpt-4".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: None,
        };
        let json: Value = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none());
        assert!(json.get("max_tokens").is_none());
    }

    #[test]
    fn assemble_content_puts_thinking_before_text_and_tool_uses() {
        let tool = ContentBlock::ToolUse {
            id: "call_0".into(),
            name: "read_file".into(),
            arguments: "{}".into(),
        };
        let content = assemble_content(
            Some("let me think".into()),
            Some("here's the answer".into()),
            vec![tool],
        );
        assert_eq!(content.len(), 3);
        assert!(matches!(
            &content[0],
            ContentBlock::Thinking { thinking, signature }
                if thinking == "let me think" && signature.is_none()
        ));
        assert!(matches!(&content[1], ContentBlock::Text { text } if text == "here's the answer"));
        assert!(matches!(&content[2], ContentBlock::ToolUse { name, .. } if name == "read_file"));
    }

    #[test]
    fn assemble_content_skips_empty_reasoning_and_text() {
        assert!(assemble_content(Some(String::new()), None, vec![]).is_empty());
        assert!(assemble_content(None, Some(String::new()), vec![]).is_empty());
    }

    #[test]
    fn response_message_deserializes_reasoning_content() {
        let msg: OpenAiResponseMessage =
            serde_json::from_str(r#"{"content":"hi","reasoning_content":"because"}"#).unwrap();
        assert_eq!(msg.reasoning_content.as_deref(), Some("because"));
    }

    #[test]
    fn response_message_reasoning_content_defaults_to_none_when_absent() {
        let msg: OpenAiResponseMessage = serde_json::from_str(r#"{"content":"hi"}"#).unwrap();
        assert!(msg.reasoning_content.is_none());
    }
}
