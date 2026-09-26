use serde::{Deserialize, Serialize};
use serde_json::Value;

fn is_false(b: &bool) -> bool {
    !b
}

/// Chat role for a conversation message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool result injected back into the conversation after a tool call.
    Tool,
}

/// A single piece of message content, shaped like Anthropic's (and ACP's)
/// content blocks. Providers with a flatter format (`core::llm::openai`)
/// convert at their own edge.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// Extended-thinking text for an assistant turn. Must be replayed
    /// verbatim (with `signature`) in its original position in the turn —
    /// with interleaved thinking a turn can hold several, between text and
    /// tool calls — or Anthropic rejects the next request.
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Thinking the provider returned encrypted (Anthropic's
    /// `redacted_thinking`). Nothing to show, but it has to be replayed
    /// verbatim in its place in the turn, like `Thinking`.
    RedactedThinking {
        data: String,
    },
    Image {
        /// Base64-encoded image data.
        data: String,
        mime_type: String,
    },
    ToolUse {
        id: String,
        name: String,
        /// JSON string of the arguments object.
        arguments: String,
        /// Opaque provider token that has to be sent back with this call on
        /// the next request (Gemini's `thoughtSignature`). Other providers
        /// ignore it. Build calls without one via [`ContentBlock::tool_use`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: String,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

impl ContentBlock {
    /// A `ToolUse` block without a provider signature.
    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            signature: None,
        }
    }
}

impl<T: Into<String>> From<T> for ContentBlock {
    fn from(value: T) -> Self {
        ContentBlock::Text { text: value.into() }
    }
}

/// A `ToolUse` block extracted from a [`Message`] for convenient iteration;
/// see [`Message::tool_calls`].
#[derive(Debug, Clone)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// The `ToolResult` block on a `Role::Tool` [`Message`]; see
/// [`Message::tool_result_block`].
#[derive(Debug, Clone)]
pub struct ToolResultBlock {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: String,
    pub is_error: bool,
}

/// A single message in a conversation thread.
///
/// `role` says what the message *is*; `content` is an ordered list of blocks
/// describing what it *contains*. A tool-result message is `role: Tool` with
/// a single `ToolResult` block rather than a distinct role.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::from(text)],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::from(text)],
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: tool_call_id.into(),
                tool_name: tool_name.into(),
                content: content.into(),
                is_error,
            }],
        }
    }

    /// Concatenation of all `Text` blocks' text, or `None` if there are none.
    pub fn text(&self) -> Option<String> {
        let joined: String = self
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        if joined.is_empty() {
            None
        } else {
            Some(joined)
        }
    }

    /// All `ToolUse` blocks in this message, in order.
    pub fn tool_calls(&self) -> Vec<ToolUseBlock> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse {
                    id,
                    name,
                    arguments,
                    ..
                } => Some(ToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// The `ToolResult` block, if this is a `Role::Tool` message.
    pub fn tool_result_block(&self) -> Option<ToolResultBlock> {
        self.content.iter().find_map(|b| match b {
            ContentBlock::ToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error,
            } => Some(ToolResultBlock {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                content: content.clone(),
                is_error: *is_error,
            }),
            _ => None,
        })
    }

    /// What a UI shows for this message when replaying saved history, in
    /// stored order (with interleaved thinking an assistant turn can go
    /// thinking → text → thinking → tool call). Skips what a live turn
    /// never showed: system messages, redacted or empty thinking, empty
    /// assistant text, and blocks in a role that doesn't display them.
    pub fn transcript(&self) -> impl Iterator<Item = TranscriptEntry<'_>> {
        self.content
            .iter()
            .filter_map(|block| TranscriptEntry::of(&self.role, block))
    }
}

/// One displayable piece of a stored [`Message`]; see [`Message::transcript`].
/// Front ends map these to their own items, so they agree on what a replayed
/// session shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranscriptEntry<'a> {
    UserText(&'a str),
    UserImage {
        /// Base64-encoded image data.
        data: &'a str,
        mime_type: &'a str,
    },
    Thinking(&'a str),
    AssistantText(&'a str),
    ToolCall {
        id: &'a str,
        name: &'a str,
        /// JSON string of the arguments object.
        arguments: &'a str,
    },
    ToolResult {
        tool_call_id: &'a str,
        tool_name: &'a str,
        content: &'a str,
        is_error: bool,
    },
}

impl<'a> TranscriptEntry<'a> {
    fn of(role: &Role, block: &'a ContentBlock) -> Option<Self> {
        match (role, block) {
            (Role::User, ContentBlock::Text { text }) => Some(Self::UserText(text)),
            (Role::User, ContentBlock::Image { data, mime_type }) => {
                Some(Self::UserImage { data, mime_type })
            }
            (Role::Assistant, ContentBlock::Thinking { thinking, .. }) if !thinking.is_empty() => {
                Some(Self::Thinking(thinking))
            }
            (Role::Assistant, ContentBlock::Text { text }) if !text.is_empty() => {
                Some(Self::AssistantText(text))
            }
            (
                Role::Assistant,
                ContentBlock::ToolUse {
                    id,
                    name,
                    arguments,
                    ..
                },
            ) => Some(Self::ToolCall {
                id,
                name,
                arguments,
            }),
            (
                Role::Tool,
                ContentBlock::ToolResult {
                    tool_call_id,
                    tool_name,
                    content,
                    is_error,
                },
            ) => Some(Self::ToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error: *is_error,
            }),
            _ => None,
        }
    }
}

/// A tool available to the agent, serialised in the OpenAI function-calling format.
#[derive(Debug, Serialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

impl Tool {
    /// Builds a function-type [`Tool`] from its name, description, and JSON
    /// Schema parameters — the shape every `ToolHandler::definition` impl
    /// needs, without spelling out `tool_type`/`FunctionDefinition` by hand.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Metadata describing a callable tool function.
#[derive(Debug, Serialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the function's parameters.
    pub parameters: Value,
}

/// Why the provider stopped generating, normalized across providers'
/// differing vocabularies (Anthropic's `stop_reason`, Gemini's
/// `finishReason`, OpenAI's `finish_reason`). Each `core::llm` provider
/// module maps its own wire values onto this once, at the response-parsing
/// boundary.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model completed its response normally.
    Stop,
    /// The model wants to invoke one or more tools.
    ToolCalls,
    /// The response was truncated because it hit the token limit (or, for
    /// Anthropic, the context window).
    MaxTokens,
    /// The model declined to answer or the provider filtered the output
    /// (Anthropic's `refusal`, OpenAI's `content_filter`, Gemini's
    /// `SAFETY`/`RECITATION`/…).
    Refusal,
    /// The provider paused a long-running turn and expects the conversation
    /// to be resent as-is to resume it (Anthropic's `pause_turn`).
    Paused,
    /// A provider-specific reason with no equivalent above (e.g. Anthropic's
    /// `stop_sequence`), passed through verbatim.
    Other(String),
}

/// A single completion choice returned by the provider.
#[derive(Debug)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<FinishReason>,
    /// Token usage for this call, when the provider reports it. `None` for
    /// providers/backends that don't return usage data (e.g. an
    /// OpenAI-compatible endpoint that ignores `stream_options`).
    pub usage: Option<Usage>,
}

/// Token usage for a single LLM call, normalized across providers'
/// differing vocabularies (Anthropic's `input_tokens`/`cache_read_input_tokens`,
/// OpenAI's `prompt_tokens`/`prompt_tokens_details.cached_tokens`, Gemini's
/// `promptTokenCount`/`cachedContentTokenCount`).
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the request that were newly processed (not served from cache).
    pub input_tokens: u64,
    /// Tokens the model generated in its response.
    pub output_tokens: u64,
    /// Tokens written to a prompt cache for reuse by a later call.
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Tokens served from a prompt cache instead of being freshly processed.
    #[serde(default)]
    pub cache_read_tokens: u64,
}

impl Usage {
    /// Sum of every token type this call accounted for.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_creation_tokens + self.cache_read_tokens
    }
}

/// Why an agent run stopped. Distinct from ACP's own `StopReason` (which this
/// maps onto at the ACP boundary) so `core` doesn't depend on the `acp` crate.
#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The LLM produced a final text response: [`FinishReason::Stop`], or a
    /// finish reason with no more specific mapping below.
    EndTurn,
    /// `config.max_iterations` was reached without the LLM stopping on its own.
    MaxIterations,
    /// The final response was cut off at the output token limit
    /// ([`FinishReason::MaxTokens`]).
    MaxTokens,
    /// The model declined to answer or its output was filtered
    /// ([`FinishReason::Refusal`]).
    Refusal,
    /// The turn was cancelled via `TurnContext::cancel`.
    Cancelled,
    /// The LLM returned a response with neither text content nor tool calls.
    NoContent,
}

impl StopReason {
    /// A short, user-facing explanation for a turn that didn't end normally,
    /// for front-ends to show under the reply. `None` for `EndTurn` and
    /// `Cancelled`, where the user already knows why the turn stopped.
    pub fn notice(self) -> Option<&'static str> {
        match self {
            StopReason::EndTurn | StopReason::Cancelled => None,
            StopReason::MaxIterations => {
                Some("stopped at the iteration limit (max_iterations) before finishing")
            }
            StopReason::MaxTokens => {
                Some("the response was cut off at the output token limit (max_tokens)")
            }
            StopReason::Refusal => Some("the model declined to respond"),
            StopReason::NoContent => Some("the model returned an empty response"),
        }
    }
}

/// Final output of a completed agent run.
#[derive(Debug, Serialize)]
pub struct AgentResult {
    pub final_response: String,
    pub iterations_used: usize,
    pub stop_reason: StopReason,
    /// Usage of the *most recent* LLM call this turn made — a snapshot of
    /// how large the context is right now (that call's `input` + `output` +
    /// cache tokens is exactly what the next call will resend as history).
    /// `None` if no call in this turn reported usage.
    pub context_usage: Option<Usage>,
}

/// An event from a running agent turn, as `run_agent` and
/// `AgentState::prompt` report it. ACP's `SessionUpdate` is derived from it
/// at the ACP edge (`acp::util::stream_event_to_session_update`).
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "event_type")]
pub enum StreamEvent {
    /// Signals the start of a new reasoning iteration.
    #[serde(rename = "iteration_start")]
    IterationStart { iteration: usize },
    /// The agent is about to invoke a tool.
    #[serde(rename = "tool_call")]
    ToolCall {
        /// Provider-assigned tool-call ID; stable across the matching
        /// [`StreamEvent::ToolResult`] and any permission check in between.
        id: String,
        tool_name: String,
        arguments: String,
    },
    /// A tool has finished executing.
    #[serde(rename = "tool_result")]
    ToolResult {
        id: String,
        tool_name: String,
        result: String,
        is_error: bool,
    },
    /// A chunk of text from the LLM.
    #[serde(rename = "llm_response")]
    LlmResponse { content: String },
    /// A chunk of the model's internal reasoning (extended thinking).
    #[serde(rename = "thinking_content")]
    ThinkingContent { content: String },
    /// Token usage for the LLM call that just completed this iteration —
    /// the current context size, not a running total (see
    /// [`AgentResult::context_usage`]).
    #[serde(rename = "usage")]
    Usage { usage: Usage },
    /// The agent has finished; `final_response` is the complete answer.
    #[serde(rename = "finished")]
    Finished {
        final_response: String,
        iterations: usize,
    },
    /// `message` was just pushed onto the turn's history: every assistant
    /// and tool-result message, in order. Persist history incrementally from
    /// here.
    #[serde(rename = "message_appended")]
    MessageAppended { message: Message },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_user_sets_correct_fields() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text().as_deref(), Some("hello"));
        assert!(msg.tool_calls().is_empty());
        assert!(msg.tool_result_block().is_none());
    }

    #[test]
    fn message_assistant_sets_correct_fields() {
        let msg = Message::assistant("response");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text().as_deref(), Some("response"));
        assert!(msg.tool_calls().is_empty());
    }

    #[test]
    fn transcript_keeps_assistant_block_order_and_skips_what_isnt_shown() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "first".into(),
                    signature: Some("sig".into()),
                },
                ContentBlock::from("Let me look."),
                ContentBlock::RedactedThinking { data: "enc".into() },
                ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                },
                ContentBlock::from(""),
                ContentBlock::tool_use("toolu_1", "read_file", "{}"),
            ],
        };
        assert_eq!(
            msg.transcript().collect::<Vec<_>>(),
            [
                TranscriptEntry::Thinking("first"),
                TranscriptEntry::AssistantText("Let me look."),
                TranscriptEntry::ToolCall {
                    id: "toolu_1",
                    name: "read_file",
                    arguments: "{}",
                },
            ]
        );
    }

    #[test]
    fn transcript_of_user_tool_and_system_messages() {
        let user = Message {
            role: Role::User,
            content: vec![
                ContentBlock::from("look at this"),
                ContentBlock::Image {
                    data: "aGk=".into(),
                    mime_type: "image/png".into(),
                },
            ],
        };
        assert_eq!(
            user.transcript().collect::<Vec<_>>(),
            [
                TranscriptEntry::UserText("look at this"),
                TranscriptEntry::UserImage {
                    data: "aGk=",
                    mime_type: "image/png",
                },
            ]
        );

        let tool = Message::tool_result("call_1", "read_file", "boom", true);
        assert_eq!(
            tool.transcript().collect::<Vec<_>>(),
            [TranscriptEntry::ToolResult {
                tool_call_id: "call_1",
                tool_name: "read_file",
                content: "boom",
                is_error: true,
            }]
        );

        let system = Message {
            role: Role::System,
            content: vec![ContentBlock::from("you are x")],
        };
        assert_eq!(system.transcript().count(), 0);
    }

    #[test]
    fn message_tool_result_sets_correct_fields() {
        let msg = Message::tool_result("call_1", "read_file", "content", false);
        assert_eq!(msg.role, Role::Tool);
        assert!(msg.tool_calls().is_empty());
        let tr = msg.tool_result_block().unwrap();
        assert_eq!(tr.tool_call_id, "call_1");
        assert_eq!(tr.tool_name, "read_file");
        assert_eq!(tr.content, "content");
        assert!(!tr.is_error);
    }

    #[test]
    fn message_text_joins_multiple_text_blocks() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "hello ".into(),
                },
                ContentBlock::tool_use("call_1", "read_file", "{}"),
                ContentBlock::Text {
                    text: "world".into(),
                },
            ],
        };
        assert_eq!(msg.text().as_deref(), Some("hello world"));
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn content_block_serializes_with_type_tag() {
        let block = ContentBlock::tool_use("call_1", "read_file", "{}");
        let json: Value = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "read_file");
        // An unset signature is left out, and a block without one reads back
        // as `None`.
        assert!(json.get("signature").is_none());
        let unsigned: ContentBlock = serde_json::from_value(serde_json::json!({
            "type": "tool_use", "id": "call_1", "name": "read_file", "arguments": "{}"
        }))
        .unwrap();
        assert_eq!(unsigned, block);

        let block = ContentBlock::Thinking {
            thinking: "hmm".into(),
            signature: Some("sig".into()),
        };
        let json: Value = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["signature"], "sig");
    }

    #[test]
    fn role_serializes_to_lowercase() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    #[test]
    fn role_deserializes_from_lowercase() {
        assert_eq!(
            serde_json::from_str::<Role>("\"system\"").unwrap(),
            Role::System
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"user\"").unwrap(),
            Role::User
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"assistant\"").unwrap(),
            Role::Assistant
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"tool\"").unwrap(),
            Role::Tool
        );
    }

    #[test]
    fn stream_event_serializes_with_event_type_tag() {
        let event = StreamEvent::IterationStart { iteration: 1 };
        let json: Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event_type"], "iteration_start");
        assert_eq!(json["iteration"], 1);

        let event = StreamEvent::Finished {
            final_response: "done".into(),
            iterations: 3,
        };
        let json: Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event_type"], "finished");
        assert_eq!(json["final_response"], "done");
        assert_eq!(json["iterations"], 3);
    }

    #[test]
    fn stream_event_tool_call_serializes() {
        let event = StreamEvent::ToolCall {
            id: "call_1".into(),
            tool_name: "read_file".into(),
            arguments: r#"{"path":"a.txt"}"#.into(),
        };
        let json: Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event_type"], "tool_call");
        assert_eq!(json["tool_name"], "read_file");
    }
}
