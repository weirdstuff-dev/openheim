use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

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

/// A single piece of message content. Close to Anthropic's own content-block
/// shape (and by extension ACP's), since both this codebase's richest
/// provider and its host protocol already think in these terms; lossless
/// providers convert directly, lossy ones (see `core::llm::openai`) flatten
/// at their own edge instead of forcing the core type to be the lossy one.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// Extended-thinking text for an assistant turn. Must be replayed
    /// verbatim (with `signature`) as the first block of the turn when it also
    /// contains `ToolUse` blocks, or Anthropic rejects the next request.
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
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
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: String,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
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
#[derive(Debug, Serialize, Deserialize, Clone)]
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
}

/// A tool available to the agent, serialised in the OpenAI function-calling format.
#[derive(Debug, Serialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

/// Metadata describing a callable tool function.
#[derive(Debug, Serialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the function's parameters.
    pub parameters: Value,
}

/// A single completion choice returned by the provider.
#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// One iteration of an agent run, including the LLM response and any tools invoked.
#[derive(Debug, Serialize, Clone)]
pub struct AgentStep {
    pub iteration: usize,
    pub message: String,
    pub tool_calls: Option<Vec<ToolExecutionResult>>,
}

/// Result of executing a single tool during an agent step.
#[derive(Debug, Serialize, Clone)]
pub struct ToolExecutionResult {
    pub tool_name: String,
    pub arguments: String,
    pub result: String,
}

/// Why an agent run stopped. Distinct from ACP's own `StopReason` (which this
/// maps onto at the ACP boundary) so `core` doesn't depend on the `acp` crate.
#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The LLM produced a final response with `finish_reason == "stop"`.
    EndTurn,
    /// `config.max_iterations` was reached without the LLM stopping on its own.
    MaxIterations,
    /// The turn was cancelled via `TurnContext::cancel`.
    Cancelled,
    /// The LLM returned a response with neither text content nor tool calls.
    NoContent,
}

/// Final output of a completed agent run.
#[derive(Debug, Serialize)]
pub struct AgentResult {
    pub final_response: String,
    pub steps: Vec<AgentStep>,
    pub iterations_used: usize,
    pub stop_reason: StopReason,
}

/// Streaming event emitted during an agent run over a WebSocket connection.
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
    /// The agent has finished; `final_response` is the complete answer.
    #[serde(rename = "finished")]
    Finished {
        final_response: String,
        iterations: usize,
    },
    /// The running list of tool-call steps for this turn, replacing any
    /// previously emitted plan. Derived from tool calls issued so far — not
    /// an upfront LLM-authored plan.
    #[serde(rename = "plan_update")]
    PlanUpdate { entries: Vec<PlanStep> },
}

/// One entry in a [`StreamEvent::PlanUpdate`]. Protocol-agnostic mirror of
/// ACP's `PlanEntry` (see [`crate::acp`] for the mapping).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanStep {
    pub content: String,
    pub status: PlanStepStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Completed,
}

// Filesystem WebSocket Models

/// Entry in the file tree
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified: Option<u64>,
}

/// Requests from the frontend to the filesystem WebSocket
#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
pub enum FsRequest {
    /// Initialize watching on a workspace directory
    #[serde(rename = "watch")]
    Watch { path: String },

    /// Stop watching
    #[serde(rename = "unwatch")]
    Unwatch,

    /// List directory contents
    #[serde(rename = "list")]
    List {
        path: String,
        recursive: Option<bool>,
    },

    /// Read file contents
    #[serde(rename = "read")]
    Read { path: String },

    /// Write file contents
    #[serde(rename = "write")]
    Write { path: String, content: String },

    /// Create a directory
    #[serde(rename = "mkdir")]
    Mkdir { path: String },

    /// Delete a file or directory
    #[serde(rename = "delete")]
    Delete { path: String },

    /// Rename/move a file or directory
    #[serde(rename = "rename")]
    Rename { from: String, to: String },
}

/// Responses/events from the filesystem WebSocket to the frontend
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum FsResponse {
    #[serde(rename = "connected")]
    Connected { message: String },

    #[serde(rename = "watching")]
    Watching { path: String },

    #[serde(rename = "unwatched")]
    Unwatched,

    #[serde(rename = "file_list")]
    FileList {
        path: String,
        entries: Vec<FileEntry>,
    },

    #[serde(rename = "file_content")]
    FileContent { path: String, content: String },

    #[serde(rename = "write_success")]
    WriteSuccess { path: String },

    #[serde(rename = "mkdir_success")]
    MkdirSuccess { path: String },

    #[serde(rename = "delete_success")]
    DeleteSuccess { path: String },

    #[serde(rename = "rename_success")]
    RenameSuccess { from: String, to: String },

    /// File system change event (from watcher)
    #[serde(rename = "fs_event")]
    FsEvent {
        event_kind: String,
        paths: Vec<String>,
    },

    #[serde(rename = "error")]
    Error { message: String },
}

/// Inbound prompt request over the agent WebSocket channel.
#[derive(Debug, Deserialize)]
pub struct WsRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub chat_id: Option<Uuid>,
    #[serde(default)]
    pub skills: Option<Vec<String>>,
}

/// Outbound response over the agent WebSocket channel.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum WsResponse {
    #[serde(rename = "event")]
    Event { data: StreamEvent },

    #[serde(rename = "error")]
    Error { message: String },

    #[serde(rename = "done")]
    Done {
        #[serde(skip_serializing_if = "Option::is_none")]
        chat_id: Option<String>,
    },
}

// Unified WebSocket envelope types

/// System-level events not tied to a specific channel.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum SystemEvent {
    #[serde(rename = "connected")]
    Connected { message: String },

    #[serde(rename = "error")]
    Error { message: String },
}

/// Inbound envelope: every message from the client wraps its payload with a channel tag.
#[derive(Debug, Deserialize)]
#[serde(tag = "channel", content = "data")]
pub enum ClientEnvelope {
    #[serde(rename = "agent")]
    Agent(WsRequest),

    #[serde(rename = "fs")]
    Fs(FsRequest),
}

/// Outbound envelope: every message to the client carries a channel tag so the
/// client can route responses without maintaining two separate connections.
#[derive(Debug, Serialize)]
#[serde(tag = "channel", content = "data")]
pub enum ServerEnvelope {
    #[serde(rename = "system")]
    System(SystemEvent),

    #[serde(rename = "agent")]
    Agent(WsResponse),

    #[serde(rename = "fs")]
    Fs(FsResponse),
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
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
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
        let block = ContentBlock::ToolUse {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: "{}".into(),
        };
        let json: Value = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "read_file");

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

    #[test]
    fn fs_request_deserializes_watch() {
        let json = r#"{"action": "watch", "path": "/tmp"}"#;
        let req: FsRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FsRequest::Watch { path } if path == "/tmp"));
    }

    #[test]
    fn fs_request_deserializes_write() {
        let json = r#"{"action": "write", "path": "a.txt", "content": "hello"}"#;
        let req: FsRequest = serde_json::from_str(json).unwrap();
        assert!(
            matches!(req, FsRequest::Write { path, content } if path == "a.txt" && content == "hello")
        );
    }

    #[test]
    fn fs_request_deserializes_rename() {
        let json = r#"{"action": "rename", "from": "a.txt", "to": "b.txt"}"#;
        let req: FsRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FsRequest::Rename { from, to } if from == "a.txt" && to == "b.txt"));
    }

    #[test]
    fn fs_response_serializes_with_type_tag() {
        let resp = FsResponse::Connected {
            message: "ok".into(),
        };
        let json: Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "connected");
        assert_eq!(json["message"], "ok");
    }

    #[test]
    fn fs_response_error_serializes() {
        let resp = FsResponse::Error {
            message: "not found".into(),
        };
        let json: Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["message"], "not found");
    }
}
