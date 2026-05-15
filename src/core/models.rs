use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

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

/// Serialised body sent to a chat-completion endpoint.
#[derive(Debug, Serialize, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// A single message in a conversation thread, compatible with the OpenAI chat format.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl Message {
    pub fn user(content: String) -> Self {
        Self {
            role: Role::User,
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn tool_result(tool_call_id: String, tool_name: String, content: String) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            tool_name: Some(tool_name),
        }
    }
}

/// A tool-call request emitted by the LLM inside an assistant message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Function name and JSON-encoded arguments from a tool call.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    /// JSON string of the arguments object.
    pub arguments: String,
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

/// Raw response from a chat-completion endpoint.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
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

/// Final output of a completed agent run.
#[derive(Debug, Serialize)]
pub struct AgentResult {
    pub final_response: String,
    pub steps: Vec<AgentStep>,
    pub iterations_used: usize,
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
        tool_name: String,
        arguments: String,
    },
    /// A tool has finished executing.
    #[serde(rename = "tool_result")]
    ToolResult { tool_name: String, result: String },
    /// A chunk of text from the LLM.
    #[serde(rename = "llm_response")]
    LlmResponse { content: String },
    /// The agent has finished; `final_response` is the complete answer.
    #[serde(rename = "finished")]
    Finished {
        final_response: String,
        iterations: usize,
    },
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
        let msg = Message::user("hello".into());
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.as_deref(), Some("hello"));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(msg.tool_name.is_none());
    }

    #[test]
    fn message_assistant_sets_correct_fields() {
        let msg = Message::assistant("response".into());
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.as_deref(), Some("response"));
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn message_tool_result_sets_correct_fields() {
        let msg = Message::tool_result("call_1".into(), "read_file".into(), "content".into());
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.content.as_deref(), Some("content"));
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msg.tool_name.as_deref(), Some("read_file"));
        assert!(msg.tool_calls.is_none());
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

    #[test]
    fn chat_request_skips_empty_tools() {
        let req = ChatRequest {
            model: "gpt-4".into(),
            messages: vec![Message::user("hi".into())],
            tools: vec![],
            max_tokens: None,
        };
        let json: Value = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none());
        assert!(json.get("max_tokens").is_none());
    }
}
