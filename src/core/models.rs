use serde::{Deserialize, Serialize};
use serde_json::Value;

// Chat API Models

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

// Agent Result Models
#[derive(Debug, Serialize, Clone)]
pub struct AgentStep {
    pub iteration: usize,
    pub message: String,
    pub tool_calls: Option<Vec<ToolExecutionResult>>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolExecutionResult {
    pub tool_name: String,
    pub arguments: String,
    pub result: String,
}

#[derive(Debug, Serialize)]
pub struct AgentResult {
    pub final_response: String,
    pub steps: Vec<AgentStep>,
    pub iterations_used: usize,
}

// Streaming Events
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "event_type")]
pub enum StreamEvent {
    #[serde(rename = "iteration_start")]
    IterationStart { iteration: usize },
    #[serde(rename = "tool_call")]
    ToolCall {
        tool_name: String,
        arguments: String,
    },
    #[serde(rename = "tool_result")]
    ToolResult { tool_name: String, result: String },
    #[serde(rename = "llm_response")]
    LlmResponse { content: String },
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
    List { path: String, recursive: Option<bool> },

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
    FileList { path: String, entries: Vec<FileEntry> },

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
    FsEvent { event_kind: String, paths: Vec<String> },

    #[serde(rename = "error")]
    Error { message: String },
}
