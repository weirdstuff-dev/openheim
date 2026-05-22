#[derive(Debug, Clone)]
pub(crate) enum AgentUpdate {
    TextChunk(String),
    ToolCall { name: String, args: String },
    ToolResult { result: String, is_error: bool },
    Done,
    Error(String),
    ModelChanged { provider: String, model: String },
}

#[derive(Debug, Clone)]
pub(crate) enum ChatItem {
    UserMessage(String),
    AssistantMessage(String),
    ToolCall { name: String, args: String },
    ToolResult { result: String, is_error: bool },
    SystemInfo(String),
    Err(String),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Status {
    Idle,
    Thinking,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Screen {
    Welcome,
    Chat,
    ModelPicker,
    ConfigViewer,
    SessionPicker,
}

#[derive(Debug, Clone)]
pub(crate) enum ConfigRow {
    Blank,
    Header(String),
    Entry { key: String, val: String },
    Item(String),
}
