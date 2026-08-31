#[derive(Debug, Clone)]
pub(crate) enum AgentUpdate {
    TextChunk(String),
    ThinkingChunk(String),
    ToolCall {
        name: String,
        args: String,
    },
    ToolResult {
        result: String,
        is_error: bool,
    },
    Done,
    Error(String),
    ModelChanged {
        provider: String,
        model: String,
    },
    /// Current context size (the most recent LLM call's usage), refreshed
    /// after a completed turn or a session switch.
    Usage(crate::core::models::Usage),
}

#[derive(Debug, Clone)]
pub(crate) enum ChatItem {
    UserMessage(String),
    AssistantMessage(String),
    Thinking(String),
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
    SkillsViewer,
    McpViewer,
    ThemePicker,
    PermissionPrompt,
}

impl Screen {
    pub(crate) fn is_overlay(self) -> bool {
        matches!(
            self,
            Screen::ModelPicker
                | Screen::ConfigViewer
                | Screen::SessionPicker
                | Screen::SkillsViewer
                | Screen::McpViewer
                | Screen::ThemePicker
                | Screen::PermissionPrompt
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ConfigRow {
    Blank,
    Header(String),
    Entry { key: String, val: String },
    Item(String),
}
