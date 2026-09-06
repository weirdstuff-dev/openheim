#[derive(Debug, Clone)]
pub(crate) enum AgentUpdate {
    /// One raw event from a live turn — see
    /// `App::handle_stream_event` for which variants the UI reacts to.
    Stream(crate::core::models::StreamEvent),
    Error(String),
    ModelChanged {
        provider: String,
        model: String,
    },
    /// Current context size, refreshed on a session switch (a live turn's
    /// footer instead updates from `StreamEvent::Usage` as it streams).
    /// `None` clears the footer — e.g. switching to a session with no
    /// completed turn yet.
    Usage(Option<crate::core::models::Usage>),
    /// Answers a `:sessions` request — persisted conversation metadata,
    /// loaded off the UI task by the agent task.
    SessionList(Vec<crate::memory::ConversationMeta>),
    /// A batch of chat items replayed from a restored session's history,
    /// appended once the agent task's `SessionHandle::restore` finishes
    /// loading it (see `App::open_session`).
    History(Vec<ChatItem>),
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
