#[derive(Debug, Clone)]
pub(crate) enum AgentUpdate {
    /// One raw event from a live turn — see
    /// `App::handle_stream_event` for which variants the UI reacts to.
    Stream(crate::core::models::StreamEvent),
    Error(String),
    /// A turn ended abnormally but without an error (truncated, refused,
    /// iteration limit, …); shown as a system line under the reply. See
    /// `StopReason::notice`.
    Notice(String),
    ModelChanged {
        provider: String,
        model: String,
    },
    /// The context size after a session switch; `None` clears the footer.
    Usage(Option<crate::core::models::Usage>),
    /// Answers a `:sessions` request — persisted conversation metadata,
    /// loaded off the UI task by the agent task.
    SessionList(Vec<crate::memory::ConversationMeta>),
    /// A restored session's history as chat items (see `App::open_session`).
    History(Vec<ChatItem>),
    /// A `:new` session was created (see `App::start_new_session`).
    NewSession(Vec<ChatItem>),
}

#[derive(Debug, Clone, PartialEq)]
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

/// The base screen, under any popup (see `state::Overlay`) or permission
/// prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Screen {
    Welcome,
    Chat,
}

#[derive(Debug, Clone)]
pub(crate) enum ConfigRow {
    Blank,
    Header(String),
    Entry { key: String, val: String },
    Item(String),
}
