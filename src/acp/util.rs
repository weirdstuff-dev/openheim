//! Small pieces of ACP vocabulary shared across the `acp` submodules:
//! session modes, stop-reason/tool-kind mapping, and history replay.

use agent_client_protocol::schema::{
    ContentBlock as AcpContentBlock, ContentChunk, ImageContent, SessionMode, SessionModeState,
    SessionUpdate, StopReason, TextContent, ToolCall as AcpToolCall, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};

use crate::{
    core::models::{ContentBlock, Message, Role, StopReason as CoreStopReason},
    error::{Error, Result},
};

/// Which tool policy a session runs under, set via `session/set_mode`.
/// [`Self::as_str`] gives the ACP wire-level mode id; [`Self::parse`] is the
/// inverse, for the boundary where that id arrives as a `&str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentMode {
    /// Full tool access; tool calls go through the permission gate as normal.
    #[default]
    Code,
    /// Read-only: only `read_file` and `list_dir` are offered to the LLM, so
    /// nothing mutating can run. Both still go through the permission gate
    /// and can trigger a `session/request_permission` prompt unless already
    /// approved.
    Architect,
}

impl AgentMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            AgentMode::Code => "code",
            AgentMode::Architect => "architect",
        }
    }

    pub fn parse(mode_id: &str) -> Result<Self> {
        match mode_id {
            "code" => Ok(AgentMode::Code),
            "architect" => Ok(AgentMode::Architect),
            other => Err(Error::ParseError(format!("unknown session mode: {other}"))),
        }
    }
}

pub(super) fn session_mode_state(current_mode: AgentMode) -> SessionModeState {
    SessionModeState::new(
        current_mode.as_str().to_string(),
        vec![
            SessionMode::new(AgentMode::Code.as_str(), "Code")
                .description("Full tool access; tool calls request permission."),
            SessionMode::new(AgentMode::Architect.as_str(), "Architect")
                .description("Read-only: inspects and plans without editing or executing."),
        ],
    )
}

/// Wraps reasoning text in a plain text block tagged `_meta.kind == "thinking"`
/// — the tunnel ACP uses for thinking content (ACP's own content model has no
/// thinking variant; the `thinking` entry in the session metadata advertised
/// by `initialize` documents this convention for clients).
pub(super) fn thinking_chunk(content: String) -> TextContent {
    let mut meta = serde_json::Map::new();
    meta.insert(
        "kind".to_string(),
        serde_json::Value::String("thinking".to_string()),
    );
    TextContent::new(content).meta(meta)
}

/// Replays persisted history to a (re)attaching connection as the same
/// stream of session updates a live turn would have produced, so a reloaded
/// session renders identically to one that stayed open — including assistant
/// thinking blocks, which lead the persisted content (`[Thinking?, Text?,
/// ToolUse*]`) and are tunneled through `agent_message_chunk` with
/// `content._meta.kind == "thinking"` exactly as the live streaming path
/// does. Without this, thinking shown during a turn vanished on reload even
/// though it was persisted.
pub(super) fn replay_history_messages<F>(messages: &[Message], on_update: &mut F)
where
    F: FnMut(SessionUpdate),
{
    for msg in messages {
        match msg.role {
            Role::User => {
                // Iterate every persisted block in order (not just `msg.text()`,
                // which only concatenates `Text` blocks) so an image attached
                // to the prompt is restored alongside the text instead of
                // silently dropped on reload.
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            on_update(SessionUpdate::UserMessageChunk(ContentChunk::new(
                                AcpContentBlock::from(text.clone()),
                            )));
                        }
                        ContentBlock::Image { data, mime_type } => {
                            on_update(SessionUpdate::UserMessageChunk(ContentChunk::new(
                                AcpContentBlock::Image(ImageContent::new(
                                    data.clone(),
                                    mime_type.clone(),
                                )),
                            )));
                        }
                        _ => {}
                    }
                }
            }
            Role::Assistant => {
                for block in &msg.content {
                    if let ContentBlock::Thinking { thinking, .. } = block {
                        on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                            AcpContentBlock::Text(thinking_chunk(thinking.clone())),
                        )));
                    }
                }
                if let Some(text) = msg.text() {
                    on_update(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        AcpContentBlock::from(text),
                    )));
                }
                for tc in msg.tool_calls() {
                    let raw_input = match serde_json::from_str(&tc.arguments) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            tracing::warn!(
                                tool_call_id = %tc.id,
                                tool_name = %tc.name,
                                "failed to parse tool call arguments: {e}"
                            );
                            None
                        }
                    };
                    on_update(SessionUpdate::ToolCall(
                        AcpToolCall::new(tc.id.clone(), &tc.name)
                            .kind(tool_kind_for(&tc.name))
                            .status(ToolCallStatus::InProgress)
                            .raw_input(raw_input),
                    ));
                }
            }
            Role::Tool => {
                if let Some(tr) = msg.tool_result_block() {
                    let status = if tr.is_error {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    };
                    on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                        tr.tool_call_id,
                        ToolCallUpdateFields::new()
                            .status(status)
                            .raw_output(serde_json::Value::String(tr.content)),
                    )));
                }
            }
            _ => {}
        }
    }
}

/// Maps a builtin/MCP tool name to the closest ACP [`ToolKind`], purely for
/// client UI treatment (icons etc.) — has no bearing on execution.
pub(super) fn tool_kind_for(tool_name: &str) -> ToolKind {
    match tool_name {
        "execute_command" => ToolKind::Execute,
        "read_file" => ToolKind::Read,
        "write_file" => ToolKind::Edit,
        _ => ToolKind::Other,
    }
}

/// Maps core's own [`CoreStopReason`] onto ACP's `StopReason`.
pub(super) fn map_stop_reason(reason: CoreStopReason) -> StopReason {
    match reason {
        CoreStopReason::EndTurn => StopReason::EndTurn,
        CoreStopReason::MaxIterations => StopReason::MaxTurnRequests,
        CoreStopReason::Cancelled => StopReason::Cancelled,
        // ACP has no "the model returned nothing usable" variant; `EndTurn`
        // is the least misleading fit (it's not cancellation or exhaustion).
        CoreStopReason::NoContent => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    fn agent_text_chunks(updates: &[SessionUpdate]) -> Vec<&ContentChunk> {
        updates
            .iter()
            .filter_map(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => Some(chunk),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn replay_emits_thinking_before_text_for_assistant_messages() {
        let messages = vec![
            Message::user("hello"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "pondering".into(),
                        signature: None,
                    },
                    ContentBlock::Text {
                        text: "the answer".into(),
                    },
                ],
            },
        ];
        let mut updates = Vec::new();
        replay_history_messages(&messages, &mut |u| updates.push(u));

        let chunks = agent_text_chunks(&updates);
        assert_eq!(chunks.len(), 2);

        match &chunks[0].content {
            AcpContentBlock::Text(t) => {
                assert_eq!(t.text, "pondering");
                assert_eq!(
                    t.meta.as_ref().and_then(|m| m.get("kind")),
                    Some(&serde_json::json!("thinking"))
                );
            }
            other => panic!("expected a text block, got {other:?}"),
        }
        match &chunks[1].content {
            AcpContentBlock::Text(t) => {
                assert_eq!(t.text, "the answer");
                assert!(t.meta.is_none());
            }
            other => panic!("expected a text block, got {other:?}"),
        }
    }

    #[test]
    fn replay_still_emits_user_text_and_tool_calls() {
        let messages = vec![
            Message::user("hello"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                }],
            },
            Message::tool_result("call_1", "read_file", "file content", false),
        ];
        let mut updates = Vec::new();
        replay_history_messages(&messages, &mut |u| updates.push(u));

        assert!(matches!(
            &updates[0],
            SessionUpdate::UserMessageChunk(c) if matches!(&c.content, AcpContentBlock::Text(t) if t.text == "hello")
        ));
        assert!(matches!(
            &updates[1],
            SessionUpdate::ToolCall(tc) if tc.raw_input.is_some()
        ));
        assert!(matches!(&updates[2], SessionUpdate::ToolCallUpdate(_)));
    }

    #[test]
    fn replay_restores_user_text_and_image_blocks_in_order() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "check this out".into(),
                },
                ContentBlock::Image {
                    data: "base64data".into(),
                    mime_type: "image/png".into(),
                },
            ],
        }];
        let mut updates = Vec::new();
        replay_history_messages(&messages, &mut |u| updates.push(u));

        assert_eq!(updates.len(), 2);
        assert!(matches!(
            &updates[0],
            SessionUpdate::UserMessageChunk(c) if matches!(&c.content, AcpContentBlock::Text(t) if t.text == "check this out")
        ));
        assert!(matches!(
            &updates[1],
            SessionUpdate::UserMessageChunk(c) if matches!(
                &c.content,
                AcpContentBlock::Image(img) if img.data == "base64data" && img.mime_type == "image/png"
            )
        ));
    }
}
