//! Small pieces of ACP vocabulary shared across the `acp` submodules:
//! session modes, stop-reason/tool-kind mapping, history replay, and the
//! `StreamEvent → SessionUpdate` mapping for a live turn.

use agent_client_protocol::schema::v1::{
    ContentBlock as AcpContentBlock, ContentChunk, ImageContent, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOption, SessionInfo, SessionMode,
    SessionModeState, SessionUpdate, StopReason, TextContent, ToolCall as AcpToolCall,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};

use crate::{
    config::AppConfig,
    core::{
        models::{Message, StopReason as CoreStopReason, StreamEvent, TranscriptEntry},
        runtime::AgentMode,
    },
    memory::ConversationMeta,
    tools::{ToolExecutor, ToolKindHint},
};

/// `session/set_config_option` id for the model selector.
pub(super) const MODEL_CONFIG_ID: &str = "model";

/// Every model configured across every provider, tagged with `provider` in
/// each entry's `_meta` — the `session/new`, `session/load`, and
/// `session/set_config_option` shape for advertising available models as a
/// `select`-kind session config option.
pub(super) fn session_model_config_option(
    app_config: &AppConfig,
    current_model: &str,
) -> SessionConfigOption {
    let available_models: Vec<SessionConfigSelectOption> = app_config
        .providers
        .iter()
        .flat_map(|(provider_name, p)| {
            p.models.iter().map(move |m| {
                let mut meta = serde_json::Map::new();
                meta.insert(
                    "provider".to_string(),
                    serde_json::Value::String(provider_name.clone()),
                );
                SessionConfigSelectOption::new(m.clone(), m.clone()).meta(meta)
            })
        })
        .collect();
    SessionConfigOption::select(
        MODEL_CONFIG_ID,
        "Model",
        current_model.to_string(),
        available_models,
    )
    .category(SessionConfigOptionCategory::Model)
}

/// Maps persisted session metadata onto the ACP `SessionInfo` shape
/// `session/list` responds with.
pub(crate) fn conversation_metas_to_session_info(metas: Vec<ConversationMeta>) -> Vec<SessionInfo> {
    metas
        .into_iter()
        .map(|m| {
            let path = m.cwd.unwrap_or_else(|| std::path::PathBuf::from("/"));
            let mut info = SessionInfo::new(m.id.to_string(), path);
            if let Some(t) = m.title {
                info = info.title(t);
            }
            info.updated_at(m.updated_at.to_rfc3339())
        })
        .collect()
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
/// thinking blocks, which are tunneled through `agent_message_chunk` with
/// `content._meta.kind == "thinking"` exactly as the live streaming path
/// does. Which blocks are replayed, and in what order, is
/// [`Message::transcript`]'s call (shared with the TUI); this only picks
/// the `SessionUpdate`.
pub(crate) fn replay_history_messages<F>(
    messages: &[Message],
    executor: &dyn ToolExecutor,
    on_update: &mut F,
) where
    F: FnMut(SessionUpdate),
{
    for entry in messages.iter().flat_map(Message::transcript) {
        let update = match entry {
            TranscriptEntry::UserText(text) => SessionUpdate::UserMessageChunk(ContentChunk::new(
                AcpContentBlock::from(text.to_string()),
            )),
            TranscriptEntry::UserImage { data, mime_type } => {
                SessionUpdate::UserMessageChunk(ContentChunk::new(AcpContentBlock::Image(
                    ImageContent::new(data.to_string(), mime_type.to_string()),
                )))
            }
            TranscriptEntry::Thinking(thinking) => SessionUpdate::AgentMessageChunk(
                ContentChunk::new(AcpContentBlock::Text(thinking_chunk(thinking.to_string()))),
            ),
            TranscriptEntry::AssistantText(text) => SessionUpdate::AgentMessageChunk(
                ContentChunk::new(AcpContentBlock::from(text.to_string())),
            ),
            TranscriptEntry::ToolCall {
                id,
                name,
                arguments,
            } => SessionUpdate::ToolCall(
                AcpToolCall::new(id.to_string(), name)
                    .kind(tool_kind_for(name, executor))
                    .status(ToolCallStatus::InProgress)
                    .raw_input(raw_input(id, name, arguments)),
            ),
            TranscriptEntry::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let status = if is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    tool_call_id.to_string(),
                    ToolCallUpdateFields::new()
                        .status(status)
                        .raw_output(serde_json::Value::String(content.to_string())),
                ))
            }
        };
        on_update(update);
    }
}

/// Maps a [`ToolKindHint`] (a tool's own [`capabilities()`](crate::tools::ToolHandler::capabilities)
/// declaration) onto the closest ACP [`ToolKind`], purely for client UI
/// treatment (icons etc.) — has no bearing on execution.
pub(super) fn acp_tool_kind(hint: ToolKindHint) -> ToolKind {
    match hint {
        ToolKindHint::Read => ToolKind::Read,
        ToolKindHint::Edit => ToolKind::Edit,
        ToolKindHint::Delete => ToolKind::Delete,
        ToolKindHint::Move => ToolKind::Move,
        ToolKindHint::Search => ToolKind::Search,
        ToolKindHint::Execute => ToolKind::Execute,
        ToolKindHint::Think => ToolKind::Think,
        ToolKindHint::Fetch => ToolKind::Fetch,
        ToolKindHint::SwitchMode => ToolKind::SwitchMode,
        ToolKindHint::Other => ToolKind::Other,
    }
}

/// Looks up `tool_name`'s declared [`ToolKindHint`] in `executor` and maps it
/// to ACP's [`ToolKind`]. Unregistered names (e.g. replaying history from a
/// tool that's no longer configured) fall back to [`ToolKind::Other`] — see
/// [`crate::tools::ToolExecutor::capabilities`]'s default.
pub(super) fn tool_kind_for(tool_name: &str, executor: &dyn ToolExecutor) -> ToolKind {
    acp_tool_kind(executor.capabilities(tool_name).kind)
}

#[cfg(test)]
mod tool_kind_tests {
    use super::*;
    use crate::tools::SystemToolExecutor;

    fn executor() -> SystemToolExecutor {
        let mut e = SystemToolExecutor::new();
        e.register_builtins(true);
        e
    }

    #[test]
    fn maps_every_builtin_tool() {
        let e = executor();
        assert_eq!(tool_kind_for("execute_command", &e), ToolKind::Execute);
        assert_eq!(tool_kind_for("read_file", &e), ToolKind::Read);
        assert_eq!(tool_kind_for("write_file", &e), ToolKind::Edit);
        assert_eq!(tool_kind_for("edit_file", &e), ToolKind::Edit);
        assert_eq!(tool_kind_for("list_dir", &e), ToolKind::Read);
        assert_eq!(tool_kind_for("search", &e), ToolKind::Search);
        assert_eq!(tool_kind_for("web_fetch", &e), ToolKind::Fetch);
    }

    #[test]
    fn unknown_tool_falls_back_to_other() {
        let e = executor();
        assert_eq!(
            tool_kind_for("some_mcp_server__custom_tool", &e),
            ToolKind::Other
        );
    }
}

/// A tool call's JSON arguments as ACP `raw_input`. Malformed arguments are
/// the model's mistake, and the tool reports them back to it, so here they
/// are logged and shown as no input rather than failing the update. The one
/// place ACP decodes arguments, so every update applies the same policy.
pub(super) fn raw_input(
    tool_call_id: &str,
    tool_name: &str,
    arguments: &str,
) -> Option<serde_json::Value> {
    match serde_json::from_str(arguments) {
        Ok(value) => Some(value),
        Err(e) => {
            tracing::warn!(
                tool_call_id,
                tool_name,
                "failed to parse tool call arguments: {e}"
            );
            None
        }
    }
}

/// Maps one core [`StreamEvent`] from a live turn onto the [`SessionUpdate`]
/// it corresponds to, if any. `IterationStart`, `Usage`, `Finished`, and
/// `MessageAppended` have no ACP wire equivalent — they're `AgentState`-
/// internal or ACP-client-facing-nothing signals (history persistence,
/// context-size bookkeeping, turn-done) — and map to `None`.
pub(crate) fn stream_event_to_session_update(
    event: StreamEvent,
    executor: &dyn ToolExecutor,
) -> Option<SessionUpdate> {
    match event {
        StreamEvent::LlmResponse { content } => Some(SessionUpdate::AgentMessageChunk(
            ContentChunk::new(AcpContentBlock::from(content)),
        )),
        StreamEvent::ThinkingContent { content } => Some(SessionUpdate::AgentMessageChunk(
            ContentChunk::new(AcpContentBlock::Text(thinking_chunk(content))),
        )),
        StreamEvent::ToolCall {
            id,
            tool_name,
            arguments,
        } => {
            // Pending, not InProgress: the permission gate (invoked by the
            // agent loop right after this event) hasn't authorized
            // execution yet at this point.
            let raw_input = raw_input(&id, &tool_name, &arguments);
            Some(SessionUpdate::ToolCall(
                AcpToolCall::new(id, &*tool_name)
                    .kind(tool_kind_for(&tool_name, executor))
                    .status(ToolCallStatus::Pending)
                    .raw_input(raw_input),
            ))
        }
        StreamEvent::ToolResult {
            id,
            result,
            is_error,
            ..
        } => {
            let status = if is_error {
                ToolCallStatus::Failed
            } else {
                ToolCallStatus::Completed
            };
            Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(status)
                    .raw_output(serde_json::Value::String(result)),
            )))
        }
        StreamEvent::IterationStart { .. }
        | StreamEvent::Usage { .. }
        | StreamEvent::Finished { .. }
        | StreamEvent::MessageAppended { .. } => None,
    }
}

/// Maps core's own [`CoreStopReason`] onto ACP's `StopReason`.
pub(super) fn map_stop_reason(reason: CoreStopReason) -> StopReason {
    match reason {
        CoreStopReason::EndTurn => StopReason::EndTurn,
        CoreStopReason::MaxIterations => StopReason::MaxTurnRequests,
        CoreStopReason::MaxTokens => StopReason::MaxTokens,
        CoreStopReason::Refusal => StopReason::Refusal,
        CoreStopReason::Cancelled => StopReason::Cancelled,
        // ACP has no "the model returned nothing usable" variant; `EndTurn`
        // is the least misleading fit (it's not cancellation or exhaustion).
        CoreStopReason::NoContent => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use crate::core::models::{ContentBlock, Role};
    use crate::tools::SystemToolExecutor;

    fn executor() -> SystemToolExecutor {
        let mut e = SystemToolExecutor::new();
        e.register_builtins(true);
        e
    }

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
        replay_history_messages(&messages, &executor(), &mut |u| updates.push(u));

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

    // Interleaved thinking stores several thinking blocks between text and
    // tool calls; replay follows the stored order instead of grouping them.
    #[test]
    fn replay_keeps_interleaved_blocks_in_order() {
        let thinking = |t: &str| ContentBlock::Thinking {
            thinking: t.into(),
            signature: Some("sig".into()),
        };
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                thinking("first"),
                ContentBlock::from("Let me look."),
                ContentBlock::RedactedThinking { data: "enc".into() },
                thinking("second"),
                ContentBlock::tool_use("toolu_1", "read_file", "{}"),
            ],
        }];
        let mut updates = Vec::new();
        replay_history_messages(&messages, &executor(), &mut |u| updates.push(u));

        let replayed: Vec<String> = updates
            .iter()
            .map(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    AcpContentBlock::Text(t) if t.meta.is_some() => format!("thinking:{}", t.text),
                    AcpContentBlock::Text(t) => format!("text:{}", t.text),
                    other => panic!("unexpected chunk {other:?}"),
                },
                SessionUpdate::ToolCall(call) => format!("tool:{}", call.title),
                other => panic!("unexpected update {other:?}"),
            })
            .collect();
        // Redacted thinking has nothing to show.
        assert_eq!(
            replayed,
            [
                "thinking:first",
                "text:Let me look.",
                "thinking:second",
                "tool:read_file"
            ]
        );
    }

    #[test]
    fn replay_still_emits_user_text_and_tool_calls() {
        let messages = vec![
            Message::user("hello"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::tool_use(
                    "call_1",
                    "read_file",
                    r#"{"path":"a.txt"}"#,
                )],
            },
            Message::tool_result("call_1", "read_file", "file content", false),
        ];
        let mut updates = Vec::new();
        replay_history_messages(&messages, &executor(), &mut |u| updates.push(u));

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
        replay_history_messages(&messages, &executor(), &mut |u| updates.push(u));

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

#[cfg(test)]
mod stream_event_tests {
    use super::*;
    use crate::core::models::Usage;
    use crate::tools::SystemToolExecutor;

    fn executor() -> SystemToolExecutor {
        let mut e = SystemToolExecutor::new();
        e.register_builtins(true);
        e
    }

    #[test]
    fn llm_response_becomes_agent_message_chunk() {
        let update = stream_event_to_session_update(
            StreamEvent::LlmResponse {
                content: "hi".into(),
            },
            &executor(),
        );
        assert!(matches!(
            update,
            Some(SessionUpdate::AgentMessageChunk(c)) if matches!(&c.content, AcpContentBlock::Text(t) if t.text == "hi")
        ));
    }

    #[test]
    fn thinking_content_is_tagged_via_meta() {
        let update = stream_event_to_session_update(
            StreamEvent::ThinkingContent {
                content: "pondering".into(),
            },
            &executor(),
        );
        match update {
            Some(SessionUpdate::AgentMessageChunk(c)) => match c.content {
                AcpContentBlock::Text(t) => {
                    assert_eq!(t.text, "pondering");
                    assert_eq!(
                        t.meta.as_ref().and_then(|m| m.get("kind")),
                        Some(&serde_json::json!("thinking"))
                    );
                }
                other => panic!("expected a text block, got {other:?}"),
            },
            other => panic!("expected an AgentMessageChunk, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_and_tool_result_map_to_their_acp_shapes() {
        let e = executor();
        let call = stream_event_to_session_update(
            StreamEvent::ToolCall {
                id: "call_1".into(),
                tool_name: "read_file".into(),
                arguments: r#"{"path":"a.txt"}"#.into(),
            },
            &e,
        );
        assert!(matches!(
            call,
            Some(SessionUpdate::ToolCall(tc)) if tc.raw_input.is_some()
        ));

        let result = stream_event_to_session_update(
            StreamEvent::ToolResult {
                id: "call_1".into(),
                tool_name: "read_file".into(),
                result: "contents".into(),
                is_error: false,
            },
            &e,
        );
        assert!(matches!(
            result,
            Some(SessionUpdate::ToolCallUpdate(u)) if u.fields.status == Some(ToolCallStatus::Completed)
        ));
    }

    #[test]
    fn events_with_no_acp_equivalent_map_to_none() {
        let e = executor();
        assert!(
            stream_event_to_session_update(StreamEvent::IterationStart { iteration: 1 }, &e)
                .is_none()
        );
        assert!(
            stream_event_to_session_update(
                StreamEvent::Usage {
                    usage: Usage::default()
                },
                &e
            )
            .is_none()
        );
        assert!(
            stream_event_to_session_update(
                StreamEvent::Finished {
                    final_response: "done".into(),
                    iterations: 1,
                },
                &e
            )
            .is_none()
        );
        assert!(
            stream_event_to_session_update(
                StreamEvent::MessageAppended {
                    message: Message::user("hi"),
                },
                &e
            )
            .is_none()
        );
    }
}
