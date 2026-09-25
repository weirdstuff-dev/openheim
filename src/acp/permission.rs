//! [`PermissionGate`] backed by ACP's `session/request_permission`.

use std::sync::Arc;

use agent_client_protocol::{
    Client, ConnectionTo,
    schema::v1::{
        PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
        RequestPermissionResponse, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
    },
};

use crate::{
    core::permission::{PermissionDecision, PermissionGate, PermissionRequest},
    tools::ToolExecutor,
};

use super::util::{raw_input, tool_kind_for};

/// Lives here (not in `core`) because it depends on the live client connection.
/// Only asks: remembering `*Always` answers is `RememberingGate`'s job, which
/// `AgentState::prompt` wraps around this.
pub(super) struct AcpPermissionGate {
    pub(super) cx: ConnectionTo<Client>,
    pub(super) session_id: String,
    /// For the tool kind shown in the permission prompt.
    pub(super) executor: Arc<dyn ToolExecutor>,
}

#[async_trait::async_trait]
impl PermissionGate for AcpPermissionGate {
    async fn check(&self, request: &PermissionRequest<'_>) -> PermissionDecision {
        let tool_name = request.tool_name;
        let tool_call = ToolCallUpdate::new(
            client_tool_call_id(request),
            ToolCallUpdateFields::new()
                .title(permission_title(request))
                .kind(tool_kind_for(tool_name, self.executor.as_ref()))
                .status(ToolCallStatus::Pending)
                .raw_input(raw_input(
                    request.tool_call_id,
                    tool_name,
                    request.arguments,
                )),
        );
        let options = vec![
            PermissionOption::new("allow_once", "Allow Once", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                "allow_always",
                "Allow Always",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(
                "reject_once",
                "Reject Once",
                PermissionOptionKind::RejectOnce,
            ),
            PermissionOption::new(
                "reject_always",
                "Reject Always",
                PermissionOptionKind::RejectAlways,
            ),
        ];

        let response = self
            .cx
            .send_request(RequestPermissionRequest::new(
                self.session_id.clone(),
                tool_call,
                options,
            ))
            .block_task()
            .await;

        match response {
            Ok(RequestPermissionResponse {
                outcome: RequestPermissionOutcome::Selected(selected),
                ..
            }) => match selected.option_id.0.as_ref() {
                "allow_once" => PermissionDecision::AllowOnce,
                "allow_always" => PermissionDecision::AllowAlways,
                "reject_always" => PermissionDecision::RejectAlways,
                _ => PermissionDecision::RejectOnce,
            },
            Ok(RequestPermissionResponse {
                outcome: RequestPermissionOutcome::Cancelled,
                ..
            }) => PermissionDecision::RejectOnce,
            // `RequestPermissionOutcome` is #[non_exhaustive]; treat any future
            // variant conservatively, same as an explicit rejection.
            Ok(_) => PermissionDecision::RejectOnce,
            Err(e) => {
                tracing::warn!("session/request_permission failed: {e}");
                PermissionDecision::RejectOnce
            }
        }
    }
}

/// The prompt's title: the tool name, plus the subagent that wants it run.
/// A subagent's tool calls are never sent as `session/update`s, so the title
/// is the only place the client learns where the call came from.
fn permission_title(request: &PermissionRequest<'_>) -> String {
    match request.subagent {
        Some(subagent) => format!("{} (subagent '{subagent}')", request.tool_name),
        None => request.tool_name.to_string(),
    }
}

/// The tool call id the client sees. A subagent's ids come from its own
/// conversation and can repeat one of the parent's, so they're prefixed with
/// the subagent's name to keep the client from mixing the two calls up.
fn client_tool_call_id(request: &PermissionRequest<'_>) -> String {
    match request.subagent {
        Some(subagent) => format!("subagent:{subagent}:{}", request.tool_call_id),
        None => request.tool_call_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_names_the_subagent_that_asked() {
        let request = PermissionRequest::new("call_1", "execute_command", "{}");
        assert_eq!(permission_title(&request), "execute_command");
        assert_eq!(
            permission_title(&request.from_subagent("reviewer")),
            "execute_command (subagent 'reviewer')"
        );
    }

    #[test]
    fn subagent_tool_call_ids_are_kept_apart_from_the_parents() {
        let request = PermissionRequest::new("call_1", "execute_command", "{}");
        assert_eq!(client_tool_call_id(&request), "call_1");
        let request = PermissionRequest::new("call_1", "execute_command", "{}");
        assert_eq!(
            client_tool_call_id(&request.from_subagent("reviewer")),
            "subagent:reviewer:call_1"
        );
    }
}
