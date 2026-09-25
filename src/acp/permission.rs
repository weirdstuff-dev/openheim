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
    core::permission::{PermissionDecision, PermissionGate},
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
    async fn check(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision {
        let tool_call = ToolCallUpdate::new(
            tool_call_id.to_string(),
            ToolCallUpdateFields::new()
                .title(tool_name)
                .kind(tool_kind_for(tool_name, self.executor.as_ref()))
                .status(ToolCallStatus::Pending)
                .raw_input(raw_input(tool_call_id, tool_name, arguments)),
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
