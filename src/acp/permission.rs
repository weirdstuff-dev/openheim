//! [`PermissionGate`] backed by ACP's `session/request_permission`.

use std::sync::Arc;

use agent_client_protocol::{
    Client, ConnectionTo,
    schema::{
        PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
        RequestPermissionResponse, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
    },
};

use crate::core::permission::{PermissionDecision, PermissionGate, approval_key};

use super::{AgentState, util::tool_kind_for};

/// Lives here (not in `core`) because it depends on the live client connection.
pub(super) struct AcpPermissionGate {
    pub(super) cx: ConnectionTo<Client>,
    pub(super) session_id: String,
    pub(super) state: Arc<AgentState>,
}

#[async_trait::async_trait]
impl PermissionGate for AcpPermissionGate {
    async fn check(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision {
        let key = approval_key(tool_name, arguments);
        if let Some(remembered) = self
            .state
            .sessions
            .read()
            .await
            .get(&self.session_id)
            .and_then(|s| s.approved_tools.get(&key).copied())
        {
            return remembered;
        }

        let raw_input = serde_json::from_str(arguments).ok();
        let tool_call = ToolCallUpdate::new(
            tool_call_id.to_string(),
            ToolCallUpdateFields::new()
                .title(tool_name)
                .kind(tool_kind_for(tool_name))
                .status(ToolCallStatus::Pending)
                .raw_input(raw_input),
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

        let decision = match response {
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
        };

        if matches!(
            decision,
            PermissionDecision::AllowAlways | PermissionDecision::RejectAlways
        ) && let Some(s) = self.state.sessions.write().await.get_mut(&self.session_id)
        {
            s.approved_tools.insert(key, decision);
        }

        decision
    }
}
