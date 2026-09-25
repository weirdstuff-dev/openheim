//! [`PermissionGate`] implementation that hands tool-call approval to the
//! interactive terminal UI instead of auto-allowing.
//!
//! The agent loop runs on a spawned task (see `tui::mod::run`), separate from
//! the render/input loop. `check()` sends a [`PendingPermission`] over a
//! channel to that render/input loop and blocks the agent task on a oneshot
//! reply, which `App` sends once the user picks an option.

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::core::permission::{PermissionDecision, PermissionGate, PermissionRequest};

/// Options shown in the permission prompt, in display/cycle order. Shared
/// between `app` (key handling) and `render` (the popup) so they can't drift.
pub(crate) const PERMISSION_OPTIONS: [(&str, PermissionDecision); 4] = [
    ("Allow Once", PermissionDecision::AllowOnce),
    ("Allow Always", PermissionDecision::AllowAlways),
    ("Reject Once", PermissionDecision::RejectOnce),
    ("Reject Always", PermissionDecision::RejectAlways),
];

/// One pending approval, sent from the agent task to the UI loop.
pub(crate) struct PendingPermission {
    pub(crate) tool_name: String,
    pub(crate) arguments: String,
    /// The `delegate_task` subagent that made the call, if any.
    pub(crate) subagent: Option<String>,
    pub(crate) respond_to: oneshot::Sender<PermissionDecision>,
}

impl PendingPermission {
    /// How the transcript refers to the call once it's answered.
    pub(crate) fn describe(&self) -> String {
        match &self.subagent {
            Some(subagent) => format!("'{}' from subagent '{subagent}'", self.tool_name),
            None => format!("'{}'", self.tool_name),
        }
    }
}

/// Only asks: "Allow Always"/"Reject Always" answers are remembered per
/// session by the runtime's `RememberingGate`, which wraps this gate.
pub(crate) struct TuiPermissionGate {
    pub(crate) tx: mpsc::UnboundedSender<PendingPermission>,
}

impl TuiPermissionGate {
    pub(crate) fn new(tx: mpsc::UnboundedSender<PendingPermission>) -> Self {
        Self { tx }
    }
}

#[async_trait]
impl PermissionGate for TuiPermissionGate {
    async fn check(&self, request: &PermissionRequest<'_>) -> PermissionDecision {
        let (respond_to, rx) = oneshot::channel();
        let request = PendingPermission {
            tool_name: request.tool_name.to_string(),
            arguments: request.arguments.to_string(),
            subagent: request.subagent.map(str::to_string),
            respond_to,
        };
        // Fail closed: if the UI loop is gone (shutting down) or drops the
        // request without answering (e.g. the app exited mid-prompt), don't
        // execute the tool call.
        if self.tx.send(request).is_err() {
            return PermissionDecision::RejectOnce;
        }
        rx.await.unwrap_or(PermissionDecision::RejectOnce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_returns_the_ui_loops_answer() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = TuiPermissionGate::new(tx);

        let check = tokio::spawn(async move {
            gate.check(&PermissionRequest::new("call_1", "read_file", "{}"))
                .await
        });

        let request = rx.recv().await.unwrap();
        assert_eq!(request.tool_name, "read_file");
        assert_eq!(request.subagent, None);
        let _ = request.respond_to.send(PermissionDecision::AllowAlways);

        assert_eq!(check.await.unwrap(), PermissionDecision::AllowAlways);
    }

    #[tokio::test]
    async fn check_passes_the_subagent_on_to_the_ui_loop() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = TuiPermissionGate::new(tx);

        tokio::spawn(async move {
            gate.check(
                &PermissionRequest::new("call_1", "read_file", "{}").from_subagent("reviewer"),
            )
            .await
        });

        let request = rx.recv().await.unwrap();
        assert_eq!(request.subagent.as_deref(), Some("reviewer"));
        assert_eq!(request.describe(), "'read_file' from subagent 'reviewer'");
    }

    #[tokio::test]
    async fn check_fails_closed_when_ui_loop_is_gone() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx); // simulate the UI loop having shut down
        let gate = TuiPermissionGate::new(tx);

        let decision = gate
            .check(&PermissionRequest::new("call_1", "execute_command", "{}"))
            .await;
        assert_eq!(decision, PermissionDecision::RejectOnce);
    }

    #[tokio::test]
    async fn check_fails_closed_when_request_is_dropped_unanswered() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = TuiPermissionGate::new(tx);

        let check = tokio::spawn(async move {
            gate.check(&PermissionRequest::new("call_1", "write_file", "{}"))
                .await
        });
        let request = rx.recv().await.unwrap();
        drop(request); // simulate the app exiting before answering

        assert_eq!(check.await.unwrap(), PermissionDecision::RejectOnce);
    }
}
