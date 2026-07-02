//! [`PermissionGate`] implementation that hands tool-call approval to the
//! interactive terminal UI instead of auto-allowing.
//!
//! The agent loop runs on a spawned task (see `tui::mod::run`), separate from
//! the render/input loop. `check()` sends a [`PermissionRequest`] over a
//! channel to that render/input loop and blocks the agent task on a oneshot
//! reply, which `App` sends once the user picks an option.

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::core::permission::{PermissionDecision, PermissionGate};

/// Options shown in the permission prompt, in display/cycle order. Shared
/// between `app` (key handling) and `render` (the popup) so they can't drift.
pub(crate) const PERMISSION_OPTIONS: [(&str, PermissionDecision); 4] = [
    ("Allow Once", PermissionDecision::AllowOnce),
    ("Allow Always", PermissionDecision::AllowAlways),
    ("Reject Once", PermissionDecision::RejectOnce),
    ("Reject Always", PermissionDecision::RejectAlways),
];

/// One pending approval, sent from the agent task to the UI loop.
pub(crate) struct PermissionRequest {
    pub(crate) tool_name: String,
    pub(crate) arguments: String,
    pub(crate) respond_to: oneshot::Sender<PermissionDecision>,
}

pub(crate) struct TuiPermissionGate {
    pub(crate) tx: mpsc::UnboundedSender<PermissionRequest>,
}

#[async_trait]
impl PermissionGate for TuiPermissionGate {
    async fn check(
        &self,
        _tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision {
        let (respond_to, rx) = oneshot::channel();
        let request = PermissionRequest {
            tool_name: tool_name.to_string(),
            arguments: arguments.to_string(),
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
        let gate = TuiPermissionGate { tx };

        let check = tokio::spawn(async move { gate.check("call_1", "read_file", "{}").await });

        let request = rx.recv().await.unwrap();
        assert_eq!(request.tool_name, "read_file");
        let _ = request.respond_to.send(PermissionDecision::AllowAlways);

        assert_eq!(check.await.unwrap(), PermissionDecision::AllowAlways);
    }

    #[tokio::test]
    async fn check_fails_closed_when_ui_loop_is_gone() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx); // simulate the UI loop having shut down
        let gate = TuiPermissionGate { tx };

        let decision = gate.check("call_1", "execute_command", "{}").await;
        assert_eq!(decision, PermissionDecision::RejectOnce);
    }

    #[tokio::test]
    async fn check_fails_closed_when_request_is_dropped_unanswered() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = TuiPermissionGate { tx };

        let check = tokio::spawn(async move { gate.check("call_1", "write_file", "{}").await });
        let request = rx.recv().await.unwrap();
        drop(request); // simulate the app exiting before answering

        assert_eq!(check.await.unwrap(), PermissionDecision::RejectOnce);
    }
}
