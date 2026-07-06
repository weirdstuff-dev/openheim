//! [`PermissionGate`] implementation that hands tool-call approval to the
//! interactive terminal UI instead of auto-allowing.
//!
//! The agent loop runs on a spawned task (see `tui::mod::run`), separate from
//! the render/input loop. `check()` sends a [`PermissionRequest`] over a
//! channel to that render/input loop and blocks the agent task on a oneshot
//! reply, which `App` sends once the user picks an option.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::core::permission::{PermissionDecision, PermissionGate, approval_key};

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
    /// Remembered `AllowAlways`/`RejectAlways` decisions, keyed by
    /// [`approval_key`], so picking "Allow Always" actually sticks for the
    /// rest of the session instead of re-prompting on every matching call.
    approved_tools: Mutex<HashMap<String, PermissionDecision>>,
}

impl TuiPermissionGate {
    pub(crate) fn new(tx: mpsc::UnboundedSender<PermissionRequest>) -> Self {
        Self {
            tx,
            approved_tools: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl PermissionGate for TuiPermissionGate {
    async fn check(
        &self,
        _tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision {
        let key = approval_key(tool_name, arguments);
        if let Some(remembered) = self.approved_tools.lock().unwrap().get(&key).copied() {
            return remembered;
        }

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
        let decision = rx.await.unwrap_or(PermissionDecision::RejectOnce);

        if matches!(
            decision,
            PermissionDecision::AllowAlways | PermissionDecision::RejectAlways
        ) {
            self.approved_tools.lock().unwrap().insert(key, decision);
        }

        decision
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn check_returns_the_ui_loops_answer() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = TuiPermissionGate::new(tx);

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
        let gate = TuiPermissionGate::new(tx);

        let decision = gate.check("call_1", "execute_command", "{}").await;
        assert_eq!(decision, PermissionDecision::RejectOnce);
    }

    #[tokio::test]
    async fn check_fails_closed_when_request_is_dropped_unanswered() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = TuiPermissionGate::new(tx);

        let check = tokio::spawn(async move { gate.check("call_1", "write_file", "{}").await });
        let request = rx.recv().await.unwrap();
        drop(request); // simulate the app exiting before answering

        assert_eq!(check.await.unwrap(), PermissionDecision::RejectOnce);
    }

    #[tokio::test]
    async fn allow_always_is_remembered_for_later_calls_to_the_same_tool() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = Arc::new(TuiPermissionGate::new(tx));

        let first_gate = gate.clone();
        let first =
            tokio::spawn(async move { first_gate.check("call_1", "read_file", "{}").await });
        let request = rx.recv().await.unwrap();
        let _ = request.respond_to.send(PermissionDecision::AllowAlways);
        assert_eq!(first.await.unwrap(), PermissionDecision::AllowAlways);

        // Second call for the same tool must not go back to the UI loop: no
        // request arrives, and the cached decision is returned directly.
        let second = gate.check("call_2", "read_file", "{}").await;
        assert_eq!(second, PermissionDecision::AllowAlways);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn allow_once_is_not_remembered() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let gate = Arc::new(TuiPermissionGate::new(tx));

        let first_gate = gate.clone();
        let first =
            tokio::spawn(async move { first_gate.check("call_1", "read_file", "{}").await });
        let request = rx.recv().await.unwrap();
        let _ = request.respond_to.send(PermissionDecision::AllowOnce);
        assert_eq!(first.await.unwrap(), PermissionDecision::AllowOnce);

        // A second call must prompt again since only Allow Always is sticky.
        let second_gate = gate.clone();
        let second =
            tokio::spawn(async move { second_gate.check("call_2", "read_file", "{}").await });
        let request = rx.recv().await.unwrap();
        let _ = request.respond_to.send(PermissionDecision::AllowOnce);
        assert_eq!(second.await.unwrap(), PermissionDecision::AllowOnce);
    }
}
