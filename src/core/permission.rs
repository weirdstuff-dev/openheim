//! Tool-call authorization: an embedder-supplied hook the agent loop consults
//! before executing any tool call the LLM requests.
//!
//! This mirrors [`crate::core::llm::LlmClient`] and [`crate::tools::ToolExecutor`]:
//! a protocol-agnostic trait defined here, with the ACP-specific implementation
//! (backed by `session/request_permission`) living in [`crate::acp`].

use async_trait::async_trait;

/// The user's (or embedder's) decision on a single tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Allow this call only.
    AllowOnce,
    /// Allow this call and remember the choice for the rest of the session.
    AllowAlways,
    /// Reject this call only.
    RejectOnce,
    /// Reject this call and remember the choice for the rest of the session.
    RejectAlways,
}

impl PermissionDecision {
    pub fn is_allowed(self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowAlways)
    }
}

/// Asked before every tool call the agent loop is about to execute.
#[async_trait]
pub trait PermissionGate: Send + Sync {
    async fn check(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision;
}

/// Default gate for contexts with no human in the loop to ask: the TUI (already
/// fully interactive/local), the library facade, `openheim run`, and subagents.
/// Always allows.
pub struct AllowAll;

#[async_trait]
impl PermissionGate for AllowAll {
    async fn check(
        &self,
        _tool_call_id: &str,
        _tool_name: &str,
        _arguments: &str,
    ) -> PermissionDecision {
        PermissionDecision::AllowOnce
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allow_all_always_allows() {
        let decision = AllowAll.check("call_1", "execute_command", "{}").await;
        assert!(decision.is_allowed());
    }
}
