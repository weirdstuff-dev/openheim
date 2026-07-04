//! Cross-cutting turn controls threaded through the agent loop and down into
//! tool execution.
//!
//! Lives outside `core::agent` so [`crate::tools::ToolExecutor`] can depend on
//! it without introducing a dependency on the agent loop itself.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::core::permission::PermissionGate;

/// Grouped into one struct so `run_agent_loop`'s (and `ToolExecutor::execute`'s)
/// parameter list doesn't grow with every new hook (cancellation, permission
/// checks, and more to come).
///
/// Tool wrappers that spawn nested agent-loop turns (namely
/// [`crate::tools::DelegateTool`] for subagents) reuse the fields here rather
/// than manufacturing their own: the subagent shares the parent turn's
/// cancellation token, so a `session/cancel` on the outer turn stops the
/// subagent too, and inherits the parent's permission gate, so subagent tool
/// calls go through the same approval flow as the orchestrator's own —
/// there is no separate "subagent trust policy".
pub struct TurnContext<'a> {
    pub cancel: &'a CancellationToken,
    pub permission_gate: &'a Arc<dyn PermissionGate>,
}
