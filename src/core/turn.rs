//! Cross-cutting turn controls threaded through the agent loop and down into
//! tool execution.
//!
//! Lives outside `core::agent` so [`crate::tools::ToolExecutor`] and
//! [`crate::tools::ToolHandler`] can depend on it without introducing a
//! dependency on the agent loop itself.

use std::path::Path;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::core::client_io::ClientIo;
use crate::core::permission::PermissionGate;

/// Everything a single prompt turn carries down to the tools it runs.
///
/// Grouped into one struct so `run_agent_loop`'s (and the tool traits')
/// parameter lists don't grow with every new hook. Every [`crate::tools::ToolHandler`]
/// receives this on `execute`, so a built-in or custom tool can honour
/// cancellation, confine itself to the work directory, and route file I/O
/// through the client without any wrapper in between.
///
/// Tools that spawn nested agent-loop turns (namely
/// [`crate::tools::DelegateTool`] for subagents) pass the same context
/// straight through rather than manufacturing their own: the subagent shares
/// the parent turn's cancellation token, so a `session/cancel` on the outer
/// turn stops the subagent too, and inherits the parent's permission gate, so
/// subagent tool calls go through the same approval flow as the
/// orchestrator's own — there is no separate "subagent trust policy".
pub struct TurnContext<'a> {
    /// Fires when the turn is cancelled; long-running tools should race
    /// their work against it.
    pub cancel: &'a CancellationToken,
    /// Approval hook consulted by the agent loop before each tool call.
    pub permission_gate: &'a Arc<dyn PermissionGate>,
    /// Sandbox boundary for filesystem tools (see
    /// [`crate::tools::sandbox::validate_path`]) and the working directory
    /// for `execute_command`.
    pub work_dir: &'a Path,
    /// Optional delegation of file reads/writes to the client (e.g. an
    /// editor's unsaved buffers); [`crate::core::client_io::NoClientIo`]
    /// when there is none.
    pub client_io: &'a dyn ClientIo,
}
