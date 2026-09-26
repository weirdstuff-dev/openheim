//! Cross-cutting turn controls threaded through the agent loop and down into
//! tool execution.
//!
//! Lives outside `core::agent` so [`crate::tools::ToolExecutor`] and
//! [`crate::tools::ToolHandler`] can depend on it without introducing a
//! dependency on the agent loop itself.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::core::client_io::ClientIo;
use crate::core::permission::PermissionGate;
use crate::error::Result;

/// Everything a single prompt turn carries down to the tools it runs.
///
/// Grouped into one struct so `run_agent`'s (and the tool traits')
/// parameter lists don't grow with every new hook. Every [`crate::tools::ToolHandler`]
/// receives this on `execute`, so a built-in or custom tool can honour
/// cancellation, confine itself to the work directory, and route file I/O
/// through the client without any wrapper in between.
///
/// Tools that spawn nested agent-loop turns (namely
/// [`crate::tools::DelegateTool`] for subagents) pass the same context
/// straight through rather than manufacturing their own: the subagent shares
/// the parent turn's cancellation token, so a `session/cancel` on the outer
/// turn stops the subagent too, and asks the parent's permission gate, so
/// subagent tool calls go through the same approval flow as the
/// orchestrator's own — there is no separate "subagent trust policy". The
/// gate is wrapped so each request's `subagent` names who is asking.
pub struct TurnContext<'a> {
    /// Fires when the turn is cancelled; long-running tools should race
    /// their work against it.
    pub cancel: &'a CancellationToken,
    /// Approval hook consulted by the agent loop before each tool call.
    pub permission_gate: &'a Arc<dyn PermissionGate>,
    /// Sandbox boundary for filesystem tools: no path they touch may lie
    /// outside it.
    pub work_dir: &'a Path,
    /// Where relative paths resolve and `execute_command` runs: the
    /// session's `cwd` when that is inside `work_dir`, otherwise `work_dir`.
    pub cwd: &'a Path,
    /// Optional delegation of file reads/writes to the client (e.g. an
    /// editor's unsaved buffers); [`crate::core::client_io::NoClientIo`]
    /// when there is none.
    pub client_io: &'a dyn ClientIo,
}

impl TurnContext<'_> {
    /// `requested` resolved against [`Self::cwd`] and checked to lie inside
    /// [`Self::work_dir`]; what every filesystem tool opens. See
    /// [`crate::tools::sandbox::validate_path_from`].
    pub fn resolve_path(&self, requested: &str) -> Result<PathBuf> {
        crate::tools::sandbox::validate_path_from(requested, self.cwd, self.work_dir)
    }
}
