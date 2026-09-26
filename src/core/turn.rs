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

/// Everything a prompt turn carries down to the tools it runs, passed to
/// every [`crate::tools::ToolHandler::execute`].
///
/// A tool that runs nested turns ([`crate::tools::DelegateTool`]) passes it
/// straight through, so a subagent is cancelled with its parent and asks the
/// same permission gate (wrapped so each request names the subagent).
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
