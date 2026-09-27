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
/// A tool that runs nested turns (`delegate_task`) passes it
/// straight through, so a subagent is cancelled with its parent and asks the
/// same permission gate (wrapped so each request names the subagent).
#[non_exhaustive]
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

impl<'a> TurnContext<'a> {
    /// A context whose `cwd` is `work_dir`; set another with
    /// [`Self::with_cwd`].
    pub fn new(
        cancel: &'a CancellationToken,
        permission_gate: &'a Arc<dyn PermissionGate>,
        work_dir: &'a Path,
        client_io: &'a dyn ClientIo,
    ) -> Self {
        Self {
            cancel,
            permission_gate,
            work_dir,
            cwd: work_dir,
            client_io,
        }
    }

    pub fn with_cwd(mut self, cwd: &'a Path) -> Self {
        self.cwd = cwd;
        self
    }

    /// `requested` resolved against [`Self::cwd`] and checked to lie inside
    /// [`Self::work_dir`] (symlinks followed); what every filesystem tool
    /// opens.
    pub fn resolve_path(&self, requested: &str) -> Result<PathBuf> {
        crate::tools::sandbox::validate_path_from(requested, self.cwd, self.work_dir)
    }
}
