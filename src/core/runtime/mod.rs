//! The agent runtime core: [`AgentState`] is the process-wide,
//! per-connection-shared handle every entry point (ACP, the other
//! transports, the library facade) is a method on; [`session`] is the live
//! session map's own state and eviction policy; [`AgentMode`] controls which
//! tools a session's turns are offered.
//!
//! `AgentState::prompt`/`load_session` still speak ACP's `SessionUpdate` and
//! reach into `crate::acp::util` for the thinking/tool-kind/replay helpers
//! that build it — collapsing that down to `core::models::StreamEvent`
//! end-to-end, so this module has no ACP dependency at all, is tracked
//! separately (PLAN.md item 5).

pub mod session;

mod state;

pub use state::AgentState;

/// Which tool policy a session runs under, set via `session/set_mode`.
/// [`Self::as_str`] gives the wire-level mode id; [`Self::parse`] is the
/// inverse, for the boundary where that id arrives as a `&str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentMode {
    /// Full tool access; tool calls go through the permission gate as normal.
    #[default]
    Code,
    /// Read-only: only `read_file`, `list_dir`, `search` (and
    /// `search_memory` with the `rag` feature) are offered to the LLM, so
    /// nothing mutating can run. All of them still go through the
    /// permission gate and can trigger a permission prompt unless already
    /// approved.
    Architect,
}

impl AgentMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            AgentMode::Code => "code",
            AgentMode::Architect => "architect",
        }
    }

    pub fn parse(mode_id: &str) -> crate::error::Result<Self> {
        match mode_id {
            "code" => Ok(AgentMode::Code),
            "architect" => Ok(AgentMode::Architect),
            other => Err(crate::error::Error::ParseError(format!(
                "unknown session mode: {other}"
            ))),
        }
    }
}
