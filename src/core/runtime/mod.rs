//! The agent runtime core: [`AgentState`] is the shared handle every entry
//! point (ACP, the other transports, the library facade) calls into;
//! [`session`] holds the live session map and its eviction policy;
//! [`AgentMode`] controls which tools a session's turns are offered.
//!
//! Only `core::models` types cross this module's boundary. Mapping them onto
//! ACP's wire types is done by the callers (`acp::serve`, `client.rs`).

pub mod session;

mod state;

pub use state::{AgentState, LoadedSession};

/// Which tool policy a session runs under, set via `session/set_mode`.
/// [`Self::as_str`] gives the wire-level mode id; [`Self::parse`] is the
/// inverse, for the boundary where that id arrives as a `&str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentMode {
    /// Full tool access; tool calls go through the permission gate as normal.
    #[default]
    Code,
    /// Read-only: only tools that declare `read_only` are offered (the
    /// built-in `read_file`, `list_dir`, `search`, `web_fetch`,
    /// `search_memory`, and any custom tool that opts in). They still go
    /// through the permission gate.
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
            other => Err(crate::error::Error::InvalidArgument(format!(
                "unknown session mode: {other}"
            ))),
        }
    }
}
