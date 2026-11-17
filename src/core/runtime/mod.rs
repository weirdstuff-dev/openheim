//! The agent runtime core: [`AgentState`] is the shared handle every entry
//! point (ACP, the other transports, the library facade) calls into;
//! [`session`] holds the live session map and its eviction policy;
//! [`AgentMode`] controls which tools a session's turns are offered.
//!
//! Only `core::models` types cross this module's boundary. Mapping them onto
//! ACP's wire types is done by the callers (`acp::serve`, `client.rs`).

pub mod session;

mod state;

pub use state::{AgentState, LoadedSession, SessionSettings};

/// Which tool policy a session runs under. [`Self::as_str`] gives the
/// wire-level mode id (also its serialized form); [`Self::parse`] is the
/// inverse, for the boundary where that id arrives as a `&str`. Saved
/// conversations that stored `"code"` or `"architect"` deserialize as
/// [`Self::Execute`] and [`Self::Plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentMode {
    /// Full tool access; tool calls go through the permission gate as normal.
    #[default]
    #[serde(alias = "code")]
    Execute,
    /// Read-only: only tools that declare `read_only` are offered (the
    /// built-in `read_file`, `list_dir`, `search`, `web_fetch`,
    /// `search_memory`, and any custom tool that opts in). They still go
    /// through the permission gate.
    #[serde(alias = "architect")]
    Plan,
}

impl AgentMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            AgentMode::Execute => "execute",
            AgentMode::Plan => "plan",
        }
    }

    pub fn parse(mode_id: &str) -> crate::error::Result<Self> {
        match mode_id {
            "execute" => Ok(AgentMode::Execute),
            "plan" => Ok(AgentMode::Plan),
            other => Err(crate::error::Error::InvalidArgument(format!(
                "unknown session mode: {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AgentMode;

    #[test]
    fn loads_the_previous_mode_ids() {
        let execute: AgentMode = serde_json::from_str("\"code\"").unwrap();
        let plan: AgentMode = serde_json::from_str("\"architect\"").unwrap();
        assert_eq!(execute, AgentMode::Execute);
        assert_eq!(plan, AgentMode::Plan);
    }
}
