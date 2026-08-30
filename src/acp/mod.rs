//! Agent Client Protocol (ACP) integration.
//!
//! Split by responsibility: [`session`] is the live session map's own state
//! and eviction policy; [`state`] is [`AgentState`], the shared handle every
//! entry point below is a method on; [`permission`]/[`client_io`] adapt ACP's
//! `session/request_permission` and `fs/*` requests to `core`'s
//! `PermissionGate`/`ClientIo` traits; [`convert`] maps ACP content blocks to
//! `core::models::ContentBlock`; [`util`] is shared ACP vocabulary (session
//! modes, stop-reason/tool-kind mapping, history replay); [`serve`] is the
//! connection loop that wires it all to the `agent-client-protocol` crate.

pub mod session;

mod client_io;
mod convert;
mod permission;
mod serve;
mod state;
mod util;

pub use serve::serve;
pub use state::AgentState;
pub use util::AgentMode;
