//! Agent Client Protocol (ACP) integration.
//!
//! The runtime core — [`crate::core::runtime::AgentState`], its session map,
//! and [`crate::core::runtime::AgentMode`] — lives outside this module now;
//! everything left here is specific to speaking ACP over the
//! `agent-client-protocol` crate: [`permission`]/[`client_io`] adapt ACP's
//! `session/request_permission` and `fs/*` requests to `core`'s
//! `PermissionGate`/`ClientIo` traits; [`convert`] maps ACP content blocks to
//! `core::models::ContentBlock`; [`util`] is shared ACP vocabulary (session
//! modes, stop-reason/tool-kind mapping, history replay) plus the
//! `StreamEvent → SessionUpdate` mapping for a live turn — the one place ACP
//! decodes `core::models::StreamEvent`, used by both `serve` and the library
//! facade's `SessionHandle::prompt`; [`serve`] is the connection loop that
//! wires it all to the `agent-client-protocol` crate.

pub(crate) mod convert;
pub(crate) mod util;

mod client_io;
mod permission;
mod serve;

pub use serve::serve;
