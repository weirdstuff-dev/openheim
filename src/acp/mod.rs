//! Agent Client Protocol (ACP) adapter over [`crate::core::runtime::AgentState`].
//!
//! - `permission`, `client_io`: ACP's `session/request_permission` and
//!   `fs/*` as `core`'s `PermissionGate` and `ClientIo`.
//! - `convert`: ACP prompt content to `core::models::ContentBlock`.
//! - `util`: shared ACP vocabulary, history replay, and the
//!   `StreamEvent → SessionUpdate` mapping (also behind
//!   `SessionHandle::acp_updates`).
//! - [`serve`]: the connection loop.

pub(crate) mod convert;
pub(crate) mod util;

mod client_io;
mod permission;
mod serve;

pub use serve::serve;

// ACP's own wire vocabulary, for library users of the ACP adapters
// (`SessionHandle::acp_updates`, `SessionHandle::acp_replay`) without a
// direct `agent-client-protocol` dependency.
pub use agent_client_protocol::schema::v1 as schema;
