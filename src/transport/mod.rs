//! Server transport implementations for the Openheim agent runtime.
//!
//! Each module wraps the same [`acp::serve`] loop in a different I/O layer:
//!
//! | Module | Transport | Use case |
//! |--------|-----------|----------|
//! | [`stdio`] | stdin / stdout (ACP) | CLI clients, IDE extensions, local tooling |
//! | [`run`] | In-process ACP client | `openheim run "<prompt>"` one-shot CLI mode |
//! | `ws` (needs the `server` feature) | WebSocket + REST (axum) | Browser frontends, remote API clients |

pub mod run;
pub mod stdio;
#[cfg(feature = "server")]
pub mod ws;
