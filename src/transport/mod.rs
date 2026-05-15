//! Server transport implementations for the Openheim agent runtime.
//!
//! Each module wraps the same [`acp::serve`] loop in a different I/O layer:
//!
//! | Module | Transport | Use case |
//! |--------|-----------|----------|
//! | [`stdio`] | stdin / stdout (ACP) | CLI clients, IDE extensions, local tooling |
//! | [`ws`] | WebSocket + REST (axum) | Browser frontends, remote API clients |
//! | [`run`] | In-process ACP client | `openheim run "<prompt>"` one-shot CLI mode |

pub mod run;
pub mod stdio;
pub mod ws;
