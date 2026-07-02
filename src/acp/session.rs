use std::collections::HashMap;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::AgentConfig;
use crate::core::permission::PermissionDecision;

#[derive(Debug)]
pub struct SessionState {
    pub chat_id: Uuid,
    pub config: AgentConfig,
    pub cwd: PathBuf,
    pub skills: Vec<String>,
    /// Cancelled when a `session/cancel` notification arrives for this session,
    /// so an in-flight prompt turn (running in its own spawned task) can stop.
    pub cancel: CancellationToken,
    /// Remembered `AllowAlways`/`RejectAlways` decisions from prior
    /// `session/request_permission` prompts, keyed by tool name, so the same
    /// tool isn't re-prompted for the rest of the session.
    pub approved_tools: HashMap<String, PermissionDecision>,
}
