use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::acp::AgentMode;
use crate::config::AgentConfig;
use crate::core::permission::PermissionDecision;
use crate::error::{Error, Result};

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
    /// `session/request_permission` prompts, so the same tool call isn't
    /// re-prompted for the rest of the session. Keyed by tool name for most
    /// tools; see [`crate::acp::approval_key`] for `execute_command`'s finer
    /// per-command-prefix scoping.
    pub approved_tools: HashMap<String, PermissionDecision>,
    /// Set via `session/set_mode`. Controls which tools are offered to the LLM.
    pub mode: AgentMode,
    /// Held for the duration of a `session/prompt` turn so a second, overlapping
    /// prompt on the same session is rejected instead of racing the first one
    /// (both would otherwise reset `cancel` and clobber the saved history).
    pub prompt_lock: Arc<Mutex<()>>,
}

impl SessionState {
    /// Acquires [`Self::prompt_lock`] for the caller's turn, or a clean error
    /// if another turn already holds it. `session_id` is only used to name
    /// the session in the error message; hold the returned guard for the
    /// duration of the turn — it releases automatically when dropped.
    pub fn try_acquire_prompt_lock(&self, session_id: &str) -> Result<OwnedMutexGuard<()>> {
        self.prompt_lock.clone().try_lock_owned().map_err(|_| {
            Error::Other(format!(
                "a prompt is already in flight for session {session_id}"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> SessionState {
        SessionState {
            chat_id: Uuid::new_v4(),
            config: AgentConfig::new(
                "mock".into(),
                "https://example.com".into(),
                "key".into(),
                "mock-model".into(),
                5,
            ),
            cwd: PathBuf::from("/tmp"),
            skills: vec![],
            cancel: CancellationToken::new(),
            approved_tools: HashMap::new(),
            mode: AgentMode::Code,
            prompt_lock: Arc::new(Mutex::new(())),
        }
    }

    #[test]
    fn second_prompt_is_rejected_while_first_is_in_flight() {
        let state = sample_state();
        let _first = state.try_acquire_prompt_lock("s1").unwrap();

        let second = state.try_acquire_prompt_lock("s1");

        assert!(second.is_err());
    }

    #[test]
    fn prompt_lock_is_available_again_once_the_first_guard_drops() {
        let state = sample_state();
        {
            let _first = state.try_acquire_prompt_lock("s1").unwrap();
        }

        assert!(state.try_acquire_prompt_lock("s1").is_ok());
    }
}
