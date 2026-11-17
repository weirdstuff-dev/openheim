use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::AgentConfig;
use crate::core::llm::LlmClient;
use crate::core::models::Effort;
use crate::core::permission::Approvals;
use crate::core::runtime::AgentMode;
use crate::error::{Error, Result};

pub struct SessionState {
    pub chat_id: Uuid,
    /// The session's model, with `effort` already applied when it's set.
    pub config: AgentConfig,
    /// The client for `config` when it names another model or effort
    /// level than the runtime's default; `None` means the default's client.
    /// Set together with `config`, and built before the sessions lock is
    /// taken.
    pub llm: Option<Arc<dyn LlmClient>>,
    /// The effort level chosen for this session, kept across model
    /// switches. `None` until one is chosen: the session then runs at its
    /// provider's `effort`.
    pub effort: Option<Effort>,
    /// The client's working directory, canonicalized when it exists on this
    /// machine; saved with the conversation and used to filter
    /// `list_sessions`.
    pub cwd: PathBuf,
    /// Where this session's tools work (`TurnContext::cwd`): `cwd` when it's
    /// a directory inside the work directory, otherwise the work directory.
    pub tool_cwd: PathBuf,
    pub skills: Vec<String>,
    /// Cancelled when a `session/cancel` notification arrives for this session,
    /// so an in-flight prompt turn (running in its own spawned task) can stop.
    pub cancel: CancellationToken,
    /// Remembered `AllowAlways`/`RejectAlways` decisions, read and written by
    /// each turn's `RememberingGate`; keyed by
    /// [`crate::core::permission::approval_key`].
    pub approved_tools: Approvals,
    /// Controls which tools are offered to the LLM. Saved with the
    /// conversation, and restored when it's loaded.
    pub mode: AgentMode,
    /// Held for a whole turn, so an overlapping prompt on the same session in
    /// this process is rejected instead of racing it. (The cross-process
    /// lease, `memory::lease`, is taken separately by `AgentState::prompt`.)
    pub prompt_lock: Arc<Mutex<()>>,
    /// Last time this session was used; orders `evict_idle_sessions`.
    /// Updated under the sessions map's write lock.
    pub last_active: Instant,
}

impl SessionState {
    /// Acquires [`Self::prompt_lock`] for the caller's turn, or
    /// `Error::SessionBusy` (naming `session_id`) if another turn holds it.
    /// The lock is released when the guard drops.
    pub fn try_acquire_prompt_lock(&self, session_id: &str) -> Result<OwnedMutexGuard<()>> {
        self.prompt_lock
            .clone()
            .try_lock_owned()
            .map_err(|_| Error::SessionBusy {
                session_id: session_id.to_string(),
            })
    }
}

/// How long a live session may sit unused before it can be evicted: a week,
/// so a TUI left open over a weekend keeps its state.
pub(crate) const SESSION_IDLE_EVICTION_AFTER: Duration = Duration::from_secs(60 * 60 * 24 * 7);

/// Hard cap on live sessions, for when every session stays active. An
/// evicted session loses nothing durable: `session/load` rebuilds it from
/// disk.
pub(crate) const MAX_LIVE_SESSIONS: usize = 512;

/// Inserts the state `build_fresh` returns under `session_id` unless that
/// session is already live, in which case the live entry is kept and only
/// its `last_active` is bumped. Returns `true` when a fresh state was
/// inserted; `build_fresh` isn't called otherwise.
///
/// The live entry wins because replacing it would orphan a running turn
/// (its cancel token), let a second turn overlap it (its `prompt_lock`), and
/// forget remembered approvals.
pub(crate) fn insert_or_keep_live(
    sessions: &mut HashMap<String, SessionState>,
    session_id: &str,
    build_fresh: impl FnOnce() -> Result<SessionState>,
) -> Result<bool> {
    match sessions.get_mut(session_id) {
        Some(live) => {
            live.last_active = Instant::now();
            Ok(false)
        }
        None => {
            sessions.insert(session_id.to_string(), build_fresh()?);
            Ok(true)
        }
    }
}

/// Evicts sessions so the live map stays bounded: first anything idle longer
/// than `idle_after`, then — if the map still exceeds `max_sessions` — the
/// least-recently-active ones until it fits. A session with a prompt in
/// flight is never evicted. Call it with the sessions map's write lock held:
/// turns take `prompt_lock` under that lock, so none can start mid-sweep.
pub(crate) fn evict_idle_sessions(
    sessions: &mut HashMap<String, SessionState>,
    now: Instant,
    idle_after: Duration,
    max_sessions: usize,
) {
    let idle: Vec<String> = sessions
        .iter()
        .filter(|(_, s)| now.duration_since(s.last_active) > idle_after && !prompt_in_flight(s))
        .map(|(id, _)| id.clone())
        .collect();
    for id in idle {
        sessions.remove(&id);
    }

    if sessions.len() > max_sessions {
        let excess = sessions.len() - max_sessions;
        let mut candidates: Vec<(String, Instant)> = sessions
            .iter()
            .filter(|(_, s)| !prompt_in_flight(s))
            .map(|(id, s)| (id.clone(), s.last_active))
            .collect();
        candidates.sort_by_key(|(_, at)| *at);
        for (id, _) in candidates.into_iter().take(excess) {
            sessions.remove(&id);
        }
    }
}

/// Whether a prompt turn currently holds this session's `prompt_lock`.
/// Probing takes the lock briefly; safe only under the map's write lock (see
/// [`evict_idle_sessions`]).
pub(crate) fn prompt_in_flight(s: &SessionState) -> bool {
    s.prompt_lock.clone().try_lock_owned().is_err()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> SessionState {
        let chat_id = Uuid::new_v4();
        SessionState {
            chat_id,
            config: AgentConfig::new(
                "mock".into(),
                "https://example.com".into(),
                "key".into(),
                "mock-model".into(),
                5,
            ),
            llm: None,
            effort: None,
            cwd: PathBuf::from("/tmp"),
            tool_cwd: PathBuf::from("/tmp"),
            skills: vec![],
            cancel: CancellationToken::new(),
            approved_tools: Approvals::default(),
            mode: AgentMode::Execute,
            prompt_lock: Arc::new(Mutex::new(())),
            last_active: Instant::now(),
        }
    }

    fn sample_state_at(last_active: Instant) -> SessionState {
        let mut s = sample_state();
        s.last_active = last_active;
        s
    }

    #[test]
    fn evict_idle_sessions_removes_idle_unlocked_entries() {
        let now = Instant::now();
        let mut sessions = HashMap::new();
        sessions.insert(
            "stale".to_string(),
            sample_state_at(now - Duration::from_secs(600)),
        );
        sessions.insert("fresh".to_string(), sample_state_at(now));

        evict_idle_sessions(&mut sessions, now, Duration::from_secs(300), usize::MAX);

        assert!(!sessions.contains_key("stale"));
        assert!(sessions.contains_key("fresh"));
    }

    #[test]
    fn evict_idle_sessions_never_removes_entries_with_prompts_in_flight() {
        let now = Instant::now();
        let mut sessions = HashMap::new();
        let stale = sample_state_at(now - Duration::from_secs(600));
        let _guard = stale.prompt_lock.clone().try_lock_owned().unwrap();
        sessions.insert("in_flight".to_string(), stale);

        evict_idle_sessions(&mut sessions, now, Duration::from_secs(300), usize::MAX);

        assert!(sessions.contains_key("in_flight"));
    }

    #[test]
    fn evict_idle_sessions_cap_evicts_least_recently_active_first() {
        let now = Instant::now();
        let mut sessions = HashMap::new();
        sessions.insert(
            "oldest".to_string(),
            sample_state_at(now - Duration::from_secs(300)),
        );
        sessions.insert(
            "middle".to_string(),
            sample_state_at(now - Duration::from_secs(200)),
        );
        sessions.insert(
            "newest".to_string(),
            sample_state_at(now - Duration::from_secs(100)),
        );

        // Idle threshold high enough that only the cap pass runs.
        evict_idle_sessions(&mut sessions, now, Duration::from_secs(3600), 1);

        assert!(!sessions.contains_key("oldest"));
        assert!(!sessions.contains_key("middle"));
        assert!(sessions.contains_key("newest"));
    }
}
