use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::AgentConfig;
use crate::core::permission::Approvals;
use crate::core::runtime::AgentMode;
use crate::error::{Error, Result};

#[derive(Debug)]
pub struct SessionState {
    pub chat_id: Uuid,
    pub config: AgentConfig,
    /// The client's working directory, as sent; saved with the conversation
    /// and used to filter `list_sessions`.
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
    /// Set via `session/set_mode`. Controls which tools are offered to the LLM.
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
    use crate::core::permission::PermissionDecision;

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
            cwd: PathBuf::from("/tmp"),
            tool_cwd: PathBuf::from("/tmp"),
            skills: vec![],
            cancel: CancellationToken::new(),
            approved_tools: Approvals::default(),
            mode: AgentMode::Code,
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

    #[test]
    fn insert_or_keep_live_inserts_when_absent() {
        let mut sessions = HashMap::new();

        assert!(insert_or_keep_live(&mut sessions, "s1", || Ok(sample_state())).unwrap());
        assert!(sessions.contains_key("s1"));
    }

    #[test]
    fn insert_or_keep_live_preserves_the_live_entrys_control_state() {
        let mut sessions = HashMap::new();
        let live = sample_state();
        let live_cancel = live.cancel.clone();
        let live_lock = Arc::clone(&live.prompt_lock);
        live.approved_tools.remember(
            "execute_command:git status".to_string(),
            PermissionDecision::AllowAlways,
        );
        sessions.insert("s1".to_string(), live);

        let inserted = insert_or_keep_live(&mut sessions, "s1", || Ok(sample_state())).unwrap();

        assert!(!inserted);
        let kept = sessions.get("s1").unwrap();
        // Same token: cancelling the pre-load handle must reach the kept entry.
        live_cancel.cancel();
        assert!(kept.cancel.is_cancelled());
        assert!(Arc::ptr_eq(&kept.prompt_lock, &live_lock));
        assert_eq!(
            kept.approved_tools.get("execute_command:git status"),
            Some(PermissionDecision::AllowAlways)
        );
    }

    #[test]
    fn insert_or_keep_live_bumps_the_live_entrys_last_active() {
        let stale_since = Instant::now() - Duration::from_secs(60);
        let mut sessions = HashMap::new();
        sessions.insert("s1".to_string(), sample_state_at(stale_since));

        insert_or_keep_live(&mut sessions, "s1", || {
            panic!("must not build a fresh state when the session is already live")
        })
        .unwrap();

        assert!(sessions.get("s1").unwrap().last_active > stale_since);
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

    #[test]
    fn evict_idle_sessions_cap_skips_in_flight_entries_even_when_over_cap() {
        let now = Instant::now();
        let mut sessions = HashMap::new();
        let in_flight = sample_state_at(now - Duration::from_secs(300));
        let _guard = in_flight.prompt_lock.clone().try_lock_owned().unwrap();
        sessions.insert("in_flight".to_string(), in_flight);
        sessions.insert(
            "newest".to_string(),
            sample_state_at(now - Duration::from_secs(100)),
        );

        evict_idle_sessions(&mut sessions, now, Duration::from_secs(3600), 1);

        // The only evictable entry is gone; the in-flight one survives even
        // though the map is still over the cap.
        assert!(!sessions.contains_key("newest"));
        assert!(sessions.contains_key("in_flight"));
    }

    #[test]
    fn prompt_in_flight_reflects_the_lock() {
        let state = sample_state();
        assert!(!prompt_in_flight(&state));

        let _guard = state.prompt_lock.clone().try_lock_owned().unwrap();
        assert!(prompt_in_flight(&state));
    }
}
