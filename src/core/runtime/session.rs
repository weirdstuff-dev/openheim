use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::AgentConfig;
use crate::core::permission::PermissionDecision;
use crate::core::runtime::AgentMode;
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
    /// tools; see
    /// [`crate::core::permission::approval_key`] for `execute_command`'s
    /// exact-command-string scoping.
    pub approved_tools: HashMap<String, PermissionDecision>,
    /// Set via `session/set_mode`. Controls which tools are offered to the LLM.
    pub mode: AgentMode,
    /// Held for the duration of a `session/prompt` turn so a second, overlapping
    /// prompt on the same session (in this process) is rejected instead of
    /// racing the first one (both would otherwise reset `cancel` and clobber
    /// the saved history). The cross-process write lease (`memory::lease`) is a
    /// separate, turn-scoped guard acquired directly in `AgentState::prompt` —
    /// not stored here — so merely loading/viewing a session never locks it
    /// against other processes; only an in-flight turn does.
    pub prompt_lock: Arc<Mutex<()>>,
    /// Last time this session was used (prompt, cancel, model/mode switch,
    /// load). Drives least-recently-active ordering in
    /// [`evict_idle_sessions`]; bumped under the sessions map's write lock.
    pub last_active: Instant,
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

/// How long a live session may sit unused before it becomes eligible for
/// eviction. Long enough that no realistic workflow (a TUI left open over a
/// weekend, an editor reconnecting Monday) loses state; short enough that a
/// long-lived server's session map doesn't accumulate stale entries forever.
pub(crate) const SESSION_IDLE_EVICTION_AFTER: Duration = Duration::from_secs(60 * 60 * 24 * 7);

/// Hard cap on live sessions. Bounds the map even under sustained churn
/// (idle eviction alone can't, if every session stays active). Evicted
/// sessions are reconstructible: their history lives on disk and
/// `session/load` re-materializes the state.
pub(crate) const MAX_LIVE_SESSIONS: usize = 512;

/// Inserts a freshly-built `SessionState` under `session_id` unless that
/// session is already live, in which case the live entry wins and only its
/// `last_active` is bumped. Returns `true` when a fresh state was inserted.
///
/// `build_fresh` is called at most once, and only once we've confirmed
/// there's no live entry to keep, so a redundant build (and the disk read it
/// implies) never happens when the *live* entry is the one that ends up
/// owning the session.
///
/// Replacing a live entry would break the cross-connection session UX: a
/// fresh `CancellationToken` orphans any in-flight turn (`session/cancel`
/// would cancel the new token, not the running turn's), a fresh `prompt_lock`
/// would let a second turn overlap a still-running one on the same chat, and
/// wiping `approved_tools` loses remembered AllowAlways decisions. The live
/// entry is also strictly newer than the disk snapshot a load builds its
/// fresh state from. Either way the caller still replays the on-disk history
/// to *its* connection — that part is per-connection, not session state.
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
/// flight is never evicted. Must be called while holding the sessions map's
/// *write* lock: `AgentState::prompt` acquires `prompt_lock` under that same
/// lock, so a lock observed free here cannot be acquired by a new turn until
/// the sweep finishes, and a running turn holds its lock for the whole turn.
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
            skills: vec![],
            cancel: CancellationToken::new(),
            approved_tools: HashMap::new(),
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
        let mut live = sample_state();
        let live_cancel = live.cancel.clone();
        let live_lock = Arc::clone(&live.prompt_lock);
        live.approved_tools.insert(
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
            Some(&PermissionDecision::AllowAlways)
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
