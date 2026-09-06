//! Advisory, per-session write lease so two `openheim` processes sharing the
//! same `~/.openheim/history/` (the desktop app's embedded core and an
//! independently-spawned CLI, in particular) don't both write the same
//! conversation at once — see `PLAN.md` §1.
//!
//! The lease is turn-scoped, not session-scoped: `core::runtime::AgentState::prompt`
//! acquires it right before running a turn and holds it only for that turn's
//! duration, so merely loading or holding a session open never locks it
//! against other processes — only an in-flight `session/prompt` does. Two
//! processes can freely view the same session, and whichever sends a prompt
//! first gets the lease; the other's `session/prompt` fails fast with
//! [`crate::error::Error::SessionLocked`] instead of racing or queuing.
//!
//! A lease is a small JSON lockfile, `{uuid}.lock`, next to the conversation's
//! `.json`/`.jsonl` files. It's advisory (nothing stops a process from
//! ignoring it and writing anyway) and lockfile-with-pid rather than an OS
//! file lock (`flock`): portable across the platforms this crate targets,
//! human-readable for debugging, and lets a stale lock (holder crashed
//! without releasing it) be detected and taken over instead of wedging the
//! session forever.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;
use uuid::Uuid;

use crate::error::{Error, Result};

/// How stale a lease may be — by file mtime, i.e. time since it was last
/// acquired or refreshed — before another process may take it over even
/// though it can't positively confirm the holder is gone (different host, or
/// this platform can't check pid liveness). On the same host, an alive pid
/// always wins regardless of age; this bound only matters as the fallback
/// for the cases pid-liveness can't cover.
const STALE_TTL: Duration = Duration::from_secs(30 * 60);

/// Contents of a `{uuid}.lock` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeaseInfo {
    pid: u32,
    hostname: String,
    acquired_at: DateTime<Utc>,
    /// Best-effort identity marker for the holder process, not a true OS
    /// process-start timestamp (getting that portably needs a dependency
    /// this crate doesn't otherwise have): the first time *this* process
    /// acquired any lease. Two different processes are extremely unlikely to
    /// share both a pid and this value, which is what it's for — a human (or
    /// a future, stricter version of this check) reading the lockfile can
    /// tell a live pid apart from an unrelated process that happens to have
    /// reused it.
    process_start: DateTime<Utc>,
}

/// This process's identity, computed once and reused for every lease it
/// acquires.
static IDENTITY: LazyLock<(String, DateTime<Utc>)> =
    LazyLock::new(|| (current_hostname(), Utc::now()));

fn current_hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

/// Whether `pid` is currently alive on this host. `None` if this platform
/// has no way to check (anything but unix, today), in which case staleness
/// falls back to [`STALE_TTL`] alone.
#[cfg(unix)]
fn pid_is_alive(pid: u32) -> Option<bool> {
    Some(!matches!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None),
        Err(nix::errno::Errno::ESRCH)
    ))
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> Option<bool> {
    None
}

fn lock_path(dir: &Path, id: &Uuid) -> PathBuf {
    dir.join(format!("{id}.lock"))
}

/// Writes `contents` to `path` by writing to a sibling temp file and
/// renaming it into place, so a reader never observes a partial write.
///
/// For *refreshing* or *taking over* an already-existing lockfile only —
/// [`acquire`] uses [`create_lease_exclusively`] instead when no lease was
/// observed, so two processes racing to claim a session for the first time
/// can't both win via a rename that neither has to contend for.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    // Per-call unique suffix (pid + a fresh uuid): a shared `.tmp` name would
    // let two concurrent writers to the *same* lockfile (e.g. this process
    // refreshing its lease while another takes it over as stale) clobber
    // each other's temp file before either gets to rename.
    let mut tmp_path = path.as_os_str().to_owned();
    tmp_path.push(format!(".{}.{}.tmp", std::process::id(), Uuid::new_v4()));
    let tmp_path = PathBuf::from(tmp_path);
    std::fs::write(&tmp_path, contents)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Atomically creates `path` with `contents`, failing with
/// [`std::io::ErrorKind::AlreadyExists`] if it's already there instead of
/// overwriting it. Used by [`acquire`] to claim a session that currently has
/// no lockfile: unlike [`write_atomic`]'s rename (which always wins,
/// unconditionally replacing whatever's at `path`), this lets two processes
/// racing to be the first to acquire the same session's lease discover the
/// race instead of silently having the second one's rename erase the first.
fn create_lease_exclusively(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(contents.as_bytes())
}

/// Holds a session's write lease for as long as it's alive; releases it
/// (best-effort — the lockfile is simply removed) on drop, which is how a
/// lease is meant to be released. `core::runtime::AgentState::prompt` holds one for
/// exactly the duration of a single turn, so normal release just means "the
/// turn finished" (or was cancelled, or errored) — no eviction or process
/// exit required. A crash (or `kill -9`) mid-turn skips this and leaves the
/// lockfile behind for the next [`acquire`] to find and take over as stale.
#[derive(Debug)]
pub struct SessionLease {
    path: PathBuf,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        // Only remove it if it still looks like ours — guards against the
        // rare case where our own lease was itself taken over as stale (e.g.
        // this process was frozen past `STALE_TTL`) by the time we get here;
        // removing it then would delete someone else's live lock.
        let Ok(data) = std::fs::read_to_string(&self.path) else {
            return;
        };
        let Ok(info) = serde_json::from_str::<LeaseInfo>(&data) else {
            return;
        };
        if info.pid == std::process::id() && info.hostname == IDENTITY.0 {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Acquires the write lease for session `id`, whose lockfile lives in `dir`
/// (a `HistoryManager`'s `history_dir`).
///
/// Succeeds immediately if there's no existing lease, the existing lease is
/// already this process's own (idempotent — re-acquiring just refreshes
/// `acquired_at`), or the existing lease is stale (its holder pid is
/// confirmed dead on this host, or — when liveness can't be checked — it's
/// older than [`STALE_TTL`]); a stale takeover is logged as a warning.
///
/// Otherwise returns [`Error::SessionLocked`] naming the holder.
pub fn acquire(dir: &Path, id: &Uuid) -> Result<SessionLease> {
    let path = lock_path(dir, id);
    let info = LeaseInfo {
        pid: std::process::id(),
        hostname: IDENTITY.0.clone(),
        acquired_at: Utc::now(),
        process_start: IDENTITY.1,
    };
    let contents = serde_json::to_string_pretty(&info)?;

    let existing = std::fs::read_to_string(&path)
        .ok()
        .and_then(|data| serde_json::from_str::<LeaseInfo>(&data).ok());

    let Some(existing) = existing else {
        // Nothing readable at `path` — either it's genuinely absent, or it
        // exists but is unreadable/corrupt. Either way there's no live claim
        // to respect, but only the "genuinely absent" case is safe to claim
        // via an exclusive create: `create_new` fails against a file that's
        // merely unparsable, so that case falls back to the unconditional
        // rename, same as a confirmed-stale takeover below.
        return match create_lease_exclusively(&path, &contents) {
            Ok(()) => Ok(SessionLease { path }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // Lost the race to claim a brand-new lease — report whoever
                // actually won it rather than a generic error, same as the
                // "live foreign lease" case below.
                let winner = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|data| serde_json::from_str::<LeaseInfo>(&data).ok());
                Err(match winner {
                    Some(w) => Error::SessionLocked {
                        session_id: id.to_string(),
                        pid: w.pid,
                        host: w.hostname,
                    },
                    None => Error::SessionLocked {
                        session_id: id.to_string(),
                        pid: 0,
                        host: "unknown".to_string(),
                    },
                })
            }
            Err(_) => {
                // Path exists but wasn't readable as a `LeaseInfo` above —
                // overwrite the corrupt file via rename.
                write_atomic(&path, &contents)?;
                Ok(SessionLease { path })
            }
        };
    };

    if existing.pid == std::process::id() && existing.hostname == IDENTITY.0 {
        // Idempotent refresh of our own lease; the file already exists, so
        // this always goes through the rename path.
        write_atomic(&path, &contents)?;
        return Ok(SessionLease { path });
    }

    let same_host = existing.hostname == IDENTITY.0;
    let alive = same_host.then(|| pid_is_alive(existing.pid)).flatten();
    let stale = match alive {
        Some(true) => false,
        Some(false) => true,
        // Can't confirm liveness (different host, or this platform can't
        // check): fall back to age.
        None => std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .map(|m| m.elapsed().unwrap_or_default() > STALE_TTL)
            .unwrap_or(true),
    };

    if !stale {
        return Err(Error::SessionLocked {
            session_id: id.to_string(),
            pid: existing.pid,
            host: existing.hostname,
        });
    }
    tracing::warn!(
        session_id = %id,
        pid = existing.pid,
        host = %existing.hostname,
        "taking over stale session lease"
    );
    write_atomic(&path, &contents)?;
    Ok(SessionLease { path })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn acquire_succeeds_when_no_lease_exists() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let lease = acquire(dir.path(), &id).unwrap();
        assert!(lock_path(dir.path(), &id).exists());
        drop(lease);
    }

    #[test]
    fn create_lease_exclusively_fails_if_the_path_already_exists() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("some.lock");
        create_lease_exclusively(&path, "first").unwrap();

        let err = create_lease_exclusively(&path, "second").unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        // The loser's write must not have clobbered the winner's content.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");
    }

    #[test]
    fn reacquiring_our_own_lease_succeeds() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let _first = acquire(dir.path(), &id).unwrap();
        // Same process, same session: must not be treated as contention.
        let _second = acquire(dir.path(), &id).unwrap();
    }

    #[test]
    fn acquire_refuses_a_live_foreign_lease() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let path = lock_path(dir.path(), &id);
        // Our own pid (guaranteed alive) but a foreign hostname: the
        // liveness check is skipped (different host) and the fresh mtime
        // keeps it under the TTL, so the lease must be refused as live.
        let info = LeaseInfo {
            pid: std::process::id(),
            hostname: "some-other-host".to_string(),
            acquired_at: Utc::now(),
            process_start: Utc::now(),
        };
        write_atomic(&path, &serde_json::to_string_pretty(&info).unwrap()).unwrap();

        let err = acquire(dir.path(), &id).unwrap_err();
        assert!(matches!(err, Error::SessionLocked { pid, ref host, .. }
            if pid == std::process::id() && host == "some-other-host"));
    }

    #[test]
    fn acquire_takes_over_a_lease_from_a_dead_pid_on_this_host() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let path = lock_path(dir.path(), &id);
        // An implausibly high pid, essentially guaranteed to be unassigned
        // (and so not alive) on any real system.
        let info = LeaseInfo {
            pid: 0x7FFF_FFFE,
            hostname: IDENTITY.0.clone(),
            acquired_at: Utc::now(),
            process_start: Utc::now(),
        };
        write_atomic(&path, &serde_json::to_string_pretty(&info).unwrap()).unwrap();

        let lease = acquire(dir.path(), &id).unwrap();
        let data = std::fs::read_to_string(&path).unwrap();
        let now_held: LeaseInfo = serde_json::from_str(&data).unwrap();
        assert_eq!(now_held.pid, std::process::id());
        drop(lease);
    }

    #[test]
    fn acquire_takes_over_a_stale_cross_host_lease_past_the_ttl() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let path = lock_path(dir.path(), &id);
        let info = LeaseInfo {
            pid: std::process::id(),
            hostname: "some-other-host".to_string(),
            acquired_at: Utc::now() - chrono::Duration::hours(2),
            process_start: Utc::now(),
        };
        write_atomic(&path, &serde_json::to_string_pretty(&info).unwrap()).unwrap();
        // Back-date the file's mtime past the TTL (write_atomic just set it to now).
        let old = std::time::SystemTime::now() - Duration::from_secs(60 * 60);
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();

        let lease = acquire(dir.path(), &id).unwrap();
        drop(lease);
    }

    #[test]
    fn drop_releases_the_lease() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let path = lock_path(dir.path(), &id);
        {
            let _lease = acquire(dir.path(), &id).unwrap();
            assert!(path.exists());
        }
        assert!(!path.exists());
    }

    #[test]
    fn drop_does_not_delete_a_lease_that_was_taken_over_by_someone_else() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        let path = lock_path(dir.path(), &id);
        let lease = acquire(dir.path(), &id).unwrap();

        // Simulate another process stealing it (e.g. after we were frozen
        // past the TTL).
        let foreign = LeaseInfo {
            pid: 424242,
            hostname: "someone-elses-host".to_string(),
            acquired_at: Utc::now(),
            process_start: Utc::now(),
        };
        write_atomic(&path, &serde_json::to_string_pretty(&foreign).unwrap()).unwrap();

        drop(lease);
        assert!(path.exists(), "must not delete another holder's lease");
    }
}
