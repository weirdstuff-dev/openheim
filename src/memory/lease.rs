//! Advisory per-conversation write lease, so two `openheim` processes
//! sharing a history directory (say, the desktop app and the CLI) don't
//! write the same conversation at once.
//!
//! `AgentState::prompt` holds it for one turn only, so viewing a session
//! never locks it; a prompt from a second process fails with
//! [`crate::error::Error::SessionLocked`] while a turn runs.
//!
//! The lease is a JSON lockfile, `{uuid}.lock`, holding the owner's pid and
//! host rather than an OS file lock: portable, readable, and a lease left by
//! a crashed process can be recognised and taken over.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;
use uuid::Uuid;

use crate::error::{Error, Result};

/// Age (by file mtime) after which a lease may be taken over when its holder
/// can't be checked (another host, or no pid check on this platform). On the
/// same host a live pid always keeps it.
const STALE_TTL: Duration = Duration::from_secs(30 * 60);

/// Age after which a lockfile that isn't a readable lease is taken over.
/// Until then it may be a lease its creator is still writing.
const CORRUPT_LEASE_GRACE: Duration = Duration::from_secs(10);

/// Contents of a `{uuid}.lock` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeaseInfo {
    pid: u32,
    hostname: String,
    acquired_at: DateTime<Utc>,
    /// When the holder process first acquired any lease: with the pid, tells
    /// the holder apart from an unrelated process that reused its pid. (Not
    /// the OS process start time, which isn't portable.)
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

/// Whether `path` was last modified more than `age` ago (`true` if its
/// modification time can't be read).
fn older_than(path: &Path, age: Duration) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .map(|m| m.elapsed().unwrap_or_default() > age)
        .unwrap_or(true)
}

fn lock_path(dir: &Path, id: &Uuid) -> PathBuf {
    dir.join(format!("{id}.lock"))
}

/// Writes `contents` to `path` via a temp file and rename, so readers never
/// see a partial write. Only for refreshing or taking over an existing
/// lockfile; a first claim uses [`create_lease_exclusively`], so two
/// processes can't both win it.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    // A unique temp name per call, so two writers to the same lockfile can't
    // clobber each other's temp file.
    let mut tmp_path = path.as_os_str().to_owned();
    tmp_path.push(format!(".{}.{}.tmp", std::process::id(), Uuid::new_v4()));
    let tmp_path = PathBuf::from(tmp_path);
    std::fs::write(&tmp_path, contents)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Creates `path` with `contents`, failing with
/// [`std::io::ErrorKind::AlreadyExists`] if it's already there, so of two
/// processes claiming a lease at once exactly one wins.
fn create_lease_exclusively(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(contents.as_bytes())
}

/// A held write lease; dropping it removes the lockfile. A crash leaves the
/// file behind for the next [`acquire`] to take over as stale.
#[derive(Debug)]
pub struct SessionLease {
    path: PathBuf,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        // Only if it's still ours: it may have been taken over as stale
        // (e.g. this process was suspended past `STALE_TTL`).
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
/// Succeeds if there's no lease, the lease is already this process's (it's
/// refreshed), or it's stale: its pid is dead on this host, or, when that
/// can't be checked, it's older than [`STALE_TTL`] (logged as a warning).
/// Otherwise fails with [`Error::SessionLocked`] naming the holder.
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
        // No readable lease: the file is absent, or isn't a lease.
        return match create_lease_exclusively(&path, &contents) {
            Ok(()) => Ok(SessionLease { path }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let holder = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|data| serde_json::from_str::<LeaseInfo>(&data).ok());
                match holder {
                    // Another process claimed it first.
                    Some(h) => Err(Error::SessionLocked {
                        session_id: id.to_string(),
                        pid: h.pid,
                        host: h.hostname,
                    }),
                    None if older_than(&path, CORRUPT_LEASE_GRACE) => {
                        tracing::warn!(session_id = %id, "taking over unreadable session lease");
                        write_atomic(&path, &contents)?;
                        Ok(SessionLease { path })
                    }
                    None => Err(Error::SessionLocked {
                        session_id: id.to_string(),
                        pid: 0,
                        host: "unknown".to_string(),
                    }),
                }
            }
            Err(e) => Err(e.into()),
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
        None => older_than(&path, STALE_TTL),
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

    /// Writes `contents` to `id`'s lockfile, last modified `age` ago.
    fn write_lockfile(dir: &Path, id: &Uuid, contents: &str, age: Duration) {
        let path = lock_path(dir, id);
        std::fs::write(&path, contents).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - age)
            .unwrap();
    }

    // A lockfile that isn't a lease (e.g. left empty by a crash right after
    // it was created) doesn't lock its session for good.
    #[test]
    fn an_unreadable_lockfile_is_taken_over_once_it_is_old_enough() {
        let dir = tempdir().unwrap();
        for contents in ["", "{not json"] {
            let id = Uuid::new_v4();
            write_lockfile(dir.path(), &id, contents, CORRUPT_LEASE_GRACE * 2);

            let lease = acquire(dir.path(), &id).unwrap();

            let data = std::fs::read_to_string(lock_path(dir.path(), &id)).unwrap();
            let info: LeaseInfo = serde_json::from_str(&data).unwrap();
            assert_eq!(info.pid, std::process::id(), "{contents:?}");
            drop(lease);
        }
    }

    // A fresh unreadable lockfile may be a lease still being written by the
    // process that just created it.
    #[test]
    fn a_fresh_unreadable_lockfile_still_locks() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        write_lockfile(dir.path(), &id, "", Duration::ZERO);

        let err = acquire(dir.path(), &id).unwrap_err();
        assert!(matches!(err, Error::SessionLocked { .. }), "{err}");
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
