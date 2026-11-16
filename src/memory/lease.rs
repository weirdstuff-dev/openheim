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
/// same host a live pid keeps it, unless the pid now belongs to a different
/// process.
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
    /// When the holder process first acquired any lease. Informational only;
    /// still written because older versions can't read a lease without it.
    process_start: DateTime<Utc>,
    /// The holder process's [`process_start_token`], which tells it apart
    /// from an unrelated process that later got the same pid. Absent in
    /// leases written by older versions or on platforms without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    start_token: Option<String>,
    /// The holder's [`pid_namespace`]: its pid only means something to a
    /// process in the same one. Absent in leases written by older versions
    /// or where there's none to tell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pid_namespace: Option<String>,
}

impl LeaseInfo {
    /// Whether the holder's pid can be checked from here: same host and,
    /// where known, the same PID namespace. Containers that share a
    /// hostname (host networking) but not a PID namespace see different
    /// processes under the same pid, so to each other they're another host.
    /// A lease from an older version, which records no namespace, is judged
    /// by hostname alone.
    fn pid_visible_here(&self) -> bool {
        self.hostname == IDENTITY.0
            && (self.pid_namespace.is_none() || self.pid_namespace == *OWN_PID_NAMESPACE)
    }

    /// Whether this process holds the lease: our pid, as seen from here.
    fn is_ours(&self) -> bool {
        self.pid == std::process::id() && self.pid_visible_here()
    }
}

/// This process's identity, computed once and reused for every lease it
/// acquires.
static IDENTITY: LazyLock<(String, DateTime<Utc>)> =
    LazyLock::new(|| (current_hostname(), Utc::now()));

/// This process's [`process_start_token`].
static OWN_START_TOKEN: LazyLock<Option<String>> =
    LazyLock::new(|| process_start_token(std::process::id()));

/// This process's [`pid_namespace`].
static OWN_PID_NAMESPACE: LazyLock<Option<String>> = LazyLock::new(pid_namespace);

/// The PID namespace this process lives in (`pid:[4026531836]`), which
/// decides what its pids refer to. `None` where there's none to tell.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn pid_namespace() -> Option<String> {
    let link = std::fs::read_link("/proc/self/ns/pid").ok()?;
    Some(link.to_string_lossy().into_owned())
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn pid_namespace() -> Option<String> {
    None
}

/// The name leases record as their holder's host: the system hostname, or,
/// on a system without one, [`fallback_host_id`] in the temp directory.
fn current_hostname() -> String {
    system_hostname().unwrap_or_else(|| fallback_host_id(&std::env::temp_dir()))
}

#[cfg(unix)]
fn system_hostname() -> Option<String> {
    let name = nix::unistd::gethostname().ok()?.into_string().ok()?;
    non_empty(name)
}

#[cfg(windows)]
fn system_hostname() -> Option<String> {
    non_empty(std::env::var("COMPUTERNAME").ok()?)
}

#[cfg(not(any(unix, windows)))]
fn system_hostname() -> Option<String> {
    None
}

fn non_empty(name: String) -> Option<String> {
    let name = name.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// A random host id kept in `dir`, created by the first process to need
/// it. `dir` is the temp directory, which belongs to one machine or
/// container: unlike the data directory, containers that share a data
/// volume don't share it, so they don't take each other for one host.
///
/// If the file can't be read or created, the id is this process's alone.
/// Other processes then treat its leases as another host's, taken over only
/// once stale, never as their own.
fn fallback_host_id(dir: &Path) -> String {
    let path = dir.join("openheim-host-id");
    let read = || std::fs::read_to_string(&path).ok().and_then(non_empty);
    if let Some(id) = read() {
        return id;
    }
    let id = format!("host-{}", Uuid::new_v4());
    match create_exclusively(&path, &id) {
        Ok(()) => id,
        // Another process created it first: use its id.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => read().unwrap_or(id),
        Err(_) => id,
    }
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

/// A value that identifies process `pid` for as long as it runs and differs
/// for a later process given the same pid: the kernel's boot id plus the
/// process's start time in ticks since boot. Only compared for equality,
/// so wall-clock changes don't affect it. `None` if the process doesn't
/// exist or can't be inspected.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn process_start_token(pid: u32) -> Option<String> {
    let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 22, `starttime`. Field 2, the command name, is parenthesised
    // and may contain spaces, so fields are counted from after its `)`:
    // the first one there is field 3.
    let start_ticks: u64 = stat[stat.rfind(')')? + 1..]
        .split_whitespace()
        .nth(19)?
        .parse()
        .ok()?;
    Some(format!("{}:{start_ticks}", boot_id.trim()))
}

/// A value that identifies process `pid` for as long as it runs and differs
/// for a later process given the same pid: its start time to the
/// microsecond, as the kernel recorded it. `None` if the process doesn't
/// exist or can't be inspected.
#[cfg(target_os = "macos")]
fn process_start_token(pid: u32) -> Option<String> {
    let pid = libc::c_int::try_from(pid).ok()?;
    // SAFETY: `proc_bsdinfo` is a plain C struct of integers and byte
    // arrays, for which all zeroes is a valid value.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: `info` is a writable buffer of `size` bytes, the size
    // PROC_PIDTBSDINFO writes.
    let written =
        unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size) };
    (written == size).then(|| format!("{}.{:06}", info.pbi_start_tvsec, info.pbi_start_tvusec))
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
fn process_start_token(_pid: u32) -> Option<String> {
    None
}

/// Whether the live process with `lease.pid` on this host is not the one
/// that wrote `lease`: both start tokens are known and differ. If either is
/// unknown, the pid is trusted.
fn pid_was_reused(lease: &LeaseInfo) -> bool {
    match (&lease.start_token, process_start_token(lease.pid)) {
        (Some(recorded), Some(current)) => *recorded != current,
        _ => false,
    }
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
/// lockfile; a first claim uses [`create_exclusively`], so two
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
/// processes claiming a lease (or creating the host id) at once exactly one
/// wins.
fn create_exclusively(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(contents.as_bytes())
}

/// A held write lease; dropping it removes the lockfile. A crash leaves the
/// file behind for the next `acquire` to take over as stale.
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
        if info.is_ours() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Acquires the write lease for session `id`, whose lockfile lives in `dir`
/// (a `HistoryManager`'s `history_dir`).
///
/// Succeeds if there's no lease, the lease is already this process's (it's
/// refreshed), or it's stale: its pid is dead on this host or now belongs
/// to another process, or, when liveness can't be checked, it's older than
/// [`STALE_TTL`] (logged as a warning).
/// Otherwise fails with [`Error::SessionLocked`] naming the holder.
pub fn acquire(dir: &Path, id: &Uuid) -> Result<SessionLease> {
    let path = lock_path(dir, id);
    let info = LeaseInfo {
        pid: std::process::id(),
        hostname: IDENTITY.0.clone(),
        acquired_at: Utc::now(),
        process_start: IDENTITY.1,
        start_token: OWN_START_TOKEN.clone(),
        pid_namespace: OWN_PID_NAMESPACE.clone(),
    };
    let contents = serde_json::to_string_pretty(&info)?;

    let existing = std::fs::read_to_string(&path)
        .ok()
        .and_then(|data| serde_json::from_str::<LeaseInfo>(&data).ok());

    let Some(existing) = existing else {
        // No readable lease: the file is absent, or isn't a lease.
        return match create_exclusively(&path, &contents) {
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

    if existing.is_ours() {
        // Idempotent refresh of our own lease; the file already exists, so
        // this always goes through the rename path.
        write_atomic(&path, &contents)?;
        return Ok(SessionLease { path });
    }

    let alive = existing
        .pid_visible_here()
        .then(|| pid_is_alive(existing.pid))
        .flatten();
    let stale = match alive {
        // A live pid keeps the lease unless it now belongs to another
        // process (after a crash and reboot, say).
        Some(true) => pid_was_reused(&existing),
        Some(false) => true,
        // Can't confirm liveness (different host or PID namespace, or this
        // platform can't check): fall back to age.
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
    fn create_exclusively_fails_if_the_path_already_exists() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("some.lock");
        create_exclusively(&path, "first").unwrap();

        let err = create_exclusively(&path, "second").unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        // The loser's write must not have clobbered the winner's content.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn system_hostname_is_found_without_a_hostname_binary() {
        let name = system_hostname().expect("the test machine has a hostname");
        assert!(!name.is_empty() && name.trim() == name);
    }

    // Processes on one machine agree on the fallback id: the first creates
    // it, later ones read it.
    #[test]
    fn fallback_host_id_is_created_once_and_then_reused() {
        let dir = tempdir().unwrap();
        let first = fallback_host_id(dir.path());
        assert!(first.starts_with("host-"), "{first}");
        assert_eq!(fallback_host_id(dir.path()), first);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("openheim-host-id")).unwrap(),
            first
        );
    }

    // Two machines (or containers) with their own temp directories get
    // different ids.
    #[test]
    fn fallback_host_ids_differ_between_machines() {
        let (a, b) = (tempdir().unwrap(), tempdir().unwrap());
        assert_ne!(fallback_host_id(a.path()), fallback_host_id(b.path()));
    }

    // Without a writable place for the id, each call (each process, since
    // it's computed once) gets its own, never a shared constant.
    #[test]
    fn fallback_host_id_without_a_usable_dir_is_unique() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("no-such-dir");
        let (a, b) = (fallback_host_id(&missing), fallback_host_id(&missing));
        assert!(a.starts_with("host-") && b.starts_with("host-"));
        assert_ne!(a, b);
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
            start_token: None,
            pid_namespace: None,
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
            start_token: None,
            pid_namespace: None,
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
            start_token: None,
            pid_namespace: None,
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

    /// A lease on this host but from another PID namespace (a container
    /// sharing the hostname), held by `pid` and last modified `age` ago.
    fn write_other_namespace_lease(dir: &Path, id: &Uuid, pid: u32, age: Duration) {
        let info = LeaseInfo {
            pid,
            hostname: IDENTITY.0.clone(),
            acquired_at: Utc::now(),
            process_start: Utc::now(),
            start_token: Some("theirs".to_string()),
            pid_namespace: Some("pid:[1]".to_string()),
        };
        write_lockfile(dir, id, &serde_json::to_string(&info).unwrap(), age);
    }

    // Its pid means nothing here: neither dead nor reused, so the lease
    // holds until it's past the TTL, as another host's would.
    #[cfg(unix)]
    #[test]
    fn a_lease_from_another_pid_namespace_is_judged_by_age() {
        let dir = tempdir().unwrap();
        for pid in [0x7FFF_FFFE, std::os::unix::process::parent_id()] {
            let id = Uuid::new_v4();
            write_other_namespace_lease(dir.path(), &id, pid, Duration::ZERO);
            let err = acquire(dir.path(), &id).unwrap_err();
            assert!(
                matches!(err, Error::SessionLocked { .. }),
                "pid {pid}: {err}"
            );
        }

        let id = Uuid::new_v4();
        write_other_namespace_lease(dir.path(), &id, 0x7FFF_FFFE, STALE_TTL * 2);
        drop(acquire(dir.path(), &id).unwrap());
    }

    // Our pid number in another namespace is another process.
    #[test]
    fn a_lease_with_our_pid_from_another_pid_namespace_is_not_ours() {
        let dir = tempdir().unwrap();
        let id = Uuid::new_v4();
        write_other_namespace_lease(dir.path(), &id, std::process::id(), Duration::ZERO);

        let err = acquire(dir.path(), &id).unwrap_err();
        assert!(matches!(err, Error::SessionLocked { .. }), "{err}");
        drop(SessionLease {
            path: lock_path(dir.path(), &id),
        });
        assert!(lock_path(dir.path(), &id).exists(), "dropped as if ours");
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn the_pid_namespace_is_known_on_linux() {
        let ns = OWN_PID_NAMESPACE
            .clone()
            .expect("readable /proc/self/ns/pid");
        assert!(ns.starts_with("pid:["), "{ns}");
    }

    // A lease written by a version without `start_token` still parses.
    #[test]
    fn a_lease_without_a_start_token_is_readable() {
        let data = r#"{"pid":1,"hostname":"h","acquired_at":"2026-01-01T00:00:00Z","process_start":"2026-01-01T00:00:00Z"}"#;
        let info: LeaseInfo = serde_json::from_str(data).unwrap();
        assert_eq!(info.start_token, None);
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    mod pid_reuse {
        use super::*;

        /// Writes a lease on this host held by our parent process, a live
        /// pid that isn't ours, carrying `start_token`.
        fn write_parent_lease(dir: &Path, id: &Uuid, start_token: Option<String>) {
            let info = LeaseInfo {
                pid: std::os::unix::process::parent_id(),
                hostname: IDENTITY.0.clone(),
                acquired_at: Utc::now(),
                process_start: Utc::now(),
                start_token,
                pid_namespace: OWN_PID_NAMESPACE.clone(),
            };
            write_atomic(
                &lock_path(dir, id),
                &serde_json::to_string_pretty(&info).unwrap(),
            )
            .unwrap();
        }

        #[test]
        fn start_tokens_tell_processes_apart() {
            let own = process_start_token(std::process::id()).expect("own token");
            assert_eq!(process_start_token(std::process::id()), Some(own.clone()));
            let parent =
                process_start_token(std::os::unix::process::parent_id()).expect("parent token");
            assert_ne!(own, parent);
            assert_eq!(process_start_token(0x7FFF_FFFE), None);
        }

        #[test]
        fn acquire_records_our_start_token() {
            let dir = tempdir().unwrap();
            let id = Uuid::new_v4();
            let _lease = acquire(dir.path(), &id).unwrap();
            let data = std::fs::read_to_string(lock_path(dir.path(), &id)).unwrap();
            let info: LeaseInfo = serde_json::from_str(&data).unwrap();
            assert_eq!(info.start_token, *OWN_START_TOKEN);
            assert!(info.start_token.is_some());
        }

        // The lease's pid is alive, but its process started at a different
        // time than the holder: the pid was reused, so the lease is stale.
        #[test]
        fn a_live_pid_with_another_start_token_is_taken_over() {
            let dir = tempdir().unwrap();
            let id = Uuid::new_v4();
            write_parent_lease(dir.path(), &id, Some("not-the-parent".to_string()));

            let lease = acquire(dir.path(), &id).unwrap();
            let data = std::fs::read_to_string(lock_path(dir.path(), &id)).unwrap();
            let info: LeaseInfo = serde_json::from_str(&data).unwrap();
            assert_eq!(info.pid, std::process::id());
            drop(lease);
        }

        #[test]
        fn a_live_pid_with_its_own_start_token_keeps_the_lease() {
            let dir = tempdir().unwrap();
            let id = Uuid::new_v4();
            let parent = std::os::unix::process::parent_id();
            write_parent_lease(dir.path(), &id, process_start_token(parent));

            let err = acquire(dir.path(), &id).unwrap_err();
            assert!(matches!(err, Error::SessionLocked { pid, .. } if pid == parent));
        }

        // A lease from an older version has no token to compare, so a live
        // pid keeps it.
        #[test]
        fn a_live_pid_without_a_start_token_keeps_the_lease() {
            let dir = tempdir().unwrap();
            let id = Uuid::new_v4();
            write_parent_lease(dir.path(), &id, None);

            let err = acquire(dir.path(), &id).unwrap_err();
            assert!(matches!(err, Error::SessionLocked { .. }), "{err}");
        }
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
            start_token: None,
            pid_namespace: None,
        };
        write_atomic(&path, &serde_json::to_string_pretty(&foreign).unwrap()).unwrap();

        drop(lease);
        assert!(path.exists(), "must not delete another holder's lease");
    }
}
