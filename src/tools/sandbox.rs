//! Work-directory path validation for the agent sandbox.

use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

/// Lexically normalizes an absolute `path` without touching the filesystem.
///
/// Strips `.` components and resolves `..` against the preceding component
/// (popping past the root of an absolute path is a no-op, matching POSIX
/// `/.. == /`). The result contains no `.` or `..` components.
///
/// This is deliberately lexical, not kernel-equivalent: `link/..` pops the
/// `link` component itself rather than following the symlink. For sandbox
/// purposes that is the safe direction — the normalized path is what callers
/// open, and it leaves no `..` segments for the kernel to re-resolve at
/// syscall time.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// Validates that `requested` resolves to a path within `work_dir`.
///
/// Relative paths are resolved against `work_dir`, and the joined path is
/// normalized *lexically* before any filesystem probing so `..` segments
/// cannot hide behind a non-existent prefix component. The normalized path
/// is what gets validated and returned; callers therefore never open a path
/// that still contains `..`.
///
/// For paths that already exist symlinks are followed and the canonicalized
/// result is checked. For paths that do not yet exist (e.g. a file about to
/// be written) the nearest existing ancestor is canonicalized and checked
/// instead.
///
/// Returns the resolved absolute path on success, or an error describing
/// why the path is rejected.
pub fn validate_path(requested: &str, work_dir: &Path) -> Result<PathBuf> {
    validate_path_from(requested, work_dir, work_dir)
}

/// [`validate_path`], with a relative `requested` taken from `cwd` instead
/// of from `work_dir`. The result must still lie inside `work_dir`, wherever
/// `cwd` is.
pub fn validate_path_from(requested: &str, cwd: &Path, work_dir: &Path) -> Result<PathBuf> {
    let work_dir_canonical = canonical_dir(work_dir)?;
    let base = if cwd == work_dir {
        work_dir_canonical.clone()
    } else {
        canonical_dir(cwd)?
    };
    let resolved = resolve(requested, &base);
    check_inside(&resolved, requested, work_dir, &work_dir_canonical)
}

/// Validates `requested` for acting on the directory entry itself (delete,
/// rename) rather than on what it points to.
///
/// Unlike [`validate_path`], a final symlink component is *not* followed:
/// the returned path names the link, so deleting or renaming it affects the
/// link and not its target (which may be a directory whose contents would
/// otherwise be removed). Every earlier component is resolved and must stay
/// inside `work_dir`, exactly as in [`validate_path`].
///
/// The work directory itself is rejected: it is inside the sandbox, but
/// deleting or moving it would take the whole workspace with it.
pub fn validate_entry(requested: &str, work_dir: &Path) -> Result<PathBuf> {
    let work_dir_canonical = canonical_dir(work_dir)?;
    let resolved = resolve(requested, &work_dir_canonical);
    let is_root = resolved == work_dir_canonical || resolved == lexical_normalize(work_dir);
    let (Some(parent), Some(name), false) = (resolved.parent(), resolved.file_name(), is_root)
    else {
        return Err(Error::ToolExecutionError(format!(
            "path '{requested}' is the work directory itself"
        )));
    };
    let parent = check_inside(parent, requested, work_dir, &work_dir_canonical)?;
    Ok(parent.join(name))
}

fn canonical_dir(dir: &Path) -> Result<PathBuf> {
    dir.canonicalize().map_err(|_| {
        Error::ToolExecutionError(format!("directory '{}' is inaccessible", dir.display()))
    })
}

/// `requested` as an absolute, lexically normalized path; relative paths
/// are taken from `base`.
fn resolve(requested: &str, base: &Path) -> PathBuf {
    let requested_path = Path::new(requested);
    let joined = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        base.join(requested_path)
    };
    // Normalize before probing the filesystem. Without this, a path like
    // `x/../../../outside/f` looks non-existent to `exists()` (the kernel
    // cannot resolve `x/..` while `x` is missing), the ancestor walk validates
    // only `work_dir`, and the raw `..`-bearing path is returned — later
    // resolved by the kernel after `create_dir_all` builds the prefix.
    lexical_normalize(&joined)
}

/// Checks that the normalized path `resolved` (following symlinks if it
/// exists, else via its nearest existing ancestor) lies inside the work
/// directory, and returns the path callers should use. `requested` and
/// `work_dir` are only for error messages.
fn check_inside(
    resolved: &Path,
    requested: &str,
    work_dir: &Path,
    work_dir_canonical: &Path,
) -> Result<PathBuf> {
    let check = if resolved.exists() {
        resolved.canonicalize().map_err(Error::IoError)?
    } else {
        // Dangling symlinks look non-existent to exists(); detect them explicitly
        // so write_file cannot create the symlink target outside the sandbox.
        if resolved
            .symlink_metadata()
            .ok()
            .is_some_and(|m| m.file_type().is_symlink())
        {
            return Err(Error::ToolExecutionError(format!(
                "path '{}' is a dangling symlink (work directory: '{}')",
                requested,
                work_dir.display()
            )));
        }
        // Walk up the tree until we find an existing ancestor, canonicalize
        // that, and verify it is within the work directory.
        let mut ancestor: &Path = resolved;
        loop {
            ancestor = ancestor.parent().ok_or_else(|| {
                Error::ToolExecutionError(format!(
                    "path '{}' has no accessible ancestor within the filesystem",
                    requested
                ))
            })?;
            if ancestor.exists() {
                let canonical_ancestor = ancestor.canonicalize().map_err(Error::IoError)?;
                if !canonical_ancestor.starts_with(work_dir_canonical) {
                    return Err(Error::ToolExecutionError(format!(
                        "path '{}' is outside the work directory '{}'",
                        requested,
                        work_dir.display()
                    )));
                }
                // The non-existing tail of the path is fine; return the
                // normalized path (no `..` components) so the caller can
                // create it without the kernel re-resolving anything.
                return Ok(resolved.to_path_buf());
            }
        }
    };

    if check.starts_with(work_dir_canonical) {
        Ok(check)
    } else {
        Err(Error::ToolExecutionError(format!(
            "path '{}' is outside the work directory '{}'",
            requested,
            work_dir.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn allows_existing_file_inside_work_dir() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("foo.txt");
        fs::write(&file, "x").unwrap();
        assert!(validate_path(file.to_str().unwrap(), dir.path()).is_ok());
    }

    #[test]
    fn allows_relative_path_inside() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        assert!(validate_path("sub", dir.path()).is_ok());
    }

    #[test]
    fn allows_new_file_path_inside_work_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(validate_path("new_file.txt", dir.path()).is_ok());
    }

    #[test]
    fn rejects_absolute_path_outside_work_dir() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let outside = other.path().join("secret.txt");
        fs::write(&outside, "x").unwrap();
        let err = validate_path(outside.to_str().unwrap(), dir.path()).unwrap_err();
        assert!(err.to_string().contains("outside the work directory"));
    }

    #[test]
    fn rejects_dotdot_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let err = validate_path("../../etc/passwd", dir.path()).unwrap_err();
        assert!(err.to_string().contains("outside the work directory"));
    }

    #[test]
    fn rejects_dangling_symlink_inside_work_dir() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("dangling_link");
        // Point the symlink at a path that does not exist so it is dangling.
        std::os::unix::fs::symlink("/nonexistent_target_path_12345", &link).unwrap();
        let err = validate_path(link.to_str().unwrap(), dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("dangling symlink"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_dotdot_behind_nonexistent_prefix() {
        let dir = tempfile::tempdir().unwrap();
        // `x` does not exist, so every ancestor containing `x/..` returns
        // ENOENT — without normalization the ancestor walk would validate
        // only the work dir and hand back the raw `..`-bearing path.
        let err = validate_path("x/../../../pwned.txt", dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_dotdot_behind_nonexistent_prefix_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let requested = canonical.join("missing/../../escape.txt");
        let err = validate_path(requested.to_str().unwrap(), dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn returns_normalized_path_without_dotdot_components() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let resolved = validate_path("sub/../new_file.txt", dir.path()).unwrap();
        assert_eq!(
            resolved,
            dir.path().canonicalize().unwrap().join("new_file.txt")
        );
        assert!(
            !resolved
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        );
    }

    #[test]
    fn relative_paths_resolve_from_cwd_but_stay_inside_work_dir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        let cwd = root.join("sub");

        assert_eq!(
            validate_path_from("a.txt", &cwd, dir.path()).unwrap(),
            root.join("sub/a.txt")
        );
        assert_eq!(
            validate_path_from("../b.txt", &cwd, dir.path()).unwrap(),
            root.join("b.txt")
        );
        let err = validate_path_from("../../c.txt", &cwd, dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "{err}"
        );
    }

    // A `cwd` outside the work directory can't widen what's reachable.
    #[test]
    fn a_cwd_outside_work_dir_grants_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "x").unwrap();

        let err = validate_path_from("secret.txt", outside.path(), dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn validate_entry_names_a_symlink_itself_not_its_target() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::create_dir(root.join("src")).unwrap();
        std::os::unix::fs::symlink(root.join("src"), root.join("link")).unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("out_link")).unwrap();
        std::os::unix::fs::symlink(&root, root.join("root_link")).unwrap();

        // `validate_path` follows the link; `validate_entry` doesn't, even
        // when the target is outside the sandbox or is the root itself.
        assert_eq!(validate_path("link", dir.path()).unwrap(), root.join("src"));
        for name in ["link", "out_link", "root_link"] {
            assert_eq!(validate_entry(name, dir.path()).unwrap(), root.join(name));
        }
    }

    #[cfg(unix)]
    #[test]
    fn validate_entry_still_resolves_symlinked_parents() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("victim.txt"), "data").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("out")).unwrap();

        let err = validate_entry("out/victim.txt", dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "{err}"
        );
    }

    #[test]
    fn validate_entry_rejects_the_work_dir_itself() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let absolute = dir.path().to_str().unwrap().to_string();
        let canonical = dir.path().canonicalize().unwrap();

        for requested in ["", ".", "sub/..", &absolute, canonical.to_str().unwrap()] {
            let err = validate_entry(requested, dir.path()).unwrap_err();
            assert!(
                err.to_string().contains("work directory itself"),
                "{requested:?}: {err}"
            );
        }
    }
}
