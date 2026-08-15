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
    let work_dir_canonical = work_dir.canonicalize().map_err(|_| {
        Error::ToolExecutionError(format!(
            "work directory '{}' is inaccessible",
            work_dir.display()
        ))
    })?;

    let requested_path = Path::new(requested);
    let joined = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        work_dir_canonical.join(requested_path)
    };
    // C1: normalize before probing the filesystem. Without this, a path like
    // `x/../../../outside/f` looks non-existent to `exists()` (the kernel
    // cannot resolve `x/..` while `x` is missing), the ancestor walk validates
    // only `work_dir`, and the raw `..`-bearing path is returned — later
    // resolved by the kernel after `create_dir_all` builds the prefix.
    let resolved = lexical_normalize(&joined);

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
        let mut ancestor: &Path = &resolved;
        loop {
            ancestor = ancestor.parent().ok_or_else(|| {
                Error::ToolExecutionError(format!(
                    "path '{}' has no accessible ancestor within the filesystem",
                    requested
                ))
            })?;
            if ancestor.exists() {
                let canonical_ancestor = ancestor.canonicalize().map_err(Error::IoError)?;
                if !canonical_ancestor.starts_with(&work_dir_canonical) {
                    return Err(Error::ToolExecutionError(format!(
                        "path '{}' is outside the work directory '{}'",
                        requested,
                        work_dir.display()
                    )));
                }
                // The non-existing tail of the path is fine; return the
                // normalized path (no `..` components) so the caller can
                // create it without the kernel re-resolving anything.
                return Ok(resolved);
            }
        }
    };

    if check.starts_with(&work_dir_canonical) {
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
        // ENOENT — the pre-normalization ancestor walk validated only the
        // work dir and handed back the raw `..`-bearing path (C1).
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
}
