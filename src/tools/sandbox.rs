//! Work-directory path validation for the agent sandbox.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Validates that `requested` resolves to a path within `work_dir`.
///
/// Relative paths are resolved against `work_dir`. For paths that already
/// exist symlinks are followed and the canonicalized result is checked.
/// For paths that do not yet exist (e.g. a file about to be written) the
/// nearest existing ancestor is canonicalized and checked instead.
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
    let resolved = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        work_dir_canonical.join(requested_path)
    };

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
                // The non-existing tail of the path is fine; return it as-is
                // so the caller (write_file) can create it.
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
}
