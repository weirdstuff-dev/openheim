use crate::config::config_dir;
use crate::error::{Error, Result};
use std::path::{Component, Path, PathBuf};

/// Checks that `name` is safe to interpolate into a skills-directory path.
/// Skill names are untrusted input — they arrive from remote ACP clients
/// (`session/new` `_meta.skills`), from `default_skills` in the config file,
/// and from persisted conversation metadata — so a name must be exactly one
/// normal path component. This rejects `""`, `.`/`..`, any `/` or `\`
/// separator (platform-specific parsing), Windows drive prefixes (`C:evil`),
/// and any `..` substring.
fn validate_skill_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    let is_single_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !is_single_normal_component || name.contains("..") || name.contains('\0') {
        return Err(Error::NotFound(format!(
            "Invalid skill name '{name}': must be a single path component without '..'"
        )));
    }
    Ok(())
}

/// Manages Markdown skill files stored in `~/.openheim/skills/`.
///
/// A skill is a named Markdown file (`{name}.md`) containing system-level
/// instructions. Skills are loaded by [`PromptBuilder::add_skill`] and injected
/// into the LLM prompt as a system message, letting users extend the agent's
/// behaviour without modifying code.
///
/// # Example skill file: `~/.openheim/skills/rust.md`
///
/// ```markdown
/// You are an expert Rust programmer. Always prefer idiomatic Rust.
/// Avoid unsafe code unless strictly necessary.
/// ```
#[derive(Clone)]
pub struct SkillsManager {
    skills_dir: PathBuf,
}

impl SkillsManager {
    /// Creates a `SkillsManager` backed by `~/.openheim/skills/`, creating the
    /// directory if it doesn't exist.
    pub fn new() -> Result<Self> {
        let dir = config_dir()?.join("skills");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { skills_dir: dir })
    }

    /// Loads the content of a single skill by name.
    ///
    /// Reads `{skills_dir}/{name}.md`. Returns an error if the file does not
    /// exist or the name is invalid (see [`validate_skill_name`]). The path is
    /// canonicalized and checked for containment so a symlink inside the
    /// skills directory cannot redirect the read outside it; the canonical
    /// (fully resolved) path is what gets read.
    pub fn load_skill(&self, name: &str) -> Result<String> {
        validate_skill_name(name)?;
        let path = self.skills_dir.join(format!("{}.md", name));
        if !path.exists() {
            return Err(Error::NotFound(format!(
                "Skill '{}' not found at {}",
                name,
                path.display()
            )));
        }
        // Both sides must be canonicalized: the file to resolve symlinks (and
        // to read the resolved path, closing the check→read swap window), the
        // directory because the skills dir may itself sit behind a symlink
        // (e.g. macOS `/var` → `/private/var`) and the comparison would
        // otherwise fail for every skill.
        let dir_canonical = self.skills_dir.canonicalize().map_err(Error::IoError)?;
        let canonical = path.canonicalize().map_err(Error::IoError)?;
        if !canonical.starts_with(&dir_canonical) {
            return Err(Error::NotFound(format!(
                "Skill '{}' resolves outside the skills directory",
                name
            )));
        }
        let content = std::fs::read_to_string(&canonical)?;
        Ok(content)
    }

    /// Loads multiple skills by name, returning `(name, content)` pairs in the same order.
    ///
    /// Returns an error on the first skill that is not found or has an invalid name.
    pub fn load_skills(&self, names: &[String]) -> Result<Vec<(String, String)>> {
        let mut skills = Vec::new();
        for name in names {
            let content = self.load_skill(name)?;
            skills.push((name.clone(), content));
        }
        Ok(skills)
    }

    /// Returns the names of all available skills, sorted alphabetically.
    ///
    /// Only `.md` files whose stem is a valid skill name (see
    /// [`Self::load_skill`]) are considered; the extension is stripped from
    /// the returned names so every advertised skill is loadable.
    pub fn list_skills(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.skills_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && validate_skill_name(stem).is_ok()
            {
                names.push(stem.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Test-only constructor pointing at a specific skills directory
    /// (mirrors `HistoryManager::with_dir`).
    #[cfg(test)]
    pub fn with_dir(dir: PathBuf) -> Self {
        Self { skills_dir: dir }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_skill_by_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("rust.md"), "Be idiomatic.").unwrap();
        let mgr = SkillsManager::with_dir(dir.path().to_path_buf());
        assert_eq!(mgr.load_skill("rust").unwrap(), "Be idiomatic.");
    }

    #[test]
    fn missing_skill_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = SkillsManager::with_dir(dir.path().to_path_buf());
        let err = mgr.load_skill("absent").unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[test]
    fn rejects_traversal_names_before_touching_filesystem() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = SkillsManager::with_dir(dir.path().to_path_buf());
        for bad in [
            "",
            ".",
            "..",
            "../escape",
            "../../etc/passwd",
            "/etc/passwd",
            "a/b",
            "sub/../x",
            "a..b",
            "..\\..\\windows\\evil",
        ] {
            let err = mgr.load_skill(bad).unwrap_err();
            assert!(
                err.to_string().contains("Invalid skill name"),
                "name {bad:?}: unexpected error {err}"
            );
        }
    }

    #[test]
    fn rejects_traversal_even_when_target_exists() {
        // Without name validation, `../secret` joined to `<dir>/skills/`
        // would reach `<dir>/secret.md` and return its content.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("skills")).unwrap();
        std::fs::write(dir.path().join("secret.md"), "TOP SECRET").unwrap();
        let mgr = SkillsManager::with_dir(dir.path().join("skills"));
        let err = mgr.load_skill("../secret").unwrap_err();
        assert!(err.to_string().contains("Invalid skill name"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escaping_skills_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("skills")).unwrap();
        let outside = dir.path().join("outside.md");
        std::fs::write(&outside, "secret").unwrap();
        std::os::unix::fs::symlink(&outside, dir.path().join("skills/link.md")).unwrap();
        let mgr = SkillsManager::with_dir(dir.path().join("skills"));
        let err = mgr.load_skill("link").unwrap_err();
        assert!(
            err.to_string().contains("outside the skills directory"),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn allows_symlink_inside_skills_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("skills")).unwrap();
        std::fs::write(dir.path().join("skills/real.md"), "real").unwrap();
        std::os::unix::fs::symlink("real.md", dir.path().join("skills/link.md")).unwrap();
        let mgr = SkillsManager::with_dir(dir.path().join("skills"));
        assert_eq!(mgr.load_skill("link").unwrap(), "real");
    }

    #[test]
    fn list_skills_skips_unloadable_names() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("good.md"), "x").unwrap();
        std::fs::write(dir.path().join("we..ird.md"), "x").unwrap();
        std::fs::write(dir.path().join("notmd.txt"), "x").unwrap();
        let mgr = SkillsManager::with_dir(dir.path().to_path_buf());
        assert_eq!(mgr.list_skills().unwrap(), vec!["good".to_string()]);
    }
}
