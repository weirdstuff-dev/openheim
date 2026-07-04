use crate::config::config_dir;
use crate::error::{Error, Result};
use std::path::PathBuf;

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
    /// Reads `{skills_dir}/{name}.md`. Returns an error if the file does not exist.
    pub fn load_skill(&self, name: &str) -> Result<String> {
        let path = self.skills_dir.join(format!("{}.md", name));
        if !path.exists() {
            return Err(Error::NotFound(format!(
                "Skill '{}' not found at {}",
                name,
                path.display()
            )));
        }
        let content = std::fs::read_to_string(&path)?;
        Ok(content)
    }

    /// Loads multiple skills by name, returning `(name, content)` pairs in the same order.
    ///
    /// Returns an error on the first skill that is not found.
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
    /// Only `.md` files are considered; the extension is stripped from the returned names.
    pub fn list_skills(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.skills_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                names.push(stem.to_string());
            }
        }
        names.sort();
        Ok(names)
    }
}
