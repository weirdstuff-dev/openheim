use crate::config::config_dir;
use crate::error::{Error, Result};
use std::path::PathBuf;

pub struct SkillsManager {
    skills_dir: PathBuf,
}

impl SkillsManager {
    pub fn new() -> Result<Self> {
        let dir = config_dir()
            .map_err(|e| Error::Other(e.to_string()))?
            .join("skills");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { skills_dir: dir })
    }

    pub fn load_skill(&self, name: &str) -> Result<String> {
        let path = self.skills_dir.join(format!("{}.md", name));
        if !path.exists() {
            return Err(Error::Other(format!(
                "Skill '{}' not found at {}",
                name,
                path.display()
            )));
        }
        let content = std::fs::read_to_string(&path)?;
        Ok(content)
    }

    pub fn load_skills(&self, names: &[String]) -> Result<Vec<(String, String)>> {
        let mut skills = Vec::new();
        for name in names {
            let content = self.load_skill(name)?;
            skills.push((name.clone(), content));
        }
        Ok(skills)
    }

    pub fn list_skills(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.skills_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }
}
