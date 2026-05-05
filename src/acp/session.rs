use std::path::PathBuf;
use uuid::Uuid;

use crate::config::AgentConfig;

#[derive(Debug)]
pub struct SessionState {
    pub chat_id: Uuid,
    pub config: AgentConfig,
    pub cwd: PathBuf,
    pub skills: Vec<String>,
}
