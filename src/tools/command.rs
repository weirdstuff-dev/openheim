use crate::error::{Error, Result};
use std::process::Command;

pub fn execute_command(command: &str) -> Result<String> {
    #[cfg(target_family = "unix")]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };

    #[cfg(target_family = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    };

    let output = cmd
        .output()
        .map_err(|e| Error::ToolExecutionError(format!("Failed to execute command: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Ok(format!(
            "Command failed:\nStdout: {}\nStderr: {}",
            stdout, stderr
        ))
    }
}
