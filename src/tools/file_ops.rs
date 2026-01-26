use crate::error::{Error, Result};
use std::path::Path;
use tokio::fs;

pub async fn read_file_async(path: &str) -> Result<String> {
    let content = fs::read_to_string(path).await.map_err(Error::IoError)?;
    Ok(content)
}

pub async fn write_file_async(path: &str, content: &str) -> Result<String> {
    fs::write(path, content).await.map_err(Error::IoError)?;
    Ok(format!("Successfully wrote to {}", path))
}

pub fn read_file(path: &str) -> Result<String> {
    if path.is_empty() {
        return Err(Error::ParseError("Empty path".to_string()));
    }

    // Use existing runtime if available, otherwise create one.
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(read_file_async(path)),
        Err(_) => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| Error::Other(format!("Failed to create runtime: {}", e)))?;
            rt.block_on(read_file_async(path))
        }
    }
}

pub fn write_file(path: &str, content: &str) -> Result<String> {
    if path.is_empty() {
        return Err(Error::ParseError("Empty path".to_string()));
    }

    // Ensure parent directory exists if a parent is provided.
    if let Some(parent) = Path::new(path).parent() {
        if !parent.exists() {
            // Try to create parent directories synchronously.
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::IoError(e))?;
        }
    }

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(write_file_async(path, content)),
        Err(_) => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| Error::Other(format!("Failed to create runtime: {}", e)))?;
            rt.block_on(write_file_async(path, content))
        }
    }
}