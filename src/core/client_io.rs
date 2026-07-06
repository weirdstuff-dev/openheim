//! Optional delegation of file I/O to the ACP client, for agents running
//! inside an editor sandbox that owns the filesystem view (unsaved buffers,
//! virtual filesystems, etc.) rather than the process's own `tokio::fs`.
//!
//! Mirrors [`crate::core::permission`]: a protocol-agnostic trait here, with
//! the ACP-specific implementation (backed by `fs/read_text_file` and
//! `fs/write_text_file`) living in [`crate::acp`].

use std::path::Path;

use async_trait::async_trait;

use crate::error::Result;

/// Asked before falling back to local filesystem I/O for `read_file` /
/// `write_file`. Returning `None` means "not available, use local I/O" —
/// either the client doesn't advertise the capability, or there's no live
/// ACP client connection at all (TUI, library embedding, subagents).
#[async_trait]
pub trait ClientIo: Send + Sync {
    async fn read_file(&self, path: &Path) -> Option<Result<String>>;
    async fn write_file(&self, path: &Path, content: &str) -> Option<Result<()>>;
}

/// Default: always defers to local filesystem I/O.
pub struct NoClientIo;

#[async_trait]
impl ClientIo for NoClientIo {
    async fn read_file(&self, _path: &Path) -> Option<Result<String>> {
        None
    }

    async fn write_file(&self, _path: &Path, _content: &str) -> Option<Result<()>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_client_io_always_defers_to_local() {
        assert!(NoClientIo.read_file(Path::new("/tmp/x")).await.is_none());
        assert!(
            NoClientIo
                .write_file(Path::new("/tmp/x"), "hi")
                .await
                .is_none()
        );
    }
}
