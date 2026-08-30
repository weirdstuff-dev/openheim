//! [`ClientIo`] backed by ACP's `fs/read_text_file` / `fs/write_text_file`.

use std::sync::Arc;

use agent_client_protocol::{
    Client, ConnectionTo,
    schema::{
        ClientCapabilities, ReadTextFileRequest, ReadTextFileResponse, WriteTextFileRequest,
        WriteTextFileResponse,
    },
};
use tokio::sync::RwLock;

use crate::{
    core::client_io::ClientIo,
    error::{Error, Result},
};

/// Only attempts a request when the client actually advertised the
/// corresponding capability at `initialize` time; otherwise defers to local I/O.
pub(super) struct AcpClientIo {
    pub(super) cx: ConnectionTo<Client>,
    pub(super) session_id: String,
    pub(super) client_capabilities: Arc<RwLock<ClientCapabilities>>,
}

#[async_trait::async_trait]
impl ClientIo for AcpClientIo {
    async fn read_file(&self, path: &std::path::Path) -> Option<Result<String>> {
        if !self.client_capabilities.read().await.fs.read_text_file {
            return None;
        }
        let response = self
            .cx
            .send_request(ReadTextFileRequest::new(
                self.session_id.clone(),
                path.to_path_buf(),
            ))
            .block_task()
            .await;
        Some(match response {
            Ok(ReadTextFileResponse { content, .. }) => Ok(content),
            Err(e) => Err(Error::Other(format!("fs/read_text_file failed: {e}"))),
        })
    }

    async fn write_file(&self, path: &std::path::Path, content: &str) -> Option<Result<()>> {
        if !self.client_capabilities.read().await.fs.write_text_file {
            return None;
        }
        let response = self
            .cx
            .send_request(WriteTextFileRequest::new(
                self.session_id.clone(),
                path.to_path_buf(),
                content,
            ))
            .block_task()
            .await;
        Some(match response {
            Ok(WriteTextFileResponse { .. }) => Ok(()),
            Err(e) => Err(Error::Other(format!("fs/write_text_file failed: {e}"))),
        })
    }
}
