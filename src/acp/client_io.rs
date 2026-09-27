//! [`ClientIo`] backed by ACP's `fs/read_text_file` / `fs/write_text_file`.

use std::sync::Arc;

use agent_client_protocol::{
    Client, ConnectionTo,
    schema::v1::{
        ClientCapabilities, ReadTextFileRequest, ReadTextFileResponse, WriteTextFileRequest,
        WriteTextFileResponse,
    },
};
use tokio::sync::RwLock;

use crate::{
    core::client_io::{ClientIo, LineRange},
    error::{Error, Result},
};

/// The `fs/read_text_file` request for `lines` of `path`.
fn read_request(session_id: &str, path: &std::path::Path, lines: LineRange) -> ReadTextFileRequest {
    ReadTextFileRequest::new(session_id.to_string(), path.to_path_buf())
        .line(lines.line)
        .limit(lines.limit)
}

/// Only attempts a request when the client actually advertised the
/// corresponding capability at `initialize` time; otherwise defers to local I/O.
pub(super) struct AcpClientIo {
    pub(super) cx: ConnectionTo<Client>,
    pub(super) session_id: String,
    pub(super) client_capabilities: Arc<RwLock<ClientCapabilities>>,
}

#[async_trait::async_trait]
impl ClientIo for AcpClientIo {
    async fn read_file(&self, path: &std::path::Path, lines: LineRange) -> Option<Result<String>> {
        if !self.client_capabilities.read().await.fs.read_text_file {
            return None;
        }
        let response = self
            .cx
            .send_request(read_request(&self.session_id, path, lines))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_request_carries_the_line_range() {
        let path = std::path::Path::new("/w/a.txt");
        let ranged = read_request(
            "s1",
            path,
            LineRange {
                line: Some(10),
                limit: Some(20),
            },
        );
        assert_eq!(ranged.path, path);
        assert_eq!(ranged.line, Some(10));
        assert_eq!(ranged.limit, Some(20));

        let whole = read_request("s1", path, LineRange::default());
        assert_eq!(whole.line, None);
        assert_eq!(whole.limit, None);
    }
}
