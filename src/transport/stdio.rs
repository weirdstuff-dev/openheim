use std::sync::Arc;

use agent_client_protocol_tokio::Stdio;

use crate::{
    acp::{self, AgentState},
    config::load_config,
    rag::RagContext,
};

pub async fn run() -> crate::error::Result<()> {
    let app_config = load_config()?;
    let agent_config = app_config.resolve(None)?;
    let rag = RagContext::new()?;
    let state = Arc::new(AgentState::new(agent_config, app_config, rag).await?);

    acp::serve(Stdio::new(), state)
        .await
        .map_err(|e| crate::error::Error::Other(e.to_string()))
}
