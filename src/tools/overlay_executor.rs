//! Single-tool override on top of any [`ToolExecutor`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::Result;

use super::{ToolCapabilities, ToolExecutor, ToolHandler};

/// Wraps an inner [`ToolExecutor`] and routes one tool name to `handler`
/// instead, adding it if the inner executor doesn't have it.
///
/// Used by `AgentState::prompt` to swap in a per-turn `delegate_task` bound
/// to the session's live model, without rebuilding the shared registry.
pub(crate) struct OverlayExecutor {
    inner: Arc<dyn ToolExecutor>,
    handler: Arc<dyn ToolHandler>,
    name: String,
}

impl OverlayExecutor {
    pub(crate) fn new(inner: Arc<dyn ToolExecutor>, handler: Arc<dyn ToolHandler>) -> Self {
        let name = handler.definition().function.name;
        Self {
            inner,
            handler,
            name,
        }
    }
}

#[async_trait]
impl ToolExecutor for OverlayExecutor {
    /// The inner list with the overlaid tool's definition swapped in at the
    /// same position, so the order matches the inner executor's.
    fn list_tools(&self) -> Vec<Tool> {
        let mut tools = self.inner.list_tools();
        match tools.iter_mut().find(|t| t.function.name == self.name) {
            Some(existing) => *existing = self.handler.definition(),
            None => tools.push(self.handler.definition()),
        }
        tools
    }

    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String> {
        if name == self.name {
            self.handler.execute(args_json, turn).await
        } else {
            self.inner.execute(name, args_json, turn).await
        }
    }

    fn capabilities(&self, name: &str) -> ToolCapabilities {
        if name == self.name {
            self.handler.capabilities()
        } else {
            self.inner.capabilities(name)
        }
    }
}
