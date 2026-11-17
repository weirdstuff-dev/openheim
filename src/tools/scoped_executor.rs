//! Tool-allowlist wrapper around any [`ToolExecutor`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::Result;

use super::{ToolCapabilities, ToolExecutor};

/// Wraps an inner [`ToolExecutor`] and restricts it to a fixed set of tool names.
///
/// `list_tools` only returns tools whose name appears in `allowed`; `execute`
/// rejects calls to any other name with a descriptive string rather than an
/// error, so the calling LLM can read and react to the restriction (see the
/// "error as content" guidance in `docs/custom-tools.md`).
///
/// Used by [`crate::tools::delegate::DelegateTool`] to enforce a subagent
/// profile's optional `tools` allowlist.
pub struct ScopedExecutor {
    inner: Arc<dyn ToolExecutor>,
    allowed: Vec<String>,
}

impl ScopedExecutor {
    pub fn new(inner: Arc<dyn ToolExecutor>, allowed: Vec<String>) -> Self {
        Self { inner, allowed }
    }

    fn is_allowed(&self, name: &str) -> bool {
        self.allowed.iter().any(|a| a == name)
    }
}

#[async_trait]
impl ToolExecutor for ScopedExecutor {
    fn list_tools(&self) -> Vec<Tool> {
        self.inner
            .list_tools()
            .into_iter()
            .filter(|t| self.is_allowed(&t.function.name))
            .collect()
    }

    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String> {
        if !self.is_allowed(name) {
            return Ok(format!(
                "Tool '{name}' is not available to this agent. Available tools: {}",
                self.allowed.join(", ")
            ));
        }
        self.inner.execute(name, args_json, turn).await
    }

    fn capabilities(&self, name: &str) -> ToolCapabilities {
        self.inner.capabilities(name)
    }
}
