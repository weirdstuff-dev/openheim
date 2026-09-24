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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_support::TurnHarness;

    /// A tool named `name` that answers with `reply`.
    struct Named {
        name: &'static str,
        reply: &'static str,
        read_only: bool,
    }

    #[async_trait]
    impl ToolHandler for Named {
        fn definition(&self) -> Tool {
            Tool::function(
                self.name,
                self.reply,
                serde_json::json!({"type": "object", "properties": {}}),
            )
        }

        async fn execute(&self, _args: &str, _turn: &TurnContext<'_>) -> Result<String> {
            Ok(self.reply.to_string())
        }

        fn capabilities(&self) -> ToolCapabilities {
            ToolCapabilities {
                read_only: self.read_only,
                ..ToolCapabilities::default()
            }
        }
    }

    fn overlay() -> OverlayExecutor {
        let mut inner = crate::tools::SystemToolExecutor::new();
        inner.register(Box::new(Named {
            name: "shared",
            reply: "old",
            read_only: true,
        }));
        inner.register(Box::new(Named {
            name: "other",
            reply: "other",
            read_only: true,
        }));
        OverlayExecutor::new(
            Arc::new(inner),
            Arc::new(Named {
                name: "shared",
                reply: "new",
                read_only: false,
            }),
        )
    }

    #[tokio::test]
    async fn routes_the_overlaid_name_to_the_handler_and_the_rest_to_inner() {
        let executor = overlay();
        let harness = TurnHarness::new();
        let turn = harness.turn();

        assert_eq!(
            executor.execute("shared", "{}", &turn).await.unwrap(),
            "new"
        );
        assert_eq!(
            executor.execute("other", "{}", &turn).await.unwrap(),
            "other"
        );
    }

    #[test]
    fn lists_the_overlaid_tool_once_with_the_handlers_definition() {
        let tools = overlay().list_tools();
        let shared: Vec<_> = tools
            .iter()
            .filter(|t| t.function.name == "shared")
            .collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].function.description, "new");
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn capabilities_come_from_the_handler_for_the_overlaid_name() {
        let executor = overlay();
        assert!(!executor.capabilities("shared").read_only);
        assert!(executor.capabilities("other").read_only);
    }
}
