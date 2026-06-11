//! Tool-allowlist wrapper around any [`ToolExecutor`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::core::models::Tool;
use crate::error::Result;

use super::ToolExecutor;

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

    async fn execute(&self, name: &str, args_json: &str) -> Result<String> {
        if !self.is_allowed(name) {
            return Ok(format!(
                "Tool '{name}' is not available to this agent. Available tools: {}",
                self.allowed.join(", ")
            ));
        }
        self.inner.execute(name, args_json).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::models::FunctionDefinition;
    use crate::error::Error;

    struct FixedExecutor(Vec<&'static str>);

    #[async_trait]
    impl ToolExecutor for FixedExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            self.0
                .iter()
                .map(|name| Tool {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: name.to_string(),
                        description: String::new(),
                        parameters: serde_json::json!({"type": "object", "properties": {}}),
                    },
                })
                .collect()
        }

        async fn execute(&self, name: &str, _args_json: &str) -> Result<String> {
            if self.0.contains(&name) {
                Ok(format!("ran {name}"))
            } else {
                Err(Error::ToolExecutionError(format!("unknown tool: {name}")))
            }
        }
    }

    #[test]
    fn list_tools_filters_to_allowlist() {
        let inner = Arc::new(FixedExecutor(vec![
            "read_file",
            "write_file",
            "execute_command",
        ]));
        let scoped = ScopedExecutor::new(inner, vec!["read_file".to_string()]);
        let names: Vec<_> = scoped
            .list_tools()
            .into_iter()
            .map(|t| t.function.name)
            .collect();
        assert_eq!(names, vec!["read_file"]);
    }

    #[tokio::test]
    async fn execute_forwards_allowed_calls() {
        let inner = Arc::new(FixedExecutor(vec!["read_file"]));
        let scoped = ScopedExecutor::new(inner, vec!["read_file".to_string()]);
        let result = scoped.execute("read_file", "{}").await.unwrap();
        assert_eq!(result, "ran read_file");
    }

    #[tokio::test]
    async fn execute_returns_descriptive_string_for_disallowed_calls() {
        let inner = Arc::new(FixedExecutor(vec!["read_file", "execute_command"]));
        let scoped = ScopedExecutor::new(inner, vec!["read_file".to_string()]);
        let result = scoped.execute("execute_command", "{}").await.unwrap();
        assert!(result.contains("not available"));
        assert!(result.contains("read_file"));
    }
}
