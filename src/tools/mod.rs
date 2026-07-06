//! Tool abstraction layer: trait definitions, built-in tools, and the runtime
//! executor that routes LLM tool calls to the correct handler.
//!
//! # Built-in tools
//!
//! Three tools are registered by default via [`SystemToolExecutor::register_builtins`]:
//!
//! | Name | Description |
//! |------|-------------|
//! | `execute_command` | Run a shell command (`sh -c` on Unix, `cmd /C` on Windows) |
//! | `read_file` | Read a file from disk |
//! | `write_file` | Write a file to disk, creating parent directories as needed |
//!
//! Additional tools are loaded from MCP servers and registered under the
//! `{server_name}__{tool_name}` namespace.
//!
//! # Implementing a custom tool
//!
//! ```rust,no_run
//! use async_trait::async_trait;
//! use serde_json::json;
//!
//! struct GreetTool;
//!
//! # use openheim::tools::ToolHandler;
//! # use openheim::core::models::{Tool, FunctionDefinition};
//! # use openheim::error::Result;
//! #[async_trait]
//! impl ToolHandler for GreetTool {
//!     fn definition(&self) -> Tool {
//!         Tool {
//!             tool_type: "function".to_string(),
//!             function: FunctionDefinition {
//!                 name: "greet".to_string(),
//!                 description: "Greet someone by name.".to_string(),
//!                 parameters: json!({
//!                     "type": "object",
//!                     "properties": { "name": { "type": "string" } },
//!                     "required": ["name"]
//!                 }),
//!             },
//!         }
//!     }
//!
//!     async fn execute(&self, args: &str) -> Result<String> {
//!         let v: serde_json::Value = serde_json::from_str(args)?;
//!         let name = v["name"].as_str().unwrap_or("world");
//!         Ok(format!("Hello, {name}!"))
//!     }
//! }
//! ```
//!
//! Register it with [`crate::client::OpenheimBuilder::tool`] when embedding
//! openheim as a library:
//!
//! ```rust,no_run
//! # use openheim::OpenheimClient;
//! # use openheim::tools::ToolHandler;
//! # struct GreetTool;
//! # #[async_trait::async_trait]
//! # impl ToolHandler for GreetTool {
//! #     fn definition(&self) -> openheim::core::models::Tool { unimplemented!() }
//! #     async fn execute(&self, _args: &str) -> openheim::error::Result<String> { unimplemented!() }
//! # }
//! # async fn wiring() -> openheim::error::Result<()> {
//! let client = OpenheimClient::builder()
//!     .tool(Box::new(GreetTool))
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! `SystemToolExecutor::register` (shown below) is the lower-level entry
//! point `OpenheimBuilder::tool` uses internally — reach for it directly only
//! if you're constructing an [`crate::acp::AgentState`] yourself instead of
//! going through the builder.

pub mod delegate;
mod execute_command;
mod read_file;
pub mod sandbox;
mod sandboxed_executor;
mod scoped_executor;
mod write_file;

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;

use crate::config::McpServerConfig;
use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

pub use delegate::{DELEGATE_TOOL_NAME, DelegateTool, with_delegation};
pub use sandboxed_executor::SandboxedExecutor;
pub use scoped_executor::ScopedExecutor;

#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Returns the tool definition (name, description, JSON-schema parameters).
    fn definition(&self) -> Tool;

    /// Executes the tool with the given JSON-encoded arguments.
    async fn execute(&self, args: &str) -> Result<String>;
}

/// Routes LLM tool-call requests to the correct [`ToolHandler`].
///
/// The production implementation is [`SystemToolExecutor`]. Tests typically use
/// lightweight mock implementations of this trait.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Returns the list of tools available to the LLM.
    fn list_tools(&self) -> Vec<Tool>;

    /// Dispatches a tool call by name with JSON-encoded arguments.
    ///
    /// `turn` carries the calling turn's cancellation token and permission
    /// gate through to wrappers/tools that need them — currently only
    /// [`DelegateTool`], which reuses both for the subagent turn it spawns
    /// rather than manufacturing its own.
    ///
    /// Returns the tool output as a string, or an error if the tool is unknown
    /// or its execution fails.
    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String>;
}

/// The default tool executor used by the agent runtime.
///
/// Maintains a registry of [`ToolHandler`]s keyed by tool name and dispatches
/// LLM tool calls to the appropriate handler. Built-in tools are registered via
/// [`register_builtins`]; MCP tools are added during [`build`].
pub struct SystemToolExecutor {
    handlers: HashMap<String, Box<dyn ToolHandler>>,
}

impl SystemToolExecutor {
    /// Creates an empty executor with no registered tools.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Builds a fully-configured executor: registers built-in tools then connects
    /// to all configured MCP servers and registers their tools.
    ///
    /// When `allow_shell` is `false` the `execute_command` tool is omitted so
    /// the LLM never sees it in its tool list.
    ///
    /// Returns the executor alongside [`McpServerStatus`] entries for each server
    /// so callers can inspect which connections succeeded.
    pub async fn build(
        mcp_configs: &BTreeMap<String, McpServerConfig>,
        allow_shell: bool,
    ) -> (Self, Vec<crate::mcp::McpServerStatus>) {
        let mut executor = Self::new();
        executor.register_builtins();
        if !allow_shell {
            executor.handlers.remove("execute_command");
        }
        let (handlers, statuses) = crate::mcp::load_mcp_tools(mcp_configs).await;
        for handler in handlers {
            executor.register(handler);
        }
        (executor, statuses)
    }

    /// Registers the three built-in tools: `execute_command`, `read_file`, `write_file`.
    pub fn register_builtins(&mut self) {
        self.register(Box::new(execute_command::ExecuteCommandTool));
        self.register(Box::new(read_file::ReadFileTool));
        self.register(Box::new(write_file::WriteFileTool));
    }

    /// Registers a single tool handler.
    ///
    /// If a tool with the same name is already registered it is overwritten and a
    /// warning is logged.
    pub fn register(&mut self, handler: Box<dyn ToolHandler>) {
        let name = handler.definition().function.name.clone();
        if self.handlers.contains_key(&name) {
            tracing::warn!(name = %name, "Tool name collision: overwriting existing handler");
        }
        self.handlers.insert(name, handler);
    }
}

impl Default for SystemToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolExecutor for SystemToolExecutor {
    fn list_tools(&self) -> Vec<Tool> {
        self.handlers.values().map(|h| h.definition()).collect()
    }

    async fn execute(
        &self,
        name: &str,
        args_json: &str,
        _turn: &TurnContext<'_>,
    ) -> Result<String> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| Error::ToolExecutionError(format!("Unknown tool: {}", name)))?;
        handler.execute(args_json).await
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use crate::core::permission::{AllowAll, PermissionGate};
    use crate::core::turn::TurnContext;

    /// Owns the pieces a test [`TurnContext`] borrows from, so callers can do
    /// `let turn = harness.turn();` without fighting temporary lifetimes.
    pub(crate) struct TurnHarness {
        cancel: CancellationToken,
        permission_gate: Arc<dyn PermissionGate>,
    }

    impl TurnHarness {
        pub(crate) fn new() -> Self {
            Self {
                cancel: CancellationToken::new(),
                permission_gate: Arc::new(AllowAll),
            }
        }

        pub(crate) fn turn(&self) -> TurnContext<'_> {
            TurnContext {
                cancel: &self.cancel,
                permission_gate: &self.permission_gate,
            }
        }

        /// A clone of the underlying token, so a test can cancel the turn
        /// from outside while `turn()` is borrowed elsewhere.
        pub(crate) fn cancel_handle(&self) -> CancellationToken {
            self.cancel.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::test_support::TurnHarness;
    use super::*;

    #[test]
    fn new_executor_is_empty() {
        let executor = SystemToolExecutor::new();
        assert_eq!(executor.handlers.len(), 0);
    }

    #[test]
    fn register_builtins_adds_three_tools() {
        let mut executor = SystemToolExecutor::new();
        executor.register_builtins();
        assert!(executor.handlers.contains_key("execute_command"));
        assert!(executor.handlers.contains_key("read_file"));
        assert!(executor.handlers.contains_key("write_file"));
        assert_eq!(executor.handlers.len(), 3);
    }

    #[tokio::test]
    async fn executor_returns_error_for_unknown_tool() {
        let executor = SystemToolExecutor::new();
        let harness = TurnHarness::new();
        let result = executor
            .execute("nonexistent_tool", "{}", &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown tool"));
    }

    #[tokio::test]
    async fn build_without_shell_omits_execute_command() {
        let (executor, _) = SystemToolExecutor::build(&BTreeMap::new(), false).await;
        assert!(!executor.handlers.contains_key("execute_command"));
        assert!(executor.handlers.contains_key("read_file"));
        assert!(executor.handlers.contains_key("write_file"));
    }
}
