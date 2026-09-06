//! Tool abstraction layer: trait definitions, built-in tools, and the runtime
//! executor that routes LLM tool calls to the correct handler.
//!
//! # Built-in tools
//!
//! Seven tools are registered by default via [`SystemToolExecutor::register_builtins`]:
//!
//! | Name | Description |
//! |------|-------------|
//! | `execute_command` | Run a shell command (`sh -c` on Unix, `cmd /C` on Windows) in the work directory, with a hard timeout, cancellation, and output caps |
//! | `read_file` | Read a file from disk |
//! | `write_file` | Write a file to disk, creating parent directories as needed |
//! | `edit_file` | Replace an exact string in a file, without rewriting the whole thing |
//! | `list_dir` | List the immediate contents of a directory |
//! | `search` | Regex search across files, ripgrep-style (built on ripgrep's own crates, `.gitignore`-aware) |
//! | `web_fetch` | Fetch a public http(s) URL and return its content as text, with an SSRF guard and size/time caps |
//!
//! Every filesystem tool validates its path against the turn's work
//! directory ([`sandbox::validate_path`]) and delegates reads/writes to the
//! turn's [`ClientIo`](crate::core::client_io::ClientIo) when one is
//! available; there is no separate sandbox wrapper.
//!
//! With the `rag` feature, `AgentState` also registers `remember`,
//! `search_memory`, and `forget` (see `crate::rag::tool`), and
//! `delegate_task` ([`DelegateTool`]) is always registered.
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
//! # use openheim::tools::args::{parse_args, require_str};
//! # use openheim::core::models::{Tool, FunctionDefinition};
//! # use openheim::core::turn::TurnContext;
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
//!     async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
//!         let args = parse_args(args)?;
//!         let name = require_str(&args, "name")?;
//!         // `turn` carries the cancel token, the work directory, and the
//!         // client I/O hook; see `TurnContext` for what to do with each.
//!         Ok(format!("Hello, {name}! (from {})", turn.work_dir.display()))
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
//! # use openheim::core::turn::TurnContext;
//! # struct GreetTool;
//! # #[async_trait::async_trait]
//! # impl ToolHandler for GreetTool {
//! #     fn definition(&self) -> openheim::core::models::Tool { unimplemented!() }
//! #     async fn execute(&self, _args: &str, _turn: &TurnContext<'_>) -> openheim::error::Result<String> { unimplemented!() }
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

pub mod args;
pub mod delegate;
mod edit_file;
mod execute_command;
mod list_dir;
mod read_file;
pub mod sandbox;
mod scoped_executor;
mod search;
mod web_fetch;
mod write_file;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::McpServerConfig;
use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

pub use delegate::{DELEGATE_TOOL_NAME, DelegateTool};
pub use scoped_executor::ScopedExecutor;

#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Returns the tool definition (name, description, JSON-schema parameters).
    fn definition(&self) -> Tool;

    /// Executes the tool with the given JSON-encoded arguments.
    ///
    /// `turn` is the calling turn's [`TurnContext`]: race long work against
    /// `turn.cancel`, confine filesystem access to `turn.work_dir` (via
    /// [`sandbox::validate_path`]), and prefer `turn.client_io` for file
    /// reads/writes so an editor-hosted client can serve its own buffers.
    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String>;
}

/// Routes LLM tool-call requests to the correct [`ToolHandler`].
///
/// The production implementation is [`SystemToolExecutor`]. Tests typically use
/// lightweight mock implementations of this trait.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Returns the list of tools available to the LLM.
    fn list_tools(&self) -> Vec<Tool>;

    /// Dispatches a tool call by name with JSON-encoded arguments, passing
    /// the calling turn's context through to the handler.
    ///
    /// Returns the tool output as a string, or an error if the tool is unknown
    /// or its execution fails.
    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String>;
}

/// The default tool executor used by the agent runtime.
///
/// Maintains a registry of [`ToolHandler`]s keyed by tool name and dispatches
/// LLM tool calls to the appropriate handler. Built-in tools are registered via
/// [`register_builtins`](Self::register_builtins); MCP tools are added during
/// [`build`](Self::build).
///
/// Cloning is cheap and yields an independent registry sharing the same
/// handlers — a snapshot of the tool set as it exists at that moment. The
/// runtime uses this to hand [`DelegateTool`] a view that predates its own
/// registration, so subagents can never delegate recursively.
#[derive(Clone)]
pub struct SystemToolExecutor {
    handlers: HashMap<String, Arc<dyn ToolHandler>>,
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
    /// This is the one place `allow_shell` is enforced: when it's `false` the
    /// `execute_command` tool is never registered, so the LLM neither sees it
    /// nor can call it.
    ///
    /// Returns the executor alongside [`McpServerStatus`](crate::mcp::McpServerStatus)
    /// entries for each server so callers can inspect which connections succeeded.
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

    /// Registers the seven built-in tools: `execute_command`, `read_file`,
    /// `write_file`, `edit_file`, `list_dir`, `search`, `web_fetch`.
    pub fn register_builtins(&mut self) {
        self.register(Box::new(execute_command::ExecuteCommandTool));
        self.register(Box::new(read_file::ReadFileTool));
        self.register(Box::new(write_file::WriteFileTool));
        self.register(Box::new(edit_file::EditFileTool));
        self.register(Box::new(list_dir::ListDirTool));
        self.register(Box::new(search::SearchTool));
        self.register(Box::new(web_fetch::WebFetchTool));
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
        self.handlers.insert(name, Arc::from(handler));
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

    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| Error::ToolExecutionError(format!("Unknown tool: {}", name)))?;
        handler.execute(args_json, turn).await
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;
    use std::sync::Arc;

    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use crate::core::client_io::{ClientIo, NoClientIo};
    use crate::core::permission::{AllowAll, PermissionGate};
    use crate::core::turn::TurnContext;
    use crate::error::Result;

    /// Owns the pieces a test [`TurnContext`] borrows from, so callers can do
    /// `let turn = harness.turn();` without fighting temporary lifetimes.
    /// The work directory is a fresh temp dir per harness.
    pub(crate) struct TurnHarness {
        cancel: CancellationToken,
        permission_gate: Arc<dyn PermissionGate>,
        work_dir: tempfile::TempDir,
        client_io: Arc<dyn ClientIo>,
    }

    impl TurnHarness {
        pub(crate) fn new() -> Self {
            Self {
                cancel: CancellationToken::new(),
                permission_gate: Arc::new(AllowAll),
                work_dir: tempfile::tempdir().expect("create temp work dir"),
                client_io: Arc::new(NoClientIo),
            }
        }

        pub(crate) fn with_client_io(mut self, client_io: Arc<dyn ClientIo>) -> Self {
            self.client_io = client_io;
            self
        }

        pub(crate) fn work_dir(&self) -> &Path {
            self.work_dir.path()
        }

        pub(crate) fn turn(&self) -> TurnContext<'_> {
            TurnContext {
                cancel: &self.cancel,
                permission_gate: &self.permission_gate,
                work_dir: self.work_dir.path(),
                client_io: &*self.client_io,
            }
        }

        /// A clone of the underlying token, so a test can cancel the turn
        /// from outside while `turn()` is borrowed elsewhere.
        pub(crate) fn cancel_handle(&self) -> CancellationToken {
            self.cancel.clone()
        }
    }

    /// Always answers from an in-memory string, never touching local disk.
    pub(crate) struct FixedClientIo(pub &'static str);

    #[async_trait]
    impl ClientIo for FixedClientIo {
        async fn read_file(&self, _path: &Path) -> Option<Result<String>> {
            Some(Ok(self.0.to_string()))
        }

        async fn write_file(&self, _path: &Path, _content: &str) -> Option<Result<()>> {
            Some(Ok(()))
        }
    }

    /// Never resolves on its own; used to prove cancellation aborts the wait
    /// on an unresponsive client rather than blocking the turn.
    pub(crate) struct HangingClientIo;

    #[async_trait]
    impl ClientIo for HangingClientIo {
        async fn read_file(&self, _path: &Path) -> Option<Result<String>> {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this wait before the sleep elapses");
        }

        async fn write_file(&self, _path: &Path, _content: &str) -> Option<Result<()>> {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this wait before the sleep elapses");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::test_support::TurnHarness;
    use super::*;
    use crate::core::models::FunctionDefinition;

    #[test]
    fn new_executor_is_empty() {
        let executor = SystemToolExecutor::new();
        assert_eq!(executor.handlers.len(), 0);
    }

    #[test]
    fn register_builtins_adds_seven_tools() {
        let mut executor = SystemToolExecutor::new();
        executor.register_builtins();
        assert!(executor.handlers.contains_key("execute_command"));
        assert!(executor.handlers.contains_key("read_file"));
        assert!(executor.handlers.contains_key("write_file"));
        assert!(executor.handlers.contains_key("edit_file"));
        assert!(executor.handlers.contains_key("list_dir"));
        assert!(executor.handlers.contains_key("search"));
        assert!(executor.handlers.contains_key("web_fetch"));
        assert_eq!(executor.handlers.len(), 7);
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

    #[test]
    fn clone_snapshots_the_registry() {
        let mut executor = SystemToolExecutor::new();
        executor.register_builtins();
        let snapshot = executor.clone();
        executor.register(Box::new(ContextEchoTool));
        assert!(executor.handlers.contains_key("context_echo"));
        assert!(!snapshot.handlers.contains_key("context_echo"));
    }

    /// A custom tool that reports what it was handed, proving the turn
    /// context reaches handlers unchanged through `SystemToolExecutor`.
    struct ContextEchoTool;

    #[async_trait]
    impl ToolHandler for ContextEchoTool {
        fn definition(&self) -> Tool {
            Tool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "context_echo".to_string(),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                },
            }
        }

        async fn execute(&self, _args: &str, turn: &TurnContext<'_>) -> Result<String> {
            Ok(format!(
                "work_dir={} cancelled={}",
                turn.work_dir.display(),
                turn.cancel.is_cancelled()
            ))
        }
    }

    #[tokio::test]
    async fn custom_tool_sees_work_dir_and_cancel_from_turn() {
        let mut executor = SystemToolExecutor::new();
        executor.register(Box::new(ContextEchoTool));
        let harness = TurnHarness::new();

        let before = executor
            .execute("context_echo", "{}", &harness.turn())
            .await
            .unwrap();
        assert_eq!(
            before,
            format!("work_dir={} cancelled=false", harness.work_dir().display())
        );

        harness.cancel_handle().cancel();
        let after = executor
            .execute("context_echo", "{}", &harness.turn())
            .await
            .unwrap();
        assert!(after.ends_with("cancelled=true"), "{after}");
    }
}
