//! Tool abstraction layer: trait definitions, built-in tools, and the runtime
//! executor that routes LLM tool calls to the correct handler.
//!
//! # Built-in tools
//!
//! Seven tools are available via [`SystemToolExecutor::register_builtins`]
//! (`execute_command` only when `allow_shell` is `true`):
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
//! Every filesystem tool resolves its path with
//! [`TurnContext::resolve_path`] (relative to the turn's `cwd`, confined to
//! its `work_dir`) and delegates reads/writes to the
//! turn's [`ClientIo`](crate::core::client_io::ClientIo) when one is
//! available; there is no separate sandbox wrapper.
//!
//! With the `rag` feature, an [`OpenheimClient`](crate::OpenheimClient) also
//! registers `remember`, `search_memory`, `edit_memory`, and `forget`, and
//! `delegate_task` is always registered.
//!
//! Additional tools are loaded from MCP servers and registered under the
//! `{server_name}__{tool_name}` namespace.
//!
//! # Implementing a custom tool
//!
//! ```rust,no_run
//! use async_trait::async_trait;
//! use serde::Deserialize;
//! use serde_json::json;
//!
//! struct GreetTool;
//!
//! #[derive(Deserialize)]
//! struct GreetArgs {
//!     name: String,
//! }
//!
//! # use openheim::tools::ToolHandler;
//! # use openheim::tools::args::parse;
//! # use openheim::core::models::Tool;
//! # use openheim::core::turn::TurnContext;
//! # use openheim::error::Result;
//! #[async_trait]
//! impl ToolHandler for GreetTool {
//!     fn definition(&self) -> Tool {
//!         Tool::function(
//!             "greet",
//!             "Greet someone by name.",
//!             json!({
//!                 "type": "object",
//!                 "properties": { "name": { "type": "string" } },
//!                 "required": ["name"]
//!             }),
//!         )
//!     }
//!
//!     async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
//!         let GreetArgs { name } = parse(args)?;
//!         // `turn` carries the cancel token, the work and working
//!         // directories, and the client I/O hook; see `TurnContext` for
//!         // what to do with each.
//!         Ok(format!("Hello, {name}! (from {})", turn.cwd.display()))
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
//! if you're driving the agent loop yourself with
//! [`run_agent`](crate::core::agent::run_agent) instead of going through the
//! builder.

pub mod args;
mod capabilities;
pub(crate) mod delegate;
mod edit_file;
mod execute_command;
mod list_dir;
mod overlay_executor;
mod read_file;
pub(crate) mod sandbox;
mod scoped_executor;
mod search;
mod web_fetch;
mod write_file;

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::McpServerConfig;
use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

pub use capabilities::{ApprovalScope, ToolCapabilities, ToolKindHint};
pub(crate) use delegate::DelegateTool;
pub use execute_command::kill_running_commands;
pub(crate) use overlay_executor::OverlayExecutor;
pub(crate) use scoped_executor::ScopedExecutor;

#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Returns the tool definition (name, description, JSON-schema parameters).
    fn definition(&self) -> Tool;

    /// Executes the tool with the given JSON-encoded arguments.
    ///
    /// `turn` is the calling turn's [`TurnContext`]: race long work against
    /// `turn.cancel`, resolve every path with [`TurnContext::resolve_path`]
    /// (which confines it to `turn.work_dir`), and prefer `turn.client_io` for file
    /// reads/writes so an editor-hosted client can serve its own buffers.
    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String>;

    /// Declares what this tool is: read-only or not, its [`ToolKindHint`],
    /// and how its approvals are scoped. See [`ToolCapabilities`] for what
    /// the default (a tool that declares nothing) means in practice.
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }
}

/// Routes LLM tool calls to the right [`ToolHandler`]; the built-in
/// implementation is [`SystemToolExecutor`].
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

    /// Returns the named tool's declared [`ToolCapabilities`], or the
    /// (conservative) default if the tool is unknown.
    fn capabilities(&self, name: &str) -> ToolCapabilities {
        let _ = name;
        ToolCapabilities::default()
    }
}

/// The default tool executor: a registry of [`ToolHandler`]s keyed by name.
///
/// Cloning is cheap and gives an independent registry sharing the same
/// handlers: a snapshot of the tool set (how `delegate_task` gets one
/// without itself in it).
#[derive(Clone)]
pub struct SystemToolExecutor {
    /// Ordered, so the tool list that leads every request is the same in
    /// every process and the provider's prompt cache stays reusable.
    handlers: BTreeMap<String, Arc<dyn ToolHandler>>,
}

impl SystemToolExecutor {
    /// Creates an empty executor with no registered tools.
    pub fn new() -> Self {
        Self {
            handlers: BTreeMap::new(),
        }
    }

    /// An executor with the built-in tools and those of every configured MCP
    /// server, plus each server's
    /// [`McpServerStatus`](crate::mcp::McpServerStatus).
    pub async fn build(
        mcp_configs: &BTreeMap<String, McpServerConfig>,
        allow_shell: bool,
    ) -> (Self, Vec<crate::mcp::McpServerStatus>) {
        let mut executor = Self::new();
        executor.register_builtins(allow_shell);
        let (handlers, statuses) = crate::mcp::load_mcp_tools(mcp_configs).await;
        for handler in handlers {
            executor.register(handler);
        }
        (executor, statuses)
    }

    /// Registers the built-in tools: `read_file`, `write_file`, `edit_file`,
    /// `list_dir`, `search`, `web_fetch`, and `execute_command` only when
    /// `allow_shell` is set (the one place it's enforced).
    pub fn register_builtins(&mut self, allow_shell: bool) {
        if allow_shell {
            self.register(Box::new(execute_command::ExecuteCommandTool));
        }
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

    fn capabilities(&self, name: &str) -> ToolCapabilities {
        self.handlers
            .get(name)
            .map(|h| h.capabilities())
            .unwrap_or_default()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use crate::core::client_io::{ClientIo, LineRange, NoClientIo};
    use crate::core::permission::{AllowAll, PermissionGate};
    use crate::core::turn::TurnContext;
    use crate::error::Result;

    /// Owns the pieces a test [`TurnContext`] borrows from, so callers can do
    /// `let turn = harness.turn();` without fighting temporary lifetimes.
    /// The work directory is a fresh temp dir per harness, and is also the
    /// working directory unless [`Self::with_cwd`] sets another.
    pub(crate) struct TurnHarness {
        cancel: CancellationToken,
        permission_gate: Arc<dyn PermissionGate>,
        work_dir: tempfile::TempDir,
        cwd: Option<PathBuf>,
        client_io: Arc<dyn ClientIo>,
    }

    impl TurnHarness {
        pub(crate) fn new() -> Self {
            Self {
                cancel: CancellationToken::new(),
                permission_gate: Arc::new(AllowAll),
                work_dir: tempfile::tempdir().expect("create temp work dir"),
                cwd: None,
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
                cwd: self.cwd.as_deref().unwrap_or(self.work_dir.path()),
                client_io: &*self.client_io,
            }
        }
    }

    /// Answers every read with `content` and records the ranges asked for.
    pub(crate) struct RecordingClientIo {
        pub(crate) content: String,
        pub(crate) reads: std::sync::Mutex<Vec<LineRange>>,
    }

    impl RecordingClientIo {
        pub(crate) fn new(content: impl Into<String>) -> Self {
            Self {
                content: content.into(),
                reads: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ClientIo for RecordingClientIo {
        async fn read_file(&self, _path: &Path, lines: LineRange) -> Option<Result<String>> {
            self.reads.lock().unwrap().push(lines);
            Some(Ok(self.content.clone()))
        }

        async fn write_file(&self, _path: &Path, _content: &str) -> Option<Result<()>> {
            Some(Ok(()))
        }
    }
}
