//! [`ToolCapabilities`]: the single declaration each [`super::ToolHandler`]
//! makes about itself, consumed wherever a tool's coarse behavior matters —
//! Architect mode's allowlist, `approval_key`'s `execute_command` special
//! case, ACP's tool-kind-for-UI mapping, and `SystemToolExecutor::build`'s
//! shell-tool gate.

/// Coarse hint of what kind of action a tool performs. Mirrors
/// `agent_client_protocol::schema::ToolKind`'s variant set without depending
/// on that crate — `core` and `tools` stay ACP-free; the mapping onto the
/// real `ToolKind` lives in `acp::util`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolKindHint {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    SwitchMode,
    #[default]
    Other,
}

/// How a permission decision for a tool call is keyed and cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalScope {
    /// One approval covers every call to this tool, keyed by tool name.
    #[default]
    ToolName,
    /// Approval only covers calls with the exact same arguments — e.g.
    /// `execute_command`, where approving `git status` must not silently
    /// cover `git status && rm -rf ~`.
    ExactArguments,
}

/// What a [`super::ToolHandler`] declares about itself: whether it's safe to
/// expose read-only (Architect mode), which [`ToolKindHint`] best describes
/// it (client UI treatment), and how its approvals should be scoped.
///
/// The default — `read_only: false`, [`ToolKindHint::Other`],
/// [`ApprovalScope::ToolName`] — is the conservative choice for a tool that
/// declares nothing, and is what MCP-sourced tools get: there's no signal in
/// an MCP tool definition to derive `read_only` or `kind` from.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolCapabilities {
    /// Safe to expose in Architect mode (no filesystem/shell/network writes,
    /// no state mutation).
    pub read_only: bool,
    pub kind: ToolKindHint,
    pub approval_scope: ApprovalScope,
}
