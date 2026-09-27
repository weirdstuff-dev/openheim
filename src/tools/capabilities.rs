//! [`ToolCapabilities`]: what each [`super::ToolHandler`] declares about
//! itself, read by Architect mode's allowlist, `approval_key`, and ACP's
//! tool-kind mapping.

/// Coarse hint of what kind of action a tool performs. Mirrors ACP's
/// `ToolKind` without depending on it; `acp::util` maps one onto the other.
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
    /// Approval only covers calls with the same arguments (compared as JSON,
    /// so key order and whitespace don't matter), e.g. `execute_command`:
    /// approving `git status` must not cover `git status && rm -rf ~`.
    ExactArguments,
}

/// What a [`super::ToolHandler`] declares about itself: whether it's
/// read-only (offered in Architect mode), its [`ToolKindHint`] (client UI),
/// and how its approvals are scoped.
///
/// The default (not read-only, [`ToolKindHint::Other`],
/// [`ApprovalScope::ToolName`]) is the conservative choice, and what MCP
/// tools get.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolCapabilities {
    /// Safe to expose in Architect mode (no filesystem/shell/network writes,
    /// no state mutation).
    pub read_only: bool,
    pub kind: ToolKindHint,
    pub approval_scope: ApprovalScope,
}
