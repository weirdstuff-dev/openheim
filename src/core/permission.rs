//! Tool-call authorization: an embedder-supplied hook the agent loop consults
//! before executing any tool call the LLM requests.
//!
//! This mirrors [`crate::core::llm::LlmClient`] and [`crate::tools::ToolExecutor`]:
//! a protocol-agnostic trait defined here, with the ACP-specific implementation
//! (backed by `session/request_permission`) living in [`crate::acp`].

use async_trait::async_trait;

/// The user's (or embedder's) decision on a single tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Allow this call only.
    AllowOnce,
    /// Allow this call and remember the choice for the rest of the session.
    AllowAlways,
    /// Reject this call only.
    RejectOnce,
    /// Reject this call and remember the choice for the rest of the session.
    RejectAlways,
}

impl PermissionDecision {
    pub fn is_allowed(self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowAlways)
    }
}

/// Asked before every tool call the agent loop is about to execute.
#[async_trait]
pub trait PermissionGate: Send + Sync {
    async fn check(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> PermissionDecision;
}

/// Key used to remember an `AllowAlways`/`RejectAlways` decision across
/// tool calls in a session. For most tools this is just the tool name — one
/// approval covers every future call to that tool. For `execute_command`
/// specifically, this is scoped by the invoked command's first word (its
/// program name, e.g. `git`, `rm`, `npm`) instead: otherwise "Allow Always" on
/// `git status` would silently approve every future shell command, including
/// `rm -rf /`. Falls back to the bare tool name if the command can't be
/// extracted from `arguments` — broader than per-command scoping, but no
/// worse than this cache's pre-existing per-tool-name behavior.
pub fn approval_key(tool_name: &str, arguments: &str) -> String {
    if tool_name != "execute_command" {
        return tool_name.to_string();
    }
    let command_prefix = serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("command")?.as_str().map(str::to_string))
        .and_then(|cmd| cmd.split_whitespace().next().map(str::to_string));
    match command_prefix {
        Some(prefix) => format!("{tool_name}:{prefix}"),
        None => tool_name.to_string(),
    }
}

/// Default gate for contexts with no human in the loop to ask: the TUI (already
/// fully interactive/local), the library facade, `openheim run`, and subagents.
/// Always allows.
pub struct AllowAll;

#[async_trait]
impl PermissionGate for AllowAll {
    async fn check(
        &self,
        _tool_call_id: &str,
        _tool_name: &str,
        _arguments: &str,
    ) -> PermissionDecision {
        PermissionDecision::AllowOnce
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allow_all_always_allows() {
        let decision = AllowAll.check("call_1", "execute_command", "{}").await;
        assert!(decision.is_allowed());
    }

    #[test]
    fn non_shell_tools_are_keyed_by_bare_tool_name_regardless_of_arguments() {
        assert_eq!(
            approval_key("read_file", r#"{"path": "a.txt"}"#),
            "read_file"
        );
        assert_eq!(
            approval_key("read_file", r#"{"path": "b.txt"}"#),
            "read_file"
        );
    }

    #[test]
    fn shell_commands_are_scoped_by_their_first_word() {
        assert_eq!(
            approval_key("execute_command", r#"{"command": "git status"}"#),
            "execute_command:git"
        );
        assert_eq!(
            approval_key("execute_command", r#"{"command": "git commit -m x"}"#),
            "execute_command:git"
        );
    }

    #[test]
    fn different_shell_commands_get_different_keys() {
        let git = approval_key("execute_command", r#"{"command": "git status"}"#);
        let rm = approval_key("execute_command", r#"{"command": "rm -rf /"}"#);
        assert_ne!(git, rm);
    }

    #[test]
    fn unparseable_shell_arguments_fall_back_to_the_bare_tool_name() {
        assert_eq!(
            approval_key("execute_command", "not json"),
            "execute_command"
        );
        assert_eq!(
            approval_key("execute_command", r#"{"no_command_field": true}"#),
            "execute_command"
        );
    }
}
