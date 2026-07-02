//! Subagent delegation: a tool that lets the orchestrating agent hand off a
//! self-contained task to a named subagent profile (see [`crate::subagents`]).
//!
//! Each call to `delegate_task` runs a fresh, isolated [`run_agent_with_history`]
//! turn — its own message history, its own system prompt (the profile's persona,
//! not the parent's `system.md`/skills), and optionally its own model/provider and
//! restricted tool set — and returns only the subagent's final answer. The
//! orchestrator never sees the subagent's intermediate steps, exactly like
//! Claude Code's `Task` subagents.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::config::{AgentConfig, AppConfig, client_for_config};
use crate::core::agent::run_agent_with_history;
use crate::core::client_io::NoClientIo;
use crate::core::llm::LlmClient;
use crate::core::models::{FunctionDefinition, Message, StopReason, Tool};
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};
use crate::rag::PromptBuilder;
use crate::subagents::AgentProfile;

use super::scoped_executor::ScopedExecutor;
use super::{SandboxedExecutor, ToolExecutor};

/// Name under which [`DelegateTool`] is exposed to the orchestrating LLM.
pub const DELEGATE_TOOL_NAME: &str = "delegate_task";

/// Routes `delegate_task` calls to a named [`AgentProfile`], running each as an
/// isolated agent-loop turn.
///
/// `base_executor` is deliberately the executor as it exists *before*
/// `delegate_task` is added to it (see [`with_delegation`]): subagents are built
/// from this delegate-free view, so `delegate_task` is structurally absent from
/// their own tool list. This rules out recursive delegation by construction —
/// no depth counters or runtime checks are needed.
pub struct DelegateTool {
    base_executor: Arc<dyn ToolExecutor>,
    work_dir: PathBuf,
    allow_shell: bool,
    profiles: Vec<AgentProfile>,
    llm: Arc<dyn LlmClient>,
    app_config: AppConfig,
    base_config: AgentConfig,
}

impl DelegateTool {
    pub fn new(
        base_executor: Arc<dyn ToolExecutor>,
        work_dir: PathBuf,
        allow_shell: bool,
        profiles: Vec<AgentProfile>,
        llm: Arc<dyn LlmClient>,
        app_config: AppConfig,
        base_config: AgentConfig,
    ) -> Self {
        Self {
            base_executor,
            work_dir,
            allow_shell,
            profiles,
            llm,
            app_config,
            base_config,
        }
    }

    fn find_profile(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Resolves the [`AgentConfig`] and [`LlmClient`] a subagent run should use,
    /// honouring the profile's optional `model`/`provider`/`max_iterations`
    /// overrides. Reuses the parent's client when the resolved provider and model
    /// match — otherwise builds a fresh one, mirroring the pattern `acp_prompt`
    /// already uses for per-session model switches (see `src/acp/mod.rs`).
    fn resolve_runtime(&self, profile: &AgentProfile) -> Result<(AgentConfig, Arc<dyn LlmClient>)> {
        let config = match (&profile.provider, &profile.model) {
            (Some(provider), Some(model)) => {
                self.app_config.resolve_with_provider(provider, model)?
            }
            (None, Some(model)) => self.app_config.resolve(Some(model))?,
            _ => self.base_config.clone(),
        };
        let config = match profile.max_iterations {
            Some(max_iterations) => config.with_max_iterations(max_iterations),
            None => config,
        };

        let llm = client_for_config(&config, &self.base_config, &self.llm)?;

        Ok((config, llm))
    }

    /// Builds the tool executor a subagent run should use: the shared,
    /// delegate-free base executor, optionally narrowed to the profile's `tools`
    /// allowlist, wrapped in the same sandbox boundary (`work_dir`/`allow_shell`)
    /// as the parent so subagents cannot escalate privileges.
    fn build_executor(&self, profile: &AgentProfile) -> Arc<dyn ToolExecutor> {
        let scoped: Arc<dyn ToolExecutor> = match &profile.tools {
            Some(allowed) => Arc::new(ScopedExecutor::new(
                self.base_executor.clone(),
                allowed.clone(),
            )),
            None => self.base_executor.clone(),
        };
        Arc::new(SandboxedExecutor::new(
            scoped,
            self.work_dir.clone(),
            self.allow_shell,
            Arc::new(NoClientIo),
        ))
    }

    /// Not a [`ToolHandler`](super::ToolHandler): unlike ordinary tools,
    /// `delegate_task` needs the calling turn's [`TurnContext`] to spawn its
    /// subagent run (see [`Self::execute`]), so it's dispatched directly by
    /// [`WithDelegate`] rather than registered into a [`super::SystemToolExecutor`].
    pub fn definition(&self) -> Tool {
        let names: Vec<String> = self.profiles.iter().map(|p| p.name.clone()).collect();

        let mut listing = String::new();
        for profile in &self.profiles {
            let description = if profile.description.is_empty() {
                "(no description provided)"
            } else {
                profile.description.as_str()
            };
            listing.push_str(&format!("\n- `{}`: {description}", profile.name));
        }

        let description = format!(
            "Delegate a self-contained task to a specialized subagent that runs independently \
             with its own context, persona, and (optionally) its own model or restricted tool \
             set. The subagent CANNOT see this conversation, so `task` must be a complete, \
             standalone brief containing every detail it needs. Only its final answer is \
             returned to you — its intermediate steps are not visible.\n\
             \n\
             Available subagents:{listing}"
        );

        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: DELEGATE_TOOL_NAME.to_string(),
                description,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": names,
                            "description": "Name of the subagent to delegate to."
                        },
                        "task": {
                            "type": "string",
                            "description": "A complete, self-contained description of the task. \
                                            Include all context the subagent needs — it cannot \
                                            see your conversation history."
                        }
                    },
                    "required": ["agent", "task"]
                }),
            },
        }
    }

    /// Runs the delegated subagent turn, reusing `turn`'s cancellation token
    /// and permission gate rather than manufacturing fresh ones: a
    /// `session/cancel` on the orchestrating turn must stop the subagent too,
    /// and there is no separate trust policy for subagent tool calls — they
    /// go through the same approval flow (e.g. `session/request_permission`)
    /// as the orchestrator's own.
    pub async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let v: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("invalid arguments: {e}")))?;

        let agent_name = v["agent"]
            .as_str()
            .ok_or_else(|| Error::ParseError("missing 'agent' argument".to_string()))?;
        let task = v["task"]
            .as_str()
            .ok_or_else(|| Error::ParseError("missing 'task' argument".to_string()))?;

        let Some(profile) = self.find_profile(agent_name) else {
            let available = self
                .profiles
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Ok(format!(
                "Unknown subagent '{agent_name}'. Available subagents: {available}"
            ));
        };

        let (config, llm) = self.resolve_runtime(profile)?;
        let executor = self.build_executor(profile);

        let mut prompt_builder = PromptBuilder::new();
        prompt_builder.set_system(profile.system_prompt.clone());

        // Fresh, isolated history — the subagent only ever sees its own task.
        let mut messages = vec![Message::user(task.to_string())];

        let result = run_agent_with_history(
            llm,
            executor,
            &config,
            &mut messages,
            Some(&prompt_builder),
            &TurnContext {
                cancel: turn.cancel,
                permission_gate: turn.permission_gate,
            },
        )
        .await?;

        if result.stop_reason == StopReason::MaxIterations {
            Ok(format!(
                "{}\n\n[Note: subagent '{agent_name}' reached its iteration limit \
                 ({}) before finishing — this answer may be incomplete.]",
                result.final_response, config.max_iterations
            ))
        } else {
            Ok(result.final_response)
        }
    }
}

/// Composes a base [`ToolExecutor`] with [`DelegateTool`], surfacing
/// `delegate_task` to the LLM alongside the base tools.
///
/// `inner` is the executor *as it exists before* this wrapper — exactly the view
/// passed to [`DelegateTool::new`] as `base_executor` — so subagents are handed a
/// tool list that structurally never contains `delegate_task`.
struct WithDelegate {
    inner: Arc<dyn ToolExecutor>,
    delegate: Arc<DelegateTool>,
}

impl WithDelegate {
    fn new(inner: Arc<dyn ToolExecutor>, delegate: Arc<DelegateTool>) -> Self {
        Self { inner, delegate }
    }
}

#[async_trait]
impl ToolExecutor for WithDelegate {
    fn list_tools(&self) -> Vec<Tool> {
        let mut tools = self.inner.list_tools();
        tools.push(self.delegate.definition());
        tools
    }

    async fn execute(&self, name: &str, args_json: &str, turn: &TurnContext<'_>) -> Result<String> {
        if name == DELEGATE_TOOL_NAME {
            self.delegate.execute(args_json, turn).await
        } else {
            self.inner.execute(name, args_json, turn).await
        }
    }
}

/// Wraps `executor` with [`DelegateTool`] support when `profiles` is non-empty;
/// otherwise returns `executor` unchanged so agents with no configured subagents
/// pay no overhead and never see a useless `delegate_task` tool.
#[allow(clippy::too_many_arguments)]
pub fn with_delegation(
    executor: Arc<dyn ToolExecutor>,
    work_dir: PathBuf,
    allow_shell: bool,
    profiles: Vec<AgentProfile>,
    llm: Arc<dyn LlmClient>,
    app_config: AppConfig,
    base_config: AgentConfig,
) -> Arc<dyn ToolExecutor> {
    if profiles.is_empty() {
        return executor;
    }
    let delegate = Arc::new(DelegateTool::new(
        executor.clone(),
        work_dir,
        allow_shell,
        profiles,
        llm,
        app_config,
        base_config,
    ));
    Arc::new(WithDelegate::new(executor, delegate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::models::{Choice, ContentBlock, Role};
    use crate::core::permission::{AllowAll, PermissionDecision, PermissionGate};
    use crate::tools::test_support::TurnHarness;
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    fn sample_app_config() -> AppConfig {
        AppConfig {
            default_provider: "mock".into(),
            max_iterations: 10,
            theme_color: None,
            providers: BTreeMap::new(),
            mcp_servers: BTreeMap::new(),
            default_skills: vec![],
            work_dir: None,
            allow_shell: false,
        }
    }

    fn sample_agent_config() -> AgentConfig {
        AgentConfig::new(
            "mock".into(),
            "https://example.com".into(),
            "key".into(),
            "mock-model".into(),
            5,
        )
    }

    fn sample_profile(name: &str, description: &str) -> AgentProfile {
        AgentProfile {
            name: name.into(),
            description: description.into(),
            model: None,
            provider: None,
            tools: None,
            max_iterations: None,
            system_prompt: "You are a test subagent.".into(),
        }
    }

    fn text_choice(content: &str, finish: &str) -> Choice {
        Choice {
            message: Message::assistant(content),
            finish_reason: Some(finish.into()),
        }
    }

    fn tool_call_choice() -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "nonexistent".into(),
                    arguments: "{}".into(),
                }],
            },
            finish_reason: Some("tool_calls".into()),
        }
    }

    struct MockLlm {
        responses: Mutex<Vec<Choice>>,
    }

    impl MockLlm {
        fn new(responses: Vec<Choice>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err(Error::ApiError("no more mock responses".into()))
            } else {
                Ok(responses.remove(0))
            }
        }
    }

    /// A tool executor with no tools — every call fails, mirroring how a
    /// subagent would see "unknown tool" for anything it tries that isn't there.
    struct EmptyExecutor;

    #[async_trait]
    impl ToolExecutor for EmptyExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            vec![]
        }

        async fn execute(
            &self,
            name: &str,
            _args_json: &str,
            _turn: &TurnContext<'_>,
        ) -> Result<String> {
            Err(Error::ToolExecutionError(format!("Unknown tool: {name}")))
        }
    }

    fn make_tool(profiles: Vec<AgentProfile>, llm: Arc<dyn LlmClient>) -> DelegateTool {
        DelegateTool::new(
            Arc::new(EmptyExecutor),
            PathBuf::from("/tmp"),
            false,
            profiles,
            llm,
            sample_app_config(),
            sample_agent_config(),
        )
    }

    #[test]
    fn definition_lists_available_profiles() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(
            vec![sample_profile("reviewer", "Reviews code for bugs.")],
            llm,
        );
        let def = tool.definition();

        assert_eq!(def.function.name, DELEGATE_TOOL_NAME);
        assert!(def.function.description.contains("reviewer"));
        assert!(def.function.description.contains("Reviews code for bugs."));

        let names = def.function.parameters["properties"]["agent"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(names, &vec![serde_json::Value::String("reviewer".into())]);
    }

    #[tokio::test]
    async fn execute_returns_message_for_unknown_agent() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);
        let harness = TurnHarness::new();

        let result = tool
            .execute(
                r#"{"agent": "ghost", "task": "do something"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();

        assert!(result.contains("Unknown subagent 'ghost'"));
        assert!(result.contains("reviewer"));
    }

    #[tokio::test]
    async fn execute_runs_subagent_in_isolated_context_and_returns_final_answer() {
        let llm = Arc::new(MockLlm::new(vec![text_choice("subagent answer", "stop")]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);
        let harness = TurnHarness::new();

        let result = tool
            .execute(
                r#"{"agent": "reviewer", "task": "look at this diff"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();

        assert_eq!(result, "subagent answer");
    }

    #[tokio::test]
    async fn execute_notes_when_subagent_hits_its_iteration_limit() {
        let llm = Arc::new(MockLlm::new(vec![tool_call_choice(), tool_call_choice()]));
        let mut profile = sample_profile("looper", "desc");
        profile.max_iterations = Some(2);
        let tool = make_tool(vec![profile], llm);
        let harness = TurnHarness::new();

        let result = tool
            .execute(
                r#"{"agent": "looper", "task": "loop forever"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();

        assert!(result.contains("reached its iteration limit (2)"));
    }

    #[tokio::test]
    async fn execute_stops_immediately_when_parent_turn_already_cancelled() {
        // Subagent must inherit the parent's cancel token, not a fresh one —
        // otherwise a `session/cancel` on the orchestrator would never reach
        // an in-flight subagent.
        let llm = Arc::new(MockLlm::new(vec![text_choice(
            "should not be seen",
            "stop",
        )]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);

        let cancel = CancellationToken::new();
        cancel.cancel();
        let permission_gate: Arc<dyn PermissionGate> = Arc::new(AllowAll);
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
        };

        let result = tool
            .execute(
                r#"{"agent": "reviewer", "task": "look at this diff"}"#,
                &turn,
            )
            .await
            .unwrap();

        // The loop bails before ever calling the LLM, so no final text is produced.
        assert_eq!(result, "");
    }

    struct RejectPermissionGate;

    #[async_trait]
    impl PermissionGate for RejectPermissionGate {
        async fn check(
            &self,
            _tool_call_id: &str,
            _tool_name: &str,
            _arguments: &str,
        ) -> PermissionDecision {
            PermissionDecision::RejectOnce
        }
    }

    #[tokio::test]
    async fn subagent_tool_calls_go_through_parent_permission_gate() {
        // Subagent tool calls must be checked by the same gate as the
        // orchestrator's own — there is no separate, more-trusting policy for
        // subagents.
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice(),
            text_choice("denied", "stop"),
        ]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);

        let cancel = CancellationToken::new();
        let permission_gate: Arc<dyn PermissionGate> = Arc::new(RejectPermissionGate);
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
        };

        let result = tool
            .execute(
                r#"{"agent": "reviewer", "task": "look at this diff"}"#,
                &turn,
            )
            .await
            .unwrap();

        assert_eq!(result, "denied");
    }

    #[test]
    fn with_delegation_returns_executor_unchanged_when_no_profiles() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let base: Arc<dyn ToolExecutor> = Arc::new(EmptyExecutor);
        let result = with_delegation(
            base.clone(),
            PathBuf::from("/tmp"),
            false,
            vec![],
            llm,
            sample_app_config(),
            sample_agent_config(),
        );
        assert!(Arc::ptr_eq(&base, &result));
    }

    #[tokio::test]
    async fn with_delegation_exposes_delegate_tool_alongside_base_tools() {
        let llm = Arc::new(MockLlm::new(vec![text_choice("done", "stop")]));
        let base: Arc<dyn ToolExecutor> = Arc::new(EmptyExecutor);
        let executor = with_delegation(
            base,
            PathBuf::from("/tmp"),
            false,
            vec![sample_profile("reviewer", "desc")],
            llm,
            sample_app_config(),
            sample_agent_config(),
        );

        let names: Vec<_> = executor
            .list_tools()
            .into_iter()
            .map(|t| t.function.name)
            .collect();
        assert!(names.contains(&DELEGATE_TOOL_NAME.to_string()));

        let harness = TurnHarness::new();
        let result = executor
            .execute(
                DELEGATE_TOOL_NAME,
                r#"{"agent": "reviewer", "task": "go"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();
        assert_eq!(result, "done");
    }
}
