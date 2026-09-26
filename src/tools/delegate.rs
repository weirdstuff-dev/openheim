//! Subagent delegation: a tool that lets the orchestrating agent hand off a
//! self-contained task to a subagent — either a named profile from
//! `~/.openheim/agents/` (see [`crate::subagents`]) or an ephemeral one the
//! orchestrator defines inline in the tool call itself (a `system_prompt` plus
//! optional model/tools overrides). Inline subagents exist only for the duration
//! of the call and are never persisted anywhere.
//!
//! Each call to `delegate_task` runs a fresh, isolated [`run_agent`]
//! turn — its own message history, its own system prompt (the profile's persona,
//! not the parent's `system.md`/skills), and optionally its own model/provider and
//! restricted tool set — and returns only the subagent's final answer. The
//! orchestrator never sees the subagent's intermediate steps, exactly like
//! Claude Code's `Task` subagents.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::config::{AgentConfig, AppConfig, client_for_config};
use crate::core::agent::run_agent;
use crate::core::llm::LlmClient;
use crate::core::models::{Message, StopReason, Tool};
use crate::core::permission::{PermissionDecision, PermissionGate, PermissionRequest};
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};
use crate::memory::PromptBuilder;
use crate::subagents::AgentProfile;

use super::args::parse;
use super::capabilities::ToolCapabilities;
use super::scoped_executor::ScopedExecutor;
use super::{ToolExecutor, ToolHandler};

/// `delegate_task`'s arguments; exactly one of `agent`/`system_prompt` must
/// be set (checked in `execute`, whose error tells the LLM how to fix it).
#[derive(Deserialize)]
struct DelegateArgs {
    task: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    max_iterations: Option<usize>,
}

/// Name under which [`DelegateTool`] is exposed to the orchestrating LLM.
pub const DELEGATE_TOOL_NAME: &str = "delegate_task";

/// Routes `delegate_task` calls to a named [`AgentProfile`], running each as an
/// isolated agent-loop turn.
///
/// `base_executor` must be a snapshot of the tool registry taken *before*
/// `delegate_task` is registered into it (see `AgentState::new`, which clones
/// the [`super::SystemToolExecutor`] for exactly this purpose): subagents are
/// built from this delegate-free view, so `delegate_task` is structurally
/// absent from their own tool list. This rules out recursive delegation by
/// construction — no depth counters or runtime checks are needed.
///
/// `llm`/`base_config` are the model a subagent runs on when its profile
/// sets no `model`/`provider` of its own. `AgentState` registers one bound to
/// the startup default (so tool listings include `delegate_task`), then
/// rebinds a copy to the session's live model every turn via
/// [`Self::for_session`], so a mid-session model switch reaches subagents too.
#[derive(Clone)]
pub struct DelegateTool {
    base_executor: Arc<dyn ToolExecutor>,
    profiles: Vec<AgentProfile>,
    llm: Arc<dyn LlmClient>,
    app_config: AppConfig,
    base_config: AgentConfig,
}

impl DelegateTool {
    pub fn new(
        base_executor: Arc<dyn ToolExecutor>,
        profiles: Vec<AgentProfile>,
        llm: Arc<dyn LlmClient>,
        app_config: AppConfig,
        base_config: AgentConfig,
    ) -> Self {
        Self {
            base_executor,
            profiles,
            llm,
            app_config,
            base_config,
        }
    }

    /// This tool with subagents falling back to `llm`/`config` (a session's
    /// live model) instead of the one it was built with.
    pub fn for_session(&self, llm: Arc<dyn LlmClient>, config: AgentConfig) -> Self {
        Self {
            base_executor: self.base_executor.clone(),
            profiles: self.profiles.clone(),
            llm,
            app_config: self.app_config.clone(),
            base_config: config,
        }
    }

    fn find_profile(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Resolves the [`AgentConfig`] and [`LlmClient`] a subagent run should use,
    /// honouring the profile's optional `model`/`provider`/`max_iterations`
    /// overrides. Reuses the parent's client when the resolved provider and model
    /// match — otherwise builds a fresh one, mirroring the pattern `AgentState::prompt`
    /// already uses for per-session model switches (see `src/core/runtime/state.rs`).
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
    /// delegate-free base executor, optionally narrowed to the profile's
    /// `tools` allowlist. The work-directory boundary needs no wrapper — it
    /// travels in the [`TurnContext`] the subagent inherits from its parent.
    fn build_executor(&self, profile: &AgentProfile) -> Arc<dyn ToolExecutor> {
        match &profile.tools {
            Some(allowed) => Arc::new(ScopedExecutor::new(
                self.base_executor.clone(),
                allowed.clone(),
            )),
            None => self.base_executor.clone(),
        }
    }
}

#[async_trait]
impl ToolHandler for DelegateTool {
    fn definition(&self) -> Tool {
        let listing = if self.profiles.is_empty() {
            "\n(none configured — define one inline via `system_prompt`)".to_string()
        } else {
            let mut listing = String::new();
            for profile in &self.profiles {
                let description = if profile.description.is_empty() {
                    "(no description provided)"
                } else {
                    profile.description.as_str()
                };
                listing.push_str(&format!("\n- `{}`: {description}", profile.name));
            }
            listing
        };

        let description = format!(
            "Delegate a self-contained task to a specialized subagent that runs independently \
             with its own context, persona, and (optionally) its own model or restricted tool \
             set. The subagent CANNOT see this conversation, so `task` must be a complete, \
             standalone brief containing every detail it needs. Only its final answer is \
             returned to you — its intermediate steps are not visible.\n\
             \n\
             Pick a pre-configured subagent by `agent` name, OR define an ephemeral one \
             inline by providing `system_prompt` (with optional `tools`, `model`, \
             `provider`, `max_iterations`). Inline subagents exist only for this one call \
             and are not saved. Exactly one of `agent` or `system_prompt` is required.\n\
             \n\
             Available subagents:{listing}"
        );

        let mut agent_schema = json!({
            "type": "string",
            "description": "Name of a pre-configured subagent to delegate to. \
                            Mutually exclusive with `system_prompt`."
        });
        if !self.profiles.is_empty() {
            let names: Vec<String> = self.profiles.iter().map(|p| p.name.clone()).collect();
            agent_schema["enum"] = json!(names);
        }

        Tool::function(
            DELEGATE_TOOL_NAME,
            description,
            json!({
                "type": "object",
                "properties": {
                    "agent": agent_schema,
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for an ephemeral inline subagent — \
                                        its persona and instructions. Mutually exclusive \
                                        with `agent`."
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Inline subagent only: restrict it to this set of \
                                        tool names. Omitted = it inherits your full tool set."
                    },
                    "model": {
                        "type": "string",
                        "description": "Inline subagent only: run it on this model instead \
                                        of yours."
                    },
                    "provider": {
                        "type": "string",
                        "description": "Inline subagent only: provider for `model`. Only \
                                        used when `model` is also set."
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Inline subagent only: cap its agent-loop iterations."
                    },
                    "task": {
                        "type": "string",
                        "description": "A complete, self-contained description of the task. \
                                        Include all context the subagent needs — it cannot \
                                        see your conversation history."
                    }
                },
                "required": ["task"]
            }),
        )
    }

    /// Runs the delegated subagent turn under the orchestrating turn's
    /// [`TurnContext`] rather than a fresh one: a `session/cancel` on the
    /// orchestrator must stop the subagent too, there is no separate trust
    /// policy for subagent tool calls (they go through the same gate, e.g.
    /// `session/request_permission`), and the subagent is confined to the
    /// same work directory and client I/O. The one change is that the gate is
    /// wrapped in a `SubagentGate`, so each request names the subagent.
    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args: DelegateArgs = parse(args)?;

        let profile = match (args.agent.as_deref(), args.system_prompt.as_deref()) {
            (Some(_), Some(_)) => {
                return Err(Error::ToolExecutionError(
                    "Provide either 'agent' (a pre-configured subagent) or 'system_prompt' \
                     (an inline one), not both."
                        .to_string(),
                ));
            }
            (Some(agent_name), None) => match self.find_profile(agent_name) {
                Some(profile) => profile.clone(),
                None => {
                    let available = self
                        .profiles
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(Error::ToolExecutionError(format!(
                        "Unknown subagent '{agent_name}'. Available subagents: {available}. \
                         Alternatively, define an inline subagent via 'system_prompt'."
                    )));
                }
            },
            (None, Some(system_prompt)) => {
                inline_profile(system_prompt, &args, self.base_config.max_iterations)
            }
            (None, None) => {
                return Err(Error::ToolExecutionError(
                    "Missing subagent: provide 'agent' (a pre-configured subagent) or \
                     'system_prompt' (an inline one)."
                        .to_string(),
                ));
            }
        };
        let profile = &profile;

        let (config, llm) = self.resolve_runtime(profile)?;
        let executor = self.build_executor(profile);

        let mut prompt_builder = PromptBuilder::new();
        prompt_builder.set_system(profile.system_prompt.clone());

        // Fresh, isolated history — the subagent only ever sees its own task.
        let mut messages = vec![Message::user(args.task.clone())];

        let permission_gate: Arc<dyn PermissionGate> = Arc::new(SubagentGate {
            parent: turn.permission_gate.clone(),
            subagent: profile.name.clone(),
        });
        let subagent_turn = TurnContext {
            permission_gate: &permission_gate,
            ..*turn
        };

        let result = run_agent(
            &*llm,
            &*executor,
            &config,
            &mut messages,
            Some(&prompt_builder),
            &subagent_turn,
            |_| {},
        )
        .await?;

        match result.stop_reason {
            StopReason::MaxIterations => Ok(format!(
                "{}\n\n[Note: subagent '{}' reached its iteration limit \
                 ({}) before finishing — this answer may be incomplete.]",
                result.final_response, profile.name, config.max_iterations
            )),
            reason => match reason.notice() {
                Some(notice) => Ok(format!(
                    "{}\n\n[Note: subagent '{}' did not finish normally: {notice}.]",
                    result.final_response, profile.name
                )),
                None => Ok(result.final_response),
            },
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        // Not read-only: a subagent's own tool set (its profile's `tools`
        // allowlist, defaulting to the parent's full set) can include
        // writes/execution, so `delegate_task` itself can't be exposed in
        // Architect mode. Kind stays the `Other` default — there's no ACP
        // `ToolKind` for "ran a subagent".
        ToolCapabilities::default()
    }
}

/// The parent turn's gate, with every request attributed to `subagent`: the
/// subagent's tool calls never appear in the parent's transcript, so without
/// this the user would be asked about calls they've never seen.
struct SubagentGate {
    parent: Arc<dyn PermissionGate>,
    subagent: String,
}

#[async_trait]
impl PermissionGate for SubagentGate {
    async fn check(&self, request: &PermissionRequest<'_>) -> PermissionDecision {
        self.parent
            .check(&request.from_subagent(&self.subagent))
            .await
    }
}

/// Builds an ephemeral [`AgentProfile`] from `delegate_task`'s inline arguments.
///
/// The profile lives only for this one call — it is never written to
/// `~/.openheim/agents/` or registered anywhere. Because it flows through the
/// same [`DelegateTool::resolve_runtime`]/[`DelegateTool::build_executor`] path
/// as named profiles, inline subagents get the identical sandbox, permission
/// gate, and no-recursion guarantees.
///
/// The LLM-supplied `max_iterations` is capped at `session_max_iterations`
/// so an orchestrator can't grant its subagent a longer run than its own.
fn inline_profile(
    system_prompt: &str,
    args: &DelegateArgs,
    session_max_iterations: usize,
) -> AgentProfile {
    AgentProfile {
        name: "inline".to_string(),
        description: String::new(),
        model: args.model.clone(),
        provider: args.provider.clone(),
        tools: args.tools.clone(),
        max_iterations: args.max_iterations.map(|n| n.min(session_max_iterations)),
        system_prompt: system_prompt.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::client_io::NoClientIo;
    use crate::core::models::{Choice, ContentBlock, FinishReason, Role};
    use crate::core::permission::AllowAll;
    use crate::tools::SystemToolExecutor;
    use crate::tools::test_support::TurnHarness;
    use std::path::Path;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    fn sample_app_config() -> AppConfig {
        AppConfig::for_tests("mock")
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

    fn text_choice(content: &str) -> Choice {
        Choice {
            message: Message::assistant(content),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        }
    }

    fn tool_call_choice() -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::tool_use("call_1", "nonexistent", "{}")],
            },
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
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
    async fn execute_errors_for_unknown_agent() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);
        let harness = TurnHarness::new();

        let err = tool
            .execute(
                r#"{"agent": "ghost", "task": "do something"}"#,
                &harness.turn(),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::ToolExecutionError(_)), "{err}");
        assert!(err.to_string().contains("Unknown subagent 'ghost'"));
        assert!(err.to_string().contains("reviewer"));
    }

    #[tokio::test]
    async fn execute_runs_subagent_in_isolated_context_and_returns_final_answer() {
        let llm = Arc::new(MockLlm::new(vec![text_choice("subagent answer")]));
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
        let llm = Arc::new(MockLlm::new(vec![text_choice("should not be seen")]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);

        let cancel = CancellationToken::new();
        cancel.cancel();
        let permission_gate: Arc<dyn PermissionGate> = Arc::new(AllowAll);
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
            work_dir: Path::new("."),
            cwd: Path::new("."),
            client_io: &NoClientIo,
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

    /// Rejects everything, recording which subagent (if any) each request
    /// named.
    #[derive(Default)]
    struct RejectPermissionGate {
        asked_by: Mutex<Vec<Option<String>>>,
    }

    #[async_trait]
    impl PermissionGate for RejectPermissionGate {
        async fn check(&self, request: &PermissionRequest<'_>) -> PermissionDecision {
            self.asked_by
                .lock()
                .unwrap()
                .push(request.subagent.map(str::to_string));
            PermissionDecision::RejectOnce
        }
    }

    #[tokio::test]
    async fn subagent_tool_calls_go_through_parent_permission_gate_named() {
        // Subagent tool calls must be checked by the same gate as the
        // orchestrator's own — there is no separate, more-trusting policy for
        // subagents — and the request must say which subagent is asking.
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice(),
            text_choice("denied"),
        ]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);

        let cancel = CancellationToken::new();
        let gate = Arc::new(RejectPermissionGate::default());
        let permission_gate: Arc<dyn PermissionGate> = gate.clone();
        let turn = TurnContext {
            cancel: &cancel,
            permission_gate: &permission_gate,
            work_dir: Path::new("."),
            cwd: Path::new("."),
            client_io: &NoClientIo,
        };

        let result = tool
            .execute(
                r#"{"agent": "reviewer", "task": "look at this diff"}"#,
                &turn,
            )
            .await
            .unwrap();

        assert_eq!(result, "denied");
        assert_eq!(
            *gate.asked_by.lock().unwrap(),
            vec![Some("reviewer".to_string())]
        );
    }

    /// Subagents are built from a snapshot of the registry taken before
    /// `delegate_task` registers itself, so they can never delegate again.
    /// This is the wiring `AgentState::new` relies on; keep it explicit
    /// here.
    #[test]
    fn subagents_never_see_delegate_task() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let mut executor = SystemToolExecutor::new();
        executor.register_builtins(true);
        let base: Arc<dyn ToolExecutor> = Arc::new(executor.clone());
        let tool = DelegateTool::new(
            base.clone(),
            vec![sample_profile("reviewer", "desc")],
            llm,
            sample_app_config(),
            sample_agent_config(),
        );
        executor.register(Box::new(tool));

        let names = |e: &dyn ToolExecutor| -> Vec<String> {
            e.list_tools()
                .into_iter()
                .map(|t| t.function.name)
                .collect()
        };
        assert!(names(&executor).contains(&DELEGATE_TOOL_NAME.to_string()));
        assert!(!names(&*base).contains(&DELEGATE_TOOL_NAME.to_string()));
        assert!(names(&*base).contains(&"read_file".to_string()));
    }

    #[test]
    fn definition_omits_enum_when_no_profiles() {
        // An empty `enum` would make `agent` unusable on strict providers; with
        // no profiles the constraint is dropped entirely.
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(vec![], llm);
        let def = tool.definition();

        assert!(def.function.parameters["properties"]["agent"]["enum"].is_null());
        assert!(def.function.description.contains("none configured"));
        let required = def.function.parameters["required"].as_array().unwrap();
        assert_eq!(required, &vec![serde_json::Value::String("task".into())]);
    }

    #[tokio::test]
    async fn execute_runs_inline_subagent_from_system_prompt() {
        let llm = Arc::new(MockLlm::new(vec![text_choice("inline answer")]));
        let tool = make_tool(vec![], llm);
        let harness = TurnHarness::new();

        let result = tool
            .execute(
                r#"{"system_prompt": "You are a poet.", "task": "write a haiku"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();

        assert_eq!(result, "inline answer");
    }

    #[tokio::test]
    async fn execute_notes_when_inline_subagent_hits_iteration_limit() {
        let llm = Arc::new(MockLlm::new(vec![tool_call_choice(), tool_call_choice()]));
        let tool = make_tool(vec![], llm);
        let harness = TurnHarness::new();

        let result = tool
            .execute(
                r#"{"system_prompt": "Loop.", "max_iterations": 2, "task": "go"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();

        assert!(result.contains("subagent 'inline' reached its iteration limit (2)"));
    }

    #[tokio::test]
    async fn execute_rejects_both_agent_and_system_prompt() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(vec![sample_profile("reviewer", "desc")], llm);
        let harness = TurnHarness::new();

        let err = tool
            .execute(
                r#"{"agent": "reviewer", "system_prompt": "You are X.", "task": "go"}"#,
                &harness.turn(),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::ToolExecutionError(_)), "{err}");
        assert!(err.to_string().contains("not both"));
    }

    #[tokio::test]
    async fn execute_rejects_neither_agent_nor_system_prompt() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(vec![], llm);
        let harness = TurnHarness::new();

        let err = tool
            .execute(r#"{"task": "go"}"#, &harness.turn())
            .await
            .unwrap_err();

        assert!(matches!(err, Error::ToolExecutionError(_)), "{err}");
        assert!(err.to_string().contains("Missing subagent"));
    }

    #[tokio::test]
    async fn inline_max_iterations_is_capped_at_the_sessions() {
        // The session allows 5 iterations; asking for 100 must not lift that.
        let llm = Arc::new(MockLlm::new((0..10).map(|_| tool_call_choice()).collect()));
        let tool = make_tool(vec![], llm);
        let harness = TurnHarness::new();

        let result = tool
            .execute(
                r#"{"system_prompt": "Loop.", "max_iterations": 100, "task": "go"}"#,
                &harness.turn(),
            )
            .await
            .unwrap();

        assert!(
            result.contains("reached its iteration limit (5)"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn execute_rejects_non_string_inline_tools() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let tool = make_tool(vec![], llm);
        let harness = TurnHarness::new();

        let err = tool
            .execute(
                r#"{"system_prompt": "X.", "tools": [1, 2], "task": "go"}"#,
                &harness.turn(),
            )
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("failed to parse arguments"),
            "{err}"
        );
    }

    #[test]
    fn inline_profile_maps_all_optional_fields() {
        let args: DelegateArgs = parse(
            r#"{
                "task": "go",
                "tools": ["read_file"],
                "model": "claude-haiku-4-5",
                "provider": "anthropic",
                "max_iterations": 3
            }"#,
        )
        .unwrap();

        let profile = inline_profile("You are X.", &args, 10);
        assert_eq!(profile.name, "inline");
        assert_eq!(profile.system_prompt, "You are X.");
        assert_eq!(profile.tools, Some(vec!["read_file".to_string()]));
        assert_eq!(profile.model.as_deref(), Some("claude-haiku-4-5"));
        assert_eq!(profile.provider.as_deref(), Some("anthropic"));
        assert_eq!(profile.max_iterations, Some(3));
    }

    #[tokio::test]
    async fn registered_delegate_tool_dispatches_through_system_executor() {
        let llm = Arc::new(MockLlm::new(vec![text_choice("done")]));
        let mut executor = SystemToolExecutor::new();
        let base: Arc<dyn ToolExecutor> = Arc::new(executor.clone());
        executor.register(Box::new(DelegateTool::new(
            base,
            vec![sample_profile("reviewer", "desc")],
            llm,
            sample_app_config(),
            sample_agent_config(),
        )));

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

    #[test]
    fn args_struct_matches_schema() {
        let llm = Arc::new(MockLlm::new(vec![]));
        crate::tools::args::assert_args_match_schema::<DelegateArgs>(
            &make_tool(vec![sample_profile("reviewer", "desc")], llm),
            serde_json::json!({
                "task": "go",
                "agent": "reviewer",
                "system_prompt": "You are X.",
                "tools": ["read_file"],
                "model": "m",
                "provider": "p",
                "max_iterations": 3
            }),
        );
    }
}
