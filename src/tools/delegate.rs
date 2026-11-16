//! Subagent delegation: the `delegate_task` tool hands a self-contained task
//! to a subagent, either a named profile from `~/.openheim/agents/` (see
//! [`crate::subagents`]) or one defined inline in the call (`system_prompt`
//! plus optional model and tool overrides), which is never saved.
//!
//! Each call runs a separate [`run_agent`] turn with its own history and
//! system prompt (not the parent's `system.md` or skills), and returns only
//! the subagent's final answer.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::config::{AgentConfig, AppConfig, HttpClients, client_for_config};
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

/// The `delegate_task` tool: runs each call as a separate agent turn.
///
/// `base_executor` must be the tool registry as it was before
/// `delegate_task` was added, so subagents can't delegate recursively.
/// `llm`/`base_config` are the model a subagent runs on when its profile
/// names none; [`Self::for_session`] rebinds them to a session's model.
#[derive(Clone)]
pub struct DelegateTool {
    base_executor: Arc<dyn ToolExecutor>,
    profiles: Vec<AgentProfile>,
    llm: Arc<dyn LlmClient>,
    app_config: AppConfig,
    base_config: AgentConfig,
    /// Where clients for profiles with their own model get their
    /// `reqwest::Client`.
    http: Arc<HttpClients>,
}

impl DelegateTool {
    pub fn new(
        base_executor: Arc<dyn ToolExecutor>,
        profiles: Vec<AgentProfile>,
        llm: Arc<dyn LlmClient>,
        app_config: AppConfig,
        base_config: AgentConfig,
        http: Arc<HttpClients>,
    ) -> Self {
        Self {
            base_executor,
            profiles,
            llm,
            app_config,
            base_config,
            http,
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
            http: self.http.clone(),
        }
    }

    fn find_profile(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// The [`AgentConfig`] and [`LlmClient`] a subagent runs on, with the
    /// profile's `model`/`provider`/`effort`/`max_iterations` overrides
    /// applied. The parent's client is reused when provider, model and
    /// effort level match.
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
        let on_session_model = config.provider_name == self.base_config.provider_name
            && config.model == self.base_config.model;
        let config = match profile.effort {
            Some(effort) => config.with_effort(effort),
            None if on_session_model => config.with_effort(self.base_config.effort),
            None => config,
        };

        let llm = client_for_config(&config, &self.base_config, &self.llm, &self.http)?;

        Ok((config, llm))
    }

    /// The subagent's tools: the base executor, narrowed to the profile's
    /// `tools` allowlist if it has one.
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

    /// Runs the subagent under the parent turn's [`TurnContext`], so it is
    /// cancelled with the parent, asks the same permission gate, and works in
    /// the same directories. The gate is wrapped in a `SubagentGate`, so each
    /// request names the subagent.
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
        // Not read-only: a subagent's tools can write and execute.
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

/// An [`AgentProfile`] for one call, built from `delegate_task`'s inline
/// arguments and never saved. The model's `max_iterations` is capped at
/// `session_max_iterations`, so a subagent can't run longer than its parent.
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
        effort: None,
        tools: args.tools.clone(),
        max_iterations: args.max_iterations.map(|n| n.min(session_max_iterations)),
        system_prompt: system_prompt.to_string(),
    }
}
