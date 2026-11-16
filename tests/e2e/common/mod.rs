//! Test fixture: a temp config dir and workspace wired to a [`FakeLlm`].

pub mod bin;
pub mod llm;
pub mod mcp;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use openheim::core::permission::{PermissionDecision, PermissionGate, PermissionRequest};
use openheim::{OpenheimClient, PromptInput, SessionHandle, StopReason, StreamEvent};
use tempfile::TempDir;

pub use llm::{
    Answer, FakeLlm, Kind, Reply, finish, rate_limited, status, text, thinking, tool, tools,
};

pub struct Env {
    pub llm: FakeLlm,
    /// Holds `config.toml` and the data (history, skills, memory.db).
    pub dir: TempDir,
    /// The agent's `work_dir`, and the cwd of its sessions and processes.
    pub work: PathBuf,
}

impl Env {
    /// An OpenAI-format provider and no extra config.
    pub async fn new() -> Self {
        Self::with(Kind::OpenAi, "").await
    }

    /// `extra` is TOML placed at the top level of the config, before the
    /// provider table.
    pub async fn with(kind: Kind, extra: &str) -> Self {
        Self::build(kind, extra, "").await
    }

    /// Like [`Self::with`]; `provider` is TOML added to the provider table.
    pub async fn build(kind: Kind, extra: &str, provider: &str) -> Self {
        let llm = FakeLlm::start(kind).await;
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        std::fs::create_dir(&work).unwrap();
        let work = work.canonicalize().unwrap();
        let config = format!(
            r#"default_provider = "fake"
work_dir = {work:?}
{extra}

[providers.fake]
kind = "{kind}"
api_base = "{base}"
api_key = "test-key"
default_model = "fake-model"
models = ["fake-model", "fake-model-2"]
{provider}
"#,
            work = work.display().to_string(),
            kind = kind.config_kind(),
            base = llm.api_base(),
        );
        std::fs::write(dir.path().join("config.toml"), config).unwrap();
        Self { llm, dir, work }
    }

    pub fn config(&self) -> PathBuf {
        self.dir.path().join("config.toml")
    }

    pub fn script(&self, replies: impl IntoIterator<Item = Reply>) -> &Self {
        self.llm.script(replies);
        self
    }

    /// Writes `contents` to `path` relative to the workspace.
    pub fn file(&self, path: &str, contents: &str) -> &Self {
        let path = self.work.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
        self
    }

    /// Reads `path` relative to the workspace.
    pub fn read(&self, path: &str) -> String {
        std::fs::read_to_string(self.work.join(path)).unwrap()
    }

    pub async fn client(&self) -> OpenheimClient {
        OpenheimClient::from_config(self.config())
            .build()
            .await
            .unwrap()
    }

    pub async fn session(&self) -> Session {
        let client = self.client().await;
        let handle = client.new_session().cwd(&self.work).start().await.unwrap();
        Session { client, handle }
    }

    pub fn path(&self, rel: &str) -> PathBuf {
        self.work.join(rel)
    }

    pub fn data(&self) -> &Path {
        self.dir.path()
    }
}

pub struct Session {
    pub client: OpenheimClient,
    pub handle: SessionHandle,
}

impl Session {
    pub fn gate(self, gate: &Arc<Gate>) -> Self {
        Self {
            handle: self.handle.permission_gate(gate.clone()),
            ..self
        }
    }

    pub async fn say(&self, input: impl Into<PromptInput>) -> Turn {
        self.try_say(input).await.unwrap()
    }

    pub async fn try_say(&self, input: impl Into<PromptInput>) -> openheim::Result<Turn> {
        let mut events = Vec::new();
        let stop = self.handle.prompt(input, |e| events.push(e)).await?;
        Ok(Turn { stop, events })
    }
}

/// A permission gate answering with scripted decisions (then `AllowOnce`),
/// or never answering if built with [`Gate::hang`]. Records the tools asked
/// about.
#[derive(Default)]
pub struct Gate {
    decisions: Mutex<VecDeque<PermissionDecision>>,
    hang: bool,
    pub asked: Mutex<Vec<String>>,
}

impl Gate {
    pub fn new(decisions: impl IntoIterator<Item = PermissionDecision>) -> Arc<Self> {
        Arc::new(Self {
            decisions: Mutex::new(decisions.into_iter().collect()),
            ..Self::default()
        })
    }

    pub fn hang() -> Arc<Self> {
        Arc::new(Self {
            hang: true,
            ..Self::default()
        })
    }

    pub fn asked(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl PermissionGate for Gate {
    async fn check(&self, request: &PermissionRequest<'_>) -> PermissionDecision {
        self.asked
            .lock()
            .unwrap()
            .push(request.tool_name.to_string());
        if self.hang {
            std::future::pending::<()>().await;
        }
        let next = self.decisions.lock().unwrap().pop_front();
        next.unwrap_or(PermissionDecision::AllowOnce)
    }
}

/// Everything one prompt produced.
#[derive(Debug)]
pub struct Turn {
    pub stop: StopReason,
    pub events: Vec<StreamEvent>,
}

impl Turn {
    /// The final response.
    pub fn text(&self) -> String {
        self.events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Finished { final_response, .. } => Some(final_response.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// `(tool, result, is_error)` for every tool call, in order.
    pub fn results(&self) -> Vec<(String, String, bool)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolResult {
                    tool_name,
                    result,
                    is_error,
                    ..
                } => Some((tool_name.clone(), result.clone(), *is_error)),
                _ => None,
            })
            .collect()
    }

    /// The single tool result, which must not be an error.
    pub fn ok_result(&self) -> String {
        let results = self.results();
        assert_eq!(results.len(), 1, "expected one tool result: {results:?}");
        let (name, result, is_error) = &results[0];
        assert!(!is_error, "{name} failed: {result}");
        result.clone()
    }

    /// The single tool result, which must be an error.
    pub fn err_result(&self) -> String {
        let results = self.results();
        assert_eq!(results.len(), 1, "expected one tool result: {results:?}");
        let (name, result, is_error) = &results[0];
        assert!(is_error, "{name} succeeded: {result}");
        result.clone()
    }
}
