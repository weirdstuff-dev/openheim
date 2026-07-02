use std::sync::Arc;

use tokio::sync::mpsc;

use crate::config::AgentConfig;
use crate::core::llm::{LlmChunk, LlmClient};
use crate::core::models::*;
use crate::core::turn::TurnContext;
use crate::error::Result;
use crate::rag::PromptBuilder;
use crate::tools::ToolExecutor;

async fn call_llm(
    llm: &Arc<dyn LlmClient>,
    messages: &[Message],
    tools: &[Tool],
    prompt_builder: Option<&PromptBuilder>,
) -> Result<Choice> {
    match prompt_builder {
        Some(builder) => {
            let built = builder.build(messages);
            llm.send(&built, tools).await
        }
        None => llm.send(messages, tools).await,
    }
}

async fn call_llm_streaming(
    llm: &Arc<dyn LlmClient>,
    messages: &[Message],
    tools: &[Tool],
    prompt_builder: Option<&PromptBuilder>,
    chunk_tx: mpsc::UnboundedSender<LlmChunk>,
) -> Result<Choice> {
    match prompt_builder {
        Some(builder) => {
            let built = builder.build(messages);
            llm.send_streaming(&built, tools, chunk_tx).await
        }
        None => llm.send_streaming(messages, tools, chunk_tx).await,
    }
}

/// Core agent loop: repeatedly calls the LLM and executes tool calls until a
/// final text response with `finish_reason == FinishReason::Stop` is produced or
/// `config.max_iterations` is reached.
///
/// Appends all assistant and tool-result messages to `messages` in place so the
/// caller retains a complete history after this returns.
///
/// If `callback` is `Some`, a [`StreamEvent`] is emitted for each significant
/// step: iteration start, tool calls, tool results, LLM text responses, and
/// the final completion.
///
/// `cancel` is checked between iterations and before each tool call, and is
/// also raced against the in-flight LLM call itself (both streaming and
/// non-streaming) so a caller (e.g. the ACP layer reacting to
/// `session/cancel`) can abort a slow or hanging request rather than waiting
/// for it to finish. The loop always returns `Ok`; [`AgentResult::stop_reason`]
/// reports why it stopped (`EndTurn` / `MaxIterations` / `Cancelled` /
/// `NoContent`) instead of callers having to reverse-engineer it.
async fn run_agent_loop<F>(
    llm: &Arc<dyn LlmClient>,
    tool_executor: &Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
    turn: &TurnContext<'_>,
    mut callback: Option<F>,
) -> Result<AgentResult>
where
    F: FnMut(StreamEvent) + Send,
{
    let tools = tool_executor.list_tools();
    let mut steps = Vec::new();
    let mut final_response = String::new();
    let mut iterations_used = 0;
    // Overwritten on every early exit; stays `MaxIterations` only if the
    // `for` loop below runs to completion without the LLM ever stopping.
    let mut stop_reason = StopReason::MaxIterations;

    'turn: for iteration in 0..config.max_iterations {
        if turn.cancel.is_cancelled() {
            stop_reason = StopReason::Cancelled;
            break;
        }

        let iter_num = iteration + 1;
        iterations_used = iter_num;

        if let Some(cb) = callback.as_mut() {
            cb(StreamEvent::IterationStart {
                iteration: iter_num,
            });
        }

        let choice = if callback.is_some() {
            let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<LlmChunk>();
            let choice_fut = call_llm_streaming(llm, messages, &tools, prompt_builder, chunk_tx);
            tokio::pin!(choice_fut);

            let mut maybe_choice: Option<Result<Choice>> = None;
            loop {
                tokio::select! {
                    _ = turn.cancel.cancelled() => {
                        // Dropping `choice_fut` here aborts the in-flight
                        // LLM request rather than waiting for it to finish.
                        stop_reason = StopReason::Cancelled;
                        break 'turn;
                    }
                    result = &mut choice_fut, if maybe_choice.is_none() => {
                        maybe_choice = Some(result);
                    }
                    maybe_chunk = chunk_rx.recv() => {
                        match maybe_chunk {
                            Some(LlmChunk::Text(text)) => {
                                if let Some(cb) = callback.as_mut() {
                                    cb(StreamEvent::LlmResponse { content: text });
                                }
                            }
                            Some(LlmChunk::Thinking(thought)) => {
                                if let Some(cb) = callback.as_mut() {
                                    cb(StreamEvent::ThinkingContent { content: thought });
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
            maybe_choice.unwrap_or_else(|| {
                Err(crate::error::Error::Other(
                    "stream ended prematurely".into(),
                ))
            })?
        } else {
            let result: Result<Choice> = tokio::select! {
                _ = turn.cancel.cancelled() => {
                    // Dropping the `call_llm` future here aborts the
                    // in-flight LLM request rather than waiting for it.
                    stop_reason = StopReason::Cancelled;
                    break 'turn;
                }
                result = call_llm(llm, messages, &tools, prompt_builder) => result,
            };
            result?
        };
        messages.push(choice.message.clone());
        if let Some(cb) = callback.as_mut() {
            cb(StreamEvent::MessageAppended {
                message: choice.message.clone(),
            });
        }

        let tool_calls = choice.message.tool_calls();
        if !tool_calls.is_empty() {
            let mut tool_results = Vec::new();

            for tool_call in &tool_calls {
                if turn.cancel.is_cancelled() {
                    stop_reason = StopReason::Cancelled;
                    break;
                }

                let id = &tool_call.id;
                let tool_name = &tool_call.name;
                let arguments = &tool_call.arguments;

                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::ToolCall {
                        id: id.clone(),
                        tool_name: tool_name.clone(),
                        arguments: arguments.clone(),
                    });
                }

                let decision = turn.permission_gate.check(id, tool_name, arguments).await;
                let (result, is_error) = if decision.is_allowed() {
                    match tool_executor.execute(tool_name, arguments, turn).await {
                        Ok(r) => (r, false),
                        Err(e) => (format!("Error: {e}"), true),
                    }
                } else {
                    ("Permission denied by user.".to_string(), true)
                };

                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::ToolResult {
                        id: id.clone(),
                        tool_name: tool_name.clone(),
                        result: result.clone(),
                        is_error,
                    });
                }

                tool_results.push(ToolExecutionResult {
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                    result: result.clone(),
                });

                let tool_result_message =
                    Message::tool_result(tool_call.id.clone(), tool_name.clone(), result, is_error);
                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::MessageAppended {
                        message: tool_result_message.clone(),
                    });
                }
                messages.push(tool_result_message);
            }

            steps.push(AgentStep {
                iteration: iter_num,
                message: "Tool calls executed".to_string(),
                tool_calls: Some(tool_results),
            });

            // Exit immediately rather than relying on next iteration's
            // top-of-loop check, which would never run if this was the
            // last allowed iteration and would misreport `MaxIterations`.
            if turn.cancel.is_cancelled() {
                stop_reason = StopReason::Cancelled;
                break 'turn;
            }
        } else if let Some(content) = choice.message.text() {
            // LlmResponse chunks already fired per-token from the streaming select
            // loop above; just record the final text here.
            final_response = content.clone();

            steps.push(AgentStep {
                iteration: iter_num,
                message: content,
                tool_calls: None,
            });

            if choice.finish_reason == Some(FinishReason::Stop) {
                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::Finished {
                        final_response: final_response.clone(),
                        iterations: iter_num,
                    });
                }

                return Ok(AgentResult {
                    final_response,
                    steps,
                    iterations_used: iter_num,
                    stop_reason: StopReason::EndTurn,
                });
            }
        } else {
            tracing::warn!(
                "Unexpected LLM response at iteration {}: no content or tool_calls",
                iter_num
            );
            stop_reason = StopReason::NoContent;
            break;
        }
    }

    if let Some(cb) = callback.as_mut() {
        cb(StreamEvent::Finished {
            final_response: final_response.clone(),
            iterations: iterations_used,
        });
    }

    Ok(AgentResult {
        final_response,
        steps,
        iterations_used,
        stop_reason,
    })
}

/// Runs the agent loop against an existing message history without streaming.
///
/// `messages` is extended in place with the full conversation turn — assistant
/// messages and tool results. The caller is responsible for persisting the
/// updated history after this returns.
///
/// # Arguments
///
/// * `llm` — LLM backend to use for inference
/// * `tool_executor` — resolves and executes tool calls made by the LLM
/// * `config` — agent settings, including `max_iterations`
/// * `messages` — conversation history; mutated in place
/// * `prompt_builder` — if `Some`, prepends skill-based system content to each LLM request
pub async fn run_agent_with_history(
    llm: Arc<dyn LlmClient>,
    tool_executor: Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
    turn: &TurnContext<'_>,
) -> Result<AgentResult> {
    run_agent_loop::<fn(StreamEvent)>(
        &llm,
        &tool_executor,
        config,
        messages,
        prompt_builder,
        turn,
        None,
    )
    .await
}

/// Streaming variant of [`run_agent_with_history`].
///
/// Identical in behaviour, but emits [`StreamEvent`]s via `callback` as the
/// agent progresses through iterations, tool calls, and LLM responses.
/// The callback is invoked synchronously on the same task and must not block.
pub async fn run_agent_streaming_with_history<F>(
    llm: Arc<dyn LlmClient>,
    tool_executor: Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
    turn: &TurnContext<'_>,
    callback: F,
) -> Result<AgentResult>
where
    F: FnMut(StreamEvent) + Send,
{
    run_agent_loop(
        &llm,
        &tool_executor,
        config,
        messages,
        prompt_builder,
        turn,
        Some(callback),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::permission::{AllowAll, PermissionDecision, PermissionGate};
    use crate::error::Error;
    use async_trait::async_trait;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio_util::sync::CancellationToken;

    fn make_config(max_iterations: usize) -> AgentConfig {
        AgentConfig {
            max_iterations,
            ..AgentConfig::default()
        }
    }

    fn allow_all() -> Arc<dyn PermissionGate> {
        Arc::new(AllowAll)
    }

    fn text_choice(content: &str) -> Choice {
        Choice {
            message: Message::assistant(content),
            finish_reason: Some(FinishReason::Stop),
        }
    }

    fn tool_call_choice(tool_name: &str, args: &str) -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: tool_name.into(),
                    arguments: args.into(),
                }],
            },
            finish_reason: Some(FinishReason::ToolCalls),
        }
    }

    /// Mock LLM that returns a sequence of choices
    struct MockLlm {
        responses: Mutex<Vec<Choice>>,
        call_count: AtomicUsize,
    }

    impl MockLlm {
        fn new(responses: Vec<Choice>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err(Error::ApiError("no more mock responses".into()))
            } else {
                Ok(responses.remove(0))
            }
        }
    }

    /// Mock ToolExecutor that returns a fixed result
    struct MockToolExecutor {
        result: String,
        calls: Mutex<Vec<(String, String)>>,
    }

    impl MockToolExecutor {
        fn new(result: &str) -> Self {
            Self {
                result: result.into(),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ToolExecutor for MockToolExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            vec![]
        }

        async fn execute(
            &self,
            name: &str,
            args_json: &str,
            _turn: &TurnContext<'_>,
        ) -> Result<String> {
            self.calls
                .lock()
                .unwrap()
                .push((name.into(), args_json.into()));
            Ok(self.result.clone())
        }
    }

    struct FailingToolExecutor;

    #[async_trait]
    impl ToolExecutor for FailingToolExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            vec![]
        }
        async fn execute(
            &self,
            _name: &str,
            _args: &str,
            _turn: &TurnContext<'_>,
        ) -> Result<String> {
            Err(Error::ApiError("tool failed".into()))
        }
    }

    #[tokio::test]
    async fn agent_stops_on_finish_reason_stop() {
        let llm = Arc::new(MockLlm::new(vec![text_choice("done")]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let config = make_config(10);
        let mut messages = vec![Message::user("hi")];

        let result = run_agent_with_history(
            llm.clone(),
            executor,
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "done");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn agent_executes_tool_calls_and_continues() {
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("read_file", r#"{"path":"a.txt"}"#),
            text_choice("here is the file content"),
        ]));
        let executor = Arc::new(MockToolExecutor::new("file data"));
        let config = make_config(10);
        let mut messages = vec![Message::user("read a.txt")];

        let result = run_agent_with_history(
            llm.clone(),
            executor.clone(),
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "here is the file content");
        assert_eq!(result.iterations_used, 2);
        // Verify tool was called
        let calls = executor.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "read_file");
    }

    #[tokio::test]
    async fn agent_respects_max_iterations() {
        // LLM always returns tool calls, never stops
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
        ]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new("data"));
        let config = make_config(3);
        let mut messages = vec![Message::user("loop")];

        let result = run_agent_with_history(
            llm,
            executor,
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
            },
        )
        .await
        .unwrap();

        assert_eq!(result.iterations_used, 3);
        assert_eq!(result.stop_reason, StopReason::MaxIterations);
    }

    #[tokio::test]
    async fn agent_streaming_emits_events() {
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("echo", r#"{"cmd":"hi"}"#),
            text_choice("all done"),
        ]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new("ok"));
        let config = make_config(10);
        let mut messages = vec![Message::user("test")];

        let mut events = Vec::new();
        let result = run_agent_streaming_with_history(
            llm,
            executor,
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
            },
            |event| events.push(event),
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "all done");

        // Check relative event ordering.
        let tool_call_idx = events
            .iter()
            .position(
                |e| matches!(e, StreamEvent::ToolCall { tool_name, .. } if tool_name == "echo"),
            )
            .unwrap();
        let tool_result_idx = events
            .iter()
            .position(
                |e| matches!(e, StreamEvent::ToolResult { tool_name, .. } if tool_name == "echo"),
            )
            .unwrap();
        let llm_response_idx = events
            .iter()
            .position(
                |e| matches!(e, StreamEvent::LlmResponse { content } if content == "all done"),
            )
            .unwrap();

        let finished_idx = events
            .iter()
            .position(|e| matches!(e, StreamEvent::Finished { .. }))
            .unwrap();
        let message_appended_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::MessageAppended { .. }))
            .count();

        assert!(matches!(
            events[0],
            StreamEvent::IterationStart { iteration: 1 }
        ));
        assert!(tool_call_idx < tool_result_idx);
        assert!(tool_result_idx < llm_response_idx);
        assert!(llm_response_idx < finished_idx);
        // One MessageAppended per pushed message: the tool-call assistant
        // message, the tool-result message, and the final text response.
        assert_eq!(message_appended_count, 3);
    }

    #[tokio::test]
    async fn agent_feeds_tool_error_back_to_llm() {
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("bad_tool", "{}"),
            text_choice("I got an error"),
        ]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(FailingToolExecutor);
        let config = make_config(10);
        let mut messages = vec![Message::user("do something")];

        // Should not propagate the error; LLM should receive it as a tool result
        let result = run_agent_with_history(
            llm,
            executor,
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "I got an error");
        // The tool result message should contain the error text
        let tool_result_msg = messages
            .iter()
            .find(|m| m.tool_result_block().is_some())
            .unwrap();
        assert!(
            tool_result_msg
                .tool_result_block()
                .unwrap()
                .content
                .contains("Error:")
        );
    }

    #[tokio::test]
    async fn agent_stops_early_when_cancelled() {
        // LLM always returns tool calls, never stops on its own.
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
        ]));
        let executor = Arc::new(MockToolExecutor::new("data"));
        let config = make_config(10);
        let mut messages = vec![Message::user("loop")];
        let cancel = CancellationToken::new();
        let cancel_signal = cancel.clone();

        let result = run_agent_streaming_with_history(
            llm,
            executor.clone(),
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &cancel,
                permission_gate: &allow_all(),
            },
            move |event| {
                if matches!(event, StreamEvent::ToolResult { .. }) {
                    cancel_signal.cancel();
                }
            },
        )
        .await
        .unwrap();

        // Only the first iteration's tool call should have run before the
        // second iteration's cancellation check stopped the loop.
        assert_eq!(result.iterations_used, 1);
        let calls = executor.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(result.stop_reason, StopReason::Cancelled);
    }

    /// LLM that never resolves on its own; used to prove cancellation aborts
    /// an in-flight call instead of waiting for it to finish.
    struct SlowLlm;

    #[async_trait]
    impl LlmClient for SlowLlm {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this call before the sleep elapses");
        }
    }

    #[tokio::test]
    async fn cancel_aborts_in_flight_llm_call_streaming() {
        let llm = Arc::new(SlowLlm);
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let config = make_config(10);
        let mut messages = vec![Message::user("hi")];
        let cancel = CancellationToken::new();
        let cancel_signal = cancel.clone();

        // Cancel as soon as the loop starts its first iteration, i.e. right
        // before the (never-resolving) LLM call is made.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_agent_streaming_with_history(
                llm,
                executor,
                &config,
                &mut messages,
                None,
                &TurnContext {
                    cancel: &cancel,
                    permission_gate: &allow_all(),
                },
                move |event| {
                    if matches!(event, StreamEvent::IterationStart { .. }) {
                        cancel_signal.cancel();
                    }
                },
            ),
        )
        .await
        .expect("run_agent_loop should abort the in-flight LLM call instead of hanging")
        .unwrap();

        assert_eq!(result.final_response, "");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(result.stop_reason, StopReason::Cancelled);
    }

    #[tokio::test]
    async fn cancel_aborts_in_flight_llm_call_non_streaming() {
        let llm = Arc::new(SlowLlm);
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let config = make_config(10);
        let mut messages = vec![Message::user("hi")];
        let cancel = CancellationToken::new();
        let cancel_signal = cancel.clone();

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel_signal.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_agent_with_history(
                llm,
                executor,
                &config,
                &mut messages,
                None,
                &TurnContext {
                    cancel: &cancel,
                    permission_gate: &allow_all(),
                },
            ),
        )
        .await
        .expect("run_agent_loop should abort the in-flight LLM call instead of hanging")
        .unwrap();

        assert_eq!(result.final_response, "");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(result.stop_reason, StopReason::Cancelled);
    }

    #[tokio::test]
    async fn agent_reports_no_content_stop_reason() {
        // An LLM response with neither text nor tool calls is an anomaly
        // that should stop the loop, not be treated as a normal completion.
        let empty_choice = Choice {
            message: Message {
                role: Role::Assistant,
                content: vec![],
            },
            finish_reason: Some(FinishReason::Stop),
        };
        let llm = Arc::new(MockLlm::new(vec![empty_choice]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let config = make_config(10);
        let mut messages = vec![Message::user("hi")];

        let result = run_agent_with_history(
            llm,
            executor,
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.stop_reason, StopReason::NoContent);
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
    async fn agent_skips_execution_when_permission_denied() {
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("execute_command", r#"{"command":"rm -rf /"}"#),
            text_choice("I was denied"),
        ]));
        let executor = Arc::new(MockToolExecutor::new("should not run"));
        let config = make_config(10);
        let mut messages = vec![Message::user("do something dangerous")];

        let result = run_agent_with_history(
            llm,
            executor.clone(),
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &(Arc::new(RejectPermissionGate) as Arc<dyn PermissionGate>),
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "I was denied");
        // The tool must never actually execute once permission is denied.
        assert!(executor.calls.lock().unwrap().is_empty());
        let tool_result_msg = messages
            .iter()
            .find(|m| m.tool_result_block().is_some())
            .unwrap()
            .tool_result_block()
            .unwrap();
        assert_eq!(tool_result_msg.content, "Permission denied by user.");
        assert!(tool_result_msg.is_error);
    }
}
