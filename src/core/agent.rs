use std::sync::Arc;

use tokio::sync::mpsc;

use crate::config::AgentConfig;
use crate::core::llm::{LlmChunk, LlmClient};
use crate::core::models::*;
use crate::core::turn::TurnContext;
use crate::error::Result;
use crate::memory::PromptBuilder;
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

/// Passes one streamed chunk to the caller's callback as the matching event.
fn forward_chunk<F: FnMut(StreamEvent)>(callback: &mut Option<F>, chunk: LlmChunk) {
    let Some(cb) = callback.as_mut() else {
        return;
    };
    match chunk {
        LlmChunk::Text(text) => cb(StreamEvent::LlmResponse { content: text }),
        LlmChunk::Thinking(thought) => cb(StreamEvent::ThinkingContent { content: thought }),
    }
}

/// How a turn ends when the LLM replies without any tool calls, from the
/// provider's finish reason and whether the reply had text. `None` means
/// "keep looping": the provider paused the turn and expects the
/// conversation resent as-is to resume it.
///
/// Any finish reason without a specific mapping (`Other`, a missing one, or
/// `ToolCalls` with no tool calls attached) ends the turn: resending a
/// history that ends in an assistant reply would at best make the model
/// repeat itself, and at worst be rejected outright.
fn stop_for_reply_without_tools(
    finish_reason: Option<&FinishReason>,
    has_text: bool,
) -> Option<StopReason> {
    match finish_reason {
        Some(FinishReason::Paused) => None,
        Some(FinishReason::MaxTokens) => Some(StopReason::MaxTokens),
        Some(FinishReason::Refusal) => Some(StopReason::Refusal),
        _ if !has_text => Some(StopReason::NoContent),
        Some(FinishReason::Stop) => Some(StopReason::EndTurn),
        other => {
            tracing::debug!(finish_reason = ?other, "treating unmapped finish reason as end of turn");
            Some(StopReason::EndTurn)
        }
    }
}

/// Core agent loop: repeatedly calls the LLM and executes tool calls until
/// the LLM replies without tool calls (see [`stop_for_reply_without_tools`]
/// for how that reply's finish reason maps to a [`StopReason`]) or
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
/// also raced against the in-flight LLM call and the pending permission-gate
/// approval (both streaming and non-streaming LLM calls) so a caller (e.g.
/// the ACP layer reacting to `session/cancel`) can abort a slow or hanging
/// request — or a turn stuck waiting on user approval — rather than waiting
/// for it to finish. The loop returns `Ok` unless an LLM call fails;
/// [`AgentResult::stop_reason`] reports why it stopped instead of callers
/// having to reverse-engineer it.
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
    let mut final_response = String::new();
    let mut iterations_used = 0;
    let mut context_usage: Option<Usage> = None;
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

            // The reply ends the wait, not the chunk channel: a client may
            // close its sender before its request finishes, and one that
            // leaks the sender somewhere must not keep the turn waiting.
            let mut chunks_open = true;
            loop {
                tokio::select! {
                    _ = turn.cancel.cancelled() => {
                        // Dropping `choice_fut` here aborts the in-flight
                        // LLM request rather than waiting for it to finish.
                        stop_reason = StopReason::Cancelled;
                        break 'turn;
                    }
                    result = &mut choice_fut => {
                        // Chunks sent before the reply completed may still be
                        // buffered; forward them, in order, before moving on.
                        while let Ok(chunk) = chunk_rx.try_recv() {
                            forward_chunk(&mut callback, chunk);
                        }
                        break result;
                    }
                    maybe_chunk = chunk_rx.recv(), if chunks_open => match maybe_chunk {
                        Some(chunk) => forward_chunk(&mut callback, chunk),
                        None => chunks_open = false,
                    },
                }
            }?
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
        if let Some(usage) = choice.usage {
            // Overwritten (not summed) every iteration: the latest call's
            // usage is what "context size right now" means, since each call
            // resends the full history as its prompt.
            context_usage = Some(usage);
            if let Some(cb) = callback.as_mut() {
                cb(StreamEvent::Usage { usage });
            }
        }

        let tool_calls = choice.message.tool_calls();
        if !tool_calls.is_empty() {
            // Phase 1a: announce every call up front, before any permission
            // check runs. Each is reported `Pending` (see the ACP mapping),
            // which is accurate for all of them the moment the LLM asks —
            // not just the one whose approval happens to be next in line.
            for tool_call in &tool_calls {
                if turn.cancel.is_cancelled() {
                    stop_reason = StopReason::Cancelled;
                    break 'turn;
                }

                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::ToolCall {
                        id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        arguments: tool_call.arguments.clone(),
                    });
                }
            }

            // Phase 1b: collect every call's permission decision
            // concurrently, mirroring Phase 2's execution model — an
            // interactive gate sees every request up front instead of
            // one at a time. The ACP gate can answer them independently and
            // out of order; the TUI gate queues concurrent requests instead
            // of dropping them (see `App::handle_permission_request`), so
            // it still only *shows* one prompt at a time.
            let decisions = tokio::select! {
                _ = turn.cancel.cancelled() => {
                    // Dropping the `join_all` here abandons every pending
                    // approval prompt at once, rather than blocking the turn
                    // (and holding `prompt_lock`) until the user responds to
                    // each in turn.
                    stop_reason = StopReason::Cancelled;
                    break 'turn;
                }
                decisions = futures::future::join_all(tool_calls.iter().map(|tool_call| {
                    turn.permission_gate.check(&tool_call.id, &tool_call.name, &tool_call.arguments)
                })) => decisions,
            };

            // Phase 2: run every allowed call concurrently (denied ones
            // short-circuit without touching the executor). This is what lets
            // several `delegate_task` sub-agents — or any other independent
            // tool calls the LLM batched into one turn — actually run in
            // parallel instead of one after another. `ToolResult` is emitted
            // as soon as each call finishes (completion order), so a caller
            // watching the stream sees fast calls report back before slow
            // ones instead of everything landing at once; message history in
            // Phase 3 stays in original tool-call order regardless.
            let mut pending: futures::stream::FuturesUnordered<_> = tool_calls
                .iter()
                .zip(&decisions)
                .enumerate()
                .map(|(index, (tool_call, decision))| async move {
                    if decision.is_allowed() {
                        match tool_executor
                            .execute(&tool_call.name, &tool_call.arguments, turn)
                            .await
                        {
                            Ok(r) => (index, r, false),
                            Err(e) => (index, format!("Error: {e}"), true),
                        }
                    } else {
                        (index, "Permission denied by user.".to_string(), true)
                    }
                })
                .collect();

            let mut outcomes: Vec<Option<(String, bool)>> = vec![None; tool_calls.len()];
            loop {
                tokio::select! {
                    _ = turn.cancel.cancelled() => {
                        // Dropping `pending` here abandons every in-flight
                        // tool call at once, rather than blocking the turn
                        // until all of them finish.
                        stop_reason = StopReason::Cancelled;
                        break 'turn;
                    }
                    next = futures::StreamExt::next(&mut pending) => {
                        let Some((index, result, is_error)) = next else {
                            break;
                        };
                        let tool_call = &tool_calls[index];
                        if let Some(cb) = callback.as_mut() {
                            cb(StreamEvent::ToolResult {
                                id: tool_call.id.clone(),
                                tool_name: tool_call.name.clone(),
                                result: result.clone(),
                                is_error,
                            });
                        }
                        outcomes[index] = Some((result, is_error));
                    }
                }
            }

            // Phase 3: replay results in original tool-call order (every
            // entry is `Some` here — the loop above only exits early via
            // `break 'turn` on cancellation), so message history and the
            // callback's `MessageAppended` stream stay deterministic
            // regardless of which call actually finished first.
            for (tool_call, outcome) in tool_calls.iter().zip(outcomes) {
                let (result, is_error) = outcome.expect("every outcome is filled before Phase 3");

                let tool_result_message = Message::tool_result(
                    tool_call.id.clone(),
                    tool_call.name.clone(),
                    result,
                    is_error,
                );
                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::MessageAppended {
                        message: tool_result_message.clone(),
                    });
                }
                messages.push(tool_result_message);
            }

            // Exit immediately rather than relying on next iteration's
            // top-of-loop check, which would never run if this was the
            // last allowed iteration and would misreport `MaxIterations`.
            if turn.cancel.is_cancelled() {
                stop_reason = StopReason::Cancelled;
                break 'turn;
            }
        } else {
            // LlmResponse chunks already fired per-token from the streaming select
            // loop above; just record the final text here.
            let text = choice.message.text();
            let has_text = text.is_some();
            if let Some(content) = text {
                final_response = content;
            }
            match stop_for_reply_without_tools(choice.finish_reason.as_ref(), has_text) {
                None => continue,
                Some(reason) => {
                    if reason == StopReason::NoContent {
                        tracing::warn!(
                            "Unexpected LLM response at iteration {}: no content or tool_calls",
                            iter_num
                        );
                    }
                    stop_reason = reason;
                    break;
                }
            }
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
        iterations_used,
        stop_reason,
        context_usage,
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
    use crate::core::client_io::NoClientIo;
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
            usage: None,
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
            usage: None,
        }
    }

    fn multi_tool_call_choice(calls: &[(&str, &str)]) -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content: calls
                    .iter()
                    .enumerate()
                    .map(|(i, (name, args))| ContentBlock::ToolUse {
                        id: format!("call_{i}"),
                        name: (*name).into(),
                        arguments: (*args).into(),
                    })
                    .collect(),
            },
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
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

    /// ToolExecutor where each tool's completion is gated behind a
    /// per-tool sleep, so a test can control which of several concurrent
    /// calls finishes first (deterministic under `start_paused = true`).
    struct DelayedToolExecutor {
        delays: std::collections::HashMap<String, std::time::Duration>,
    }

    #[async_trait]
    impl ToolExecutor for DelayedToolExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            vec![]
        }
        async fn execute(
            &self,
            name: &str,
            _args: &str,
            _turn: &TurnContext<'_>,
        ) -> Result<String> {
            if let Some(delay) = self.delays.get(name) {
                tokio::time::sleep(*delay).await;
            }
            Ok(format!("{name}-done"))
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "done");
        assert_eq!(result.iterations_used, 1);
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
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

    #[tokio::test(start_paused = true)]
    async fn tool_results_emit_in_completion_order_not_call_order() {
        // The LLM batches a slow call and a fast call into one turn; the fast
        // one should be reported to the caller as soon as it finishes,
        // without waiting on the slow one — even though it was requested
        // second.
        let llm = Arc::new(MockLlm::new(vec![
            multi_tool_call_choice(&[("slow", "{}"), ("fast", "{}")]),
            text_choice("done"),
        ]));
        let delays = std::collections::HashMap::from([
            ("slow".to_string(), std::time::Duration::from_secs(10)),
            ("fast".to_string(), std::time::Duration::from_millis(1)),
        ]);
        let executor = Arc::new(DelayedToolExecutor { delays });
        let config = make_config(10);
        let mut messages = vec![Message::user("go")];
        let completion_order = Arc::new(Mutex::new(Vec::new()));
        let completion_order_cb = completion_order.clone();

        run_agent_streaming_with_history(
            llm,
            executor,
            &config,
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
            },
            move |event| {
                if let StreamEvent::ToolResult { tool_name, .. } = event {
                    completion_order_cb.lock().unwrap().push(tool_name);
                }
            },
        )
        .await
        .unwrap();

        assert_eq!(*completion_order.lock().unwrap(), vec!["fast", "slow"]);
        // Message history still reflects the original tool-call order,
        // regardless of completion order.
        let tool_result_names: Vec<_> = messages
            .iter()
            .filter_map(|m| m.tool_result_block().map(|tr| tr.tool_name.clone()))
            .collect();
        assert_eq!(tool_result_names, vec!["slow", "fast"]);
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
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

    /// LLM that never resolves on its own; proves cancellation aborts
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
                    work_dir: std::path::Path::new("."),
                    client_io: &NoClientIo,
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

    /// ToolExecutor that never resolves; proves cancellation aborts an
    /// in-flight tool call (Phase 2) instead of waiting for it to finish.
    struct SlowToolExecutor;

    #[async_trait]
    impl ToolExecutor for SlowToolExecutor {
        fn list_tools(&self) -> Vec<Tool> {
            vec![]
        }
        async fn execute(
            &self,
            _name: &str,
            _args: &str,
            _turn: &TurnContext<'_>,
        ) -> Result<String> {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this call before the sleep elapses");
        }
    }

    #[tokio::test]
    async fn cancel_aborts_in_flight_tool_call() {
        let llm = Arc::new(MockLlm::new(vec![tool_call_choice("read_file", "{}")]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(SlowToolExecutor);
        let config = make_config(10);
        let mut messages = vec![Message::user("hi")];
        let cancel = CancellationToken::new();
        let cancel_signal = cancel.clone();

        // Cancel right after the tool call is announced (Phase 1a), i.e.
        // right before Phase 2 starts executing it.
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
                    work_dir: std::path::Path::new("."),
                    client_io: &NoClientIo,
                },
                move |event| {
                    if matches!(event, StreamEvent::ToolCall { .. }) {
                        cancel_signal.cancel();
                    }
                },
            ),
        )
        .await
        .expect("run_agent_loop should abort the in-flight tool call instead of hanging")
        .unwrap();

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
                    work_dir: std::path::Path::new("."),
                    client_io: &NoClientIo,
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
            usage: None,
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.final_response, "");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.stop_reason, StopReason::NoContent);
    }

    fn choice_with(content: Vec<ContentBlock>, finish_reason: Option<FinishReason>) -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content,
            },
            finish_reason,
            usage: None,
        }
    }

    // Regression test: only `FinishReason::Stop` used to end the turn; any
    // other finish on a tool-free reply resent the history (ending in that
    // assistant reply) until `max_iterations` ran out.
    #[tokio::test]
    async fn reply_without_tools_ends_turn_per_finish_reason() {
        let text = || vec![ContentBlock::from("partial answer")];
        let cases = [
            (text(), Some(FinishReason::Stop), StopReason::EndTurn),
            (text(), Some(FinishReason::MaxTokens), StopReason::MaxTokens),
            (text(), Some(FinishReason::Refusal), StopReason::Refusal),
            (
                text(),
                Some(FinishReason::Other("stop_sequence".into())),
                StopReason::EndTurn,
            ),
            (text(), None, StopReason::EndTurn),
            // A thinking model can spend the whole budget before emitting
            // any text; that's truncation, not an empty reply.
            (vec![], Some(FinishReason::MaxTokens), StopReason::MaxTokens),
            (vec![], Some(FinishReason::Refusal), StopReason::Refusal),
        ];

        for (content, finish_reason, expected) in cases {
            let has_text = !content.is_empty();
            let llm = Arc::new(MockLlm::new(vec![choice_with(
                content,
                finish_reason.clone(),
            )]));
            let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
            let mut messages = vec![Message::user("hi")];

            let result = run_agent_with_history(
                llm.clone(),
                executor,
                &make_config(10),
                &mut messages,
                None,
                &TurnContext {
                    cancel: &CancellationToken::new(),
                    permission_gate: &allow_all(),
                    work_dir: std::path::Path::new("."),
                    client_io: &NoClientIo,
                },
            )
            .await
            .unwrap();

            assert_eq!(result.stop_reason, expected, "{finish_reason:?}");
            assert_eq!(
                llm.call_count.load(Ordering::SeqCst),
                1,
                "{finish_reason:?} must not re-call the LLM"
            );
            if has_text {
                assert_eq!(result.final_response, "partial answer");
            }
        }
    }

    #[tokio::test]
    async fn paused_reply_resumes_the_turn() {
        let llm = Arc::new(MockLlm::new(vec![
            choice_with(
                vec![ContentBlock::from("working on it")],
                Some(FinishReason::Paused),
            ),
            text_choice("done"),
        ]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let mut messages = vec![Message::user("hi")];

        let result = run_agent_with_history(
            llm.clone(),
            executor,
            &make_config(10),
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.final_response, "done");
        assert_eq!(llm.call_count.load(Ordering::SeqCst), 2);
    }

    /// Streams one chunk, closes its sender, and only then finishes the
    /// request — legal for an `LlmClient`, since nothing requires the sender
    /// to live until the reply is returned.
    struct EarlyCloseLlm;

    #[async_trait]
    impl LlmClient for EarlyCloseLlm {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            unreachable!("the streaming path is under test")
        }

        async fn send_streaming(
            &self,
            _messages: &[Message],
            _tools: &[Tool],
            chunk_tx: mpsc::UnboundedSender<LlmChunk>,
        ) -> Result<Choice> {
            chunk_tx.send(LlmChunk::Text("hel".into())).unwrap();
            drop(chunk_tx);
            // Let the agent loop observe the closed channel before the
            // reply is ready.
            for _ in 0..10 {
                tokio::task::yield_now().await;
            }
            Ok(text_choice("hello"))
        }
    }

    // Regression test: the loop gave up as soon as the chunk channel
    // closed, failing the turn with "stream ended prematurely" even though
    // the reply was still on its way.
    #[tokio::test]
    async fn reply_arriving_after_the_chunk_channel_closes_still_completes() {
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let mut messages = vec![Message::user("hi")];
        let mut chunks = Vec::new();

        let result = run_agent_streaming_with_history(
            Arc::new(EarlyCloseLlm),
            executor,
            &make_config(10),
            &mut messages,
            None,
            &TurnContext {
                cancel: &CancellationToken::new(),
                permission_gate: &allow_all(),
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
            },
            |event| {
                if let StreamEvent::LlmResponse { content } = event {
                    chunks.push(content);
                }
            },
        )
        .await
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.final_response, "hello");
        assert_eq!(chunks, vec!["hel".to_string()]);
    }

    struct HangingPermissionGate;

    #[async_trait]
    impl PermissionGate for HangingPermissionGate {
        async fn check(
            &self,
            _tool_call_id: &str,
            _tool_name: &str,
            _arguments: &str,
        ) -> PermissionDecision {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this wait before the sleep elapses");
        }
    }

    #[tokio::test]
    async fn cancel_aborts_pending_permission_approval() {
        let llm = Arc::new(MockLlm::new(vec![tool_call_choice(
            "execute_command",
            r#"{"command":"echo hi"}"#,
        )]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new("should not run"));
        let config = make_config(10);
        let mut messages = vec![Message::user("do something")];
        let cancel = CancellationToken::new();
        let cancel_signal = cancel.clone();
        let gate: Arc<dyn PermissionGate> = Arc::new(HangingPermissionGate);

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
                    permission_gate: &gate,
                    work_dir: std::path::Path::new("."),
                    client_io: &NoClientIo,
                },
            ),
        )
        .await
        .expect("run_agent_loop should abort the pending permission check instead of hanging")
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::Cancelled);
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
                work_dir: std::path::Path::new("."),
                client_io: &NoClientIo,
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
