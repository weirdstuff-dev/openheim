use tokio::sync::mpsc;

use crate::config::AgentConfig;
use crate::core::llm::{LlmChunk, LlmClient};
use crate::core::models::*;
use crate::core::permission::PermissionRequest;
use crate::core::turn::TurnContext;
use crate::error::Result;
use crate::memory::PromptBuilder;
use crate::tools::ToolExecutor;

/// The result recorded for a tool call whose turn was cancelled before it
/// finished.
const CANCELLED_TOOL_RESULT: &str = "Cancelled by user.";

/// The result sent in place of one missing from the history, e.g. because
/// the process stopped mid-turn.
const MISSING_TOOL_RESULT: &str =
    "No result: the turn was interrupted before this tool call finished.";

/// Longest tool result kept, in bytes. A larger one is cut here, whatever
/// the tool: it goes into the history and is resent with every later
/// request, so one oversized result would push every request in the
/// conversation past the model's context window.
const MAX_TOOL_RESULT_BYTES: usize = 128 * 1024;

/// `result` cut to [`MAX_TOOL_RESULT_BYTES`] (at a character boundary), with
/// a note saying how much was left out.
fn cap_tool_result(mut result: String) -> String {
    if result.len() <= MAX_TOOL_RESULT_BYTES {
        return result;
    }
    let total = result.len();
    let mut end = MAX_TOOL_RESULT_BYTES;
    while !result.is_char_boundary(end) {
        end -= 1;
    }
    result.truncate(end);
    result.push_str(&format!(
        "\n[tool output truncated: showing the first {end} of {total} bytes]"
    ));
    result
}

/// Whether a tool call's arguments can be run: a JSON object, or nothing at
/// all for a call without arguments.
fn arguments_are_usable(arguments: &str) -> bool {
    arguments.trim().is_empty()
        || matches!(
            serde_json::from_str(arguments),
            Ok(serde_json::Value::Object(_))
        )
}

/// The result recorded for a call whose arguments aren't a JSON object.
/// Such a call is neither run nor put to the permission gate.
fn malformed_arguments_result(cut_off: bool) -> String {
    let mut result =
        "Error: the arguments of this call are not a valid JSON object, so it was not run."
            .to_string();
    if cut_off {
        result.push_str(
            " The reply reached the output token limit before the call was complete. \
             Make the call again with smaller arguments, splitting the work across several calls.",
        );
    }
    result
}

/// `messages` with an error result added for every tool call that has none,
/// right after the results that did arrive, or `None` if every call is
/// answered. Providers reject a history with an unanswered tool call, so
/// without this one interrupted turn would break every later request in the
/// conversation. Applied to each request only; the stored history is left as
/// it is.
fn answer_missing_tool_results(messages: &[Message]) -> Option<Vec<Message>> {
    // (index to insert before, results to insert), in ascending index order.
    let mut insertions: Vec<(usize, Vec<Message>)> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let calls = match messages[i].role {
            Role::Assistant => messages[i].tool_calls(),
            _ => Vec::new(),
        };
        i += 1;
        if calls.is_empty() {
            continue;
        }
        let mut answered = std::collections::HashSet::new();
        while let Some(result) = messages.get(i).and_then(Message::tool_result_block) {
            answered.insert(result.tool_call_id);
            i += 1;
        }
        let missing: Vec<Message> = calls
            .into_iter()
            .filter(|call| !answered.contains(&call.id))
            .map(|call| Message::tool_result(call.id, call.name, MISSING_TOOL_RESULT, true))
            .collect();
        if !missing.is_empty() {
            insertions.push((i, missing));
        }
    }
    if insertions.is_empty() {
        return None;
    }

    let mut repaired = Vec::with_capacity(messages.len() + insertions.len());
    let mut copied = 0;
    for (at, missing) in insertions {
        repaired.extend_from_slice(&messages[copied..at]);
        repaired.extend(missing);
        copied = at;
    }
    repaired.extend_from_slice(&messages[copied..]);
    Some(repaired)
}

/// Every LLM call streams, even when nobody watches the chunks: the HTTP
/// timeout is per read (see `build_http_client`), so only a streamed reply
/// keeps a long generation from timing out.
async fn call_llm_streaming(
    llm: &dyn LlmClient,
    messages: &[Message],
    tools: &[Tool],
    prompt_builder: Option<&PromptBuilder>,
    chunk_tx: mpsc::UnboundedSender<LlmChunk>,
) -> Result<Choice> {
    let repaired = answer_missing_tool_results(messages);
    let messages = repaired.as_deref().unwrap_or(messages);
    match prompt_builder {
        Some(builder) => {
            let built = builder.build(messages);
            llm.send_streaming(&built, tools, chunk_tx).await
        }
        None => llm.send_streaming(messages, tools, chunk_tx).await,
    }
}

/// Passes one streamed chunk to the caller's callback as the matching event.
fn forward_chunk<F: FnMut(StreamEvent)>(callback: &mut F, chunk: LlmChunk) {
    match chunk {
        LlmChunk::Text(text) => callback(StreamEvent::LlmResponse { content: text }),
        LlmChunk::Thinking(thought) => callback(StreamEvent::ThinkingContent { content: thought }),
    }
}

/// How a turn ends when the LLM replies without any tool calls, from the
/// provider's finish reason and whether the reply had text. `None` means
/// "keep looping": the provider paused the turn and expects the
/// conversation resent as-is to resume it.
///
/// Any other finish reason (`Other`, none, or `ToolCalls` without calls)
/// ends the turn: resending a history that ends in an assistant reply would
/// make the model repeat itself, or be rejected.
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

/// Runs one agent turn: repeatedly calls the LLM and executes the tool calls
/// it asks for until it replies without any (see `stop_for_reply_without_tools`
/// for how that reply's finish reason maps to a [`StopReason`]) or
/// `config.max_iterations` is reached.
///
/// Appends all assistant and tool-result messages to `messages` in place;
/// persisting them is the caller's job. `prompt_builder`, if set, adds the
/// system prompt and skills to each request.
///
/// `callback` receives a [`StreamEvent`] for each step: iteration start,
/// streamed text and thinking, tool calls, tool results, and completion. It
/// runs on the loop's own task and must not block; pass `|_| {}` to ignore
/// events.
///
/// `turn.cancel` stops the turn at any point: between iterations, during an
/// LLM call, while waiting for permission, or while tools run. Every tool
/// call the LLM asked for is still answered in `messages` (cancelled ones
/// with an error result), so the history stays valid to send.
///
/// Returns `Ok` unless an LLM call fails; [`AgentResult::stop_reason`] says
/// why the turn ended.
pub async fn run_agent<F>(
    llm: &dyn LlmClient,
    tool_executor: &dyn ToolExecutor,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
    turn: &TurnContext<'_>,
    mut callback: F,
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

        callback(StreamEvent::IterationStart {
            iteration: iter_num,
        });

        // Scoped so the in-flight request's borrow of `messages` ends before
        // the reply is pushed onto it.
        let choice = {
            let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<LlmChunk>();
            let choice_fut = call_llm_streaming(llm, messages, &tools, prompt_builder, chunk_tx);
            tokio::pin!(choice_fut);

            // The reply ends the wait, not the chunk channel, which a client
            // may close early or keep open.
            let mut chunks_open = true;
            loop {
                tokio::select! {
                    _ = turn.cancel.cancelled() => {
                        // Dropping `choice_fut` here aborts the in-flight LLM
                        // request rather than waiting for it to finish.
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
        };
        messages.push(choice.message.clone());
        callback(StreamEvent::MessageAppended {
            message: choice.message.clone(),
        });
        if let Some(usage) = choice.usage {
            // Overwritten, not summed: each call resends the whole history,
            // so the latest call's usage is the current context size.
            context_usage = Some(usage);
            callback(StreamEvent::Usage { usage });
        }

        let tool_calls = choice.message.tool_calls();
        if !tool_calls.is_empty() {
            // Announce every call before any permission check runs, so a UI
            // shows all of them as pending, not just the one being asked about.
            for tool_call in &tool_calls {
                callback(StreamEvent::ToolCall {
                    id: tool_call.id.clone(),
                    tool_name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                });
            }

            // The assistant message asking for these calls is already in
            // `messages`, and providers reject a history with an unanswered
            // tool call. So cancellation leaves this block, not the turn, and
            // every call without an outcome is answered as cancelled below.
            let mut outcomes: Vec<Option<(String, bool)>> = vec![None; tool_calls.len()];
            let cut_off = choice.finish_reason == Some(FinishReason::MaxTokens);
            'calls: {
                if turn.cancel.is_cancelled() {
                    break 'calls;
                }

                // Ask about every call concurrently, so a gate sees all the
                // requests at once. A call with malformed arguments gets no
                // decision: it can't run.
                let decisions = tokio::select! {
                    // Dropping the `join_all` abandons every pending approval.
                    _ = turn.cancel.cancelled() => break 'calls,
                    decisions = futures::future::join_all(tool_calls.iter().map(|tool_call| async {
                        if !arguments_are_usable(&tool_call.arguments) {
                            return None;
                        }
                        let request = PermissionRequest::new(
                            &tool_call.id,
                            &tool_call.name,
                            &tool_call.arguments,
                        );
                        Some(turn.permission_gate.check(&request).await)
                    })) => decisions,
                };

                // Run the allowed calls concurrently. `ToolResult` goes out as
                // each finishes; the history below keeps the calls' order.
                let mut pending: futures::stream::FuturesUnordered<_> = tool_calls
                    .iter()
                    .zip(&decisions)
                    .enumerate()
                    .map(|(index, (tool_call, decision))| async move {
                        let (result, is_error) = match decision {
                            None => (malformed_arguments_result(cut_off), true),
                            Some(decision) if decision.is_allowed() => {
                                match tool_executor
                                    .execute(&tool_call.name, &tool_call.arguments, turn)
                                    .await
                                {
                                    Ok(r) => (r, false),
                                    Err(e) => (format!("Error: {e}"), true),
                                }
                            }
                            Some(_) => ("Permission denied by user.".to_string(), true),
                        };
                        (index, cap_tool_result(result), is_error)
                    })
                    .collect();

                loop {
                    tokio::select! {
                        // Dropping `pending` abandons every in-flight tool
                        // call at once; calls that already finished keep
                        // their results.
                        _ = turn.cancel.cancelled() => break 'calls,
                        next = futures::StreamExt::next(&mut pending) => {
                            let Some((index, result, is_error)) = next else {
                                break 'calls;
                            };
                            let tool_call = &tool_calls[index];
                            callback(StreamEvent::ToolResult {
                                id: tool_call.id.clone(),
                                tool_name: tool_call.name.clone(),
                                result: result.clone(),
                                is_error,
                            });
                            outcomes[index] = Some((result, is_error));
                        }
                    }
                }
            }

            // Record results in the calls' original order, whichever finished
            // first. A call with no outcome was cancelled; it gets its
            // `ToolResult` here, so a UI showing it as pending sees it end.
            for (tool_call, outcome) in tool_calls.iter().zip(outcomes) {
                let (result, is_error) = outcome.unwrap_or_else(|| {
                    callback(StreamEvent::ToolResult {
                        id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        result: CANCELLED_TOOL_RESULT.to_string(),
                        is_error: true,
                    });
                    (CANCELLED_TOOL_RESULT.to_string(), true)
                });

                let tool_result_message = Message::tool_result(
                    tool_call.id.clone(),
                    tool_call.name.clone(),
                    result,
                    is_error,
                );
                callback(StreamEvent::MessageAppended {
                    message: tool_result_message.clone(),
                });
                messages.push(tool_result_message);
            }

            // Checked here too: on the last iteration the top-of-loop check
            // never runs, and the turn would be reported as `MaxIterations`.
            if turn.cancel.is_cancelled() {
                stop_reason = StopReason::Cancelled;
                break 'turn;
            }
        } else {
            // The text was already streamed above; just keep the final reply.
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

    callback(StreamEvent::Finished {
        final_response: final_response.clone(),
        iterations: iterations_used,
    });

    Ok(AgentResult {
        final_response,
        iterations_used,
        stop_reason,
        context_usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::client_io::NoClientIo;
    use crate::core::permission::{AllowAll, PermissionDecision, PermissionGate};
    use crate::error::Error;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    /// What a test turn borrows, with the defaults most tests want: ten
    /// iterations, a fresh cancel token, and a gate that allows everything.
    struct Harness {
        config: AgentConfig,
        cancel: CancellationToken,
        gate: Arc<dyn PermissionGate>,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                config: AgentConfig {
                    max_iterations: 10,
                    ..AgentConfig::default()
                },
                cancel: CancellationToken::new(),
                gate: Arc::new(AllowAll),
            }
        }

        fn max_iterations(mut self, max_iterations: usize) -> Self {
            self.config.max_iterations = max_iterations;
            self
        }

        fn gate(mut self, gate: impl PermissionGate + 'static) -> Self {
            self.gate = Arc::new(gate);
            self
        }

        /// A clone of the turn's cancel token, for cancelling it from a
        /// callback or another task.
        fn cancel_handle(&self) -> CancellationToken {
            self.cancel.clone()
        }

        async fn run(
            &self,
            llm: &dyn LlmClient,
            executor: &dyn ToolExecutor,
            messages: &mut Vec<Message>,
            callback: impl FnMut(StreamEvent) + Send,
        ) -> Result<AgentResult> {
            let turn = TurnContext {
                cancel: &self.cancel,
                permission_gate: &self.gate,
                work_dir: std::path::Path::new("."),
                cwd: std::path::Path::new("."),
                client_io: &NoClientIo,
            };
            run_agent(llm, executor, &self.config, messages, None, &turn, callback).await
        }
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
                content: vec![ContentBlock::tool_use("call_1", tool_name, args)],
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
                    .map(|(i, (name, args))| {
                        ContentBlock::tool_use(format!("call_{i}"), *name, *args)
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
        let llm = MockLlm::new(vec![text_choice("done")]);
        let mut messages = vec![Message::user("hi")];

        let result = Harness::new()
            .run(&llm, &MockToolExecutor::new(""), &mut messages, |_| {})
            .await
            .unwrap();

        assert_eq!(result.final_response, "done");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn agent_executes_tool_calls_and_continues() {
        let llm = MockLlm::new(vec![
            tool_call_choice("read_file", r#"{"path":"a.txt"}"#),
            text_choice("here is the file content"),
        ]);
        let executor = MockToolExecutor::new("file data");
        let mut messages = vec![Message::user("read a.txt")];

        let result = Harness::new()
            .run(&llm, &executor, &mut messages, |_| {})
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
        let llm = MockLlm::new(vec![
            multi_tool_call_choice(&[("slow", "{}"), ("fast", "{}")]),
            text_choice("done"),
        ]);
        let delays = std::collections::HashMap::from([
            ("slow".to_string(), std::time::Duration::from_secs(10)),
            ("fast".to_string(), std::time::Duration::from_millis(1)),
        ]);
        let executor = DelayedToolExecutor { delays };
        let mut messages = vec![Message::user("go")];
        let mut completion_order = Vec::new();

        Harness::new()
            .run(&llm, &executor, &mut messages, |event| {
                if let StreamEvent::ToolResult { tool_name, .. } = event {
                    completion_order.push(tool_name);
                }
            })
            .await
            .unwrap();

        assert_eq!(completion_order, vec!["fast", "slow"]);
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
        let llm = MockLlm::new(vec![
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
        ]);
        let mut messages = vec![Message::user("loop")];

        let result = Harness::new()
            .max_iterations(3)
            .run(&llm, &MockToolExecutor::new("data"), &mut messages, |_| {})
            .await
            .unwrap();

        assert_eq!(result.iterations_used, 3);
        assert_eq!(result.stop_reason, StopReason::MaxIterations);
    }

    #[tokio::test]
    async fn agent_streaming_emits_events() {
        let llm = MockLlm::new(vec![
            tool_call_choice("echo", r#"{"cmd":"hi"}"#),
            text_choice("all done"),
        ]);
        let mut messages = vec![Message::user("test")];

        let mut events = Vec::new();
        let result = Harness::new()
            .run(&llm, &MockToolExecutor::new("ok"), &mut messages, |event| {
                events.push(event)
            })
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
        let llm = MockLlm::new(vec![
            tool_call_choice("bad_tool", "{}"),
            text_choice("I got an error"),
        ]);
        let mut messages = vec![Message::user("do something")];

        // Should not propagate the error; LLM should receive it as a tool result
        let result = Harness::new()
            .run(&llm, &FailingToolExecutor, &mut messages, |_| {})
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
        let llm = MockLlm::new(vec![
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
            tool_call_choice("read_file", "{}"),
        ]);
        let executor = MockToolExecutor::new("data");
        let mut messages = vec![Message::user("loop")];
        let harness = Harness::new();
        let cancel = harness.cancel_handle();

        let result = harness
            .run(&llm, &executor, &mut messages, move |event| {
                if matches!(event, StreamEvent::ToolResult { .. }) {
                    cancel.cancel();
                }
            })
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
        let executor = MockToolExecutor::new("");
        let mut messages = vec![Message::user("hi")];
        let harness = Harness::new();
        let cancel = harness.cancel_handle();

        // Cancel as soon as the loop starts its first iteration, i.e. right
        // before the (never-resolving) LLM call is made.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            harness.run(&SlowLlm, &executor, &mut messages, move |event| {
                if matches!(event, StreamEvent::IterationStart { .. }) {
                    cancel.cancel();
                }
            }),
        )
        .await
        .expect("run_agent should abort the in-flight LLM call instead of hanging")
        .unwrap();

        assert_eq!(result.final_response, "");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(result.stop_reason, StopReason::Cancelled);
    }

    /// ToolExecutor that never resolves; proves cancellation aborts a
    /// running tool call instead of waiting for it to finish.
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
        let llm = MockLlm::new(vec![tool_call_choice("read_file", "{}")]);
        let mut messages = vec![Message::user("hi")];
        let harness = Harness::new();
        let cancel = harness.cancel_handle();

        // Cancel as soon as the tool call is announced, just before it runs.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            harness.run(&llm, &SlowToolExecutor, &mut messages, move |event| {
                if matches!(event, StreamEvent::ToolCall { .. }) {
                    cancel.cancel();
                }
            }),
        )
        .await
        .expect("run_agent should abort the in-flight tool call instead of hanging")
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert_every_call_answered_as_cancelled(&messages);
    }

    /// The history ends with the assistant's tool call followed by a
    /// cancelled error result for it, so the next request is still valid.
    fn assert_every_call_answered_as_cancelled(messages: &[Message]) {
        let [.., call, result] = messages else {
            panic!("expected a tool call and its result, got {messages:?}");
        };
        assert_eq!(call.tool_calls().len(), 1);
        let result = result.tool_result_block().expect("a tool result");
        assert_eq!(result.tool_call_id, call.tool_calls()[0].id);
        assert_eq!(result.content, CANCELLED_TOOL_RESULT);
        assert!(result.is_error);
    }

    // Cancelling after one of two calls finished must leave every call
    // answered: providers reject a history with an unanswered tool call.
    #[tokio::test(start_paused = true)]
    async fn cancel_keeps_finished_results_and_answers_the_rest() {
        let llm = MockLlm::new(vec![multi_tool_call_choice(&[
            ("slow", "{}"),
            ("fast", "{}"),
        ])]);
        let delays = std::collections::HashMap::from([
            ("slow".to_string(), std::time::Duration::from_secs(3600)),
            ("fast".to_string(), std::time::Duration::from_millis(1)),
        ]);
        let executor = DelayedToolExecutor { delays };
        let mut messages = vec![Message::user("go")];
        let harness = Harness::new();
        let cancel = harness.cancel_handle();
        let mut results = Vec::new();
        let mut appended = 0;

        let result = harness
            .run(&llm, &executor, &mut messages, |event| match event {
                StreamEvent::ToolResult {
                    tool_name, result, ..
                } => {
                    if tool_name == "fast" {
                        cancel.cancel();
                    }
                    results.push((tool_name, result));
                }
                StreamEvent::MessageAppended { .. } => appended += 1,
                _ => {}
            })
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Cancelled);
        // Every call's `ToolResult` fires, the cancelled one included.
        assert_eq!(
            results,
            [
                ("fast".to_string(), "fast-done".to_string()),
                ("slow".to_string(), CANCELLED_TOOL_RESULT.to_string()),
            ]
        );
        let recorded: Vec<_> = messages
            .iter()
            .filter_map(Message::tool_result_block)
            .map(|r| (r.tool_name, r.content, r.is_error))
            .collect();
        assert_eq!(
            recorded,
            [
                ("slow".to_string(), CANCELLED_TOOL_RESULT.to_string(), true),
                ("fast".to_string(), "fast-done".to_string(), false),
            ]
        );
        // The assistant message plus both results reach the history log.
        assert_eq!(appended, 3);
    }

    fn calls(ids: &[&str]) -> Message {
        Message {
            role: Role::Assistant,
            content: ids
                .iter()
                .map(|id| ContentBlock::tool_use(*id, "read_file", "{}"))
                .collect(),
        }
    }

    fn answer(id: &str) -> Message {
        Message::tool_result(id, "read_file", "ok", false)
    }

    #[test]
    fn complete_history_is_sent_unchanged() {
        let messages = vec![
            Message::user("hi"),
            calls(&["a", "b"]),
            answer("a"),
            answer("b"),
            Message::assistant("done"),
        ];
        assert!(answer_missing_tool_results(&messages).is_none());
    }

    #[test]
    fn missing_results_are_answered_after_the_ones_that_arrived() {
        let messages = vec![
            Message::user("hi"),
            calls(&["a", "b"]),
            answer("a"),
            Message::user("still there?"),
            calls(&["c"]),
        ];
        let repaired = answer_missing_tool_results(&messages).unwrap();

        let missing = |id: &str| Message::tool_result(id, "read_file", MISSING_TOOL_RESULT, true);
        assert_eq!(
            repaired,
            [
                Message::user("hi"),
                calls(&["a", "b"]),
                answer("a"),
                missing("b"),
                Message::user("still there?"),
                calls(&["c"]),
                missing("c"),
            ]
        );
    }

    // A stored turn with an unanswered call (e.g. cut short by a crash) must
    // not break the conversation's next request.
    #[tokio::test]
    async fn request_answers_a_stored_unanswered_call() {
        struct RecordingLlm(Mutex<Vec<Vec<Message>>>);

        #[async_trait]
        impl LlmClient for RecordingLlm {
            async fn send(&self, messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
                self.0.lock().unwrap().push(messages.to_vec());
                Ok(text_choice("ok"))
            }
        }

        let llm = RecordingLlm(Mutex::new(Vec::new()));
        let mut messages = vec![Message::user("hi"), calls(&["a"]), Message::user("again")];

        Harness::new()
            .run(&llm, &MockToolExecutor::new(""), &mut messages, |_| {})
            .await
            .unwrap();

        let sent = &llm.0.lock().unwrap()[0];
        assert_eq!(
            sent[2],
            Message::tool_result("a", "read_file", MISSING_TOOL_RESULT, true)
        );
        // The stored history itself isn't rewritten.
        assert_eq!(messages[2], Message::user("again"));
    }

    #[tokio::test]
    async fn cancel_aborts_in_flight_llm_call_without_a_callback() {
        let executor = MockToolExecutor::new("");
        let mut messages = vec![Message::user("hi")];
        let harness = Harness::new();
        let cancel = harness.cancel_handle();

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            harness.run(&SlowLlm, &executor, &mut messages, |_| {}),
        )
        .await
        .expect("run_agent should abort the in-flight LLM call instead of hanging")
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
        let llm = MockLlm::new(vec![empty_choice]);
        let mut messages = vec![Message::user("hi")];

        let result = Harness::new()
            .run(&llm, &MockToolExecutor::new(""), &mut messages, |_| {})
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

    // Every finish reason on a reply without tool calls ends the turn
    // (except `Paused`), instead of resending a history that ends in an
    // assistant reply.
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
            let llm = MockLlm::new(vec![choice_with(content, finish_reason.clone())]);
            let mut messages = vec![Message::user("hi")];

            let result = Harness::new()
                .run(&llm, &MockToolExecutor::new(""), &mut messages, |_| {})
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
        let llm = MockLlm::new(vec![
            choice_with(
                vec![ContentBlock::from("working on it")],
                Some(FinishReason::Paused),
            ),
            text_choice("done"),
        ]);
        let mut messages = vec![Message::user("hi")];

        let result = Harness::new()
            .run(&llm, &MockToolExecutor::new(""), &mut messages, |_| {})
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

    // The loop waits for the reply, not the chunk channel: a client may
    // close its sender while the reply is still on its way.
    #[tokio::test]
    async fn reply_arriving_after_the_chunk_channel_closes_still_completes() {
        let mut messages = vec![Message::user("hi")];
        let mut chunks = Vec::new();

        let result = Harness::new()
            .run(
                &EarlyCloseLlm,
                &MockToolExecutor::new(""),
                &mut messages,
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

    // A run whose callback ignores events (as `delegate_task` subagents do)
    // still streams, so long replies don't hit the per-read timeout.
    // `EarlyCloseLlm::send` panics, so this only passes if the request
    // streamed.
    #[tokio::test]
    async fn runs_without_a_callback_still_stream() {
        let mut messages = vec![Message::user("hi")];

        let result = Harness::new()
            .run(
                &EarlyCloseLlm,
                &MockToolExecutor::new(""),
                &mut messages,
                |_| {},
            )
            .await
            .unwrap();

        assert_eq!(result.final_response, "hello");
    }

    struct HangingPermissionGate;

    #[async_trait]
    impl PermissionGate for HangingPermissionGate {
        async fn check(&self, _request: &PermissionRequest<'_>) -> PermissionDecision {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            unreachable!("cancellation should abort this wait before the sleep elapses");
        }
    }

    #[tokio::test]
    async fn cancel_aborts_pending_permission_approval() {
        let llm = MockLlm::new(vec![tool_call_choice(
            "execute_command",
            r#"{"command":"echo hi"}"#,
        )]);
        let executor = MockToolExecutor::new("should not run");
        let mut messages = vec![Message::user("do something")];
        let harness = Harness::new().gate(HangingPermissionGate);
        let cancel = harness.cancel_handle();

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            harness.run(&llm, &executor, &mut messages, |_| {}),
        )
        .await
        .expect("run_agent should abort the pending permission check instead of hanging")
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert_every_call_answered_as_cancelled(&messages);
    }

    struct RejectPermissionGate;

    #[async_trait]
    impl PermissionGate for RejectPermissionGate {
        async fn check(&self, _request: &PermissionRequest<'_>) -> PermissionDecision {
            PermissionDecision::RejectOnce
        }
    }

    #[tokio::test]
    async fn agent_skips_execution_when_permission_denied() {
        let llm = MockLlm::new(vec![
            tool_call_choice("execute_command", r#"{"command":"rm -rf /"}"#),
            text_choice("I was denied"),
        ]);
        let executor = MockToolExecutor::new("should not run");
        let mut messages = vec![Message::user("do something dangerous")];

        let result = Harness::new()
            .gate(RejectPermissionGate)
            .run(&llm, &executor, &mut messages, |_| {})
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

    /// Counts the permission checks that reach it, allowing each.
    struct CountingGate(Arc<AtomicUsize>);

    #[async_trait]
    impl PermissionGate for CountingGate {
        async fn check(&self, _request: &PermissionRequest<'_>) -> PermissionDecision {
            self.0.fetch_add(1, Ordering::SeqCst);
            PermissionDecision::AllowOnce
        }
    }

    // A call cut off at the output limit is answered with an error the model
    // can act on, without being run or put to the user.
    #[tokio::test]
    async fn call_with_malformed_arguments_is_answered_without_running() {
        let mut cut_off = tool_call_choice("write_file", r#"{"path":"a.txt","content":"par"#);
        cut_off.finish_reason = Some(FinishReason::MaxTokens);
        let llm = MockLlm::new(vec![cut_off, text_choice("retrying")]);
        let executor = MockToolExecutor::new("should not run");
        let asked = Arc::new(AtomicUsize::new(0));
        let mut messages = vec![Message::user("write a.txt")];

        let result = Harness::new()
            .gate(CountingGate(asked.clone()))
            .run(&llm, &executor, &mut messages, |_| {})
            .await
            .unwrap();

        assert_eq!(result.final_response, "retrying");
        assert!(executor.calls.lock().unwrap().is_empty());
        assert_eq!(asked.load(Ordering::SeqCst), 0);
        let answer = messages
            .iter()
            .find_map(Message::tool_result_block)
            .unwrap();
        assert!(answer.is_error);
        assert!(
            answer.content.contains("output token limit"),
            "{}",
            answer.content
        );
    }

    #[tokio::test]
    async fn call_without_arguments_still_runs() {
        let llm = MockLlm::new(vec![tool_call_choice("list_dir", ""), text_choice("done")]);
        let executor = MockToolExecutor::new("a.txt");
        let mut messages = vec![Message::user("list")];

        Harness::new()
            .run(&llm, &executor, &mut messages, |_| {})
            .await
            .unwrap();

        assert_eq!(executor.calls.lock().unwrap().len(), 1);
    }

    // An oversized result is cut before it reaches the history, so it can't
    // push every later request past the context window.
    #[tokio::test]
    async fn oversized_tool_result_is_truncated() {
        let llm = MockLlm::new(vec![
            tool_call_choice("read_file", r#"{"path":"big.txt"}"#),
            text_choice("done"),
        ]);
        let executor = MockToolExecutor::new(&"é".repeat(MAX_TOOL_RESULT_BYTES));
        let mut messages = vec![Message::user("read it")];
        let mut streamed = String::new();

        Harness::new()
            .run(&llm, &executor, &mut messages, |event| {
                if let StreamEvent::ToolResult { result, .. } = event {
                    streamed = result;
                }
            })
            .await
            .unwrap();

        let stored = messages
            .iter()
            .find_map(Message::tool_result_block)
            .unwrap()
            .content;
        assert_eq!(stored, streamed);
        let (kept, note) = stored.split_once("\n[tool output truncated").unwrap();
        assert_eq!(kept, "é".repeat(MAX_TOOL_RESULT_BYTES / 2));
        assert!(
            note.contains(&format!("of {} bytes", 2 * MAX_TOOL_RESULT_BYTES)),
            "{note}"
        );
    }

    #[test]
    fn small_tool_results_are_kept_whole() {
        assert_eq!(cap_tool_result("ok".to_string()), "ok");
        let exact = "a".repeat(MAX_TOOL_RESULT_BYTES);
        assert_eq!(cap_tool_result(exact.clone()), exact);
    }
}
