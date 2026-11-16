use tokio::sync::mpsc;

use crate::config::AgentConfig;
use crate::core::context::{ContextFitter, Fitted, MAX_OVERFLOW_RETRIES};
use crate::core::llm::{LlmChunk, LlmClient};
use crate::core::models::*;
use crate::core::permission::PermissionRequest;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};
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

/// One LLM request as it's sent.
struct Request {
    /// The system prompt, then the history fitted to the context window.
    messages: Vec<Message>,
    fitted: Fitted,
    /// Estimated size, system prompt and tools included.
    bytes: usize,
}

/// The request for `messages`: unanswered tool calls answered, as fitted by
/// `fitter` (see [`ContextFitter::fit`]), and the system prompt from
/// `prompt_builder` in front.
fn prepare_request(
    messages: &[Message],
    prompt_builder: Option<&PromptBuilder>,
    fitter: &ContextFitter,
) -> Request {
    let repaired = answer_missing_tool_results(messages);
    let mut fitted = fitter.fit(repaired.as_deref().unwrap_or(messages));
    let bytes = fitter.request_bytes(&fitted.messages);
    let messages = match prompt_builder {
        Some(builder) => builder.build(&fitted.messages),
        None => std::mem::take(&mut fitted.messages),
    };
    Request {
        messages,
        fitted,
        bytes,
    }
}

/// The error for a request that stays too long with every earlier turn,
/// and the outputs of the current turn's older tool results, left out.
fn turn_too_long(provider_message: &str) -> Error {
    Error::ContextOverflow(format!(
        "the current turn doesn't fit in the model's context window even with \
         earlier turns and older tool outputs left out; start a new session \
         (provider said: {provider_message})"
    ))
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
/// A request that doesn't fit the context window leaves out the oldest whole
/// turns, never the one in progress; if that alone is still too long, the
/// outputs of its older tool results are replaced with a placeholder.
/// `messages` itself keeps everything. That happens before sending when
/// `config.context_window` is set and the request is estimated at over 90%
/// of it, and after the provider rejects a request as too long (up to two
/// retries per call). Each change in how many messages, or tool outputs,
/// are left out is reported as [`StreamEvent::ContextTrimmed`] (or
/// [`StreamEvent::ToolOutputsOmitted`]).
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
/// why the turn ended. [`Error::ContextOverflow`] means the turn in progress
/// is too long for the model even with those outputs left out.
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
    let system = prompt_builder.map(|b| b.build(&[])).unwrap_or_default();
    let mut fitter = ContextFitter::new(config.context_window, &system, &tools);
    // The last `ContextTrimmed` and `ToolOutputsOmitted` counts sent, so
    // each goes out only on a change.
    let mut reported_dropped = 0;
    let mut reported_omitted = 0;
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

        // A request rejected as too long is rebuilt with more left out and
        // sent again, a bounded number of times.
        let mut overflow_retries = 0;
        let choice = loop {
            let request = prepare_request(messages, prompt_builder, &fitter);
            if request.fitted.dropped > 0 && request.fitted.dropped != reported_dropped {
                reported_dropped = request.fitted.dropped;
                callback(StreamEvent::ContextTrimmed {
                    dropped: reported_dropped,
                });
            }
            // Back to 0 too: a recalibrated estimate can fit them all again.
            let omitted = request.fitted.omitted_outputs;
            if omitted != reported_omitted {
                reported_omitted = omitted;
                callback(StreamEvent::ToolOutputsOmitted { omitted });
            }

            let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<LlmChunk>();
            // Every LLM call streams, even when nobody watches the chunks: the
            // HTTP timeout is per read (see `build_http_client`), so only a
            // streamed reply keeps a long generation from timing out.
            let choice_fut = llm.send_streaming(&request.messages, &tools, chunk_tx);
            tokio::pin!(choice_fut);

            // The reply ends the wait, not the chunk channel, which a client
            // may close early or keep open.
            let mut chunks_open = true;
            let result = loop {
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
            };

            match result {
                Ok(choice) => {
                    if let Some(usage) = &choice.usage {
                        fitter.record_usage(request.bytes, usage);
                    }
                    break choice;
                }
                Err(Error::ContextOverflow(provider_message)) => {
                    if overflow_retries == MAX_OVERFLOW_RETRIES
                        || !fitter.record_overflow(&request.fitted, request.bytes)
                    {
                        return Err(turn_too_long(&provider_message));
                    }
                    overflow_retries += 1;
                    tracing::warn!(
                        dropped = request.fitted.dropped,
                        omitted_outputs = request.fitted.omitted_outputs,
                        "request exceeded the context window; retrying with less of the history"
                    );
                }
                Err(e) => return Err(e),
            }
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
    use crate::core::permission::{AllowAll, PermissionGate};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    async fn run(
        llm: &dyn LlmClient,
        executor: &dyn ToolExecutor,
        messages: &mut Vec<Message>,
        callback: impl FnMut(StreamEvent) + Send,
    ) -> AgentResult {
        let gate: Arc<dyn PermissionGate> = Arc::new(AllowAll);
        let turn = TurnContext {
            cancel: &CancellationToken::new(),
            permission_gate: &gate,
            work_dir: std::path::Path::new("."),
            cwd: std::path::Path::new("."),
            client_io: &NoClientIo,
        };
        let config = AgentConfig {
            max_iterations: 10,
            ..AgentConfig::default()
        };
        run_agent(llm, executor, &config, messages, None, &turn, callback)
            .await
            .unwrap()
    }

    fn text_choice(content: &str) -> Choice {
        Choice {
            message: Message::assistant(content),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        }
    }

    struct ScriptedLlm(Mutex<Vec<Choice>>);

    #[async_trait]
    impl LlmClient for ScriptedLlm {
        async fn send(&self, _messages: &[Message], _tools: &[Tool]) -> Result<Choice> {
            Ok(self.0.lock().unwrap().remove(0))
        }
    }

    /// Each tool sleeps for its delay, then returns `<name>-done`.
    struct DelayedTools(std::collections::HashMap<&'static str, std::time::Duration>);

    #[async_trait]
    impl ToolExecutor for DelayedTools {
        fn list_tools(&self) -> Vec<Tool> {
            vec![]
        }
        async fn execute(&self, name: &str, _: &str, _: &TurnContext<'_>) -> Result<String> {
            if let Some(delay) = self.0.get(name) {
                tokio::time::sleep(*delay).await;
            }
            Ok(format!("{name}-done"))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn tool_results_emit_in_completion_order_not_call_order() {
        let calls = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::tool_use("c0", "slow", "{}"),
                ContentBlock::tool_use("c1", "fast", "{}"),
            ],
        };
        let llm = ScriptedLlm(Mutex::new(vec![
            Choice {
                message: calls,
                finish_reason: Some(FinishReason::ToolCalls),
                usage: None,
            },
            text_choice("done"),
        ]));
        let executor = DelayedTools(std::collections::HashMap::from([
            ("slow", std::time::Duration::from_secs(10)),
            ("fast", std::time::Duration::from_millis(1)),
        ]));
        let mut messages = vec![Message::user("go")];
        let mut emitted = Vec::new();

        run(&llm, &executor, &mut messages, |event| {
            if let StreamEvent::ToolResult { tool_name, .. } = event {
                emitted.push(tool_name);
            }
        })
        .await;

        assert_eq!(emitted, ["fast", "slow"]);
        let stored: Vec<_> = messages
            .iter()
            .filter_map(|m| m.tool_result_block().map(|tr| tr.tool_name.clone()))
            .collect();
        assert_eq!(stored, ["slow", "fast"]);
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
        let executor = DelayedTools(Default::default());

        let result = run(&EarlyCloseLlm, &executor, &mut messages, |event| {
            if let StreamEvent::LlmResponse { content } = event {
                chunks.push(content);
            }
        })
        .await;

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.final_response, "hello");
        assert_eq!(chunks, ["hel"]);
    }
}
