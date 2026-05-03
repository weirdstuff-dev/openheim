use std::sync::Arc;

use crate::config::AgentConfig;
use crate::core::llm::LlmClient;
use crate::core::models::*;
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

async fn run_agent_loop<F>(
    llm: &Arc<dyn LlmClient>,
    tool_executor: &Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
    verbose: bool,
    mut callback: Option<F>,
) -> Result<AgentResult>
where
    F: FnMut(StreamEvent) + Send,
{
    let tools = tool_executor.list_tools();
    let mut steps = Vec::new();
    let mut final_response = String::new();

    if verbose {
        println!("🤖 Continuing conversation...\n");
    }

    for iteration in 0..config.max_iterations {
        let iter_num = iteration + 1;

        if verbose {
            println!("--- Iteration {} ---", iter_num);
        }
        if let Some(cb) = callback.as_mut() {
            cb(StreamEvent::IterationStart { iteration: iter_num });
        }

        let choice = call_llm(llm, messages, &tools, prompt_builder).await?;
        messages.push(choice.message.clone());

        if let Some(tool_calls) = &choice.message.tool_calls {
            if verbose {
                println!("🛠️  LLM wants to call {} tool(s)", tool_calls.len());
            }

            let mut tool_results = Vec::new();

            for tool_call in tool_calls {
                let tool_name = &tool_call.function.name;
                let arguments = &tool_call.function.arguments;

                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::ToolCall {
                        tool_name: tool_name.clone(),
                        arguments: arguments.clone(),
                    });
                }

                let result = tool_executor.execute(tool_name, arguments).await?;

                if verbose {
                    println!("✅ Tool {}: {}\n", tool_name, result);
                }
                if let Some(cb) = callback.as_mut() {
                    cb(StreamEvent::ToolResult {
                        tool_name: tool_name.clone(),
                        result: result.clone(),
                    });
                }

                tool_results.push(ToolExecutionResult {
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                    result: result.clone(),
                });

                messages.push(Message::tool_result(tool_call.id.clone(), tool_name.clone(), result));
            }

            steps.push(AgentStep {
                iteration: iter_num,
                message: "Tool calls executed".to_string(),
                tool_calls: Some(tool_results),
            });
        } else if let Some(content) = &choice.message.content {
            if verbose {
                println!("💬 LLM Response:\n{}\n", content);
            }
            if let Some(cb) = callback.as_mut() {
                cb(StreamEvent::LlmResponse {
                    content: content.clone(),
                });
            }

            final_response = content.clone();

            steps.push(AgentStep {
                iteration: iter_num,
                message: content.clone(),
                tool_calls: None,
            });

            if choice.finish_reason.as_deref() == Some("stop") {
                if verbose {
                    println!("✨ Agent finished successfully!");
                }
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
                });
            }
        } else {
            tracing::warn!(
                "Unexpected LLM response at iteration {}: no content or tool_calls",
                iter_num
            );
            break;
        }
    }

    if let Some(cb) = callback.as_mut() {
        cb(StreamEvent::Finished {
            final_response: final_response.clone(),
            iterations: config.max_iterations,
        });
    }

    Ok(AgentResult {
        final_response,
        steps,
        iterations_used: config.max_iterations,
    })
}

pub async fn run_agent_with_history(
    llm: Arc<dyn LlmClient>,
    tool_executor: Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    verbose: bool,
    prompt_builder: Option<&PromptBuilder>,
) -> Result<AgentResult> {
    run_agent_loop::<fn(StreamEvent)>(
        &llm,
        &tool_executor,
        config,
        messages,
        prompt_builder,
        verbose,
        None,
    )
    .await
}

pub async fn run_agent_streaming_with_history<F>(
    llm: Arc<dyn LlmClient>,
    tool_executor: Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
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
        false,
        Some(callback),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn make_config(max_iterations: usize) -> AgentConfig {
        AgentConfig {
            max_iterations,
            ..AgentConfig::default()
        }
    }

    fn text_choice(content: &str, finish: &str) -> Choice {
        Choice {
            message: Message::assistant(content.into()),
            finish_reason: Some(finish.into()),
        }
    }

    fn tool_call_choice(tool_name: &str, args: &str) -> Choice {
        Choice {
            message: Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: tool_name.into(),
                        arguments: args.into(),
                    },
                }]),
                tool_call_id: None,
                tool_name: None,
            },
            finish_reason: Some("tool_calls".into()),
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

        async fn execute(&self, name: &str, args_json: &str) -> Result<String> {
            self.calls.lock().unwrap().push((name.into(), args_json.into()));
            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn agent_stops_on_finish_reason_stop() {
        let llm = Arc::new(MockLlm::new(vec![
            text_choice("done", "stop"),
        ]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new(""));
        let config = make_config(10);
        let mut messages = vec![Message::user("hi".into())];

        let result = run_agent_with_history(
            llm.clone(), executor, &config, &mut messages, false, None,
        ).await.unwrap();

        assert_eq!(result.final_response, "done");
        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.steps.len(), 1);
    }

    #[tokio::test]
    async fn agent_executes_tool_calls_and_continues() {
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("read_file", r#"{"path":"a.txt"}"#),
            text_choice("here is the file content", "stop"),
        ]));
        let executor = Arc::new(MockToolExecutor::new("file data"));
        let config = make_config(10);
        let mut messages = vec![Message::user("read a.txt".into())];

        let result = run_agent_with_history(
            llm.clone(), executor.clone(), &config, &mut messages, false, None,
        ).await.unwrap();

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
        let mut messages = vec![Message::user("loop".into())];

        let result = run_agent_with_history(
            llm, executor, &config, &mut messages, false, None,
        ).await.unwrap();

        assert_eq!(result.iterations_used, 3);
    }

    #[tokio::test]
    async fn agent_streaming_emits_events() {
        let llm = Arc::new(MockLlm::new(vec![
            tool_call_choice("echo", r#"{"cmd":"hi"}"#),
            text_choice("all done", "stop"),
        ]));
        let executor: Arc<dyn ToolExecutor> = Arc::new(MockToolExecutor::new("ok"));
        let config = make_config(10);
        let mut messages = vec![Message::user("test".into())];

        let mut events = Vec::new();
        let result = run_agent_streaming_with_history(
            llm, executor, &config, &mut messages, None,
            |event| events.push(event),
        ).await.unwrap();

        assert_eq!(result.final_response, "all done");

        // Check event sequence
        assert!(matches!(events[0], StreamEvent::IterationStart { iteration: 1 }));
        assert!(matches!(&events[1], StreamEvent::ToolCall { tool_name, .. } if tool_name == "echo"));
        assert!(matches!(&events[2], StreamEvent::ToolResult { tool_name, .. } if tool_name == "echo"));
        assert!(matches!(events[3], StreamEvent::IterationStart { iteration: 2 }));
        assert!(matches!(&events[4], StreamEvent::LlmResponse { content } if content == "all done"));
        assert!(matches!(&events[5], StreamEvent::Finished { .. }));
    }
}
