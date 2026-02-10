use std::sync::Arc;

use crate::config::AgentConfig;
use crate::core::llm::LlmClient;
use crate::core::models::*;
use crate::error::Result;
use crate::rag::PromptBuilder;
use crate::tools::{get_available_tools, ToolExecutor};

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

async fn run_agent_loop(
    llm: &Arc<dyn LlmClient>,
    tool_executor: &Arc<dyn ToolExecutor>,
    config: &AgentConfig,
    messages: &mut Vec<Message>,
    prompt_builder: Option<&PromptBuilder>,
    verbose: bool,
    mut callback: Option<&mut dyn FnMut(StreamEvent)>,
) -> Result<AgentResult> {
    let tools = get_available_tools();
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
        if let Some(cb) = callback.as_deref_mut() {
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

                if let Some(cb) = callback.as_deref_mut() {
                    cb(StreamEvent::ToolCall {
                        tool_name: tool_name.clone(),
                        arguments: arguments.clone(),
                    });
                }

                let result = tool_executor.execute(tool_name, arguments).await?;

                if verbose {
                    println!("✅ Tool {}: {}\n", tool_name, result);
                }
                if let Some(cb) = callback.as_deref_mut() {
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
            if let Some(cb) = callback.as_deref_mut() {
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
                if let Some(cb) = callback.as_deref_mut() {
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

    if let Some(cb) = callback.as_deref_mut() {
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
    run_agent_loop(
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
    mut callback: F,
) -> Result<AgentResult>
where
    F: FnMut(StreamEvent),
{
    run_agent_loop(
        &llm,
        &tool_executor,
        config,
        messages,
        prompt_builder,
        false,
        Some(&mut callback),
    )
    .await
}
