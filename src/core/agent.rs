use std::sync::Arc;

use crate::config::AgentConfig;
use crate::error::Result;
use crate::core::llm::LlmClient;
use crate::core::models::*;
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

async fn process_tool_calls(
    tool_calls: &[ToolCall],
    messages: &mut Vec<Message>,
    steps: &mut Vec<AgentStep>,
    iteration: usize,
    verbose: bool,
    tool_executor: &Arc<dyn ToolExecutor>,
) -> Result<()> {
    if verbose {
        println!("🛠️  LLM wants to call {} tool(s)", tool_calls.len());
    }

    let mut tool_results = Vec::new();

    for tool_call in tool_calls {
        let result = tool_executor
            .execute(&tool_call.function.name, &tool_call.function.arguments)
            .await?;

        if verbose {
            println!("✅ Tool {}: {}\n", tool_call.function.name, result);
        }

        tool_results.push(ToolExecutionResult {
            tool_name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.clone(),
            result: result.clone(),
        });

        messages.push(Message::tool_result(tool_call.id.clone(), result));
    }

    steps.push(AgentStep {
        iteration,
        message: "Tool calls executed".to_string(),
        tool_calls: Some(tool_results),
    });

    Ok(())
}

fn process_content_response(content: &str, steps: &mut Vec<AgentStep>, iteration: usize, verbose: bool)
{
    if verbose {
        println!("💬 LLM Response:\n{}\n", content);
    }

    steps.push(AgentStep {
        iteration,
        message: content.to_string(),
        tool_calls: None,
    });
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
    let tools = get_available_tools();
    let mut steps = Vec::new();
    let mut final_response = String::new();

    for iteration in 0..config.max_iterations {
        callback(StreamEvent::IterationStart {
            iteration: iteration + 1,
        });

        let choice = call_llm(&llm, messages, &tools, prompt_builder).await?;
        messages.push(choice.message.clone());

        if let Some(tool_calls) = &choice.message.tool_calls {
            let mut tool_results = Vec::new();

            for tool_call in tool_calls {
                let tool_name = &tool_call.function.name;
                let arguments = &tool_call.function.arguments;

                callback(StreamEvent::ToolCall {
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                });

                let result = tool_executor.execute(tool_name, arguments).await?;

                callback(StreamEvent::ToolResult {
                    tool_name: tool_name.clone(),
                    result: result.clone(),
                });

                tool_results.push(ToolExecutionResult {
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                    result: result.clone(),
                });

                messages.push(Message::tool_result(tool_call.id.clone(), result));
            }

            steps.push(AgentStep {
                iteration: iteration + 1,
                message: "Tool calls executed".to_string(),
                tool_calls: Some(tool_results),
            });
        } else if let Some(content) = &choice.message.content {
            callback(StreamEvent::LlmResponse {
                content: content.clone(),
            });

            final_response = content.clone();

            steps.push(AgentStep {
                iteration: iteration + 1,
                message: content.clone(),
                tool_calls: None,
            });

            if choice.finish_reason.as_deref() == Some("stop") {
                callback(StreamEvent::Finished {
                    final_response: final_response.clone(),
                    iterations: iteration + 1,
                });

                return Ok(AgentResult {
                    final_response,
                    steps,
                    iterations_used: iteration + 1,
                });
            }
        } else {
            break;
        }
    }

    callback(StreamEvent::Finished {
        final_response: final_response.clone(),
        iterations: config.max_iterations,
    });

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
    let tools = get_available_tools();
    let mut steps = Vec::new();
    let mut final_response = String::new();

    if verbose {
        println!("🤖 Continuing conversation...\n");
    }

    for iteration in 0..config.max_iterations {
        if verbose {
            println!("--- Iteration {} ---", iteration + 1);
        }

        let choice = call_llm(&llm, messages, &tools, prompt_builder).await?;
        messages.push(choice.message.clone());

        if let Some(tool_calls) = &choice.message.tool_calls {
            process_tool_calls(
                tool_calls,
                messages,
                &mut steps,
                iteration + 1,
                verbose,
                &tool_executor,
            )
            .await?;
        } else if let Some(content) = &choice.message.content {
            process_content_response(content, &mut steps, iteration + 1, verbose);
            final_response = content.clone();

            if choice.finish_reason.as_deref() == Some("stop") {
                if verbose {
                    println!("✨ Agent finished successfully!");
                }
                return Ok(AgentResult {
                    final_response,
                    steps,
                    iterations_used: iteration + 1,
                });
            }
        } else {
            if verbose {
                println!("⚠️  Unexpected response format");
            }
            break;
        }
    }

    Ok(AgentResult {
        final_response,
        steps,
        iterations_used: config.max_iterations,
    })
}