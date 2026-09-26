//! Headless transport: runs a single agent prompt in-process and streams output to stdout.
//!
//! Used by `openheim run "<prompt>"`.

use std::io::Write as _;

use crate::client::OpenheimClient;
use crate::core::models::StreamEvent;

/// Runs the agent against `prompt` in a fresh session and prints the
/// streamed response to stdout as it arrives.
///
/// Every tool call is allowed: there's no one to ask, and running the
/// command was the consent.
pub async fn run_headless(client: OpenheimClient, prompt: String) -> crate::error::Result<()> {
    let session = client.new_session().start().await?;
    let stop_reason = session
        .prompt(prompt, |event| {
            if let StreamEvent::LlmResponse { content } = event {
                print!("{content}");
                let _ = std::io::stdout().flush();
            }
        })
        .await?;
    println!();
    // On stderr, so piping the answer somewhere doesn't capture the notice.
    if let Some(notice) = stop_reason.notice() {
        eprintln!("[openheim: {notice}]");
    }
    Ok(())
}
