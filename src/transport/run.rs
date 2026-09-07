//! Headless transport: runs a single agent prompt in-process and streams output to stdout.
//!
//! Used by `openheim run "<prompt>"`.

use std::io::Write as _;

use crate::client::OpenheimClient;
use crate::core::models::StreamEvent;

/// Runs the agent against `prompt` in a fresh session and prints the
/// streamed response to stdout as it arrives.
///
/// `client` is caller-built — via `OpenheimClient::builder().model(..)` for a
/// model override, or with custom tools / a custom `LlmClient` an embedder
/// needs — so this transport doesn't have to hand-roll its own build path.
/// The session's permission gate defaults to `AllowAll`: `openheim run` is a
/// one-shot, non-interactive CLI invocation with no human to prompt, and the
/// user already consented to this run by invoking it.
pub async fn run_headless(client: OpenheimClient, prompt: String) -> crate::error::Result<()> {
    let session = client.new_session().start().await?;
    session
        .prompt_events(&prompt, |event| {
            if let StreamEvent::LlmResponse { content } = event {
                print!("{content}");
                let _ = std::io::stdout().flush();
            }
        })
        .await?;
    println!();
    Ok(())
}
