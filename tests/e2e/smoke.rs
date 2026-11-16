//! One tool round trip through every entry point and wire format.

use serde_json::json;

use crate::common::bin::{Acp, Server, run};
use crate::common::{Env, Kind, mcp, text, tool};

async fn round_trip(kind: Kind) {
    let env = Env::with(kind, "").await;
    env.file("a.txt", "hello from disk");
    env.script([tool("read_file", json!({"path": "a.txt"})), text("done")]);

    let turn = env.session().await.say("read a.txt").await;

    assert_eq!(turn.text(), "done");
    assert!(turn.ok_result().contains("hello from disk"));
    let sent = env.llm.last().to_string();
    assert!(
        sent.contains("hello from disk"),
        "tool result not sent back: {sent}"
    );
}

#[tokio::test]
async fn openai_round_trip() {
    round_trip(Kind::OpenAi).await;
}

#[tokio::test]
async fn anthropic_round_trip() {
    round_trip(Kind::Anthropic).await;
}

#[tokio::test]
async fn gemini_round_trip() {
    round_trip(Kind::Gemini).await;
}

#[tokio::test]
async fn run_prints_the_answer() {
    let env = Env::new().await;
    env.script([text("hi there")]);

    let out = run(&env, "hello").await;

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi there");
}

#[tokio::test]
async fn acp_over_stdio() {
    let env = Env::new().await;
    env.script([text("hi over acp")]);

    let mut acp = Acp::stdio(&env).await;
    let session = acp.new_session(&env).await;

    assert_eq!(acp.prompt(&session, "hello").await, "end_turn");
    assert_eq!(acp.agent_text(), "hi over acp");
}

#[tokio::test]
async fn serve_api_and_acp_over_websocket() {
    let env = Env::new().await;
    env.script([text("hi over ws")]);

    let server = Server::start(&env).await;
    let models = server.get("/api/models").await.to_string();
    assert!(models.contains("fake-model-2"), "{models}");

    let mut acp = Acp::ws(server).await;
    let session = acp.new_session(&env).await;
    assert_eq!(acp.prompt(&session, "hello").await, "end_turn");
    assert_eq!(acp.agent_text(), "hi over ws");
}

#[tokio::test]
async fn mcp_tool_round_trip() {
    let url = mcp::start().await;
    let env = Env::with(Kind::OpenAi, &format!("[mcp_servers.stub]\nurl = {url:?}")).await;
    env.script([tool("stub__echo", json!({"text": "pong"})), text("done")]);

    let turn = env.session().await.say("echo").await;

    assert_eq!(turn.ok_result(), "pong");
}
