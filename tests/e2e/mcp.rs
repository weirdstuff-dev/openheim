//! MCP servers: their tools reach the model namespaced, and a server that
//! fails doesn't take the agent down.

use serde_json::json;

use crate::common::{Env, Kind, mcp, text};

#[tokio::test]
async fn mcp_tools_are_namespaced_and_their_failures_are_tool_errors() {
    let url = mcp::start().await;
    let env = Env::with(Kind::OpenAi, &format!("[mcp_servers.stub]\nurl = {url:?}")).await;
    env.script([text("ok")]);
    env.session().await.say("x").await;
    let tools = env.llm.last()["tools"].to_string();
    assert!(
        tools.contains("stub__echo") && tools.contains("stub__fail"),
        "{tools}"
    );

    assert!(
        env.err("stub__fail", json!({}))
            .await
            .contains("stub failure")
    );
    assert_eq!(env.ok("stub__echo", json!({"text": "pong"})).await, "pong");
}

#[tokio::test]
async fn an_unreachable_server_is_reported_and_the_rest_still_works() {
    let env = Env::with(
        Kind::OpenAi,
        "[mcp_servers.down]\nurl = \"http://127.0.0.1:1/mcp\"",
    )
    .await;
    env.file("a.txt", "local");

    let client = env.client().await;

    let statuses = serde_json::to_string(client.mcp_servers()).unwrap();
    assert!(statuses.contains("down"), "{statuses}");
    assert!(
        !client
            .tools()
            .iter()
            .any(|t| t.function.name.starts_with("down__"))
    );
    assert!(
        env.ok("read_file", json!({"path": "a.txt"}))
            .await
            .contains("local")
    );
}
