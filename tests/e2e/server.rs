//! `openheim serve`: the REST API, and `/ws` with its agent and
//! filesystem channels.

use serde_json::json;

use crate::common::bin::{Acp, Server};
use crate::common::{Env, Kind, mcp, text};

#[tokio::test]
async fn the_api_describes_the_agent_without_secrets() {
    let url = mcp::start().await;
    let env = Env::with(Kind::OpenAi, &format!("[mcp_servers.stub]\nurl = {url:?}")).await;
    std::fs::create_dir_all(env.data().join("skills")).unwrap();
    std::fs::write(env.data().join("skills/rust.md"), "Rust.").unwrap();
    env.script([text("hi")]);
    let server = Server::start(&env).await;

    let config = server.get("/api/config").await.to_string();
    assert!(
        config.contains("fake-model") && !config.contains("test-key"),
        "{config}"
    );
    assert!(
        server
            .get("/api/models")
            .await
            .to_string()
            .contains("fake-model-2")
    );
    assert_eq!(server.get("/api/skills").await, json!(["rust"]));
    let tools = server.get("/api/tools").await.to_string();
    assert!(
        tools.contains("read_file") && tools.contains("stub__echo"),
        "{tools}"
    );
    assert!(
        server
            .get("/api/mcp-servers")
            .await
            .to_string()
            .contains("stub")
    );
    assert_eq!(server.get("/api/sessions").await, json!([]));

    let addr = server.addr.clone();
    let mut acp = Acp::ws(server).await;
    let id = acp.new_session(&env).await;
    acp.prompt(&id, "hello").await;
    let sessions = get(&addr, "/api/sessions").await.text().await.unwrap();
    assert!(sessions.contains(&id));
    let session = get(&addr, &format!("/api/sessions/{id}"))
        .await
        .text()
        .await
        .unwrap();
    assert!(session.contains("hello") && session.contains("hi"));
    assert!(!get(&addr, "/api/sessions/nope").await.status().is_success());
}

async fn get(addr: &str, path: &str) -> reqwest::Response {
    reqwest::get(format!("http://{addr}{path}")).await.unwrap()
}

#[tokio::test]
async fn ws_carries_agent_frames_in_an_envelope() {
    let env = Env::new().await;
    let server = Server::start(&env).await;
    let mut ws = server.ws().await;

    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1, "clientCapabilities": {}}});
    ws.send(json!({"channel": "agent", "data": init})).await;

    let reply = ws.recv("agent").await;
    assert_eq!(reply["id"], 1);
    assert!(reply["result"]["protocolVersion"].is_number(), "{reply}");
}

#[tokio::test]
async fn the_fs_channel_works_inside_the_work_dir_only() {
    let env = Env::new().await;
    env.file("a.txt", "hello").file("dir/b.txt", "b");
    std::os::unix::fs::symlink(env.path("a.txt"), env.path("link")).unwrap();
    let server = Server::start(&env).await;
    let mut ws = server.ws().await;
    assert_eq!(ws.event("connected").await["type"], "connected");

    let reply = ws.fs(json!({"action": "read", "path": "a.txt"})).await;
    assert_eq!(
        (&reply["type"], &reply["content"]),
        (&json!("file_content"), &json!("hello"))
    );
    let list = ws.fs(json!({"action": "list", "path": "dir"})).await;
    assert!(list["entries"].to_string().contains("b.txt"), "{list}");
    for req in [
        json!({"action": "mkdir", "path": "new"}),
        json!({"action": "write", "path": "new/c.txt", "content": "c"}),
        json!({"action": "rename", "from": "new/c.txt", "to": "new/d.txt"}),
        json!({"action": "delete", "path": "link"}),
    ] {
        let reply = ws.fs(req.clone()).await;
        assert!(
            reply["type"].as_str().unwrap().ends_with("success"),
            "{req}: {reply}"
        );
    }
    assert_eq!(env.read("new/d.txt"), "c");
    assert!(!env.path("link").exists() && env.path("a.txt").exists());

    let outside = env.data().join("outside.txt");
    std::fs::write(&outside, "secret").unwrap();
    let outside = outside.to_str().unwrap();
    for req in [
        json!({"action": "read", "path": outside}),
        json!({"action": "read", "path": "../outside.txt"}),
        json!({"action": "write", "path": outside, "content": "x"}),
        json!({"action": "delete", "path": outside}),
        json!({"action": "watch", "path": env.data()}),
        json!({"action": "delete", "path": "."}),
        json!({"action": "rename", "from": ".", "to": "moved"}),
    ] {
        let reply = ws.fs(req.clone()).await;
        assert_eq!(reply["type"], "error", "{req}: {reply}");
    }
    assert_eq!(std::fs::read_to_string(outside).unwrap(), "secret");

    let watching = ws.fs(json!({"action": "watch", "path": "."})).await;
    assert_eq!(watching["type"], "watching");
    std::fs::write(env.path("watched.txt"), "w").unwrap();
    // The watcher may first report earlier changes to the tree.
    while !ws.event("fs_event").await["paths"]
        .to_string()
        .contains("watched.txt")
    {}
}
