//! `openheim acp`: sessions, permissions, the client's files, config
//! options, history replay and errors, as an ACP client sees them.

use serde_json::{Value, json};

use crate::common::bin::Acp;
use crate::common::{Env, Kind, Reply, status, text, thinking, tool};

fn prompt_params(session: &str, blocks: Value) -> Value {
    json!({"sessionId": session, "prompt": blocks})
}

#[tokio::test]
async fn tool_calls_ask_the_client_and_report_their_kind() {
    let env = Env::new().await;
    agent_profile(&env);
    let write = |n: u32| {
        tool(
            "write_file",
            json!({"path": format!("{n}.txt"), "content": "x"}),
        )
    };
    env.script([
        write(1),
        text("a"),
        write(2),
        write(3),
        text("b"),
        tool("delegate_task", json!({"agent": "helper", "task": "t"})),
        tool("read_file", json!({"path": "1.txt"})),
        text("sub"),
        text("c"),
    ]);
    let mut acp = Acp::stdio(&env).await;
    let id = acp.new_session(&env).await;

    acp.permission = "reject_once";
    acp.prompt(&id, "one").await;
    assert!(!env.path("1.txt").exists());
    acp.permission = "allow_always";
    acp.prompt(&id, "two").await;
    assert!(env.path("2.txt").exists() && env.path("3.txt").exists());
    assert_eq!(acp.permission_requests.len(), 2);
    acp.permission = "allow_once";
    acp.prompt(&id, "three").await;

    let titles: Vec<_> = acp
        .permission_requests
        .iter()
        .map(|r| r["toolCall"]["title"].to_string())
        .collect();
    assert!(titles.last().unwrap().contains("helper"), "{titles:?}");
    // The subagent's own call stays out of the session's transcript.
    let kinds: Vec<_> = acp
        .updates("tool_call")
        .map(|u| u["kind"].clone())
        .collect();
    assert_eq!(kinds.len(), 4, "{kinds:?}");
    assert!(kinds[..3].iter().all(|k| k == "edit"), "{kinds:?}");
    let done = acp
        .updates("tool_call_update")
        .filter(|u| u["status"] == "completed")
        .count();
    assert!(done >= 3);
}

fn agent_profile(env: &Env) {
    std::fs::create_dir_all(env.data().join("agents")).unwrap();
    std::fs::write(env.data().join("agents/helper.md"), "Helper.").unwrap();
}

#[tokio::test]
async fn the_clients_open_buffers_are_read_and_written_instead_of_disk() {
    let env = Env::new().await;
    env.file("on_disk.txt", "disk text")
        .file("open.txt", "stale disk text");
    env.script([
        tool(
            "read_file",
            json!({"path": "open.txt", "offset": 2, "limit": 1}),
        ),
        tool(
            "write_file",
            json!({"path": "new.txt", "content": "via editor"}),
        ),
        tool("read_file", json!({"path": "on_disk.txt"})),
        text("done"),
    ]);
    let caps = json!({"fs": {"readTextFile": true, "writeTextFile": true}});
    let mut acp = Acp::stdio_with(&env, caps).await;
    let open = env.path("open.txt").display().to_string();
    acp.files
        .insert(open, "buffer line 1\nbuffer line 2\nbuffer line 3".into());
    let id = acp.new_session(&env).await;

    acp.prompt(&id, "go").await;

    let requests = env.llm.requests();
    assert!(requests[1].body.to_string().contains("buffer line 2"));
    let new = env.path("new.txt").display().to_string();
    assert_eq!(acp.files.get(&new).map(String::as_str), Some("via editor"));
    assert!(!env.path("new.txt").exists());
    assert!(
        requests[3].body.to_string().contains("not open"),
        "unopened file not delegated"
    );
}

#[tokio::test]
async fn config_options_and_modes_change_the_session() {
    let env = Env::new().await;
    env.script([text("a"), text("b")]);
    let mut acp = Acp::stdio(&env).await;
    let created = acp
        .call("session/new", json!({"cwd": env.work, "mcpServers": []}))
        .await
        .unwrap();
    let id = created["sessionId"].as_str().unwrap().to_string();
    let options = created["configOptions"].to_string();
    assert!(options.contains("fake/fake-model-2") && options.contains("thought_level"));

    let set =
        |config: &str, value: &str| json!({"sessionId": id, "configId": config, "value": value});
    acp.call(
        "session/set_config_option",
        set("model", "fake/fake-model-2"),
    )
    .await
    .unwrap();
    acp.call("session/set_config_option", set("effort", "high"))
        .await
        .unwrap();
    assert!(
        acp.call("session/set_config_option", set("model", "fake/nope"))
            .await
            .is_err()
    );
    assert!(
        acp.call("session/set_config_option", set("colour", "red"))
            .await
            .is_err()
    );
    acp.call(
        "session/set_mode",
        json!({"sessionId": id, "modeId": "plan"}),
    )
    .await
    .unwrap();
    acp.prompt(&id, "x").await;

    let body = env.llm.last();
    assert_eq!(
        (&body["model"], &body["reasoning_effort"]),
        (&json!("fake-model-2"), &json!("high"))
    );
    assert!(!body["tools"].to_string().contains("write_file"));
}

#[tokio::test]
async fn a_loaded_session_replays_its_history_in_order() {
    let env = Env::new().await;
    env.file("a.txt", "x");
    env.script([
        Reply::Turn(crate::common::Answer {
            thinking: Some("pondering".into()),
            tools: vec![("read_file".into(), json!({"path": "a.txt"}))],
            ..Default::default()
        }),
        thinking("concluding", "the answer"),
    ]);
    let id = {
        let mut acp = Acp::stdio(&env).await;
        let id = acp.new_session(&env).await;
        acp.prompt(&id, "question").await;
        id
    };

    let mut acp = Acp::stdio(&env).await;
    let listed = acp.call("session/list", json!({})).await.unwrap();
    assert!(listed.to_string().contains(&id));
    let load = json!({"sessionId": id, "cwd": env.work, "mcpServers": []});
    acp.call("session/load", load).await.unwrap();

    let order: Vec<_> = acp
        .notifications
        .iter()
        .map(|n| &n["params"]["update"])
        .filter_map(|u| {
            let kind = u["sessionUpdate"].as_str()?;
            let text = u["content"]["text"]
                .as_str()
                .or(u["title"].as_str())
                .unwrap_or("");
            Some(format!("{kind}:{text}"))
        })
        .filter(|s| !s.starts_with("tool_call_update") && !s.starts_with("available"))
        .collect();
    let at = |s: &str| {
        order
            .iter()
            .position(|o| o.contains(s))
            .unwrap_or_else(|| panic!("{s}: {order:?}"))
    };
    assert!(at("user_message_chunk:question") < at("pondering"));
    assert!(at("pondering") < at("tool_call:"));
    assert!(
        at("tool_call:") < at("concluding")
            && at("concluding") < at("agent_message_chunk:the answer")
    );
}

#[tokio::test]
async fn prompt_content_blocks_are_accepted_or_refused() {
    let env = Env::new().await;
    env.script([text("ok")]);
    let mut acp = Acp::stdio(&env).await;
    let id = acp.new_session(&env).await;

    let ok = json!([
        {"type": "text", "text": "look"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {"type": "resource_link", "uri": "file:///tmp/notes.md", "name": "notes.md"},
    ]);
    acp.call("session/prompt", prompt_params(&id, ok))
        .await
        .unwrap();
    let sent = env.llm.last().to_string();
    assert!(
        sent.contains("aGVsbG8=") && sent.contains("notes.md"),
        "{sent}"
    );

    let audio = json!([{"type": "audio", "data": "AAAA", "mimeType": "audio/wav"}]);
    assert!(
        acp.call("session/prompt", prompt_params(&id, audio))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn failures_reach_the_client_as_errors() {
    let env = Env::new().await;
    env.script([status(401, "{\"error\": \"bad key\"}"), Reply::Hang]);
    let mut acp = Acp::stdio(&env).await;
    let id = acp.new_session(&env).await;

    let unknown = prompt_params(
        &uuid::Uuid::new_v4().to_string(),
        json!([{"type": "text", "text": "x"}]),
    );
    assert!(acp.call("session/prompt", unknown).await.is_err());
    let err = acp
        .call(
            "session/prompt",
            prompt_params(&id, json!([{"type": "text", "text": "x"}])),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");

    let pending = acp
        .send(
            "session/prompt",
            prompt_params(&id, json!([{"type": "text", "text": "y"}])),
        )
        .await;
    env.llm.wait_for(2).await;
    acp.notify("session/cancel", json!({"sessionId": id})).await;
    let cancelled = acp.response(pending).await.unwrap();
    assert_eq!(cancelled["stopReason"], "cancelled");
}

#[tokio::test]
async fn acp_works_with_every_provider() {
    for kind in [Kind::Anthropic, Kind::Gemini] {
        let env = Env::with(kind, "").await;
        env.script([text("hello there")]);
        let mut acp = Acp::stdio(&env).await;
        let id = acp.new_session(&env).await;
        assert_eq!(acp.prompt(&id, "hi").await, "end_turn");
        assert_eq!(acp.agent_text(), "hello there", "{kind:?}");
    }
}
