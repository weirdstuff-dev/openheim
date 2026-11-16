//! Sessions: persistence, resume, listing, deletion, live setting changes
//! and locking across processes.

use openheim::{AgentMode, Effort, Error, StopReason, StreamEvent};
use serde_json::json;

use crate::common::bin::Acp;
use crate::common::{Env, Kind, Reply, text, tool};

#[tokio::test]
async fn a_session_is_saved_and_resumed_by_a_new_client() {
    let env = Env::new().await;
    env.script([text("one"), text("two"), text("three")]);
    let session = env.session().await;
    session.say("first question").await;
    session.say("second question").await;
    let id = session.handle.id().to_string();
    assert!(session.handle.context_usage().await.unwrap().is_some());

    let client = env.client().await;
    let listed = client.list_sessions(Some(&env.work)).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id.to_string(), id);
    assert_eq!(listed[0].title.as_deref(), Some("first question"));
    let elsewhere = env.data().join("elsewhere");
    assert!(
        client
            .list_sessions(Some(&elsewhere))
            .await
            .unwrap()
            .is_empty()
    );

    let (handle, loaded) = client.resume_session(&id, env.work.clone()).await.unwrap();
    assert_eq!(loaded.messages.len(), 4);
    handle.prompt("third question", |_| {}).await.unwrap();

    let sent = env.llm.last().to_string();
    for part in [
        "first question",
        "one",
        "second question",
        "two",
        "third question",
    ] {
        assert!(sent.contains(part), "missing {part}");
    }
    assert_eq!(client.get_session(&id).await.unwrap().messages.len(), 6);
}

#[tokio::test]
async fn a_damaged_log_tail_keeps_earlier_messages() {
    let env = Env::new().await;
    env.script([text("kept"), text("after")]);
    let session = env.session().await;
    session.say("hello").await;
    let id = session.handle.id().to_string();
    let log = env.data().join(format!("history/{id}.jsonl"));
    let mut bytes = std::fs::read(&log).unwrap();
    bytes.extend_from_slice(b"{\"role\":\"assist");
    std::fs::write(&log, bytes).unwrap();

    let client = env.client().await;
    let (handle, loaded) = client.resume_session(&id, env.work.clone()).await.unwrap();
    assert_eq!(loaded.messages.len(), 2);
    handle.prompt("again", |_| {}).await.unwrap();

    let reloaded = env.client().await.get_session(&id).await.unwrap();
    assert_eq!(reloaded.messages.len(), 4);
}

#[tokio::test]
async fn deleting_a_session_removes_its_files() {
    let env = Env::new().await;
    env.script([text("x")]);
    let session = env.session().await;
    session.say("hello").await;
    let id = session.handle.id().to_string();

    session.client.delete_session(&id).await.unwrap();

    assert!(session.client.list_sessions(None).await.unwrap().is_empty());
    assert!(session.client.get_session(&id).await.is_err());
    let left = std::fs::read_dir(env.data().join("history"))
        .unwrap()
        .count();
    assert_eq!(left, 0);
}

#[tokio::test]
async fn deleting_a_session_cancels_its_live_turn() {
    let env = Env::new().await;
    env.script([text("saved"), Reply::Hang]);
    let session = env.session().await;
    session.say("first").await;
    let id = session.handle.id().to_string();

    let (turn, deleted) = tokio::join!(session.try_say("second"), async {
        env.llm.wait_for(2).await;
        session.client.delete_session(&id).await
    });

    deleted.unwrap();
    assert_eq!(turn.unwrap().stop, StopReason::Cancelled);
    assert!(session.client.list_sessions(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn tools_use_the_session_cwd_only_inside_work_dir() {
    let env = Env::new().await;
    env.file("a.txt", "at root").file("sub/a.txt", "in sub");
    let read = || tool("read_file", json!({"path": "a.txt"}));
    env.script([read(), text("ok"), read(), text("ok")]);
    let client = env.client().await;
    let outside = tempfile::tempdir().unwrap();

    for (cwd, expected) in [
        (env.path("sub"), "in sub"),
        (outside.path().into(), "at root"),
    ] {
        let handle = client.new_session().cwd(cwd).start().await.unwrap();
        let mut results = vec![];
        handle
            .prompt("read", |e| {
                if let StreamEvent::ToolResult { result, .. } = e {
                    results.push(result);
                }
            })
            .await
            .unwrap();
        assert!(results[0].contains(expected), "{results:?}");
    }
}

#[tokio::test]
async fn unknown_sessions_models_and_skills_are_errors() {
    let env = Env::new().await;
    let client = env.client().await;
    let unknown = uuid::Uuid::new_v4().to_string();

    assert!(
        client
            .resume_session(&unknown, env.work.clone())
            .await
            .is_err()
    );
    assert!(client.new_session().model("nope").start().await.is_err());
    assert!(
        client
            .new_session()
            .skills(vec!["missing".into()])
            .start()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn model_mode_and_effort_change_live_and_are_restored() {
    let env = Env::new().await;
    env.script([text("a"), text("b"), text("c")]);
    let session = env.session().await;
    let (client, id) = (&session.client, session.handle.id().to_string());

    client
        .switch_model(&id, "fake", "fake-model-2")
        .await
        .unwrap();
    client.set_session_effort(&id, Effort::High).await.unwrap();
    client.set_session_mode(&id, AgentMode::Plan).await.unwrap();
    session.say("x").await;

    let body = env.llm.last();
    assert_eq!(body["model"], "fake-model-2");
    assert_eq!(body["reasoning_effort"], "high");
    let tools = body["tools"].to_string();
    assert!(
        tools.contains("read_file") && !tools.contains("write_file"),
        "{tools}"
    );
    assert!(client.switch_model(&id, "fake", "nope").await.is_err());

    let (handle, loaded) = env
        .client()
        .await
        .resume_session(&id, env.work.clone())
        .await
        .unwrap();
    assert_eq!(loaded.model, "fake-model-2");
    assert_eq!(
        (loaded.mode, loaded.effort),
        (AgentMode::Plan, Effort::High)
    );
    handle.prompt("y", |_| {}).await.unwrap();
    assert_eq!(env.llm.last()["reasoning_effort"], "high");
}

#[tokio::test]
async fn an_unchosen_effort_follows_the_provider() {
    let env = Env::build(Kind::OpenAi, "", "effort = \"low\"").await;
    env.script([text("a")]);
    let session = env.session().await;
    session.say("x").await;

    let meta = session.client.list_sessions(None).await.unwrap().remove(0);

    assert_eq!(meta.effort, None);
    assert_eq!(env.llm.last()["reasoning_effort"], "low");
}

#[tokio::test]
async fn a_second_prompt_is_rejected_while_one_runs() {
    let env = Env::new().await;
    env.script([Reply::Hang]);
    let session = env.session().await;

    let (first, second) = tokio::join!(session.try_say("one"), async {
        env.llm.wait_for(1).await;
        let second = session.try_say("two").await;
        session.handle.cancel().await;
        second
    });

    assert!(second.is_err());
    assert!(first.is_ok());
    assert_eq!(env.llm.requests().len(), 1);
}

#[tokio::test]
async fn a_turn_in_another_process_locks_the_session() {
    let env = Env::new().await;
    env.script([Reply::Hang]);
    let mut acp = Acp::stdio(&env).await;
    let id = acp.new_session(&env).await;
    let params = json!({"sessionId": id, "prompt": [{"type": "text", "text": "x"}]});
    acp.send("session/prompt", params).await;
    env.llm.wait_for(1).await;

    let client = env.client().await;
    let deleted = client.delete_session(&id).await;
    assert!(
        matches!(deleted, Err(Error::SessionLocked { .. })),
        "{deleted:?}"
    );
    let (handle, _) = client.resume_session(&id, env.work.clone()).await.unwrap();
    assert!(matches!(
        handle.prompt("y", |_| {}).await,
        Err(Error::SessionLocked { .. })
    ));

    drop(acp);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    client.delete_session(&id).await.unwrap();
}
