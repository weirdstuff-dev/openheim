//! `delegate_task`: profile and inline subagents, their isolation, limits,
//! models and permissions.

use openheim::Effort;
use serde_json::json;

use crate::common::{Env, Gate, Kind, text, tool};

const REVIEWER: &str = r#"+++
description = "Reviews files"
tools = ["read_file"]
max_iterations = 2
+++
You are REVIEWER."#;

fn agent(env: &Env, name: &str, profile: &str) {
    let dir = env.data().join("agents");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{name}.md")), profile).unwrap();
}

fn delegate(args: serde_json::Value) -> crate::common::Reply {
    tool("delegate_task", args)
}

#[tokio::test]
async fn a_profile_subagent_works_in_its_own_context() {
    let env = Env::new().await;
    agent(&env, "reviewer", REVIEWER);
    agent(&env, "broken", "+++\nunterminated");
    env.file("a.txt", "the diff");
    env.script([
        delegate(json!({"agent": "reviewer", "task": "review a.txt"})),
        tool("read_file", json!({"path": "a.txt"})),
        text("LGTM"),
        text("done"),
    ]);
    let gate = Gate::new([]);

    let turn = env.session().await.gate(&gate).say("parent secret").await;

    assert!(turn.ok_result().contains("LGTM"));
    let requests = env.llm.requests();
    let enum_ = requests[0].body["tools"].to_string();
    assert!(
        enum_.contains("reviewer") && !enum_.contains("broken"),
        "{enum_}"
    );
    let sub = requests[1].body.to_string();
    assert!(sub.contains("You are REVIEWER") && sub.contains("review a.txt"));
    assert!(!sub.contains("parent secret"));
    let sub_tools = requests[1].body["tools"].to_string();
    assert!(sub_tools.contains("read_file"));
    assert!(!sub_tools.contains("write_file") && !sub_tools.contains("delegate_task"));
    assert!(requests[3].body.to_string().contains("LGTM"));
    assert_eq!(gate.asked(), ["delegate_task", "reviewer:read_file"]);
}

#[tokio::test]
async fn subagents_stop_at_their_iteration_limit() {
    let env = Env::with(Kind::OpenAi, "max_iterations = 3").await;
    agent(&env, "reviewer", REVIEWER);
    let list = || tool("read_file", json!({"path": "missing"}));
    env.script([
        delegate(json!({"agent": "reviewer", "task": "t"})),
        list(),
        list(),
        delegate(json!({"system_prompt": "Loop.", "max_iterations": 50, "task": "t"})),
        list(),
        list(),
        list(),
        text("done"),
    ]);

    let turn = env.session().await.say("x").await;

    let results = turn.results();
    let delegated: Vec<_> = results.iter().filter(|r| r.0 == "delegate_task").collect();
    assert!(
        delegated[0].1.contains("reached its iteration limit (2)"),
        "{results:?}"
    );
    assert!(
        delegated[1].1.contains("reached its iteration limit (3)"),
        "{results:?}"
    );
    assert_eq!(env.llm.requests().len(), 8);
}

#[tokio::test]
async fn an_inline_subagent_gets_only_the_tools_it_names() {
    let env = Env::new().await;
    env.script([
        delegate(json!({"system_prompt": "You are INLINE.", "tools": ["list_dir"], "task": "t"})),
        tool("write_file", json!({"path": "x", "content": "x"})),
        text("refused"),
        text("done"),
    ]);

    let turn = env.session().await.say("x").await;

    assert!(turn.ok_result().contains("refused"));
    let sub = &env.llm.requests()[1].body;
    assert!(sub.to_string().contains("You are INLINE."));
    assert!(!sub["tools"].to_string().contains("write_file"));
    assert!(
        env.llm.requests()[2]
            .body
            .to_string()
            .contains("not available")
    );
    assert!(!env.path("x").exists());
}

#[tokio::test]
async fn bad_delegations_are_tool_errors() {
    let env = Env::new().await;
    agent(&env, "reviewer", REVIEWER);
    for args in [
        json!({"agent": "ghost", "task": "t"}),
        json!({"agent": "reviewer", "system_prompt": "x", "task": "t"}),
        json!({"task": "t"}),
        json!({"system_prompt": "x", "tools": [1], "task": "t"}),
    ] {
        env.err("delegate_task", args).await;
    }
}

#[tokio::test]
async fn subagents_run_on_their_own_model_and_effort_or_the_sessions() {
    let env = Env::new().await;
    agent(
        &env,
        "fast",
        "+++\nmodel = \"fake-model-2\"\neffort = \"high\"\n+++\nFast.",
    );
    agent(&env, "plain", "Plain.");
    env.script([
        delegate(json!({"agent": "fast", "task": "t"})),
        text("a"),
        delegate(json!({"agent": "plain", "task": "t"})),
        text("b"),
        text("done"),
    ]);
    let session = env.session().await;
    let id = session.handle.id().to_string();
    session
        .client
        .set_session_effort(&id, Effort::Low)
        .await
        .unwrap();

    session.say("x").await;

    let requests = env.llm.requests();
    let (fast, plain) = (&requests[1].body, &requests[3].body);
    assert_eq!(
        (&fast["model"], &fast["reasoning_effort"]),
        (&json!("fake-model-2"), &json!("high"))
    );
    assert_eq!(
        (&plain["model"], &plain["reasoning_effort"]),
        (&json!("fake-model"), &json!("low"))
    );
    assert_eq!(requests[4].body["model"], "fake-model");
}
