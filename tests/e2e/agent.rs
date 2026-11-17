//! The agent loop: tool rounds, stop reasons, cancellation, permissions and
//! fitting requests into the context window.

use openheim::StopReason;
use openheim::StreamEvent;
use openheim::core::permission::PermissionDecision;
use serde_json::{Value, json};

use crate::common::{Env, Gate, Kind, Reply, eventually, finish, status, text, tool, tools};

const SHELL: &str = "allow_shell = true";

/// The `messages` (or Anthropic/Gemini equivalent) of request `n`.
fn messages(env: &Env, n: usize) -> Vec<Value> {
    let body = &env.llm.requests()[n].body;
    let list = body.get("messages").or(body.get("contents")).unwrap();
    list.as_array().unwrap().clone()
}

#[tokio::test]
async fn parallel_and_sequential_tool_rounds() {
    let env = Env::new().await;
    env.file("a.txt", "A").file("b.txt", "B");
    env.script([
        tools(&[
            ("read_file", json!({"path": "a.txt"})),
            ("read_file", json!({"path": "b.txt"})),
        ]),
        tool("write_file", json!({"path": "c.txt", "content": "AB"})),
        text("done"),
    ]);

    let turn = env.session().await.say("go").await;

    assert_eq!(turn.stop, StopReason::EndTurn);
    assert_eq!(turn.results().len(), 3);
    assert!(turn.results().iter().all(|(_, _, err)| !err));
    assert_eq!(env.read("c.txt"), "AB");
    assert_eq!(env.llm.requests().len(), 3);
}

#[tokio::test]
async fn stops_at_max_iterations() {
    let env = Env::with(Kind::OpenAi, "max_iterations = 2").await;
    env.script([
        tool("list_dir", json!({"path": "."})),
        tool("list_dir", json!({"path": "."})),
    ]);

    let turn = env.session().await.say("loop").await;

    assert_eq!(turn.stop, StopReason::MaxIterations);
    assert_eq!(env.llm.requests().len(), 2);
}

#[tokio::test]
async fn finish_reasons_map_to_stop_reasons() {
    let cases = [
        (Kind::OpenAi, "length", StopReason::MaxTokens),
        (Kind::OpenAi, "content_filter", StopReason::Refusal),
        (Kind::Anthropic, "max_tokens", StopReason::MaxTokens),
        (Kind::Anthropic, "refusal", StopReason::Refusal),
        (Kind::Gemini, "MAX_TOKENS", StopReason::MaxTokens),
        (Kind::Gemini, "SAFETY", StopReason::Refusal),
    ];
    for (kind, reason, stop) in cases {
        let env = Env::with(kind, "").await;
        env.script([finish("partial", reason)]);
        let turn = env.session().await.say("x").await;
        assert_eq!(turn.stop, stop, "{kind:?} {reason}");
    }
}

#[tokio::test]
async fn paused_reply_resumes_the_turn() {
    let env = Env::with(Kind::Anthropic, "").await;
    env.script([finish("working on it", "pause_turn"), text("done")]);

    let turn = env.session().await.say("x").await;

    assert_eq!(
        (turn.stop, turn.text()),
        (StopReason::EndTurn, "done".into())
    );
    assert_eq!(env.llm.requests().len(), 2);
}

#[tokio::test]
async fn empty_reply_is_no_content() {
    let env = Env::new().await;
    env.script([text("")]);

    assert_eq!(
        env.session().await.say("x").await.stop,
        StopReason::NoContent
    );
}

#[tokio::test]
async fn tool_errors_go_back_to_the_model() {
    let env = Env::new().await;
    env.script([
        tool("read_file", json!({"path": "missing.txt"})),
        tool("no_such_tool", json!({})),
        tool("write_file", Value::String("{\"path\": \"x".into())),
        text("recovered"),
    ]);

    let turn = env.session().await.say("x").await;

    assert_eq!(turn.text(), "recovered");
    let results = turn.results();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|(_, _, err)| *err), "{results:?}");
    assert!(results[2].1.contains("not a valid JSON object"));
    assert!(!env.path("x").exists());
    assert!(
        env.llm
            .last()
            .to_string()
            .contains("not a valid JSON object")
    );
}

#[tokio::test]
async fn call_without_arguments_runs() {
    let env = Env::new().await;
    env.script([tool("list_dir", Value::String(String::new())), text("ok")]);

    env.session().await.say("x").await.ok_result();
}

#[tokio::test]
async fn oversized_tool_output_is_capped() {
    let url = crate::common::mcp::start().await;
    let env = Env::with(Kind::OpenAi, &format!("[mcp_servers.stub]\nurl = {url:?}")).await;
    let big = "z".repeat(300_000);
    env.script([tool("stub__echo", json!({"text": big})), text("ok")]);

    let result = env.session().await.say("x").await.ok_result();

    assert!(result.len() < 140_000, "{} bytes", result.len());
}

#[tokio::test]
async fn denied_call_is_not_run() {
    let env = Env::new().await;
    env.script([
        tool("write_file", json!({"path": "x.txt", "content": "no"})),
        text("ok"),
    ]);
    let gate = Gate::new([PermissionDecision::RejectOnce]);

    let turn = env.session().await.gate(&gate).say("x").await;

    turn.err_result();
    assert!(!env.path("x.txt").exists());
    assert_eq!(gate.asked(), ["write_file"]);
}

#[tokio::test]
async fn always_decisions_are_remembered_for_the_session() {
    let env = Env::new().await;
    let call = || tool("list_dir", json!({"path": "."}));
    env.script([call(), call(), text("ok"), call(), text("ok")]);
    let gate = Gate::new([PermissionDecision::AllowAlways]);
    let session = env.session().await.gate(&gate);

    session.say("one").await;
    session.say("two").await;

    assert_eq!(gate.asked().len(), 1);
}

#[tokio::test]
async fn always_allowed_tools_skip_the_gate() {
    let env = Env::with(Kind::OpenAi, r#"always_allowed_tools = ["list_dir"]"#).await;
    env.script([tool("list_dir", json!({"path": "."})), text("ok")]);
    let gate = Gate::new([PermissionDecision::RejectOnce]);

    env.session().await.gate(&gate).say("x").await.ok_result();

    assert!(gate.asked().is_empty());
}

/// Runs a prompt and cancels it once `ready` holds.
async fn cancel_when(
    session: &crate::common::Session,
    what: &str,
    ready: impl Fn() -> bool,
) -> crate::common::Turn {
    let (turn, ()) = tokio::join!(session.say("x"), async {
        eventually(what, &ready).await;
        session.handle.cancel().await;
    });
    turn
}

#[tokio::test]
async fn cancel_ends_a_hanging_llm_call() {
    let env = Env::new().await;
    env.script([Reply::Hang]);
    let session = env.session().await;

    let turn = cancel_when(&session, "the request", || env.llm.requests().len() == 1).await;

    assert_eq!(turn.stop, StopReason::Cancelled);
}

#[tokio::test]
async fn cancel_ends_a_pending_permission_prompt() {
    let env = Env::new().await;
    env.script([tool("list_dir", json!({"path": "."}))]);
    let gate = Gate::hang();
    let session = env.session().await.gate(&gate);

    let turn = cancel_when(&session, "the permission check", || {
        !gate.asked().is_empty()
    })
    .await;

    assert_eq!(turn.stop, StopReason::Cancelled);
}

#[tokio::test]
async fn cancelled_tool_calls_are_answered_on_the_next_turn() {
    let env = Env::with(Kind::OpenAi, SHELL).await;
    env.script([
        tools(&[
            ("list_dir", json!({"path": "."})),
            (
                "execute_command",
                json!({"command": "touch started; sleep 30"}),
            ),
        ]),
        text("next"),
    ]);
    let session = env.session().await;

    let started = std::time::Instant::now();
    let turn = cancel_when(&session, "the command", || env.path("started").exists()).await;
    assert_eq!(turn.stop, StopReason::Cancelled);
    assert!(started.elapsed().as_secs() < 10);

    assert_eq!(session.say("again").await.text(), "next");
    let sent = env.llm.last().to_string();
    assert!(sent.contains("call_1_0") && sent.contains("call_1_1"));
    assert!(sent.contains("Cancelled by user."), "{sent}");
}

#[tokio::test]
async fn overflow_retries_without_earlier_turns() {
    let env = Env::new().await;
    env.script([
        text("first answer"),
        status(400, "context_length_exceeded"),
        text("second answer"),
    ]);
    let session = env.session().await;
    session.say("first question").await;

    let turn = session.say("second question").await;

    assert_eq!(turn.text(), "second answer");
    let (before, after) = (messages(&env, 1), messages(&env, 2));
    assert!(after.len() < before.len());
    assert!(!Value::Array(after).to_string().contains("first question"));
    assert!(
        turn.events
            .iter()
            .any(|e| matches!(e, StreamEvent::ContextTrimmed { .. }))
    );
}

#[tokio::test]
async fn overflow_of_the_only_turn_is_an_error() {
    let env = Env::new().await;
    env.script([status(400, "prompt is too long")]);

    assert!(env.session().await.try_say("x").await.is_err());
}

#[tokio::test]
async fn known_window_trims_old_turns_before_sending() {
    let env = Env::build(Kind::OpenAi, "", "context_window = 4000").await;
    env.script([text("a"), text("b")]);
    let session = env.session().await;
    session.say("x".repeat(20_000)).await;

    let turn = session.say("short").await;

    assert!(!env.llm.last().to_string().contains(&"x".repeat(100)));
    assert!(
        turn.events
            .iter()
            .any(|e| matches!(e, StreamEvent::ContextTrimmed { .. }))
    );
}

#[tokio::test]
async fn long_tool_loop_leaves_out_its_oldest_outputs() {
    let env = Env::build(Kind::OpenAi, "", "context_window = 6000").await;
    env.file("big.txt", &"y".repeat(12_000));
    let read = || tool("read_file", json!({"path": "big.txt"}));
    env.script([read(), read(), read(), text("ok")]);

    let turn = env.session().await.say("x").await;

    assert_eq!(turn.text(), "ok");
    assert!(
        turn.events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolOutputsOmitted { omitted } if *omitted > 0))
    );
    assert!(
        env.llm
            .last()
            .to_string()
            .matches(&"y".repeat(12_000))
            .count()
            < 3
    );
}
