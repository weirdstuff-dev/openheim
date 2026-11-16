//! What reaches each provider's API, and how its replies and failures are
//! read back.

use openheim::{PromptInput, StopReason, StreamEvent};
use serde_json::{Value, json};

use crate::common::{Answer, Env, Kind, Reply, rate_limited, status, text, thinking, tool};

fn sent(env: &Env, n: usize) -> String {
    env.llm.requests()[n].body.to_string()
}

#[tokio::test]
async fn requests_carry_model_auth_and_tools() {
    for kind in Kind::ALL {
        let env = Env::with(kind, "").await;
        env.script([text("hi")]);
        env.session().await.say("hello").await;

        let req = &env.llm.requests()[0];
        let header = |name: &str| {
            req.headers
                .get(name)
                .map(|v| v.to_str().unwrap().to_string())
        };
        let (path, auth) = match kind {
            Kind::OpenAi | Kind::Compatible => ("/v1/chat/completions", header("authorization")),
            Kind::Anthropic => ("/v1/messages", header("x-api-key")),
            Kind::Gemini => (
                "/v1beta/models/fake-model:streamGenerateContent",
                header("x-goog-api-key"),
            ),
        };
        assert!(req.path.starts_with(path), "{kind:?}: {}", req.path);
        assert!(auth.unwrap().ends_with("test-key"), "{kind:?}");
        let body = req.body.to_string();
        assert!(
            body.contains("hello") && body.contains("read_file"),
            "{kind:?}: {body}"
        );
        if kind != Kind::Gemini {
            assert_eq!(req.body["model"], "fake-model");
            assert_eq!(req.body["stream"], true);
        }
    }
}

#[tokio::test]
async fn conversation_and_usage_round_trip() {
    for kind in Kind::ALL {
        let env = Env::with(kind, "").await;
        env.file("a.txt", "file body");
        env.script([
            text("first answer"),
            tool("read_file", json!({"path": "a.txt"})),
            text("second answer"),
        ]);
        let session = env.session().await;
        session.say("first question").await;

        let turn = session.say("second question").await;

        assert_eq!(turn.text(), "second answer", "{kind:?}");
        let last = sent(&env, 2);
        for part in [
            "first question",
            "first answer",
            "second question",
            "call_2_0",
            "file body",
        ] {
            assert!(last.contains(part), "{kind:?} lacks {part}: {last}");
        }
        let usage = turn.events.iter().find_map(|e| match e {
            StreamEvent::Usage { usage } => Some(*usage),
            _ => None,
        });
        let usage = usage.unwrap_or_else(|| panic!("{kind:?}: no usage"));
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (10, 5),
            "{kind:?}"
        );
    }
}

#[tokio::test]
async fn images_are_sent_in_each_format() {
    for (kind, key) in [
        (Kind::OpenAi, "image_url"),
        (Kind::Anthropic, "media_type"),
        (Kind::Gemini, "inlineData"),
    ] {
        let env = Env::with(kind, "").await;
        env.script([text("a cat")]);
        let input = PromptInput::text("what is this?").image("aGVsbG8=", "image/png");

        env.session().await.say(input).await;

        let body = sent(&env, 0);
        assert!(
            body.contains("aGVsbG8=") && body.contains(key),
            "{kind:?}: {body}"
        );
    }
}

#[tokio::test]
async fn thinking_is_streamed_and_signed_thinking_replayed_in_the_tool_loop() {
    for (kind, provider) in [
        (Kind::Anthropic, ""),
        (Kind::Gemini, ""),
        (Kind::Compatible, "replay_thinking = true"),
    ] {
        let env = Env::build(kind, "", provider).await;
        env.script([
            Reply::Turn(Answer {
                thinking: Some("pondering".into()),
                tools: vec![("list_dir".into(), json!({"path": "."}))],
                ..Answer::default()
            }),
            text("done"),
            text("again"),
        ]);
        let session = env.session().await;

        let turn = session.say("x").await;
        session.say("next").await;

        if kind != Kind::Gemini {
            assert!(turn.events.iter().any(
                |e| matches!(e, StreamEvent::ThinkingContent { content } if content == "pondering")
            ));
        }
        let requests = env.llm.requests();
        assert_eq!(signatures(&requests[1].body), ["sig"], "{kind:?}");
        if kind == Kind::Compatible {
            assert!(
                signatures(&requests[2].body).is_empty(),
                "replayed outside its tool loop"
            );
        }
    }
}

/// Every thinking signature in a request body, in order: Anthropic's and
/// OpenAI-compatible `signature`, Gemini's `thoughtSignature`.
fn signatures(body: &Value) -> Vec<String> {
    match body {
        Value::Object(map) => map
            .iter()
            .flat_map(|(key, value)| match (key.as_str(), value) {
                ("signature" | "thoughtSignature", Value::String(s)) => vec![s.clone()],
                _ => signatures(value),
            })
            .collect(),
        Value::Array(items) => items.iter().flat_map(signatures).collect(),
        _ => vec![],
    }
}

#[tokio::test]
async fn thinking_is_not_replayed_by_default_on_openai() {
    let env = Env::new().await;
    env.script([thinking("pondering", "a"), text("b")]);
    let session = env.session().await;
    session.say("one").await;
    session.say("two").await;

    let last = &env.llm.requests()[1].body;
    assert!(!last.to_string().contains("pondering"), "{last}");
    assert!(signatures(last).is_empty(), "{last}");
}

#[tokio::test]
async fn effort_and_token_limit_reach_the_request() {
    // (kind, pointer to the effort field, pointer to the limit field)
    let cases = [
        (Kind::OpenAi, "/reasoning_effort", "/max_completion_tokens"),
        (Kind::Compatible, "/reasoning_effort", "/max_tokens"),
        (Kind::Anthropic, "/output_config/effort", "/max_tokens"),
        (
            Kind::Gemini,
            "/generationConfig/thinkingConfig/thinkingLevel",
            "/generationConfig/maxOutputTokens",
        ),
    ];
    for (kind, effort, limit) in cases {
        let env = Env::build(kind, "", "effort = \"high\"\nmax_tokens = 123").await;
        env.script([text("a")]);
        env.session().await.say("x").await;
        let body = env.llm.last();
        assert_eq!(
            body.pointer(effort),
            Some(&json!("high")),
            "{kind:?}: {body}"
        );
        assert_eq!(body.pointer(limit), Some(&json!(123)), "{kind:?}: {body}");

        let env = Env::build(kind, "", "effort = \"default\"").await;
        env.script([text("a")]);
        env.session().await.say("x").await;
        let body = env.llm.last();
        assert_eq!(body.pointer(effort), None, "{kind:?}: {body}");
        assert!(body.get("thinking").is_none(), "{kind:?}: {body}");
    }
}

/// A cut-off stream is retried after the first backoff (1s), so the kinds
/// run concurrently.
#[tokio::test]
async fn transient_failures_are_retried() {
    futures::future::join_all(Kind::ALL.map(|kind| async move {
        let env = Env::with(kind, "").await;
        let unavailable = Reply::Status {
            status: 503,
            retry_after: Some(0),
            body: "busy".into(),
        };
        env.script([
            Reply::CutOff(String::new()),
            rate_limited(0),
            unavailable,
            text("ok"),
        ]);

        assert_eq!(env.session().await.say("x").await.text(), "ok", "{kind:?}");
        assert_eq!(env.llm.requests().len(), 4, "{kind:?}");
    }))
    .await;
}

#[tokio::test]
async fn retries_give_up_after_four_attempts() {
    let env = Env::new().await;
    env.script((0..5).map(|_| rate_limited(0)));

    assert!(env.session().await.try_say("x").await.is_err());
    assert_eq!(env.llm.requests().len(), 4);
}

#[tokio::test]
async fn permanent_failures_are_not_retried() {
    for reply in [status(401, "bad key"), rate_limited(3600)] {
        let env = Env::new().await;
        env.script([reply.clone(), text("unused")]);

        let err = env.session().await.try_say("x").await.unwrap_err();

        assert_eq!(env.llm.requests().len(), 1, "{reply:?}");
        assert!(
            err.to_string()
                .contains(if matches!(reply, Reply::Status { status: 401, .. }) {
                    "bad key"
                } else {
                    "rate limited"
                })
        );
    }
}

#[tokio::test]
async fn stream_cut_off_after_output_is_an_error() {
    for kind in Kind::ALL {
        let env = Env::with(kind, "").await;
        env.script([Reply::CutOff("partial".into()), text("unused")]);

        let result = env.session().await.try_say("x").await;

        assert!(result.is_err(), "{kind:?}: {result:?}");
        assert_eq!(env.llm.requests().len(), 1, "{kind:?}");
    }
}

#[tokio::test]
async fn stream_options_are_dropped_when_rejected() {
    let env = Env::with(Kind::Compatible, "").await;
    env.script([
        status(
            400,
            "Unrecognized request argument supplied: stream_options",
        ),
        text("ok"),
    ]);

    assert_eq!(env.session().await.say("x").await.text(), "ok");
    let requests = env.llm.requests();
    assert!(requests[0].body.get("stream_options").is_some());
    assert!(requests[1].body.get("stream_options").is_none());
}

#[tokio::test]
async fn malformed_tool_arguments_are_resent_as_an_empty_object() {
    for kind in [Kind::Anthropic, Kind::Gemini] {
        let env = Env::with(kind, "").await;
        let bad = if kind == Kind::Gemini {
            Value::String("{oops".into())
        } else {
            Value::String("{\"path\": ".into())
        };
        env.script([tool("write_file", bad), text("ok")]);

        let turn = env.session().await.say("x").await;

        assert_eq!(turn.stop, StopReason::EndTurn, "{kind:?}");
        turn.err_result();
        let pointer = if kind == Kind::Gemini {
            "args"
        } else {
            "input"
        };
        let last = sent(&env, 1).replace(' ', "");
        assert!(
            last.contains(&format!("\"{pointer}\":{{}}")),
            "{kind:?}: {last}"
        );
    }
}
