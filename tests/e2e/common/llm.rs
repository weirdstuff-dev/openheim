//! A fake LLM provider: an HTTP server on a loopback port that answers each
//! request with the next scripted [`Reply`], in the wire format of its
//! [`Kind`], and records every request it receives.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    OpenAi,
    /// OpenAI's wire format, configured as `openai_compatible`.
    Compatible,
    Anthropic,
    Gemini,
}

impl Kind {
    pub const ALL: [Kind; 4] = [
        Kind::OpenAi,
        Kind::Compatible,
        Kind::Anthropic,
        Kind::Gemini,
    ];

    pub fn config_kind(self) -> &'static str {
        match self {
            Kind::OpenAi => "openai",
            Kind::Compatible => "openai_compatible",
            Kind::Anthropic => "anthropic",
            Kind::Gemini => "gemini",
        }
    }
}

/// One scripted answer.
#[derive(Clone, Debug, Default)]
pub struct Answer {
    /// Sent signed where the wire format supports it.
    pub thinking: Option<String>,
    pub text: String,
    /// `(name, arguments)`; a string argument is sent as the raw argument
    /// text, so it can be malformed JSON.
    pub tools: Vec<(String, Value)>,
    /// The provider's own finish reason, instead of the usual one.
    pub finish: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub enum Reply {
    Turn(Answer),
    /// A stream that ends after `text`, before the provider's end marker.
    CutOff(String),
    /// A non-200 response.
    Status {
        status: u16,
        retry_after: Option<u64>,
        body: String,
    },
    /// Never answers.
    Hang,
}

pub fn text(text: &str) -> Reply {
    Reply::Turn(Answer {
        text: text.into(),
        ..Answer::default()
    })
}

pub fn tool(name: &str, args: Value) -> Reply {
    tools(&[(name, args)])
}

pub fn tools(calls: &[(&str, Value)]) -> Reply {
    Reply::Turn(Answer {
        tools: calls
            .iter()
            .map(|(n, a)| (n.to_string(), a.clone()))
            .collect(),
        ..Answer::default()
    })
}

pub fn thinking(thinking: &str, text: &str) -> Reply {
    Reply::Turn(Answer {
        thinking: Some(thinking.into()),
        text: text.into(),
        ..Answer::default()
    })
}

/// `text` ending with the provider's finish reason `finish`.
pub fn finish(text: &str, finish: &'static str) -> Reply {
    Reply::Turn(Answer {
        text: text.into(),
        finish: Some(finish),
        ..Answer::default()
    })
}

pub fn status(status: u16, body: &str) -> Reply {
    Reply::Status {
        status,
        retry_after: None,
        body: body.into(),
    }
}

/// A 429 asking to retry after `secs`.
pub fn rate_limited(secs: u64) -> Reply {
    Reply::Status {
        status: 429,
        retry_after: Some(secs),
        body: "rate limited".into(),
    }
}

/// A request as the fake provider received it.
#[derive(Clone, Debug)]
pub struct Request {
    pub path: String,
    pub headers: HeaderMap,
    pub body: Value,
}

#[derive(Default)]
struct Shared {
    script: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<Request>>,
    /// Texts embedded so far, and whether `/embeddings` answers 503.
    embedded: Mutex<Vec<String>>,
    embeddings_down: std::sync::atomic::AtomicBool,
}

#[derive(Clone)]
pub struct FakeLlm {
    base: String,
    shared: Arc<Shared>,
}

impl FakeLlm {
    pub async fn start(kind: Kind) -> Self {
        let shared = Arc::new(Shared::default());
        let app = Router::new()
            .fallback(handle)
            .with_state((kind, shared.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let prefix = if kind == Kind::Gemini { "v1beta" } else { "v1" };
        Self {
            base: format!("http://{addr}/{prefix}"),
            shared,
        }
    }

    pub fn api_base(&self) -> &str {
        &self.base
    }

    /// Queues replies, answered in order, one per request.
    pub fn script(&self, replies: impl IntoIterator<Item = Reply>) {
        self.shared.script.lock().unwrap().extend(replies);
    }

    pub fn requests(&self) -> Vec<Request> {
        self.shared.requests.lock().unwrap().clone()
    }

    /// The body of the last request received.
    pub fn last(&self) -> Value {
        self.requests()
            .last()
            .expect("no request received")
            .body
            .clone()
    }

    /// Every text sent to `/embeddings` so far.
    pub fn embedded(&self) -> Vec<String> {
        self.shared.embedded.lock().unwrap().clone()
    }

    pub fn set_embeddings_down(&self, down: bool) {
        self.shared.embeddings_down.store(down, Ordering::SeqCst);
    }

    /// Waits until `n` requests have arrived.
    pub async fn wait_for(&self, n: usize) {
        for _ in 0..1000 {
            if self.requests().len() >= n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("only {} requests arrived", self.requests().len());
    }
}

async fn handle(
    State((kind, shared)): State<(Kind, Arc<Shared>)>,
    uri: axum::http::Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let body = serde_json::from_slice(&body).unwrap_or(Value::Null);
    if uri.path().ends_with("/embeddings") {
        return embeddings(&shared, &body);
    }
    let call = {
        let mut requests = shared.requests.lock().unwrap();
        requests.push(Request {
            path: uri.to_string(),
            headers,
            body,
        });
        requests.len()
    };
    let reply = shared.script.lock().unwrap().pop_front();
    match reply {
        None => (StatusCode::BAD_REQUEST, "fake llm: no reply scripted").into_response(),
        Some(Reply::Hang) => std::future::pending().await,
        Some(Reply::Status {
            status,
            retry_after,
            body,
        }) => {
            let mut response = (StatusCode::from_u16(status).unwrap(), body).into_response();
            if let Some(secs) = retry_after {
                response
                    .headers_mut()
                    .insert("retry-after", secs.to_string().parse().unwrap());
            }
            response
        }
        Some(Reply::Turn(turn)) => sse(encode(kind, call, &turn, true)),
        Some(Reply::CutOff(text)) => {
            let turn = Answer {
                text,
                ..Answer::default()
            };
            sse(encode(kind, call, &turn, false))
        }
    }
}

/// OpenAI's embeddings endpoint. A text's vector counts its letters, so
/// texts spelled alike are near each other.
fn embeddings(shared: &Shared, body: &Value) -> Response {
    if shared.embeddings_down.load(Ordering::SeqCst) {
        return (StatusCode::BAD_REQUEST, "embeddings down").into_response();
    }
    let inputs: Vec<String> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_string())
        .collect();
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let mut v = [0f32; 26];
            for b in text.to_lowercase().bytes().filter(u8::is_ascii_lowercase) {
                v[(b - b'a') as usize] += 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
            let v: Vec<f32> = v.iter().map(|x| x / norm).collect();
            json!({"object": "embedding", "index": index, "embedding": v})
        })
        .collect();
    shared.embedded.lock().unwrap().extend(inputs);
    axum::Json(json!({"object": "list", "data": data})).into_response()
}

fn sse(events: Vec<Value>) -> Response {
    let body: String = events
        .into_iter()
        .map(|e| match e {
            Value::String(s) => format!("data: {s}\n\n"),
            e => format!("data: {e}\n\n"),
        })
        .collect();
    ([("content-type", "text/event-stream")], body).into_response()
}

fn raw_args(args: &Value) -> String {
    match args {
        Value::String(s) => s.clone(),
        args => args.to_string(),
    }
}

/// The SSE payloads of one reply. Tool call ids are `call_<request>_<n>`.
fn encode(kind: Kind, call: usize, turn: &Answer, complete: bool) -> Vec<Value> {
    let id = |i: usize| format!("call_{call}_{i}");
    let has_tools = !turn.tools.is_empty();
    let mut out = Vec::new();
    match kind {
        Kind::OpenAi | Kind::Compatible => {
            if let Some(t) = &turn.thinking {
                out.push(json!({"choices": [{"delta": {"reasoning_content": t}}]}));
                out.push(json!({"choices": [{"delta": {"thinking_blocks": [
                    {"type": "thinking", "thinking": t, "signature": "sig"}
                ]}}]}));
            }
            if !turn.text.is_empty() {
                out.push(json!({"choices": [{"delta": {"content": turn.text}}]}));
            }
            for (i, (name, args)) in turn.tools.iter().enumerate() {
                out.push(json!({"choices": [{"delta": {"tool_calls": [{
                    "index": i, "id": id(i), "type": "function",
                    "function": {"name": name, "arguments": raw_args(args)}
                }]}}]}));
            }
            if complete {
                let usual = if has_tools { "tool_calls" } else { "stop" };
                let finish = turn.finish.unwrap_or(usual);
                out.push(json!({"choices": [{"delta": {}, "finish_reason": finish}]}));
                out.push(
                    json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
                );
                out.push(json!("[DONE]"));
            }
        }
        Kind::Anthropic => {
            out.push(json!({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}));
            let mut index = 0;
            let mut block = |out: &mut Vec<Value>, start: Value, deltas: Vec<Value>| {
                out.push(
                    json!({"type": "content_block_start", "index": index, "content_block": start}),
                );
                for delta in deltas {
                    out.push(
                        json!({"type": "content_block_delta", "index": index, "delta": delta}),
                    );
                }
                out.push(json!({"type": "content_block_stop", "index": index}));
                index += 1;
            };
            if let Some(t) = &turn.thinking {
                block(
                    &mut out,
                    json!({"type": "thinking", "thinking": ""}),
                    vec![
                        json!({"type": "thinking_delta", "thinking": t}),
                        json!({"type": "signature_delta", "signature": "sig"}),
                    ],
                );
            }
            if !turn.text.is_empty() {
                block(
                    &mut out,
                    json!({"type": "text", "text": ""}),
                    vec![json!({"type": "text_delta", "text": turn.text})],
                );
            }
            for (i, (name, args)) in turn.tools.iter().enumerate() {
                block(
                    &mut out,
                    json!({"type": "tool_use", "id": id(i), "name": name, "input": {}}),
                    vec![json!({"type": "input_json_delta", "partial_json": raw_args(args)})],
                );
            }
            if complete {
                let usual = if has_tools { "tool_use" } else { "end_turn" };
                let stop = turn.finish.unwrap_or(usual);
                out.push(json!({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}));
                out.push(json!({"type": "message_stop"}));
            }
        }
        Kind::Gemini => {
            let mut parts = Vec::new();
            if !turn.text.is_empty() {
                parts.push(json!({"text": turn.text}));
            }
            for (i, (name, args)) in turn.tools.iter().enumerate() {
                let mut part = json!({"functionCall": {"id": id(i), "name": name, "args": args}});
                if i == 0 && turn.thinking.is_some() {
                    part["thoughtSignature"] = json!("sig");
                }
                parts.push(part);
            }
            let mut candidate = json!({"content": {"role": "model", "parts": parts}});
            if complete {
                candidate["finishReason"] = json!(turn.finish.unwrap_or("STOP"));
            }
            out.push(json!({
                "candidates": [candidate],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
            }));
        }
    }
    out
}
