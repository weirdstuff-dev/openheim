//! Drivers for the `openheim` binary: `run`, `acp` over stdio, and `serve`
//! over WebSocket.

use std::process::{Output, Stdio};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::TcpStream;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, tungstenite::Message};

use super::Env;

const TIMEOUT: Duration = Duration::from_secs(30);

/// `openheim --config <env> <args>`, run in the workspace with `HOME` in
/// the temp dir.
pub fn openheim(env: &Env, args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_openheim"));
    cmd.arg("--config")
        .arg(env.config())
        .args(args)
        .current_dir(&env.work)
        .env("HOME", env.data())
        .env_remove("RUST_LOG")
        .kill_on_drop(true);
    cmd
}

pub async fn run(env: &Env, prompt: &str) -> Output {
    let out = openheim(env, &["run", prompt]).output();
    tokio::time::timeout(TIMEOUT, out).await.unwrap().unwrap()
}

enum Wire {
    Stdio {
        stdin: ChildStdin,
        lines: Lines<BufReader<ChildStdout>>,
    },
    Ws(Box<WebSocketStream<MaybeTlsStream<TcpStream>>>),
}

impl Wire {
    async fn send(&mut self, msg: &Value) {
        match self {
            Wire::Stdio { stdin, .. } => {
                stdin
                    .write_all(format!("{msg}\n").as_bytes())
                    .await
                    .unwrap();
            }
            Wire::Ws(ws) => ws.send(Message::text(msg.to_string())).await.unwrap(),
        }
    }

    async fn recv(&mut self) -> Value {
        let next = async {
            match self {
                Wire::Stdio { lines, .. } => {
                    let line = lines
                        .next_line()
                        .await
                        .unwrap()
                        .expect("agent closed stdout");
                    serde_json::from_str(&line).unwrap()
                }
                Wire::Ws(ws) => loop {
                    match ws.next().await.expect("agent closed the socket").unwrap() {
                        Message::Text(t) => break serde_json::from_str(&t).unwrap(),
                        _ => continue,
                    }
                },
            }
        };
        tokio::time::timeout(TIMEOUT, next)
            .await
            .expect("no message from the agent")
    }
}

/// An ACP client over stdio (`openheim acp`) or WebSocket (`/acp`). It
/// answers permission requests with `permission` (an option kind such as
/// `allow_once` or `reject_once`) and keeps every notification.
pub struct Acp {
    wire: Wire,
    next_id: u64,
    pub permission: &'static str,
    pub notifications: Vec<Value>,
    pub permission_requests: Vec<Value>,
    _child: Child,
}

impl Acp {
    pub async fn stdio(env: &Env) -> Self {
        let mut child = openheim(env, &["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        Self::new(Wire::Stdio { stdin, lines }, child).await
    }

    pub async fn ws(server: Server) -> Self {
        let url = format!("ws://{}/acp", server.addr);
        let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        Self::new(Wire::Ws(Box::new(ws)), server.child).await
    }

    async fn new(wire: Wire, child: Child) -> Self {
        let mut acp = Self {
            wire,
            next_id: 0,
            permission: "allow_once",
            notifications: vec![],
            permission_requests: vec![],
            _child: child,
        };
        acp.call(
            "initialize",
            json!({"protocolVersion": 1, "clientCapabilities": {}}),
        )
        .await
        .unwrap();
        acp
    }

    /// Sends a request without waiting for its response; returns its id.
    pub async fn send(&mut self, method: &str, params: Value) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.wire.send(&msg).await;
        id
    }

    /// Sends a request and returns its `result`, or its `error`.
    pub async fn call(&mut self, method: &str, params: Value) -> Result<Value, Value> {
        let id = self.send(method, params).await;
        self.response(id).await
    }

    /// Reads messages until the response to request `id`, answering the
    /// agent's requests and keeping its notifications on the way.
    pub async fn response(&mut self, id: u64) -> Result<Value, Value> {
        loop {
            let msg = self.wire.recv().await;
            if msg.get("method").is_some() && msg.get("id").is_some() {
                self.answer(msg).await;
            } else if msg.get("method").is_some() {
                self.notifications.push(msg);
            } else if msg["id"] == json!(id) {
                return match msg.get("error") {
                    Some(e) => Err(e.clone()),
                    None => Ok(msg["result"].clone()),
                };
            }
        }
    }

    async fn answer(&mut self, req: Value) {
        let result = match req["method"].as_str().unwrap() {
            "session/request_permission" => {
                let option = req["params"]["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|o| o["kind"] == self.permission)
                    .expect("no matching permission option");
                let id = option["optionId"].clone();
                self.permission_requests.push(req["params"].clone());
                json!({"outcome": {"outcome": "selected", "optionId": id}})
            }
            other => panic!("unexpected agent request {other}"),
        };
        let reply = json!({"jsonrpc": "2.0", "id": req["id"], "result": result});
        self.wire.send(&reply).await;
    }

    pub async fn new_session(&mut self, env: &Env) -> String {
        let r = self
            .call("session/new", json!({"cwd": env.work, "mcpServers": []}))
            .await
            .unwrap();
        r["sessionId"].as_str().unwrap().to_string()
    }

    /// Sends a text prompt and returns the stop reason.
    pub async fn prompt(&mut self, session: &str, text: &str) -> String {
        let r = self
            .call(
                "session/prompt",
                json!({"sessionId": session, "prompt": [{"type": "text", "text": text}]}),
            )
            .await
            .unwrap();
        r["stopReason"].as_str().unwrap().to_string()
    }

    /// The text of every `agent_message_chunk` so far, joined.
    pub fn agent_text(&self) -> String {
        self.updates("agent_message_chunk")
            .filter_map(|u| u["content"]["text"].as_str())
            .collect()
    }

    pub fn updates<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
        self.notifications
            .iter()
            .map(|n| &n["params"]["update"])
            .filter(move |u| u["sessionUpdate"] == kind)
    }
}

/// A running `openheim serve` on a free loopback port.
pub struct Server {
    pub addr: String,
    child: Child,
}

impl Server {
    pub async fn start(env: &Env) -> Self {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let child = openheim(
            env,
            &["serve", "--host", "127.0.0.1", "--port", &port.to_string()],
        )
        .spawn()
        .unwrap();
        let addr = format!("127.0.0.1:{port}");
        let server = Self { addr, child };
        let deadline = tokio::time::Instant::now() + TIMEOUT;
        while TcpStream::connect(&server.addr).await.is_err() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "serve never came up"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        server
    }

    pub async fn get(&self, path: &str) -> Value {
        reqwest::get(format!("http://{}{path}", self.addr))
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }
}
