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
/// `allow_once` or `reject_once`) and keeps every notification. With the
/// `fs` capability it serves `fs/*` requests from `files`, its open
/// editor buffers keyed by absolute path.
pub struct Acp {
    wire: Wire,
    next_id: u64,
    pub permission: &'static str,
    pub notifications: Vec<Value>,
    pub permission_requests: Vec<Value>,
    pub files: std::collections::HashMap<String, String>,
    child: Child,
}

impl Acp {
    pub async fn stdio(env: &Env) -> Self {
        Self::stdio_with(env, json!({})).await
    }

    /// `openheim acp`, initialized with `capabilities`.
    pub async fn stdio_with(env: &Env, capabilities: Value) -> Self {
        let mut child = openheim(env, &["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        Self::new(Wire::Stdio { stdin, lines }, child, capabilities).await
    }

    pub async fn ws(server: Server) -> Self {
        let url = format!("ws://{}/acp", server.addr);
        let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        Self::new(Wire::Ws(Box::new(ws)), server.child, json!({})).await
    }

    async fn new(wire: Wire, child: Child, capabilities: Value) -> Self {
        let mut acp = Self {
            wire,
            next_id: 0,
            permission: "allow_once",
            notifications: vec![],
            permission_requests: vec![],
            files: Default::default(),
            child,
        };
        acp.call(
            "initialize",
            json!({"protocolVersion": 1, "clientCapabilities": capabilities}),
        )
        .await
        .unwrap();
        acp
    }

    /// Kills the agent process and waits until it's reaped, so its pid is
    /// dead from then on.
    pub async fn kill(mut self) {
        self.child.kill().await.unwrap();
    }

    /// Sends a notification.
    pub async fn notify(&mut self, method: &str, params: Value) {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.wire.send(&msg).await;
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
            "fs/read_text_file" => {
                let path = req["params"]["path"].as_str().unwrap();
                match self.files.get(path) {
                    Some(content) => json!({"content": content}),
                    None => {
                        let error = json!({"code": -32002, "message": "not open"});
                        let reply = json!({"jsonrpc": "2.0", "id": req["id"], "error": error});
                        return self.wire.send(&reply).await;
                    }
                }
            }
            "fs/write_text_file" => {
                let params = &req["params"];
                let path = params["path"].as_str().unwrap().to_string();
                self.files
                    .insert(path, params["content"].as_str().unwrap().into());
                json!({})
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
    /// The port is picked by binding port 0 and freeing it, so another
    /// listener in this test binary can take it before `serve` binds. The
    /// server only counts as up once it answers as one (`/api/models`
    /// listing the env's models); if it exits because the port was taken,
    /// it's started again on a new one.
    pub async fn start(env: &Env) -> Self {
        // Its log goes to a file, quoted if it fails.
        let log = env.data().join("serve.log");
        let read_log = || std::fs::read_to_string(&log).unwrap_or_default();
        for _ in 0..3 {
            let port = {
                let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                l.local_addr().unwrap().port()
            };
            let port_arg = port.to_string();
            let child = openheim(env, &["serve", "--host", "127.0.0.1", "--port", &port_arg])
                .stderr(std::fs::File::create(&log).unwrap())
                .spawn()
                .unwrap();
            let mut server = Self {
                addr: format!("127.0.0.1:{port}"),
                child,
            };
            let deadline = tokio::time::Instant::now() + TIMEOUT;
            loop {
                if server.is_ours().await {
                    return server;
                }
                if let Some(status) = server.child.try_wait().unwrap() {
                    if read_log().contains("Failed to bind") {
                        break;
                    }
                    panic!("serve exited with {status}:\n{}", read_log());
                }
                if tokio::time::Instant::now() > deadline {
                    panic!("serve never came up:\n{}", read_log());
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        panic!("serve couldn't get a free port:\n{}", read_log());
    }

    async fn is_ours(&self) -> bool {
        let Ok(response) = reqwest::get(format!("http://{}/api/models", self.addr)).await else {
            return false;
        };
        response.status().is_success()
            && response
                .text()
                .await
                .is_ok_and(|body| body.contains("fake-model-2"))
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

    /// Opens `/ws`, the enveloped agent + fs socket.
    pub async fn ws(&self) -> Ws {
        let url = format!("ws://{}/ws", self.addr);
        let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        Ws {
            wire: Wire::Ws(Box::new(ws)),
            next_id: 0,
            unsolicited: vec![],
        }
    }
}

/// A `/ws` client. Fs messages without an id (the greeting, watcher
/// events) are kept in `unsolicited`.
pub struct Ws {
    wire: Wire,
    next_id: u64,
    pub unsolicited: Vec<Value>,
}

impl Ws {
    pub async fn send(&mut self, msg: Value) {
        self.wire.send(&msg).await;
    }

    /// The next message on `channel`.
    pub async fn recv(&mut self, channel: &str) -> Value {
        loop {
            let msg = self.wire.recv().await;
            if msg["channel"] == channel {
                return msg["data"].clone();
            }
        }
    }

    /// Sends fs request `req` and returns its reply.
    pub async fn fs(&mut self, mut req: Value) -> Value {
        self.next_id += 1;
        req["id"] = json!(self.next_id);
        self.send(json!({"channel": "fs", "data": req})).await;
        loop {
            let data = self.recv("fs").await;
            if data["id"] == json!(self.next_id) {
                return data;
            }
            self.unsolicited.push(data);
        }
    }

    /// Waits for an unsolicited fs message of `kind`.
    pub async fn event(&mut self, kind: &str) -> Value {
        if let Some(i) = self.unsolicited.iter().position(|m| m["type"] == kind) {
            return self.unsolicited.remove(i);
        }
        loop {
            let data = self.recv("fs").await;
            if data["type"] == kind && data.get("id").is_none() {
                return data;
            }
        }
    }
}
