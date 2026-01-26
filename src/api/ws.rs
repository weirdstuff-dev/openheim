use actix::{Actor, ActorContext, AsyncContext, Handler, Message as ActixMessage, StreamHandler, WrapFuture};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::agent::run_agent_streaming;
use crate::config::AgentConfig;
use crate::core::llm::LlmClient;
use crate::core::models::StreamEvent;
use crate::tools::ToolExecutor;

#[derive(Debug, Deserialize)]
pub struct WsRequest {
    pub prompt: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

fn default_max_iterations() -> usize {
    10
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum WsResponse {
    #[serde(rename = "connected")]
    Connected { message: String },

    #[serde(rename = "event")]
    Event { data: StreamEvent },

    #[serde(rename = "error")]
    Error { message: String },

    #[serde(rename = "done")]
    Done,
}

pub struct AgentWebSocket {
    llm: Arc<dyn LlmClient>,
    executor: Arc<dyn ToolExecutor>,
    config: AgentConfig,
}

impl AgentWebSocket {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        executor: Arc<dyn ToolExecutor>,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm,
            executor,
            config,
        }
    }

    fn send_json(&self, msg: &WsResponse, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(json) = serde_json::to_string(&msg) {
            ctx.text(json);
        }
    }
}

impl Actor for AgentWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let msg = WsResponse::Connected {
            message: "Connected to Openheim Agent".to_string(),
        };
        self.send_json(&msg, ctx);
    }
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct ExecuteAgent {
    prompt: String,
    max_iterations: usize,
}

impl Handler<ExecuteAgent> for AgentWebSocket {
    type Result = ();

    fn handle(&mut self, msg: ExecuteAgent, ctx: &mut Self::Context) {
        // Clone handles we'll move into the async task.
        let llm = self.llm.clone();
        let executor = self.executor.clone();
        let config = self.config.with_max_iterations(msg.max_iterations);
        let prompt = msg.prompt.clone();

        // Actor address to send back text messages
        let addr = ctx.address();
        let addr_for_closure = addr.clone();

        ctx.spawn(
            async move {
                // run_agent_streaming expects a synchronous callback FnMut(StreamEvent).
                // We forward events to the actor by serializing them and sending SendText messages.
                let result = run_agent_streaming(
                    llm,
                    executor,
                    &config,
                    &prompt,
                    move |event: StreamEvent| {
                        let ws_msg = WsResponse::Event { data: event };
                        if let Ok(json) = serde_json::to_string(&ws_msg) {
                            // Ignore send failures (actor may be stopping)
                            addr_for_closure.do_send(SendText { text: json });
                        }
                    },
                )
                .await;

                match result {
                    Ok(_) => {
                        let done_msg = WsResponse::Done;
                        if let Ok(json) = serde_json::to_string(&done_msg) {
                            addr.do_send(SendText { text: json });
                        }
                    }
                    Err(e) => {
                        let error_msg = WsResponse::Error {
                            message: e.to_string(),
                        };
                        if let Ok(json) = serde_json::to_string(&error_msg) {
                            addr.do_send(SendText { text: json });
                        }
                    }
                }
            }
            .into_actor(self),
        );
    }
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct SendText {
    text: String,
}

impl Handler<SendText> for AgentWebSocket {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        ctx.text(msg.text);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for AgentWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => match serde_json::from_str::<WsRequest>(&text) {
                Ok(req) => {
                    ctx.notify(ExecuteAgent {
                        prompt: req.prompt,
                        max_iterations: req.max_iterations,
                    });
                }
                Err(e) => {
                    let error = WsResponse::Error {
                        message: format!("Invalid request format: {}", e),
                    };
                    self.send_json(&error, ctx);
                }
            },
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

pub async fn ws_handler(
    req: HttpRequest,
    stream: web::Payload,
    llm: web::Data<Arc<dyn LlmClient>>,
    executor: web::Data<Arc<dyn ToolExecutor>>,
    config: web::Data<AgentConfig>,
) -> Result<HttpResponse, Error> {
    let ws = AgentWebSocket::new(
        llm.get_ref().clone(),
        executor.get_ref().clone(),
        config.get_ref().clone(),
    );
    ws::start(ws, &req, stream)
}