use actix::{Actor, ActorContext, AsyncContext, Handler, Message as ActixMessage, StreamHandler, WrapFuture};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::agent::run_agent_streaming_with_history;
use crate::config::{AgentConfig, AppConfig, resolve_client_and_config};
use crate::core::llm::LlmClient;
use crate::core::models::{Message, StreamEvent};
use crate::rag::RagContext;
use crate::tools::ToolExecutor;

#[derive(Debug, Deserialize)]
pub struct WsRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Optional model name to override the default.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub chat_id: Option<Uuid>,
    #[serde(default)]
    pub skills: Option<Vec<String>>,
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
    Done {
        #[serde(skip_serializing_if = "Option::is_none")]
        chat_id: Option<String>,
    },
}

pub struct AgentWebSocket {
    llm: Arc<dyn LlmClient>,
    executor: Arc<dyn ToolExecutor>,
    config: AgentConfig,
    app_config: AppConfig,
    http_client: ReqwestClient,
}

impl AgentWebSocket {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        executor: Arc<dyn ToolExecutor>,
        config: AgentConfig,
        app_config: AppConfig,
        http_client: ReqwestClient,
    ) -> Self {
        Self {
            llm,
            executor,
            config,
            app_config,
            http_client,
        }
    }

    fn send_json(&self, msg: &WsResponse, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(json) = serde_json::to_string(&msg) {
            ctx.text(json);
        }
    }

    fn resolve_request(&self, req: &WsRequest) -> Result<(Arc<dyn LlmClient>, AgentConfig), String> {
        resolve_client_and_config(
            req.model.as_deref(),
            req.max_iterations,
            &self.app_config,
            &self.http_client,
            self.llm.clone(),
            &self.config,
        )
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
    llm: Arc<dyn LlmClient>,
    config: AgentConfig,
    prompt: String,
    chat_id: Option<Uuid>,
    skills: Vec<String>,
}

impl Handler<ExecuteAgent> for AgentWebSocket {
    type Result = ();

    fn handle(&mut self, msg: ExecuteAgent, ctx: &mut Self::Context) {
        let llm = msg.llm;
        let executor = self.executor.clone();
        let config = msg.config;
        let prompt = msg.prompt;
        let chat_id = msg.chat_id;
        let skills = msg.skills;

        let addr = ctx.address();
        let addr_for_closure = addr.clone();

        ctx.spawn(
            async move {
                let rag = match RagContext::new() {
                    Ok(r) => r,
                    Err(e) => {
                        let error_msg = WsResponse::Error {
                            message: e.to_string(),
                        };
                        if let Ok(json) = serde_json::to_string(&error_msg) {
                            addr.do_send(SendText { text: json });
                        }
                        return;
                    }
                };

                let (mut conversation, prompt_builder) = match rag.prepare(
                    chat_id,
                    &skills,
                    Some(config.model.clone()),
                    Some(config.provider_name.clone()),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        let error_msg = WsResponse::Error {
                            message: e.to_string(),
                        };
                        if let Ok(json) = serde_json::to_string(&error_msg) {
                            addr.do_send(SendText { text: json });
                        }
                        return;
                    }
                };

                conversation.messages.push(Message::user(prompt));
                let conv_id = conversation.meta.id;

                let result = run_agent_streaming_with_history(
                    llm,
                    executor,
                    &config,
                    &mut conversation.messages,
                    Some(&prompt_builder),
                    move |event: StreamEvent| {
                        let ws_msg = WsResponse::Event { data: event };
                        if let Ok(json) = serde_json::to_string(&ws_msg) {
                            addr_for_closure.do_send(SendText { text: json });
                        }
                    },
                )
                .await;

                let _ = rag.history.save_conversation(&conversation);

                match result {
                    Ok(_) => {
                        let done_msg = WsResponse::Done {
                            chat_id: Some(conv_id.to_string()),
                        };
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
                    match self.resolve_request(&req) {
                        Ok((llm, config)) => {
                            ctx.notify(ExecuteAgent {
                                llm,
                                config,
                                prompt: req.prompt,
                                chat_id: req.chat_id,
                                skills: req.skills.unwrap_or_default(),
                            });
                        }
                        Err(e) => {
                            let error = WsResponse::Error { message: e };
                            self.send_json(&error, ctx);
                        }
                    }
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
    app_config: web::Data<AppConfig>,
    http_client: web::Data<ReqwestClient>,
) -> Result<HttpResponse, Error> {
    let ws = AgentWebSocket::new(
        llm.get_ref().clone(),
        executor.get_ref().clone(),
        config.get_ref().clone(),
        app_config.get_ref().clone(),
        http_client.get_ref().clone(),
    );
    ws::start(ws, &req, stream)
}
