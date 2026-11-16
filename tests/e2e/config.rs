//! Loading the config file: validation, provider resolution, keys, paths,
//! skills and the system prompt.

use openheim::OpenheimClient;
use serde_json::json;

use crate::common::bin::{openheim, run};
use crate::common::{Env, Kind, text, tool};

/// Writes a config with `work_dir` and one provider, `[providers.<name>]`
/// holding `body`. `{base}` in `body` becomes the fake provider's address,
/// `{m}` a default model `m` that `models` lists.
fn write_config(env: &Env, work_dir: &str, name: &str, body: &str) {
    let body = body
        .replace("{base}", env.llm.api_base())
        .replace("{m}", "default_model = \"m\"\nmodels = [\"m\"]");
    let config = format!(
        "default_provider = \"{name}\"\nwork_dir = {work_dir:?}\n\n[providers.{name}]\n{body}\n"
    );
    std::fs::write(env.config(), config).unwrap();
}

fn work(env: &Env) -> String {
    env.work.display().to_string()
}

async fn build(env: &Env) -> openheim::Result<OpenheimClient> {
    OpenheimClient::from_config(env.config()).build().await
}

#[tokio::test]
async fn invalid_configs_fail_the_build_with_a_reason() {
    let env = Env::new().await;
    env.file("a-file", "");
    let base = "api_base = \"{base}\"\n{m}";
    let cases = [
        ("api_base = \"\"\n{m}", "api_base"),
        ("api_base = \"ftp://x\"\n{m}", "api_base"),
        (
            "api_base = \"http://example.com/v1\"\napi_key = \"k\"\n{m}",
            "http",
        ),
        (
            "api_base = \"{base}\"\ndefault_model = \"m\"\nmodels = [\"other\"]",
            "default_model",
        ),
        (&format!("{base}\ncontext_window = 0"), "context_window"),
        (&format!("{base}\nkind = \"bogus\""), "bogus"),
    ];
    for (body, reason) in cases {
        write_config(&env, &work(&env), "p", body);
        let err = build(&env)
            .await
            .err()
            .unwrap_or_else(|| panic!("{body} built"));
        assert!(err.to_string().contains(reason), "{body}: {err}");
    }
    for work_dir in ["/no/such/dir", &env.path("a-file").display().to_string()] {
        write_config(&env, work_dir, "p", base);
        let err = build(&env)
            .await
            .err()
            .unwrap_or_else(|| panic!("{work_dir} built"));
        assert!(err.to_string().contains("work_dir"), "{err}");
    }
}

#[tokio::test]
async fn a_provider_named_after_an_api_speaks_it_without_a_kind() {
    let env = Env::with(Kind::Anthropic, "").await;
    let body = "api_base = \"{base}\"\napi_key = \"k\"\n{m}";
    write_config(&env, &work(&env), "anthropic", body);
    env.script([text("hi")]);

    env.session().await.say("x").await;

    assert!(env.llm.requests()[0].path.starts_with("/v1/messages"));
}

#[tokio::test]
async fn the_binary_reads_keys_from_the_environment_and_picks_models() {
    let env = Env::new().await;
    let body = "kind = \"openai\"\napi_base = \"{base}\"\nenv_var = \"E2E_KEY\"\n\
                default_model = \"m1\"\nmodels = [\"m1\", \"m2\"]";
    write_config(&env, &work(&env), "p", body);
    env.script([text("one"), text("two")]);

    let out = openheim(&env, &["run", "hello"])
        .env("E2E_KEY", "from-env")
        .output()
        .await
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = openheim(&env, &["run", "--model", "m2", "hello"])
        .env("E2E_KEY", "from-env")
        .output()
        .await
        .unwrap();
    assert!(out.status.success());

    let requests = env.llm.requests();
    assert_eq!(requests[0].headers["authorization"], "Bearer from-env");
    assert_eq!(
        (&requests[0].body["model"], &requests[1].body["model"]),
        (&json!("m1"), &json!("m2"))
    );
    let out = openheim(&env, &["run", "--model", "m3", "hello"])
        .output()
        .await
        .unwrap();
    assert!(!out.status.success());
}

#[tokio::test]
async fn init_writes_a_config_once_and_a_missing_config_is_explained() {
    let env = Env::new().await;
    let fresh = env.data().join("fresh/config.toml");
    let init = |path: &std::path::Path| {
        let mut cmd = openheim(&env, &["init"]);
        cmd.as_std_mut().args(["--config", path.to_str().unwrap()]);
        cmd
    };

    assert!(init(&fresh).output().await.unwrap().status.success());
    assert!(fresh.exists() && env.data().join("fresh/system.md").exists());
    OpenheimClient::from_config(&fresh).build().await.unwrap();
    assert!(!init(&fresh).output().await.unwrap().status.success());

    std::fs::remove_file(env.config()).unwrap();
    let out = run(&env, "x").await;
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("init"));
}

#[tokio::test]
async fn paths_resolve_from_the_cwd_and_data_can_live_elsewhere() {
    let env = Env::new().await;
    let data = env.data().join("elsewhere");
    let config = std::fs::read_to_string(env.config()).unwrap().replacen(
        &format!("work_dir = {:?}", work(&env)),
        &format!(
            "work_dir = \"sub\"\ndata_dir = {:?}",
            data.display().to_string()
        ),
        1,
    );
    std::fs::write(env.config(), config).unwrap();
    env.file("sub/a.txt", "inside sub");
    env.script([tool("read_file", json!({"path": "a.txt"})), text("done")]);

    let out = run(&env, "read").await;

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(env.llm.last().to_string().contains("inside sub"));
    assert_eq!(std::fs::read_dir(data.join("history")).unwrap().count(), 2);
}

#[tokio::test]
async fn system_md_and_skills_shape_the_system_prompt() {
    let env = Env::with(Kind::OpenAi, "default_skills = [\"base\", \"rust\"]").await;
    std::fs::create_dir_all(env.data().join("skills")).unwrap();
    std::fs::write(env.data().join("system.md"), "You are E2E-AGENT.").unwrap();
    for (name, body) in [
        ("base", "BASE-SKILL"),
        ("rust", "RUST-SKILL"),
        ("go", "GO-SKILL"),
    ] {
        std::fs::write(env.data().join(format!("skills/{name}.md")), body).unwrap();
    }
    std::os::unix::fs::symlink("/etc/hosts", env.data().join("skills/escape.md")).unwrap();
    env.script([text("a")]);
    let client = env.client().await;
    let listed = client.skills().unwrap();
    assert!(
        ["base", "go", "rust"]
            .iter()
            .all(|s| listed.contains(&s.to_string()))
    );

    let skills = vec!["go".to_string(), "rust".to_string()];
    let handle = client
        .new_session()
        .skills(skills)
        .cwd(&env.work)
        .start()
        .await
        .unwrap();
    handle.prompt("x", |_| {}).await.unwrap();

    let system = env.llm.last()["messages"][0].to_string();
    let at = |s: &str| {
        system
            .find(s)
            .unwrap_or_else(|| panic!("{s} missing: {system}"))
    };
    assert!(at("E2E-AGENT") < at("BASE-SKILL") && at("BASE-SKILL") < at("RUST-SKILL"));
    assert_eq!(system.matches("RUST-SKILL").count(), 1);
    assert!(system.contains("GO-SKILL"));
    for bad in ["../system", "escape"] {
        let start = client.new_session().skills(vec![bad.into()]).start().await;
        assert!(start.is_err(), "{bad}");
    }
}

/// Reports the directories its turn hands it.
struct WhereAmI;

#[async_trait::async_trait]
impl openheim::tools::ToolHandler for WhereAmI {
    fn definition(&self) -> openheim::Tool {
        openheim::Tool::function(
            "where_am_i",
            "Reports its directories",
            json!({"type": "object"}),
        )
    }

    async fn execute(
        &self,
        _args: &str,
        turn: &openheim::core::turn::TurnContext<'_>,
    ) -> openheim::Result<String> {
        Ok(format!(
            "work={} cwd={}",
            turn.work_dir.display(),
            turn.cwd.display()
        ))
    }
}

#[tokio::test]
async fn an_embedder_configures_everything_in_code() {
    let env = Env::new().await;
    let mcp = crate::common::mcp::start().await;
    env.script([
        tool("where_am_i", json!({})),
        tool("stub__echo", json!({"text": "hi"})),
        text("ok"),
    ]);

    let client = OpenheimClient::builder()
        .provider("fake")
        .api_base(env.llm.api_base())
        .api_key("code-key")
        .model("code-model")
        .max_tokens(77)
        .work_dir(&env.work)
        .data_dir(env.data().join("data"))
        .mcp_server("stub", openheim::McpServerConfig::http(mcp))
        .tool(Box::new(WhereAmI))
        .build()
        .await
        .unwrap();
    let handle = client.new_session().cwd(&env.work).start().await.unwrap();
    handle.prompt("x", |_| {}).await.unwrap();

    let first = &env.llm.requests()[0];
    assert_eq!(first.headers["authorization"], "Bearer code-key");
    assert_eq!(
        (&first.body["model"], &first.body["max_tokens"]),
        (&json!("code-model"), &json!(77))
    );
    let last = env.llm.last().to_string();
    let work = env.work.display().to_string();
    assert!(last.contains(&format!("work={work} cwd={work}")), "{last}");
    assert!(env.data().join("data/history").exists());

    let mixed = OpenheimClient::builder()
        .app_config(openheim::AppConfig::new("fake"))
        .provider("fake")
        .build()
        .await;
    assert!(mixed.is_err());
}
