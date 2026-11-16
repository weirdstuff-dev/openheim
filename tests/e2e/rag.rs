//! Long-term memory through its tools: keyword search out of the box,
//! semantic search with an embedding provider, and outages.

use serde_json::json;

use crate::common::{Env, Kind};

const EMBEDDINGS: &str = r#"[memory]
embedding_provider = "fake"
embedding_model = "emb-1""#;

async fn remember(env: &Env, note: &str) {
    let saved = env.ok("remember", json!({"content": note})).await;
    assert!(saved.starts_with("Remembered as memory #"), "{saved}");
}

async fn search(env: &Env, query: &str) -> String {
    env.ok("search_memory", json!({"query": query})).await
}

#[tokio::test]
async fn notes_are_remembered_edited_and_forgotten_across_clients() {
    let env = Env::new().await;
    assert!(search(&env, "anything").await.contains("empty"));
    remember(&env, "The deploy key lives in vault").await;
    remember(&env, "Lunch is at noon").await;
    assert!(env.data().join("memory.db").exists());

    let found = search(&env, "vault").await;
    assert!(found.contains("[#1]") && found.contains("deploy key") && !found.contains("Lunch"));
    assert!(found.contains("rank"), "{found}");

    let edited = env
        .ok(
            "edit_memory",
            json!({"id": 1, "content": "The key is in the safe"}),
        )
        .await;
    assert_eq!(edited, "Updated memory #1.");
    assert!(search(&env, "vault").await.contains("No relevant memories"));
    assert!(search(&env, "safe").await.contains("[#1]"));

    assert_eq!(
        env.ok("forget", json!({"id": 1})).await,
        "Forgot memory #1."
    );
    assert!(
        env.ok("forget", json!({"id": 1}))
            .await
            .contains("No memory #1")
    );
    assert!(
        env.ok("edit_memory", json!({"id": 9, "content": "x"}))
            .await
            .contains("No memory #9")
    );
    env.err("remember", json!({"content": "  "})).await;
    env.err("remember", json!({"content": "x".repeat(100_000)}))
        .await;
    env.err("search_memory", json!({})).await;
    assert!(env.llm.embedded().is_empty());
}

#[tokio::test]
async fn embeddings_make_search_semantic() {
    let env = Env::with(Kind::OpenAi, EMBEDDINGS).await;
    remember(&env, "apples and bananas").await;
    remember(&env, "zebra quiz").await;

    let found = search(&env, "banana").await;

    assert!(found.contains("similarity"), "{found}");
    let (apples, zebra) = (found.find("apples").unwrap(), found.find("zebra").unwrap());
    assert!(apples < zebra, "{found}");
    assert!(
        env.llm
            .embedded()
            .contains(&"apples and bananas".to_string())
    );
    let one = env
        .ok("search_memory", json!({"query": "banana", "top_k": 1}))
        .await;
    assert!(!one.contains("zebra"));
}

#[tokio::test]
async fn an_embeddings_outage_falls_back_to_keywords_and_backfills_later() {
    let env = Env::with(Kind::OpenAi, EMBEDDINGS).await;
    env.llm.set_embeddings_down(true);
    remember(&env, "apples and bananas").await;
    env.ok(
        "edit_memory",
        json!({"id": 1, "content": "apples and bananas!"}),
    )
    .await;

    let during = search(&env, "apples").await;
    assert!(
        during.contains("keyword search") && during.contains("apples"),
        "{during}"
    );

    env.llm.set_embeddings_down(false);
    let after = search(&env, "banana").await;
    assert!(
        after.contains("similarity") && after.contains("apples"),
        "{after}"
    );
    assert!(
        env.llm
            .embedded()
            .contains(&"apples and bananas!".to_string())
    );
}

#[tokio::test]
async fn keyword_notes_are_embedded_once_embeddings_are_configured() {
    let env = Env::new().await;
    remember(&env, "apples and bananas").await;
    let config = std::fs::read_to_string(env.config()).unwrap();
    let embedded_config = format!("{config}\n{EMBEDDINGS}\n");
    let switched = embedded_config.replace("emb-1", "emb-2");

    for config in [embedded_config, switched] {
        std::fs::write(env.config(), config).unwrap();
        let before = env.llm.embedded().len();
        let found = search(&env, "banana").await;
        assert!(
            found.contains("similarity") && found.contains("apples"),
            "{found}"
        );
        assert!(env.llm.embedded().len() > before + 1, "not re-embedded");
    }
}
