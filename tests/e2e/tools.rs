//! The built-in tools against a real temp workspace, as the model calls
//! them.

use std::os::unix::fs::PermissionsExt;

use serde_json::json;

use crate::common::{Env, Kind, text, tool};

const SHELL: &str = "allow_shell = true";

#[tokio::test]
async fn read_file_reads_line_ranges_and_refuses_bad_input() {
    let env = Env::new().await;
    env.file("a.txt", "one\ntwo\nthree");
    std::fs::write(env.path("bin"), [0xff, 0xfe, 0x00]).unwrap();
    env.file("big.txt", &"line of text\n".repeat(20_000));

    let all = env.ok("read_file", json!({"path": "a.txt"})).await;
    assert!(all.contains("one") && all.contains("three"));
    let middle = env
        .ok(
            "read_file",
            json!({"path": "a.txt", "offset": 2, "limit": 1}),
        )
        .await;
    assert!(middle.contains("two") && !middle.contains("one") && !middle.contains("three"));
    let past = env
        .ok("read_file", json!({"path": "a.txt", "offset": 10}))
        .await;
    assert!(past.contains("3 lines"), "{past}");
    let big = env.ok("read_file", json!({"path": "big.txt"})).await;
    assert!(big.len() < 110_000, "{} bytes", big.len());

    for args in [
        json!({"path": "missing.txt"}),
        json!({"path": "a.txt", "offset": 0}),
        json!({"path": "a.txt", "limit": 0}),
        json!({"path": "bin"}),
        json!({}),
    ] {
        env.err("read_file", args).await;
    }
}

#[tokio::test]
async fn write_file_creates_overwrites_and_keeps_file_identity() {
    let env = Env::new().await;
    env.ok("write_file", json!({"path": "d/e/f.txt", "content": "new"}))
        .await;
    assert_eq!(env.read("d/e/f.txt"), "new");

    env.file("script.sh", "old");
    std::fs::set_permissions(env.path("script.sh"), PermissionsExt::from_mode(0o755)).unwrap();
    std::fs::hard_link(env.path("script.sh"), env.path("link.sh")).unwrap();
    std::os::unix::fs::symlink("script.sh", env.path("sym.sh")).unwrap();
    env.ok(
        "write_file",
        json!({"path": "sym.sh", "content": "replaced"}),
    )
    .await;
    assert_eq!(env.read("link.sh"), "replaced");
    assert!(env.path("sym.sh").is_symlink());
    let mode = std::fs::metadata(env.path("script.sh"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o755);

    let long = format!("{}.txt", "n".repeat(250));
    env.ok("write_file", json!({"path": long, "content": "x"}))
        .await;

    env.file("ro.txt", "keep");
    std::fs::set_permissions(env.path("ro.txt"), PermissionsExt::from_mode(0o444)).unwrap();
    env.err("write_file", json!({"path": "ro.txt", "content": "no"}))
        .await;
    assert_eq!(env.read("ro.txt"), "keep");

    let names: Vec<_> = std::fs::read_dir(&env.work)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(
        names.iter().all(|n| !n.contains(".tmp")),
        "temp file left: {names:?}"
    );
    env.err("write_file", json!({"path": "x.txt"})).await;
}

#[tokio::test]
async fn edit_file_replaces_exactly_what_was_asked() {
    let env = Env::new().await;
    env.file("a.rs", "let a = 1;\nlet b = 1;\n");
    let edit = |old: &str, new: &str, all: bool| json!({"path": "a.rs", "old_string": old, "new_string": new, "replace_all": all});

    env.ok("edit_file", edit("a = 1", "a = 2", false)).await;
    assert_eq!(env.read("a.rs"), "let a = 2;\nlet b = 1;\n");
    // Not found, ambiguous, empty, unchanged.
    for (old, new) in [("missing", "x"), ("let", "var"), ("", "x"), ("a", "a")] {
        env.err("edit_file", edit(old, new, false)).await;
    }
    env.ok("edit_file", edit("let", "var", true)).await;
    assert_eq!(env.read("a.rs"), "var a = 2;\nvar b = 1;\n");
    env.err(
        "edit_file",
        json!({"path": "nope.rs", "old_string": "a", "new_string": "b"}),
    )
    .await;
}

#[tokio::test]
async fn list_dir_lists_sorted_and_shows_links() {
    let env = Env::new().await;
    env.file("b.txt", "").file("a/inner.txt", "");
    std::os::unix::fs::symlink("b.txt", env.path("c")).unwrap();

    let listing = env.ok("list_dir", json!({})).await;
    let (a, b, c) = (
        listing.find("a/").unwrap(),
        listing.find("b.txt").unwrap(),
        listing.find("c").unwrap(),
    );
    assert!(a < b && b < c, "{listing}");
    assert!(listing.contains("-> b.txt"), "{listing}");
    assert!(
        env.ok("list_dir", json!({"path": "a"}))
            .await
            .contains("inner.txt")
    );
    env.err("list_dir", json!({"path": "missing"})).await;
}

#[tokio::test]
async fn search_finds_matches_and_respects_gitignore() {
    let env = Env::new().await;
    std::fs::create_dir(env.path(".git")).unwrap();
    env.file(".gitignore", "ignored.txt\n")
        .file("ignored.txt", "secret needle\n")
        .file("src/kept.txt", "visible needle\nNEEDLE upper\n");

    let found = env.ok("search", json!({"pattern": "needle"})).await;
    assert!(
        found.contains("kept.txt") && !found.contains("ignored.txt"),
        "{found}"
    );
    assert!(!found.contains("upper"));
    let any_case = env
        .ok(
            "search",
            json!({"pattern": "needle", "case_insensitive": true}),
        )
        .await;
    assert!(any_case.contains("upper"));
    let none = env
        .ok("search", json!({"pattern": "absent", "path": "src"}))
        .await;
    assert!(!none.contains("kept.txt"));
    env.err("search", json!({"pattern": "("})).await;
}

#[tokio::test]
async fn paths_cannot_leave_the_work_dir() {
    let env = Env::new().await;
    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("secret.txt");
    std::fs::write(&secret, "TOP-SECRET-CONTENT").unwrap();
    std::os::unix::fs::symlink(&secret, env.path("link")).unwrap();
    std::os::unix::fs::symlink(outside.path().join("gone"), env.path("dangling")).unwrap();

    let secret = secret.to_str().unwrap();
    for (tool_name, args) in [
        ("read_file", json!({"path": secret})),
        ("read_file", json!({"path": "../secret.txt"})),
        ("read_file", json!({"path": "link"})),
        ("write_file", json!({"path": "dangling", "content": "x"})),
        (
            "write_file",
            json!({"path": "nope/../../x", "content": "x"}),
        ),
        (
            "edit_file",
            json!({"path": secret, "old_string": "s", "new_string": "t"}),
        ),
        ("list_dir", json!({"path": outside.path()})),
        (
            "search",
            json!({"pattern": "secret", "path": outside.path()}),
        ),
    ] {
        let error = env.err(tool_name, args.clone()).await;
        assert!(!error.contains("TOP-SECRET"), "{tool_name} {args}: {error}");
    }
    assert_eq!(
        std::fs::read_to_string(secret).unwrap(),
        "TOP-SECRET-CONTENT"
    );
    assert!(!outside.path().join("x").exists() && !env.data().join("x").exists());
}

#[tokio::test]
async fn execute_command_runs_in_the_session_cwd_and_reports_failure() {
    let env = Env::with(Kind::OpenAi, SHELL).await;

    let pwd = env.ok("execute_command", json!({"command": "pwd"})).await;
    assert_eq!(pwd.trim(), env.work.to_str().unwrap());
    let both = env
        .ok(
            "execute_command",
            json!({"command": "echo out; echo err >&2"}),
        )
        .await;
    assert!(both.contains("out") && both.contains("err"), "{both}");
    let failed = env
        .err(
            "execute_command",
            json!({"command": "echo partial; exit 3"}),
        )
        .await;
    assert!(failed.contains("partial"), "{failed}");
    // Whether the writer dies of the closed pipe or finishes first is a
    // race, so only the cut is checked, not the exit status.
    let (big, _) = env
        .call(
            "execute_command",
            json!({"command": "printf '%0100000d' 0"}),
        )
        .await;
    assert!(
        big.contains("truncated") && big.len() < 70_000,
        "{} bytes",
        big.len()
    );

    let started = std::time::Instant::now();
    env.ok(
        "execute_command",
        json!({"command": "sleep 30 & echo started"}),
    )
    .await;
    assert!(started.elapsed().as_secs() < 10);
}

#[tokio::test]
async fn the_shell_is_offered_only_when_allowed() {
    for (extra, offered) in [("", false), (SHELL, true)] {
        let env = Env::with(Kind::OpenAi, extra).await;
        env.script([text("ok")]);
        env.session().await.say("x").await;
        let tools = env.llm.last()["tools"].to_string();
        assert_eq!(tools.contains("execute_command"), offered, "{extra:?}");
    }
    let env = Env::new().await;
    env.script([
        tool("execute_command", json!({"command": "touch x"})),
        text("ok"),
    ]);
    env.session().await.say("x").await.err_result();
    assert!(!env.path("x").exists());
}

#[tokio::test]
async fn web_fetch_refuses_local_and_private_addresses() {
    let env = Env::new().await;
    for url in [
        "file:///etc/passwd",
        "http://127.0.0.1:1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
    ] {
        env.err("web_fetch", json!({"url": url})).await;
    }
}

/// Models send `null` for an optional argument they leave unset.
#[tokio::test]
async fn optional_arguments_accept_null() {
    let env = Env::with(Kind::OpenAi, SHELL).await;
    env.file("a.txt", "abc");
    for (tool_name, args) in [
        (
            "read_file",
            json!({"path": "a.txt", "offset": null, "limit": null}),
        ),
        ("list_dir", json!({"path": null})),
        (
            "search",
            json!({"pattern": "abc", "path": null, "case_insensitive": null}),
        ),
        (
            "edit_file",
            json!({"path": "a.txt", "old_string": "abc", "new_string": "abd", "replace_all": null}),
        ),
        ("search_memory", json!({"query": "x", "top_k": null})),
    ] {
        env.ok(tool_name, args).await;
    }
    env.script([
        tool(
            "delegate_task",
            json!({"agent": null, "system_prompt": "Sub.", "tools": null, "model": null,
                   "provider": null, "max_iterations": null, "task": "t"}),
        ),
        text("sub done"),
        text("ok"),
    ]);
    let turn = env.session().await.say("x").await;
    assert!(turn.ok_result().contains("sub done"));
}
