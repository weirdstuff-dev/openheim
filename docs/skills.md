# Skills & Agent Identity

This document covers two related concepts:

- **System identity** (`~/.openheim/system.md`) — the agent's base character, loaded on every session.
- **Skills** (`~/.openheim/skills/*.md`) — domain-specific instructions injected on demand.

---

## Agent Identity (`system.md`)

`~/.openheim/system.md` defines who the agent is. It is loaded at the start of every session and is **required** — openheim will error on startup if the file is missing.

Run `openheim init` to create it with a default starting point, then edit it to match how you want the agent to behave:

```markdown
You are a senior software engineer specialising in Rust and distributed systems.
You write clean, idiomatic code and prefer simple solutions over clever ones.
Ask clarifying questions before making large changes.
```

The identity is placed at the top of the system message, before any skills.

---

## Skills

Skills are Markdown files that inject domain-specific instructions into the system prompt. They let you shape the agent's behaviour — tone, domain expertise, constraints, output format — without touching code or configuration.

### Where skills live

Skills are stored as `.md` files in `~/.openheim/skills/`:

```
~/.openheim/skills/
├── rust.md
├── tdd.md
└── concise.md
```

The filename (without extension) is the skill name. Names are case-sensitive and must not contain spaces — use hyphens or underscores instead. Names must also be a single path component: `/`, `\`, and `..` are rejected, and a skill file must resolve inside the skills directory (a symlink pointing elsewhere is refused). Skill names come from config, the CLI, and remote ACP clients, so they are validated as untrusted input.

---

## How the system message is assembled

When a session has an identity and skills, the LLM receives a single system message structured like this:

```
You are a general purpose multiprovider LLM agent.

---

The user has given you the following identity:

<system.md content>

---

These are the skills you have mastered:

### rust

<rust.md content>

---

### tdd

<tdd.md content>
```

The system message is reconstructed on every LLM call in the session, so it is always present regardless of how long the conversation runs.

---

## Writing a skill file

A skill file is plain Markdown. Write it as instructions to the model — directive sentences work better than descriptive ones.

**Good:**
```markdown
You are an expert in systems programming with deep knowledge of Rust and C.

- Always check for integer overflow in arithmetic operations.
- Prefer stack allocation over heap allocation where possible.
- When suggesting unsafe code, explain exactly why it is safe.
- Output code in fenced blocks with the language tag set.
```

**Less effective:**
```markdown
This skill is about systems programming. The assistant knows Rust and C.
```

### Tips

- **Be specific about output format.** If you want JSON, say so. If you want short answers, say so.
- **State constraints explicitly.** "Never use `unwrap()` in library code" is more useful than "write safe code".
- **Keep each skill focused.** Compose narrow skills rather than writing one mega-skill. A `rust.md` + `tdd.md` combo is easier to maintain and reuse than a single `rust-with-tdd.md`.
- **Test with `openheim run`.** Quickly validate a skill's effect: `openheim run --skills rust "What is the idiomatic way to handle errors in Rust?"`

---

## Enabling skills

### Default skills (loaded every session)

Set `default_skills` in `~/.openheim/config.toml` to load skills automatically in every new session — no `--skills` flag needed:

```toml
default_skills = ["rules", "concise"]
```

Default skills are merged with any per-session skills. Duplicates are deduplicated; defaults always appear first.

### Per-session via CLI

Pass `--skills` as a comma-separated list of skill names:

```bash
openheim --skills rust,tdd
openheim run --skills concise "Summarise the project structure"
```

### Via the Rust library

Pass skill names when creating a session:

```rust
let session = client
    .new_session()
    .skills(vec!["rust".into(), "tdd".into()])
    .start()
    .await?;
```

Skills are persisted in the conversation metadata (`ConversationMeta.skills`), so they are restored when you resume a session with `load_session`.

---

## Listing available skills

```bash
# API (REST when running openheim serve)
curl http://localhost:1217/api/skills
# → ["concise","rust","tdd"]
```

Via the Rust library:

```rust
let skills = client.memory().skills.list_skills()?;
// → ["concise", "rust", "tdd"]
```

---

## Example files

### `~/.openheim/system.md`

```markdown
You are a pragmatic software engineer who writes clean, well-tested code.
You prefer simple solutions and always consider the maintenance burden of your suggestions.
When you are uncertain, say so rather than guessing.
```

### `~/.openheim/skills/concise.md`

```markdown
Be concise. Answer in as few words as possible without omitting essential information.
Do not add closing summaries or sign-offs. Stop as soon as the answer is complete.
```

### `~/.openheim/skills/rust.md`

```markdown
You are an expert Rust programmer targeting Rust 2024 edition.

- Prefer `?` over `unwrap()` in all non-test code.
- Use `thiserror` for library errors and `anyhow` for application errors.
- Avoid unnecessary heap allocation; prefer iterators over collecting into `Vec` when the result is immediately consumed.
- When writing async code, use `tokio` and annotate blocking calls with `spawn_blocking`.
- Always include `#[must_use]` on types and functions where ignoring the return value is a likely mistake.
```

### `~/.openheim/skills/security-review.md`

```markdown
You are performing a security review. For every code change you analyse:

1. Check for injection vulnerabilities (SQL, shell, path traversal).
2. Check for insecure defaults (plaintext secrets, world-readable files).
3. Check for missing input validation at trust boundaries.
4. If you find no issues, say so explicitly rather than staying silent.

Output findings as a numbered list. For each finding include: severity (critical/high/medium/low), location, and a one-sentence remediation.
```
