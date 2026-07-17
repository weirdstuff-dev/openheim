# Subagents

Subagents let the agent delegate a self-contained task to another, independently
configured agent — its own persona, its own model/provider, optionally its own
restricted tool set — and get back only the final answer. They're useful for
splitting work across specialised personas (a reviewer, a researcher, a
fact-checker, …) or for routing routine subtasks to a cheaper/faster model while
the orchestrator runs on a stronger one.

This mirrors how [skills](skills.md) work — drop a Markdown file in a directory and
it's picked up automatically — except a subagent profile defines a *whole separate
agent* rather than a snippet appended to the current one.

Subagents come in two flavours:

- **Named profiles** — `.md` files in `~/.openheim/agents/`, written by you ahead
  of time.
- **Inline subagents** — defined by the orchestrating agent itself, on the fly, in
  the `delegate_task` call (a `system_prompt` plus optional overrides). These are
  ephemeral: they exist only for that one call and are never persisted.

---

## Where subagent profiles live

Profiles are `.md` files in `~/.openheim/agents/`:

```
~/.openheim/agents/
├── code-reviewer.md
├── researcher.md
└── fact-checker.md
```

The filename (without extension) is the subagent's name. Names are case-sensitive
and must not contain spaces.

The `delegate_task` tool's description lists every available profile by name and
description, so the LLM knows what's available and when to reach for it. The tool
is always present — even with no profiles configured, the orchestrator can still
spin up inline subagents (see below).

---

## Writing a profile

A profile file is Markdown with an optional `+++`-delimited TOML frontmatter block,
followed by the subagent's system prompt:

```markdown
+++
description = "Reviews code changes for correctness, security, and style. Use for a second opinion on a diff."
model = "claude-haiku-4-5"
provider = "anthropic"
tools = ["read_file"]
max_iterations = 6
+++
You are a meticulous code reviewer focused on correctness, security, and idiomatic style.
Always cite file:line references for issues you find.
```

All frontmatter fields are optional:

| Field | Type | Effect |
|---|---|---|
| `description` | string | Shown to the orchestrator so it knows when to delegate to this subagent. Omitting it is allowed but makes the subagent harder for the LLM to discover. |
| `model` | string | Overrides the parent's model for this subagent's run. |
| `provider` | string | Overrides the parent's provider. Only used when `model` is also set. |
| `tools` | array of strings | Restricts the subagent to this set of tool names. Omitted = inherits the full tool set the parent has access to. |
| `max_iterations` | integer | Caps the subagent's agent-loop iterations. Omitted = inherits the parent's `max_iterations`. |

A file with no frontmatter is treated as a profile with an empty description whose
entire contents are the system prompt. A profile whose frontmatter fails to parse is
skipped (with a warning logged) rather than preventing startup.

---

## Inline subagents

The orchestrating agent can also create a subagent on the fly, without any profile
file, by calling `delegate_task` with a `system_prompt` instead of an `agent` name:

```json
{
  "system_prompt": "You are a fact-checker. Respond with a verdict and a one-paragraph explanation.",
  "tools": ["read_file"],
  "model": "claude-haiku-4-5",
  "max_iterations": 4,
  "task": "Verify the claim: water boils at 100°C at sea level."
}
```

`tools`, `model`, `provider`, and `max_iterations` are optional and mean exactly
what the corresponding profile frontmatter fields mean. Exactly one of `agent` or
`system_prompt` must be provided.

Inline subagents are **ephemeral by design**: nothing is written to
`~/.openheim/agents/` and nothing survives the call. They run through the same
machinery as named profiles, so all the sandboxing, permission, and no-recursion
guarantees below apply identically.

---

## How a delegated task runs

When the orchestrator calls `delegate_task` with a `task` and either an `agent`
name or an inline `system_prompt`, openheim:

1. Resolves the subagent's model/provider/iteration cap from the profile (falling
   back to the parent's configuration for anything left unset).
2. Builds a tool executor scoped to the profile's `tools` allowlist (if any) and
   wrapped in the **same sandbox boundary** (`work_dir` / `allow_shell`) as the
   parent — subagents cannot escalate privileges beyond what the parent has.
3. Starts a **fresh, isolated** agent run: its own message history (just the `task`
   as the first user message) and its own system prompt (the profile's persona —
   not the parent's `system.md` or skills) — but the **same cancellation token
   and permission gate** as the orchestrating turn. Every tool call the subagent
   makes is checked exactly like the orchestrator's own (e.g. an interactive
   client still sees `session/request_permission` for a subagent's shell
   command), and a `session/cancel` on the parent turn stops an in-flight
   subagent too. There is no separate, more-trusting policy for subagents.
4. Runs that agent loop to completion and returns only its final answer to the
   orchestrator. Intermediate tool calls and reasoning are not visible to the
   parent — the subagent genuinely runs "in the background", much like Claude
   Code's `Task` subagents.

Because the subagent is built from a tool executor that does not include
`delegate_task` itself, subagents cannot delegate further — there is no recursive
chain to worry about.

> **Tip:** since the subagent cannot see the parent conversation, write `task`
> descriptions in the calling prompt (or trust the orchestrator to write them) as
> complete, self-contained briefs. The LLM is told this in the tool description, but
> it helps to reinforce it in your `system.md` or skills if you rely on subagents
> heavily.

---

## Example

`~/.openheim/agents/fact-checker.md`:

```markdown
+++
description = "Verifies a specific factual claim and reports whether it's accurate, with sources if possible."
max_iterations = 4
+++
You are a fact-checker. Given a claim, determine whether it is accurate.
Respond with a verdict (Accurate / Inaccurate / Uncertain) followed by a one-paragraph explanation.
Be concise — you have a limited number of steps to reach a conclusion.
```

With this file in place, asking the orchestrator something like "use the
fact-checker subagent to verify that water boils at 100°C at sea level" causes it to
call `delegate_task` with `{"agent": "fact-checker", "task": "Verify the claim: water boils at 100°C at sea level."}`
and relay the fact-checker's verdict back to you.
