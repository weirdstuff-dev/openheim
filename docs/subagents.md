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

If the directory contains at least one valid profile, the orchestrating agent gains
a `delegate_task` tool whose description lists every available subagent by name and
description, so the LLM knows what's available and when to reach for it. If the
directory is empty, `delegate_task` is not added — there's no overhead for agents
that don't use subagents.

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

## How a delegated task runs

When the orchestrator calls `delegate_task` with an `agent` name and a `task`
description, openheim:

1. Resolves the subagent's model/provider/iteration cap from the profile (falling
   back to the parent's configuration for anything left unset).
2. Builds a tool executor scoped to the profile's `tools` allowlist (if any) and
   wrapped in the **same sandbox boundary** (`work_dir` / `allow_shell`) as the
   parent — subagents cannot escalate privileges beyond what the parent has.
3. Starts a **fresh, isolated** agent run: its own message history (just the `task`
   as the first user message) and its own system prompt (the profile's persona —
   not the parent's `system.md` or skills).
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
