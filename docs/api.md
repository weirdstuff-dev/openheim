# Openheim API Specification

> **Version:** 0.1.0
> **Base URL:** `http://{host}:{port}` (default `0.0.0.0:1217`)
> **Protocol:** REST + multiplexed WebSocket over ACP (Agent Client Protocol)

This document describes every HTTP and WebSocket endpoint the Openheim server exposes. It is intended for the frontend team to implement a complete UI client.

---

## Table of Contents

1. [Overview](#1-overview)
2. [REST API](#2-rest-api)
   - [GET /api/config](#21-get-apiconfig)
   - [GET /api/models](#22-get-apimodels)
   - [GET /api/skills](#23-get-apiskills)
   - [GET /api/tools](#24-get-apitools)
   - [GET /api/mcp-servers](#25-get-apimcp-servers)
   - [GET /api/sessions](#26-get-apisessions)
   - [GET /api/sessions/:id](#27-get-apisessionsid)
3. [WebSocket — Multiplexed Connection](#3-websocket--multiplexed-connection)
   - [Wire Format](#31-wire-format)
   - [Agent Channel (ACP)](#32-agent-channel-acp)
     - [Initialize](#321-initialize)
     - [Create Session](#322-create-session)
     - [List Sessions](#323-list-sessions)
     - [Load Session](#324-load-session)
     - [Send Prompt](#325-send-prompt)
     - [Streaming Updates (Server → Client)](#326-streaming-updates-server--client)
   - [Filesystem Channel](#33-filesystem-channel)
     - [Watch / Unwatch](#331-watch--unwatch)
     - [List Directory](#332-list-directory)
     - [Read File](#333-read-file)
     - [Write File](#334-write-file)
     - [Create Directory](#335-create-directory)
     - [Delete](#336-delete)
     - [Rename / Move](#337-rename--move)
     - [Filesystem Events (Server → Client)](#338-filesystem-events-server--client)
4. [TypeScript Interfaces](#4-typescript-interfaces)
5. [Sequence Diagrams](#5-sequence-diagrams)
6. [Error Handling](#6-error-handling)

---

## 1. Overview

Openheim exposes a single multiplexed WebSocket at `/ws` and a small set of REST endpoints at `/api/*`.

The WebSocket carries two logical channels over one physical connection:

| Channel | Purpose |
|---|---|
| **agent** | ACP (Agent Client Protocol) — initialize, create sessions, send prompts, receive streamed LLM responses + tool call updates |
| **fs** | Filesystem operations — CRUD, directory listing, live file watching |

All WS messages are JSON envelopes tagged with a `channel` field so the client can route them without opening multiple connections.

---

## 2. REST API

### 2.1 `GET /api/config`

Returns the public server configuration. API keys and sensitive env vars are stripped/redacted.

**Response `200`:**

```json
{
  "default_provider": "openai",
  "max_iterations": 10,
  "providers": {
    "openai": {
      "api_base": "https://api.openai.com/v1",
      "default_model": "gpt-4",
      "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
      "env_var": "OPENAI_API_KEY",
      "timeout_secs": 120,
      "max_tokens": 4096
    },
    "anthropic": {
      "api_base": "https://api.anthropic.com/v1",
      "default_model": "claude-3-5-sonnet-20241022",
      "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
      "env_var": "ANTHROPIC_API_KEY"
    }
  },
  "mcp_servers": {}
}
```

> **Note:** `api_key` fields are never included. `env` values inside `mcp_servers` are replaced with `"<redacted>"`.

---

### 2.2 `GET /api/models`

Returns available models grouped by provider.

**Response `200`:**

```json
{
  "default_provider": "openai",
  "providers": {
    "openai": {
      "default_model": "gpt-4",
      "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    }
  }
}
```

---

### 2.3 `GET /api/skills`

Returns a sorted list of installed skill names (loaded from `~/.openheim/skills/*.md`).

**Response `200`:**

```json
["debugging", "rust", "vue"]
```

---

### 2.4 `GET /api/tools`

Returns all registered tool definitions (built-in + MCP). Each tool follows the OpenAI function-calling schema.

**Response `200`:**

```json
[
  {
    "type": "function",
    "function": {
      "name": "execute_command",
      "description": "Execute a shell command (e.g., ls, pwd, echo). Use this for listing directories and running system commands.",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The shell command to execute"
          }
        },
        "required": ["command"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read the contents of a file at the specified path.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The path to the file to read"
          }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Write content to a file at the specified path. Creates the file if it doesn't exist.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The path to the file to write"
          },
          "content": {
            "type": "string",
            "description": "The content to write to the file"
          }
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "filesystem__read_file",
      "description": "... (from MCP server)",
      "parameters": { "..." : "..." }
    }
  }
]
```

> MCP tools are names-spaced as `{server_name}__{tool_name}` (double underscore). The server name is sanitized: hyphens and spaces become underscores.

---

### 2.5 `GET /api/mcp-servers`

Returns the connection status of all configured MCP servers.

**Response `200`:**

```json
[
  {
    "name": "filesystem",
    "transport": "stdio",
    "command": "npx",
    "url": null,
    "connected": true,
    "tool_count": 3,
    "error": null
  },
  {
    "name": "remote-tools",
    "transport": "http",
    "command": null,
    "url": "http://localhost:8080/mcp",
    "connected": false,
    "tool_count": 0,
    "error": "connection refused"
  }
]
```

---

### 2.6 `GET /api/sessions`

Returns a list of all persisted conversation sessions, sorted newest-first by `updated_at`. This is the REST equivalent of `session/list` over ACP.

**Response `200`:**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-05-10T14:22:15Z",
    "updated_at": "2025-05-10T14:35:42Z",
    "title": "Refactor the auth module",
    "model": "gpt-4-turbo",
    "provider": "openai",
    "skills": ["rust", "debugging"],
    "cwd": "/home/user/my-project"
  },
  {
    "id": "661f9511-f3ac-52e5-b827-557766551111",
    "created_at": "2025-05-09T09:00:00Z",
    "updated_at": "2025-05-09T09:15:30Z",
    "title": "Explain the project structure",
    "model": null,
    "provider": null,
    "skills": [],
    "cwd": null
  }
]
```

| Field | Type | Description |
|---|---|---|
| `id` | `string` (UUID) | Unique session identifier — use as `sessionId` in ACP calls |
| `created_at` | `string` (ISO 8601) | When the session was created |
| `updated_at` | `string` (ISO 8601) | When the session was last active |
| `title` | `string \| null` | Auto-generated from the first user message (up to 80 chars) |
| `model` | `string \| null` | Model used in this session |
| `provider` | `string \| null` | Provider used in this session |
| `skills` | `string[]` | Skills loaded for this session |
| `cwd` | `string \| null` | Working directory — populated after the first prompt in the session |

> Sessions are persisted to `~/.openheim/history/{uuid}.json` and survive server restarts.

---

### 2.7 `GET /api/sessions/:id`

Returns the full conversation for a session, including all messages.

**Path parameter:** `:id` — the UUID of the session.

**Response `200`:**

```json
{
  "meta": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-05-10T14:22:15Z",
    "updated_at": "2025-05-10T14:35:42Z",
    "title": "Refactor the auth module",
    "model": "gpt-4-turbo",
    "provider": "openai",
    "skills": ["rust"],
    "cwd": "/home/user/my-project"
  },
  "messages": [
    {
      "role": "user",
      "content": "Refactor the auth module to use JWTs.",
      "tool_calls": null,
      "tool_call_id": null,
      "tool_name": null
    },
    {
      "role": "assistant",
      "content": "I'll help you refactor the auth module...",
      "tool_calls": null,
      "tool_call_id": null,
      "tool_name": null
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "function": {
            "name": "read_file",
            "arguments": "{\"path\": \"src/auth.rs\"}"
          }
        }
      ],
      "tool_call_id": null,
      "tool_name": null
    },
    {
      "role": "tool",
      "content": "use actix_web::...\n// file contents",
      "tool_calls": null,
      "tool_call_id": "call_abc123",
      "tool_name": "read_file"
    }
  ]
}
```

**Message roles:**

| `role` | Description |
|---|---|
| `"user"` | Message sent by the human |
| `"assistant"` | LLM response text or tool call request |
| `"tool"` | Tool execution result fed back to the LLM |
| `"system"` | System prompt injected by the agent (skills, context) |

**Error `400`** — if `:id` is not a valid UUID:

```json
{ "error": "invalid session id" }
```

**Error `404`** — if the session does not exist:

```json
{ "error": "Conversation 550e8400-... not found at ..." }
```

---

## 3. WebSocket — Multiplexed Connection

### Endpoint

```
WS ws://{host}:{port}/ws
```

### 3.1 Wire Format

Every message in **both directions** is a JSON envelope:

**Client → Server (inbound):**

```json
{
  "channel": "agent",   // "agent" | "fs"
  "data": { ... }       // channel-specific payload
}
```

**Server → Client (outbound):**

```json
{
  "channel": "system",  // "system" | "agent" | "fs"
  "data": { ... }
}
```

On connection, the server immediately sends:

```json
{
  "channel": "fs",
  "data": {
    "type": "connected",
    "message": "Connected to Openheim"
  }
}
```

---

### 3.2 Agent Channel (ACP)

The agent channel uses the **Agent Client Protocol (ACP)**, which is JSON-RPC 2.0 under the hood. Messages on the `agent` channel's `data` field are raw ACP JSON-RPC messages.

#### ACP JSON-RPC Wire Format

**Request (Client → Agent):**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": { ... }
}
```

**Response (Agent → Client):**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... }
}
```

**Notification (Agent → Client, no `id`):**

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": { ... }
}
```

**Error Response:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": "session not found: abc123"
  }
}
```

#### 3.2.1 Initialize

Handshake to negotiate capabilities. Must be called before creating sessions.

**Full WS message (Client → Server):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "0.1.0",
      "clientCapabilities": {},
      "clientInfo": {
        "name": "openheim-ui",
        "version": "1.0.0"
      }
    }
  }
}
```

**Full WS message (Server → Client):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "protocolVersion": "0.1.0",
      "agentCapabilities": {
        "loadSession": true,
        "sessionCapabilities": {
          "list": {}
        }
      },
      "agentInfo": {
        "name": "openheim",
        "version": "0.1.0"
      },
      "_meta": {
        "models": {
          "default_provider": "openai",
          "providers": {
            "openai": {
              "default_model": "gpt-4",
              "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
            }
          }
        },
        "mcp_servers": [
          {
            "name": "filesystem",
            "transport": "stdio",
            "connected": true,
            "tool_count": 3
          }
        ],
        "skills": ["debugging", "rust"],
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "execute_command",
              "description": "Execute a shell command...",
              "parameters": { "...": "..." }
            }
          }
        ]
      }
    }
  }
}
```

**`agentCapabilities` fields:**

| Field | Type | Description |
|---|---|---|
| `loadSession` | `true` | Agent supports `session/load` to resume past conversations |
| `sessionCapabilities.list` | `{}` | Agent supports `session/list` to enumerate past conversations |

> The `_meta` field on the initialize response contains a snapshot of all available models, MCP servers, skills, and tools. The same data is available via the REST endpoints.

---

#### 3.2.2 Create Session

Creates a new blank conversation session. Returns a `sessionId` used for all subsequent prompts. To resume an existing session instead, see [§3.2.4 Load Session](#324-load-session).

**Full WS message (Client → Server):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "session/new",
    "params": {
      "cwd": "/path/to/workspace",
      "_meta": {
        "model": "gpt-4-turbo",
        "skills": ["rust", "debugging"]
      }
    }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `cwd` | `string` | No | Working directory for the session |
| `_meta.model` | `string` | No | Model override (must match a model from a configured provider) |
| `_meta.skills` | `string[]` | No | Skills to load for this session |

**Full WS message (Server → Client):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

> The `sessionId` is a UUID. Store it — it's required for all prompt requests.

---

#### 3.2.3 List Sessions

Returns all persisted sessions known to the agent, optionally filtered by `cwd`. Requires the agent to have advertised `sessionCapabilities.list` in its `initialize` response (Openheim always does).

**Full WS message (Client → Server):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "session/list",
    "params": {}
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `cwd` | `string` | No | If set, only return sessions whose stored `cwd` matches exactly |
| `cursor` | `string` | No | Opaque pagination cursor from a previous response's `nextCursor` |

**Full WS message (Server → Client):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
      "sessions": [
        {
          "sessionId": "550e8400-e29b-41d4-a716-446655440000",
          "cwd": "/home/user/my-project",
          "title": "Refactor the auth module",
          "updatedAt": "2025-05-10T14:35:42Z"
        },
        {
          "sessionId": "661f9511-f3ac-52e5-b827-557766551111",
          "cwd": "/",
          "title": "Explain the project structure",
          "updatedAt": "2025-05-09T09:15:30Z"
        }
      ],
      "nextCursor": null
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `sessions[].sessionId` | `string` (UUID) | Use as `sessionId` in `session/load` or `session/prompt` |
| `sessions[].cwd` | `string` | Working directory (`"/"` if never recorded) |
| `sessions[].title` | `string \| null` | Auto-generated title from the first user message |
| `sessions[].updatedAt` | `string \| null` | ISO 8601 timestamp of last activity |
| `nextCursor` | `string \| null` | If present, pass as `cursor` in the next request to get the next page |

> The sessions list is sorted newest-first. `cwd` is populated from the working directory of the first prompt sent in the session; sessions that were listed but never prompted will show `"/"`.

> For most use-cases the REST endpoint `GET /api/sessions` is simpler. Use `session/list` over ACP when you need filtering, pagination, or want to keep everything on a single connection.

---

#### 3.2.4 Load Session

Resumes a previously persisted session. Requires `loadSession: true` in the agent's capabilities (Openheim always advertises this).

The flow is:
1. Client sends `session/load` with the `sessionId` and the current `cwd`.
2. Agent replays the full conversation history as a series of `session/update` notifications (`user_message_chunk` and `agent_message_chunk`).
3. Agent responds with an empty result to signal that history replay is complete.
4. The session is now active — subsequent `session/prompt` requests use the loaded `sessionId`.

**Full WS message (Client → Server):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 4,
    "method": "session/load",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "cwd": "/home/user/my-project",
      "mcpServers": []
    }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `sessionId` | `string` | Yes | UUID of the session to resume |
| `cwd` | `string` | Yes | Current working directory (used for any subsequent prompts) |
| `mcpServers` | `array` | No | MCP server overrides for the loaded session (usually `[]`) |

**History replay notifications (Server → Client, before the response):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "sessionUpdate": "user_message_chunk",
      "content": { "type": "text", "text": "Refactor the auth module to use JWTs." }
    }
  }
}
```

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "sessionUpdate": "agent_message_chunk",
      "content": { "type": "text", "text": "I'll help you refactor the auth module..." }
    }
  }
}
```

> Tool call messages from the original session are **not** replayed — only user and assistant text. Use `GET /api/sessions/:id` if you need the raw tool call history.

**Response (after all history has been replayed):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 4,
    "result": null
  }
}
```

**Error — session not found:**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 4,
    "error": {
      "code": -32603,
      "message": "Internal error",
      "data": "Conversation 550e8400-... not found at ..."
    }
  }
}
```

---

#### 3.2.5 Send Prompt

Send a user message to the agent within a session. The agent will stream back response chunks and tool call updates as `session/update` notifications.

**Full WS message (Client → Server):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "session/prompt",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "prompt": [
        {
          "type": "text",
          "text": "List all Rust files in the src directory and explain the project structure."
        }
      ]
    }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `sessionId` | `string` | Yes | Session ID from `session/new` |
| `prompt` | `ContentBlock[]` | Yes | Array of content blocks (currently only `text` type is supported) |

**Response (after all streaming is complete):**

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
      "stopReason": "end_turn"
    }
  }
}
```

| `stopReason` value | Description |
|---|---|
| `"end_turn"` | Agent completed successfully |
| `"tool_use"` | Agent stopped to request tool execution (shouldn't happen — openheim auto-executes tools) |

---

#### 3.2.6 Streaming Updates (Server → Client)

While processing a prompt, the server sends **notifications** (JSON-RPC messages without an `id` field) via the `agent` channel. These arrive between the prompt request and its response.

All streaming notifications use `method: "session/update"` and contain a `sessionUpdate` discriminator.

##### Agent Message Chunk

Streamed text from the LLM. Accumulate these to build the full response.

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "sessionUpdate": "agent_message_chunk",
      "content": {
        "type": "text",
        "text": "Here is the project structure:\n\n"
      }
    }
  }
}
```

> Multiple `agent_message_chunk` notifications will arrive in sequence. The `content.text` values should be concatenated to form the full assistant response.

##### Tool Call Created

When the LLM requests a tool execution, a tool call is created with `in_progress` status.

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "sessionUpdate": "tool_call",
      "toolCallId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "title": "execute_command",
      "status": "in_progress",
      "rawInput": {
        "command": "find src -name '*.rs' -type f"
      }
    }
  }
}
```

##### Tool Call Update (Completed)

When the tool finishes, a `tool_call_update` is sent with the result.

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "sessionUpdate": "tool_call_update",
      "toolCallId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "status": "completed",
      "rawOutput": "src/main.rs\nsrc/lib.rs\nsrc/agent.rs\n..."
    }
  }
}
```

##### Tool Call Failed

If the tool execution errors:

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "sessionUpdate": "tool_call_update",
      "toolCallId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "status": "failed",
      "rawOutput": "Unknown tool: nonexistent"
    }
  }
}
```

##### Summary of `sessionUpdate` Types

| `sessionUpdate` | Direction | Description |
|---|---|---|
| `agent_message_chunk` | Server → Client | Streamed LLM text chunk |
| `user_message_chunk` | Server → Client | Echo of user message (not currently used) |
| `agent_thought_chunk` | Server → Client | Internal reasoning (not currently used) |
| `tool_call` | Server → Client | New tool call started (`status: "in_progress"`) |
| `tool_call_update` | Server → Client | Tool call status/result update |
| `plan` | Server → Client | Agent execution plan (not currently used) |

##### Tool Call Status Lifecycle

```
pending → in_progress → completed
                      → failed
```

---

### 3.3 Filesystem Channel

All filesystem operations are sent over the `fs` channel. Paths are relative to the workspace root (set via `watch`). Absolute paths must be within the workspace.

#### 3.3.1 Watch / Unwatch

Start watching a workspace directory. This sets the workspace root for all subsequent path validations and enables live file change events.

**Watch (Client → Server):**

```json
{
  "channel": "fs",
  "data": {
    "action": "watch",
    "path": "/path/to/workspace"
  }
}
```

**Watch Success (Server → Client):**

```json
{
  "channel": "fs",
  "data": {
    "type": "watching",
    "path": "/path/to/workspace"
  }
}
```

**Unwatch (Client → Server):**

```json
{
  "channel": "fs",
  "data": {
    "action": "unwatch"
  }
}
```

**Unwatch Success (Server → Client):**

```json
{
  "channel": "fs",
  "data": {
    "type": "unwatched"
  }
}
```

> You must call `watch` before any file operations. All paths in subsequent requests must be within the watched directory. You can only watch one directory at a time; calling `watch` again replaces the previous watch.

---

#### 3.3.2 List Directory

**Request:**

```json
{
  "channel": "fs",
  "data": {
    "action": "list",
    "path": "src",
    "recursive": false
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `string` | Yes | — | Directory path (relative to workspace root or absolute within workspace) |
| `recursive` | `boolean` | No | `false` | If true, lists all descendants recursively |

**Response:**

```json
{
  "channel": "fs",
  "data": {
    "type": "file_list",
    "path": "src",
    "entries": [
      {
        "path": "/path/to/workspace/src/main.rs",
        "name": "main.rs",
        "is_dir": false,
        "size": 2048,
        "modified": 1700000000
      },
      {
        "path": "/path/to/workspace/src/core",
        "name": "core",
        "is_dir": true,
        "size": null,
        "modified": 1700000050
      }
    ]
  }
}
```

**`FileEntry` fields:**

| Field | Type | Description |
|---|---|---|
| `path` | `string` | Full path to the entry |
| `name` | `string` | Filename or directory name |
| `is_dir` | `boolean` | `true` if directory |
| `size` | `number \| null` | File size in bytes (null for directories) |
| `modified` | `number \| null` | Last modified as Unix timestamp in seconds |

---

#### 3.3.3 Read File

**Request:**

```json
{
  "channel": "fs",
  "data": {
    "action": "read",
    "path": "src/main.rs"
  }
}
```

**Response:**

```json
{
  "channel": "fs",
  "data": {
    "type": "file_content",
    "path": "src/main.rs",
    "content": "fn main() {\n    println!(\"Hello\");\n}"
  }
}
```

---

#### 3.3.4 Write File

Creates the file if it doesn't exist. Creates parent directories automatically if needed.

**Request:**

```json
{
  "channel": "fs",
  "data": {
    "action": "write",
    "path": "src/new_module.rs",
    "content": "pub fn hello() -> &str {\n    \"world\"\n}"
  }
}
```

**Response:**

```json
{
  "channel": "fs",
  "data": {
    "type": "write_success",
    "path": "src/new_module.rs"
  }
}
```

---

#### 3.3.5 Create Directory

Creates the directory and all parent directories (equivalent to `mkdir -p`).

**Request:**

```json
{
  "channel": "fs",
  "data": {
    "action": "mkdir",
    "path": "src/utils/helpers"
  }
}
```

**Response:**

```json
{
  "channel": "fs",
  "data": {
    "type": "mkdir_success",
    "path": "src/utils/helpers"
  }
}
```

---

#### 3.3.6 Delete

Deletes a file or directory (recursively if directory).

**Request:**

```json
{
  "channel": "fs",
  "data": {
    "action": "delete",
    "path": "src/old_file.rs"
  }
}
```

**Response:**

```json
{
  "channel": "fs",
  "data": {
    "type": "delete_success",
    "path": "src/old_file.rs"
  }
}
```

---

#### 3.3.7 Rename / Move

**Request:**

```json
{
  "channel": "fs",
  "data": {
    "action": "rename",
    "from": "src/old_name.rs",
    "to": "src/new_name.rs"
  }
}
```

**Response:**

```json
{
  "channel": "fs",
  "data": {
    "type": "rename_success",
    "from": "src/old_name.rs",
    "to": "src/new_name.rs"
  }
}
```

---

#### 3.3.8 Filesystem Events (Server → Client)

When a workspace is being watched, file change events are pushed automatically:

```json
{
  "channel": "fs",
  "data": {
    "type": "fs_event",
    "eventKind": "Create(File)",
    "paths": [
      "/path/to/workspace/src/new_file.rs"
    ]
  }
}
```

| `eventKind` examples | Description |
|---|---|
| `"Create(File)"` | New file created |
| `"Create(Folder)"` | New directory created |
| `"Modify(File)"` | File modified |
| `"Remove(File)"` | File deleted |
| `"Remove(Folder)"` | Directory deleted |
| `"Any"` | Other/combined events |

> Events are polled at ~1 second intervals. Multiple rapid changes may be batched into a single event.

---

#### 3.3.9 Filesystem Error Response

Any fs operation can return an error:

```json
{
  "channel": "fs",
  "data": {
    "type": "error",
    "message": "Path not within workspace or does not exist"
  }
}
```

Common error messages:

| Message | Cause |
|---|---|
| `"Path not within workspace or does not exist"` | Path outside watched workspace or `watch` not called |
| `"Invalid directory: ..."` | Watch path doesn't exist or isn't a directory |
| `"Failed to read: ..."` | File read error (permissions, missing, etc.) |
| `"Failed to write: ..."` | File write error |
| `"Failed to create dirs: ..."` | Parent directory creation failed |
| `"Failed to mkdir: ..."` | Directory creation failed |
| `"Failed to delete file: ..."` / `"Failed to delete dir: ..."` | Deletion error |
| `"Failed to rename: ..."` | Rename/move error |

---

## 4. TypeScript Interfaces

```typescript
// ─── REST API ───────────────────────────────────────────────

interface ProviderConfig {
  api_base: string;
  default_model: string;
  models: string[];
  env_var?: string;
  timeout_secs?: number;
  max_tokens?: number;
  // api_key is NEVER included in responses
}

interface AppConfig {
  default_provider: string;
  max_iterations: number;
  providers: Record<string, ProviderConfig>;
  mcp_servers: Record<string, McpServerConfig>;
}

interface McpServerConfig {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
}

interface ProviderModels {
  default_model: string;
  models: string[];
}

interface ModelsInfo {
  default_provider: string;
  providers: Record<string, ProviderModels>;
}

// GET /api/skills
type SkillsResponse = string[];

interface FunctionDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>; // JSON Schema object
}

interface Tool {
  type: "function";
  function: FunctionDefinition;
}

// GET /api/tools
type ToolsResponse = Tool[];

interface McpServerStatus {
  name: string;
  transport: "stdio" | "http" | "unknown";
  command?: string | null;
  url?: string | null;
  connected: boolean;
  tool_count: number;
  error?: string | null;
}

// GET /api/mcp-servers
type McpServersResponse = McpServerStatus[];

// ─── Session History ─────────────────────────────────────────

interface ConversationMeta {
  id: string;           // UUID
  created_at: string;   // ISO 8601
  updated_at: string;   // ISO 8601
  title?: string | null;
  model?: string | null;
  provider?: string | null;
  skills: string[];
  cwd?: string | null;  // populated after first prompt in the session
}

interface Message {
  role: "user" | "assistant" | "tool" | "system";
  content: string | null;
  tool_calls?: ToolCall[] | null;
  tool_call_id?: string | null; // present on role:"tool" messages
  tool_name?: string | null;    // present on role:"tool" messages
}

interface ToolCall {
  id: string;
  function: {
    name: string;
    arguments: string; // JSON string
  };
}

interface Conversation {
  meta: ConversationMeta;
  messages: Message[];
}

// GET /api/sessions
type SessionsResponse = ConversationMeta[];

// GET /api/sessions/:id
type SessionResponse = Conversation;

// ─── WebSocket Envelopes ────────────────────────────────────

// Inbound (Client → Server)
interface ClientEnvelope {
  channel: "agent" | "fs";
  data: AgentData | FsRequest;
}

// Outbound (Server → Client)
interface ServerEnvelope {
  channel: "system" | "agent" | "fs";
  data: SystemEvent | AgentData | FsResponse;
}

// ─── System Events ──────────────────────────────────────────

interface SystemEvent {
  type: "connected" | "error";
  message: string;
}

// ─── ACP JSON-RPC ───────────────────────────────────────────

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: JsonRpcError;
}

interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params: Record<string, unknown>;
}

interface JsonRpcError {
  code: number;
  message: string;
  data?: string;
}

// ─── ACP Types ──────────────────────────────────────────────

interface InitializeRequest {
  protocolVersion: string;
  clientCapabilities: Record<string, unknown>;
  clientInfo?: {
    name: string;
    version: string;
    title?: string;
  };
  _meta?: Record<string, unknown>;
}

interface AgentCapabilities {
  loadSession: boolean;           // true — agent supports session/load
  sessionCapabilities: {
    list?: {};                    // present — agent supports session/list
  };
  promptCapabilities?: Record<string, unknown>;
  mcpCapabilities?: Record<string, unknown>;
}

interface InitializeResponse {
  protocolVersion: string;
  agentCapabilities: AgentCapabilities;
  agentInfo?: {
    name: string;
    version: string;
    title?: string;
  };
  _meta?: {
    models?: ModelsInfo;
    mcp_servers?: McpServerStatus[];
    skills?: string[];
    tools?: Tool[];
  };
}

interface NewSessionRequest {
  cwd?: string;
  _meta?: {
    model?: string;
    skills?: string[];
  };
}

interface NewSessionResponse {
  sessionId: string; // UUID
}

// ─── ACP: session/list ──────────────────────────────────────

interface ListSessionsRequest {
  cwd?: string;       // filter by exact working directory path
  cursor?: string;    // opaque pagination cursor
}

interface SessionInfo {
  sessionId: string;       // UUID — use in session/load or session/prompt
  cwd: string;             // "/" if not yet recorded
  title?: string | null;
  updatedAt?: string | null; // ISO 8601
}

interface ListSessionsResponse {
  sessions: SessionInfo[];
  nextCursor?: string | null; // pass as cursor in next request for pagination
}

// ─── ACP: session/load ──────────────────────────────────────

interface LoadSessionRequest {
  sessionId: string;     // UUID of the session to resume
  cwd: string;           // working directory to use for subsequent prompts
  mcpServers?: unknown[]; // MCP server overrides (usually [])
}

// Response is null / empty result — history is delivered via session/update notifications

interface ContentBlock {
  type: "text" | "image" | "audio" | "resource_link" | "resource";
  text?: string;       // for type: "text"
  data?: string;       // for type: "image" | "audio"
  mimeType?: string;   // for type: "image" | "audio"
}

interface PromptRequest {
  sessionId: string;
  prompt: ContentBlock[];
}

interface PromptResponse {
  stopReason: "end_turn" | "tool_use";
}

// ─── Session Update Notifications ───────────────────────────

interface AgentMessageChunk {
  sessionUpdate: "agent_message_chunk";
  content: ContentBlock;
}

interface ToolCallNotification {
  sessionUpdate: "tool_call";
  toolCallId: string;
  title: string;
  kind?: "read" | "edit" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "switch_mode" | "other";
  status: "pending" | "in_progress" | "completed" | "failed";
  rawInput?: Record<string, unknown>;
  rawOutput?: Record<string, unknown>;
}

interface ToolCallUpdateNotification {
  sessionUpdate: "tool_call_update";
  toolCallId: string;
  status?: "pending" | "in_progress" | "completed" | "failed";
  title?: string;
  rawInput?: Record<string, unknown>;
  rawOutput?: Record<string, unknown>;
}

interface SessionNotification {
  sessionId: string;
  sessionUpdate: AgentMessageChunk | ToolCallNotification | ToolCallUpdateNotification;
  // ... plus other variant fields inline
}

// ─── Filesystem Types ───────────────────────────────────────

interface FileEntry {
  path: string;
  name: string;
  is_dir: boolean;
  size?: number | null;
  modified?: number | null; // Unix timestamp (seconds)
}

type FsRequest =
  | { action: "watch"; path: string }
  | { action: "unwatch" }
  | { action: "list"; path: string; recursive?: boolean }
  | { action: "read"; path: string }
  | { action: "write"; path: string; content: string }
  | { action: "mkdir"; path: string }
  | { action: "delete"; path: string }
  | { action: "rename"; from: string; to: string };

type FsResponse =
  | { type: "connected"; message: string }
  | { type: "watching"; path: string }
  | { type: "unwatched" }
  | { type: "file_list"; path: string; entries: FileEntry[] }
  | { type: "file_content"; path: string; content: string }
  | { type: "write_success"; path: string }
  | { type: "mkdir_success"; path: string }
  | { type: "delete_success"; path: string }
  | { type: "rename_success"; from: string; to: string }
  | { type: "fs_event"; eventKind: string; paths: string[] }
  | { type: "error"; message: string };
```

---

## 5. Sequence Diagrams

### 5.1 Typical Chat Session

```
Frontend                        Openheim Server                  LLM
   │                                  │                           │
   │  WS connect to /ws               │                           │
   │─────────────────────────────────►│                           │
   │  { channel:"fs", type:"connected"}│                           │
   │◄─────────────────────────────────│                           │
   │                                  │                           │
   │  { channel:"agent",              │                           │
   │    method:"initialize", ... }     │                           │
   │─────────────────────────────────►│                           │
   │  { channel:"agent",              │                           │
   │    result: { ... _meta } }       │                           │
   │◄─────────────────────────────────│                           │
   │                                  │                           │
   │  { channel:"agent",              │                           │
   │    method:"session/new", ... }    │                           │
   │─────────────────────────────────►│                           │
   │  { channel:"agent",              │                           │
   │    result:{sessionId:"uuid"} }   │                           │
   │◄─────────────────────────────────│                           │
   │                                  │                           │
   │  { channel:"agent",              │                           │
   │    method:"session/prompt",      │                           │
   │    params:{sessionId,prompt} }   │                           │
   │─────────────────────────────────►│                           │
   │                                  │  POST /chat/completions   │
   │                                  │──────────────────────────►│
   │                                  │                           │
   │  session/update:                 │  (streaming tokens)       │
   │  agent_message_chunk             │◄──────────────────────────│
   │◄─────────────────────────────────│                           │
   │  session/update:                 │                           │
   │  agent_message_chunk             │  (tool call requested)    │
   │◄─────────────────────────────────│◄──────────────────────────│
   │                                  │                           │
   │  session/update:                 │                           │
   │  tool_call (in_progress)         │                           │
   │◄─────────────────────────────────│                           │
   │                                  │  (executes tool locally)  │
   │  session/update:                 │                           │
   │  tool_call_update (completed)    │                           │
   │◄─────────────────────────────────│                           │
   │                                  │  POST (feed tool result)  │
   │                                  │──────────────────────────►│
   │  session/update:                 │  (more streaming tokens)  │
   │  agent_message_chunk             │◄──────────────────────────│
   │◄─────────────────────────────────│                           │
   │                                  │                           │
   │  prompt response:                │                           │
   │  { stopReason:"end_turn" }       │                           │
   │◄─────────────────────────────────│                           │
```

### 5.2 Resuming a Past Session

```
Frontend                        Openheim Server
   │                                  │
   │  GET /api/sessions               │
   │─────────────────────────────────►│
   │  [ { id, title, updatedAt }, ... ]│
   │◄─────────────────────────────────│
   │                                  │
   │  (user picks a session from list) │
   │                                  │
   │  { channel:"agent",              │
   │    method:"session/load",        │
   │    params:{sessionId, cwd} }     │
   │─────────────────────────────────►│
   │                                  │
   │  session/update:                 │
   │  user_message_chunk (msg 1)      │
   │◄─────────────────────────────────│
   │  session/update:                 │
   │  agent_message_chunk (reply 1)   │
   │◄─────────────────────────────────│
   │  session/update: ... (all msgs)  │
   │◄─────────────────────────────────│
   │                                  │
   │  { result: null }  (load done)   │
   │◄─────────────────────────────────│
   │                                  │
   │  { channel:"agent",              │
   │    method:"session/prompt",      │
   │    params:{sessionId, prompt} }  │
   │─────────────────────────────────►│
   │  (continues conversation ...)    │
```

### 5.3 Filesystem Operations

```
Frontend                        Openheim Server
   │                                  │
   │  { channel:"fs", action:"watch",│
   │    path:"/workspace" }           │
   │─────────────────────────────────►│
   │  { channel:"fs", type:"watching",│
   │    path:"/workspace" }           │
   │◄─────────────────────────────────│
   │                                  │
   │  { channel:"fs", action:"list", │
   │    path:"src" }                  │
   │─────────────────────────────────►│
   │  { channel:"fs", type:"file_list",│
   │    entries:[...] }               │
   │◄─────────────────────────────────│
   │                                  │
   │  { channel:"fs", action:"read", │
   │    path:"src/main.rs" }          │
   │─────────────────────────────────►│
   │  { channel:"fs",                │
   │    type:"file_content",          │
   │    content:"fn main() {...}" }   │
   │◄─────────────────────────────────│
   │                                  │
   │  ... (user edits file externally)│
   │  { channel:"fs",                │
   │    type:"fs_event",              │
   │    eventKind:"Modify(File)",     │
   │    paths:["/workspace/src/main.rs"]}
   │◄─────────────────────────────────│
```

---

## 6. Error Handling

### REST Errors

All REST endpoints return `200` with JSON body on success. If the server is misconfigured (e.g., missing config file), the connection will be refused entirely.

### WebSocket Errors

**Invalid payload:**

```json
{
  "channel": "fs",
  "data": {
    "type": "error",
    "message": "Invalid payload: missing field `action`"
  }
}
```

**ACP errors** are returned as JSON-RPC error responses:

```json
{
  "channel": "agent",
  "data": {
    "jsonrpc": "2.0",
    "id": 3,
    "error": {
      "code": -32603,
      "message": "Internal error",
      "data": "session not found: invalid-session-id"
    }
  }
}
```

**Common JSON-RPC error codes:**

| Code | Meaning |
|---|---|
| `-32700` | Parse error (invalid JSON) |
| `-32600` | Invalid request |
| `-32601` | Method not found |
| `-32602` | Invalid params |
| `-32603` | Internal error |

### Connection Lifecycle

- The WebSocket stays open until the client disconnects or sends a `Close` frame.
- If the server shuts down (SIGINT), it drains gracefully.
- If the connection drops, the client should reconnect and re-initialize (call `initialize`, then either `session/new` for a fresh session or `session/load` to resume an existing one).
- Conversation history is persisted to disk — sessions survive WebSocket reconnections and server restarts.

---

## Appendix: Quick Reference

### REST Endpoints Summary

| Method | Path | Response Type | Description |
|---|---|---|---|
| `GET` | `/api/config` | `AppConfig` (sanitized) | Server configuration |
| `GET` | `/api/models` | `ModelsInfo` | Available models by provider |
| `GET` | `/api/skills` | `string[]` | Installed skill names |
| `GET` | `/api/tools` | `Tool[]` | All tool definitions |
| `GET` | `/api/mcp-servers` | `McpServerStatus[]` | MCP server statuses |
| `GET` | `/api/sessions` | `ConversationMeta[]` | All persisted sessions, newest-first |
| `GET` | `/api/sessions/:id` | `Conversation` | Full conversation including all messages |

### WebSocket Message Types Summary

**Inbound (Client → Server):**

| Channel | Action/Method | Description |
|---|---|---|
| `agent` | `initialize` | Handshake — negotiate capabilities |
| `agent` | `session/new` | Create a new blank session |
| `agent` | `session/list` | List persisted sessions (ACP native) |
| `agent` | `session/load` | Resume a persisted session + replay history |
| `agent` | `session/prompt` | Send a message in the active session |
| `fs` | `watch` | Start watching directory |
| `fs` | `unwatch` | Stop watching |
| `fs` | `list` | List directory contents |
| `fs` | `read` | Read file |
| `fs` | `write` | Write file |
| `fs` | `mkdir` | Create directory |
| `fs` | `delete` | Delete file/directory |
| `fs` | `rename` | Rename/move file/directory |

**Outbound (Server → Client):**

| Channel | Type/Method | Description |
|---|---|---|
| `fs` | `connected` | Initial connection greeting |
| `fs` | `watching` | Watch confirmed |
| `fs` | `unwatched` | Unwatch confirmed |
| `fs` | `file_list` | Directory listing result |
| `fs` | `file_content` | File read result |
| `fs` | `write_success` | File write confirmed |
| `fs` | `mkdir_success` | Directory creation confirmed |
| `fs` | `delete_success` | Deletion confirmed |
| `fs` | `rename_success` | Rename confirmed |
| `fs` | `fs_event` | Live file change event |
| `fs` | `error` | Filesystem error |
| `agent` | JSON-RPC response | Responses to initialize / session/* requests |
| `agent` | `session/update` | Streaming: message chunks, tool calls, history replay |
