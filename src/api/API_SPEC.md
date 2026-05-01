# Openheim API

---

## REST API

Base URL: `http://<host>:<port>` (default port `1217`)

---

### POST /query

Run the agent with a prompt. Blocking — returns when the agent finishes.

**Request**

```json
{
  "prompt": "Write a hello world in Python",
  "model": "gpt-4o",           // optional — overrides server default
  "max_iterations": 10,        // optional — overrides server default
  "chat_id": "uuid-string",    // optional — continue an existing conversation
  "skills": ["web_search"]     // optional — list of skill names to enable
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | yes | User message |
| `model` | string | no | Model override |
| `max_iterations` | number | no | Max agent loop iterations |
| `chat_id` | string (UUID) | no | Resume a prior conversation |
| `skills` | string[] | no | Skills to enable |

**Response `200 OK`**

```json
{
  "success": true,
  "result": {
    "final_response": "Here is your Python file...",
    "iterations": 3
  },
  "chat_id": "uuid-string"
}
```

**Response `500 Internal Server Error`**

```json
{
  "success": false,
  "error": "LLM request timed out",
  "chat_id": "uuid-string"
}
```

---

### PUT /config

Validate and persist a new configuration to `~/.openheim/config.toml`. Does not restart the server — call `POST /restart` after to apply changes.

**Request** — full `AppConfig` object

```json
{
  "default_provider": "anthropic",
  "max_iterations": 10,
  "providers": {
    "anthropic": {
      "api_base": "https://api.anthropic.com/v1",
      "default_model": "claude-3-5-sonnet-20241022",
      "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
      "env_var": "ANTHROPIC_API_KEY",
      "timeout_secs": 120,
      "max_tokens": 8096
    },
    "ollama": {
      "api_base": "http://localhost:11434/v1",
      "default_model": "llama3.2",
      "models": ["llama3.2", "mistral"]
    }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `default_provider` | string | yes | Must match a key in `providers` |
| `max_iterations` | number | no | Global default (default: `10`) |
| `providers` | object | yes | At least one provider required |

**Provider fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `api_base` | string | yes | Base URL for the provider API |
| `default_model` | string | yes | Default model for this provider |
| `models` | string[] | yes | Available models |
| `env_var` | string | no | Env var name holding the API key (recommended) |
| `api_key` | string | no | Inline API key (not recommended) |
| `timeout_secs` | number | no | Request timeout in seconds (default: `120`) |
| `max_tokens` | number | no | Max output tokens |

**Response `200 OK`**

```json
{ "status": "saved" }
```

**Response `400 Bad Request`**

```json
{ "error": "default_provider 'openai' not found in providers" }
```

---

### POST /restart

Notify all active WebSocket clients with a `restarting` system event (close code `1012`), then exit the process. The container is expected to restart automatically (Docker `restart: unless-stopped`), loading the updated config from disk.

**Request** — no body

**Response `200 OK`** — returned immediately before shutdown begins

```json
{ "status": "restarting" }
```

---

## WebSocket API

```
GET /ws
Upgrade: websocket
```

All communication uses a single WebSocket connection. Messages are JSON text frames.

Upon connection, the server immediately sends a `system` message:

```json
{ "channel": "system", "data": { "type": "connected", "message": "Connected to Openheim" } }
```

---

## Envelope Format

Every message — both client-to-server and server-to-client — is wrapped in a channel envelope.

### Client → Server

```ts
{
  channel: "agent" | "fs",
  data: AgentRequest | FsRequest
}
```

### Server → Client

```ts
{
  channel: "system" | "agent" | "fs",
  data: SystemEvent | AgentResponse | FsResponse
}
```

---

## Channel: `agent`

### Client → Server: Run agent

```json
{
  "channel": "agent",
  "data": {
    "prompt": "Write a hello world in Python",
    "model": "gpt-4o",           // optional — overrides server default
    "max_iterations": 10,        // optional — overrides server default
    "chat_id": "uuid-string",    // optional — continue an existing conversation
    "skills": ["web_search"]     // optional — list of skill names to enable
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | yes | User message |
| `model` | string | no | Model override |
| `max_iterations` | number | no | Max agent loop iterations |
| `chat_id` | string (UUID) | no | Resume a prior conversation |
| `skills` | string[] | no | Skills/tools to enable |

### Server → Client: Agent responses

The server streams a sequence of events for each run, followed by a terminal `done` or `error`.

#### `event` — streaming update

```json
{
  "channel": "agent",
  "data": {
    "type": "event",
    "data": { ...StreamEvent }
  }
}
```

**StreamEvent types** (discriminated by `event_type`):

```json
{ "event_type": "iteration_start", "iteration": 1 }
{ "event_type": "tool_call", "tool_name": "read_file", "arguments": "{\"path\":\"main.py\"}" }
{ "event_type": "tool_result", "tool_name": "read_file", "result": "print('hello')" }
{ "event_type": "llm_response", "content": "Here is your Python file..." }
{ "event_type": "finished", "final_response": "Done! Here is your file.", "iterations": 3 }
```

#### `done` — run completed

```json
{
  "channel": "agent",
  "data": {
    "type": "done",
    "chat_id": "uuid-string"
  }
}
```

#### `error` — run failed

```json
{
  "channel": "agent",
  "data": {
    "type": "error",
    "message": "Invalid model: gpt-99"
  }
}
```

**Typical event sequence:**

```
event(iteration_start) → event(tool_call) → event(tool_result) → event(llm_response) → event(finished) → done
```

---

## Channel: `fs`

All filesystem operations require a `watch` to be set first — it establishes the workspace root and restricts all path operations to within that directory (path traversal is rejected).

Paths may be absolute or relative. Relative paths are resolved against the workspace root.

### Client → Server: Filesystem requests

All requests share the shape `{ "channel": "fs", "data": { "action": "...", ...fields } }`.

#### `watch` — set workspace root and start watching for changes

```json
{ "channel": "fs", "data": { "action": "watch", "path": "/home/user/myproject" } }
```

#### `unwatch` — stop watching

```json
{ "channel": "fs", "data": { "action": "unwatch" } }
```

#### `list` — list directory contents

```json
{ "channel": "fs", "data": { "action": "list", "path": "src", "recursive": false } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Directory to list |
| `recursive` | boolean | no | Walk subdirectories (default: `false`) |

#### `read` — read a file

```json
{ "channel": "fs", "data": { "action": "read", "path": "src/main.py" } }
```

#### `write` — write a file (creates parent directories if needed)

```json
{ "channel": "fs", "data": { "action": "write", "path": "src/main.py", "content": "print('hello')" } }
```

#### `mkdir` — create a directory (including parents)

```json
{ "channel": "fs", "data": { "action": "mkdir", "path": "src/utils" } }
```

#### `delete` — delete a file or directory

```json
{ "channel": "fs", "data": { "action": "delete", "path": "src/old.py" } }
```

Directories are deleted recursively.

#### `rename` — rename or move a file or directory

```json
{ "channel": "fs", "data": { "action": "rename", "from": "src/old.py", "to": "src/new.py" } }
```

Both `from` and `to` must be within the workspace.

---

### Server → Client: Filesystem responses

#### `watching`

```json
{ "channel": "fs", "data": { "type": "watching", "path": "/home/user/myproject" } }
```

#### `unwatched`

```json
{ "channel": "fs", "data": { "type": "unwatched" } }
```

#### `file_list`

```json
{
  "channel": "fs",
  "data": {
    "type": "file_list",
    "path": "src",
    "entries": [
      {
        "path": "/home/user/myproject/src/main.py",
        "name": "main.py",
        "is_dir": false,
        "size": 1024,
        "modified": 1712345678
      }
    ]
  }
}
```

`size` is bytes, absent for directories. `modified` is a Unix timestamp in seconds, may be absent.

#### `file_content`

```json
{ "channel": "fs", "data": { "type": "file_content", "path": "src/main.py", "content": "print('hello')" } }
```

#### `write_success`

```json
{ "channel": "fs", "data": { "type": "write_success", "path": "src/main.py" } }
```

#### `mkdir_success`

```json
{ "channel": "fs", "data": { "type": "mkdir_success", "path": "src/utils" } }
```

#### `delete_success`

```json
{ "channel": "fs", "data": { "type": "delete_success", "path": "src/old.py" } }
```

#### `rename_success`

```json
{ "channel": "fs", "data": { "type": "rename_success", "from": "src/old.py", "to": "src/new.py" } }
```

#### `fs_event` — real-time file change notification (pushed after `watch`)

```json
{
  "channel": "fs",
  "data": {
    "type": "fs_event",
    "event_kind": "Modify(Data(Any))",
    "paths": ["/home/user/myproject/src/main.py"]
  }
}
```

`event_kind` is a debug string from the [`notify`](https://docs.rs/notify) crate. Common values: `Create(Any)`, `Modify(Data(Any))`, `Remove(Any)`. Treat unknown values gracefully.

#### `error`

```json
{ "channel": "fs", "data": { "type": "error", "message": "Path not within workspace or does not exist" } }
```

---

## Channel: `system`

Server-only. Clients cannot send on this channel.

#### `connected`

```json
{ "channel": "system", "data": { "type": "connected", "message": "Connected to Openheim" } }
```

Sent immediately on connection.

#### `restarting`

```json
{ "channel": "system", "data": { "type": "restarting", "reason": "config_update" } }
```

Sent to all connected clients before the server exits. The WebSocket connection is then closed with code `1012` (Service Restart). Clients should reconnect automatically.

`reason` values: `"config_update"` (triggered via `PUT /config` + `POST /restart`), `"manual"` (triggered via `POST /restart` directly).

#### `error` — malformed message

```json
{ "channel": "system", "data": { "type": "error", "message": "Invalid request: ..." } }
```

Sent when the server cannot parse a client message.

---

## Error Handling

| Situation | Channel | `type` |
|---|---|---|
| Unparseable client message | `system` | `error` |
| Agent config / model invalid | `agent` | `error` |
| Agent runtime failure | `agent` | `error` |
| No workspace set, invalid path | `fs` | `error` |
| File I/O failure | `fs` | `error` |

---

## Reconnection

The server does not persist WebSocket state across connections. On reconnect:

- Re-send `watch` to restore filesystem watching.
- Use the `chat_id` returned in `done` to resume a conversation.
- A `1012` close code means the server restarted intentionally — reconnect after a short delay.

---

## TypeScript Types (reference)

```ts
// REST

interface AppConfig {
  default_provider: string;
  max_iterations?: number;
  providers: Record<string, ProviderConfig>;
}

interface ProviderConfig {
  api_base: string;
  default_model: string;
  models: string[];
  env_var?: string;
  api_key?: string;
  timeout_secs?: number;
  max_tokens?: number;
}

interface AgentResult {
  final_response: string;
  iterations: number;
}

// POST /query
interface QueryRequest {
  prompt: string;
  model?: string;
  max_iterations?: number;
  chat_id?: string;
  skills?: string[];
}

interface QueryResponse {
  success: boolean;
  result?: AgentResult;
  error?: string;
  chat_id?: string;
}

// WebSocket envelopes
type ClientEnvelope =
  | { channel: "agent"; data: AgentRequest }
  | { channel: "fs"; data: FsRequest };

type ServerEnvelope =
  | { channel: "system"; data: SystemEvent }
  | { channel: "agent"; data: AgentResponse }
  | { channel: "fs"; data: FsResponse };

// System
type SystemEvent =
  | { type: "connected"; message: string }
  | { type: "restarting"; reason: string }
  | { type: "error"; message: string };

// Agent
interface AgentRequest {
  prompt: string;
  model?: string;
  max_iterations?: number;
  chat_id?: string;
  skills?: string[];
}

type StreamEvent =
  | { event_type: "iteration_start"; iteration: number }
  | { event_type: "tool_call"; tool_name: string; arguments: string }
  | { event_type: "tool_result"; tool_name: string; result: string }
  | { event_type: "llm_response"; content: string }
  | { event_type: "finished"; final_response: string; iterations: number };

type AgentResponse =
  | { type: "event"; data: StreamEvent }
  | { type: "done"; chat_id?: string }
  | { type: "error"; message: string };

// Filesystem
type FsRequest =
  | { action: "watch"; path: string }
  | { action: "unwatch" }
  | { action: "list"; path: string; recursive?: boolean }
  | { action: "read"; path: string }
  | { action: "write"; path: string; content: string }
  | { action: "mkdir"; path: string }
  | { action: "delete"; path: string }
  | { action: "rename"; from: string; to: string };

interface FileEntry {
  path: string;
  name: string;
  is_dir: boolean;
  size?: number;
  modified?: number;
}

type FsResponse =
  | { type: "watching"; path: string }
  | { type: "unwatched" }
  | { type: "file_list"; path: string; entries: FileEntry[] }
  | { type: "file_content"; path: string; content: string }
  | { type: "write_success"; path: string }
  | { type: "mkdir_success"; path: string }
  | { type: "delete_success"; path: string }
  | { type: "rename_success"; from: string; to: string }
  | { type: "fs_event"; event_kind: string; paths: string[] }
  | { type: "error"; message: string };
```
