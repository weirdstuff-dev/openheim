# Configuration Reference

Openheim loads its configuration from `~/.openheim/config.toml`. Generate a default file with:

```bash
cargo run -- init
```

---

## Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `default_provider` | string | — | Provider to use when no `--model` override is given (must match a key under `[providers]`) |
| `max_iterations` | integer | `10` | Maximum number of agent loop iterations per prompt before stopping |
| `default_skills` | string[] | `[]` | Skills loaded automatically in every new session. Merged with per-session `--skills`; defaults appear first, duplicates removed. |
| `theme_color` | string | `"white"` | TUI accent color. Valid values: `white`, `gray`, `blue`, `cyan`, `magenta`, `green`, `yellow`, `red`, `pink`. Can also be changed at runtime with `:theme` |

```toml
default_provider = "anthropic"
max_iterations = 20

# Always load these skills without passing --skills each time
default_skills = ["rules", "concise"]
```

---

## `[providers.<name>]`

Each key under `[providers]` defines a provider. The key name is used as the provider identifier (e.g. `"openai"`, `"anthropic"`, `"ollama"`).

| Field | Type | Required | Description |
|---|---|---|---|
| `api_base` | string | Yes | Base URL for the API |
| `default_model` | string | Yes | Model used when none is specified for this provider |
| `models` | string[] | Yes | List of available models (used for validation and the `/api/models` endpoint) |
| `env_var` | string | No | Name of the environment variable holding the API key (recommended) |
| `api_key` | string | No | Inline API key — `env_var` takes precedence if both are set |
| `timeout_secs` | integer | `120` | Request timeout in seconds |
| `max_tokens` | integer | No | Maximum output tokens per response (provider default if omitted) |

Key resolution order: `env_var` (if set and non-empty) → `api_key` → empty string (for keyless providers like Ollama).

### Examples

```toml
[providers.openai]
api_base = "https://api.openai.com/v1"
default_model = "gpt-4o"
models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
env_var = "OPENAI_API_KEY"
timeout_secs = 120
max_tokens = 4096

[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-6"
models = ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"]
env_var = "ANTHROPIC_API_KEY"

[providers.gemini]
api_base = "https://generativelanguage.googleapis.com/v1beta"
default_model = "gemini-2.5-flash"
models = ["gemini-2.5-flash", "gemini-2.5-pro"]
env_var = "GEMINI_API_KEY"

# OpenAI-compatible local model — no API key needed
[providers.ollama]
api_base = "http://localhost:11434/v1"
default_model = "llama3"
models = ["llama3", "mistral", "codellama"]
```

---

## `[mcp_servers.<name>]`

Each key under `[mcp_servers]` connects an MCP server. The key name becomes the tool-name prefix: a tool named `read_file` on a server named `filesystem` is exposed as `filesystem__read_file`.

Use either `command` (stdio transport) or `url` (Streamable HTTP transport), not both.

| Field | Type | Required | Description |
|---|---|---|---|
| `command` | string | Stdio only | Binary to spawn (e.g. `"npx"`, `"uvx"`) |
| `args` | string[] | No | Arguments passed to `command` |
| `env` | table | No | Extra environment variables for the spawned process |
| `url` | string | HTTP only | Base URL for Streamable HTTP transport |

```toml
# stdio — spawn a local process
[mcp_servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

# stdio with env vars
[mcp_servers.my-db]
command = "uvx"
args = ["mcp-server-postgres"]
env = { DATABASE_URL = "postgresql://localhost/mydb" }

# Streamable HTTP
[mcp_servers.remote-tools]
url = "http://localhost:8080/mcp"
```

The `name` key is sanitized when building tool names: hyphens and spaces become underscores. So `my-db` → `my_db__query_table`.

---

## Complete example

```toml
default_provider = "anthropic"
max_iterations = 15

[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-6"
models = ["claude-sonnet-4-6", "claude-opus-4-7"]
env_var = "ANTHROPIC_API_KEY"

[providers.openai]
api_base = "https://api.openai.com/v1"
default_model = "gpt-4o"
models = ["gpt-4o", "gpt-4-turbo"]
env_var = "OPENAI_API_KEY"

[providers.ollama]
api_base = "http://localhost:11434/v1"
default_model = "llama3"
models = ["llama3", "mistral"]

[mcp_servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
```

---

## Custom config path

To load a config from a non-default path (useful when embedding openheim as a library):

```rust
let client = OpenheimClient::from_config("/etc/myapp/openheim.toml")
    .build()
    .await?;
```

See [library.md](./library.md) for the full library API.
