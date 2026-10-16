# Deployment

Openheim ships as a single binary with four subcommands. This guide covers running the server mode (`openheim serve`) in home-cloud and enterprise environments.

---

## Subcommands

| Subcommand | Description |
|-----------|-------------|
| `openheim` | Interactive TUI (no subcommand) |
| `openheim acp` | ACP agent over stdin/stdout — for IDE extensions (Zed, Claude Code, etc.) |
| `openheim run "<prompt>"` | One-shot headless prompt, output to stdout |
| `openheim serve` | WebSocket + REST server |
| `openheim init` | Create a default `~/.openheim/config.toml` |

`~/.openheim` is `.openheim` in the home directory of the user running openheim (`$HOME`; the user profile folder on Windows). Every subcommand takes `--config <FILE>` to use another config file; see [Config and data locations](#config-and-data-locations).

Server mode is the primary deployment target for hosted and enterprise use.

---

## Interactive TUI

Running `openheim` with no subcommand opens the interactive terminal UI. Type a message and press Enter to chat. Type `:help` to see all commands.

| Command | Description |
|---|---|
| `:help` | Show all commands |
| `:q` / `:quit` | Exit |
| `:new` | Start a new session (keeps the current model) |
| `:sessions` | Browse and restore saved sessions (interactive picker) |
| `:config` | Show current configuration |
| `:models` | Open model picker (arrow keys + Enter to switch mid-session) |
| `:models <name>` | Switch to a model directly without opening the picker |
| `:mcp` | Show MCP server statuses |
| `:skills` | List available skills |
| `:theme` | Open theme color picker |
| `:theme <name>` | Apply a theme color directly and save it to `config.toml` |

**Keyboard shortcuts:** `↑`/`↓` scroll · `PgUp`/`PgDn` page · `Ctrl+C` cancel the running turn · `Ctrl+C` twice (within two seconds) quit · `Esc` close any overlay. Pasted text keeps its line breaks (shown as `↵` in the input bar) and is sent only when you press `Enter`.

**Theme colors:** `white`, `gray`, `blue`, `cyan`, `magenta`, `green`, `yellow`, `red`, `pink`

The selected theme is persisted to the config file in use (`~/.openheim/config.toml`, or the `--config` file) as `theme_color` under `[tui]`.

---

## Running the server

```bash
# Default: binds 0.0.0.0:1217
openheim serve

# Custom host/port
openheim serve --host 127.0.0.1 --port 8080
```

The server exposes:
- `ws://{host}:{port}/ws` — WebSocket endpoint (ACP agent + filesystem sidecar)
- `http://{host}:{port}/api/*` — REST endpoints (config, models, skills, tools, mcp-servers, sessions)

The filesystem sidecar is sandboxed to the configured `work_dir` (see [configuration.md](./configuration.md)) — connected clients cannot read or write outside it.

---

## Config and data locations

An openheim process uses two locations:

- **The config file**, `~/.openheim/config.toml` by default. `--config <FILE>` names another one.
- **The data directory**, which holds everything openheim writes:

  | Path | Contents |
  |---|---|
  | `system.md` | The agent's identity |
  | `skills/` | Skills |
  | `agents/` | Subagent profiles |
  | `history/` | Saved conversations |
  | `memory.db` | Long-term memory, unless `[memory].db_path` is set |
  | `openheim.log` | The TUI's log |

  It's the directory the config file's [`data_dir`](./configuration.md#top-level-fields) key names, or, without that key, the config file's own directory: `~/.openheim` by default, or wherever the `--config` file is.

`--config` works with every subcommand, before or after it (`openheim --config FILE run "…"` and `openheim run --config FILE "…"` are the same). A relative path is taken from the directory openheim is started in. openheim doesn't expand `~` itself; write `--config ~/agents/coder.toml` with a space so your shell expands it, or use an absolute path.

`openheim init` writes the default config file and a `system.md` next to it, creating the directory if needed. Every other subcommand fails if the config file is missing; with `--config`, the error names the `init` command to run. If you then set `data_dir`, move `system.md` there, since that's where it's read from. `:theme` in the TUI saves to the config file in use.

### Running several agents

Give each agent its own config file. Without a `data_dir` key, each agent's data lives next to its config, so agents in separate directories share nothing:

```bash
openheim --config ~/agents/coder/config.toml init
openheim --config ~/agents/research/config.toml init
# edit both config files

# Run both at once
openheim --config ~/agents/coder/config.toml
openheim --config ~/agents/research/config.toml serve --port 1218
```

To keep an agent's data apart from its config, for example a read-only config in `/etc` and data under `/var/lib`, set `data_dir` in the config:

```toml
# /etc/openheim/coder.toml
data_dir = "/var/lib/openheim/coder"
```

```bash
openheim --config /etc/openheim/coder.toml serve
```

Some settings are not tied to the config or data location, so check them when two agents run side by side:

- **Ports.** `openheim serve` binds port 1217 by default. Give each server its own `--port`.
- **`[memory].db_path`.** An absolute path puts the memory database there instead of the data directory. Two configs with the same `db_path` share one memory. Leave it unset to keep each agent's memory separate.
- **`work_dir`.** It's the directory the agent's tools may read and write, not where its data lives, and it defaults to the directory openheim was started in. Two agents started in the same directory work on the same files, and nothing coordinates their edits. Set `work_dir` in each config to keep them apart.
- **Environment variables.** Provider keys read from the environment (`ANTHROPIC_API_KEY`, `env_var = "…"`) and `RUST_LOG` come from each process's own environment, so set them per process to give agents different keys.

Give each agent its own data directory too. Two agents whose config files sit in the same directory (and set no `data_dir`), or whose configs set the same `data_dir`, share one data directory.

---

## Configuration

Openheim reads `~/.openheim/config.toml` at startup, or the file `--config` names. In server deployments, place the config at the path the process user can read. See [configuration.md](./configuration.md) for the full reference.

### API keys via environment variables

Provider API keys can be supplied as environment variables instead of being stored in the config file:

| Environment variable | Provider |
|----------------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` | Google Gemini |

Any provider block in `config.toml` with `env_var = "MY_VAR"` reads from that variable at startup. This is the recommended approach for production — store secrets in your secret manager and inject them at runtime.

```toml
[providers.anthropic]
api_base = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-6"
models = ["claude-sonnet-4-6", "claude-opus-4-7"]
env_var = "ANTHROPIC_API_KEY"
```

### Logging

Openheim uses `tracing` with `RUST_LOG` for log level control:

```bash
RUST_LOG=info openheim serve
RUST_LOG=openheim=debug openheim serve   # debug only openheim internals
```

Logs go to stderr, except in the TUI: once it's on screen, they're appended to `openheim.log` in the data directory (`~/.openheim` by default; see [Config and data locations](#config-and-data-locations)) instead.

---

## Docker

The repo ships a multi-stage `Dockerfile` and a `docker-compose.yml`. The image runs as a non-root `openheim` user, and `docker-entrypoint.sh` generates a default config (`openheim init`) on first start if no config volume is mounted.

```bash
# Build and start with docker compose
docker compose up --build

# Or run manually
docker build -t openheim .
docker run -p 1217:1217 \
  -e OPENAI_API_KEY=sk-your-key \
  -v openheim-data:/home/openheim/.openheim \
  openheim serve
```

### Dockerfile (as shipped)

```dockerfile
FROM rust:1.92-slim-bookworm AS builder
WORKDIR /workspace
RUN apt-get update && apt-get install -y pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /workspace/target/release/openheim /usr/local/bin/openheim
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && useradd -m -u 1000 openheim
USER openheim
EXPOSE 1217
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["openheim", "serve"]
```

### docker-compose.yml (as shipped)

```yaml
services:
  openheim:
    build: .
    container_name: openheim-agent
    ports:
      - "1217:1217"
    environment:
      - RUST_LOG=info
      # API keys are read from config.toml, but can also be set here
      # and referenced via env_var in your config
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
    restart: unless-stopped
    volumes:
      # Config directory — mount your own config.toml here;
      # a default config is generated on first run if not provided
      - openheim-config:/home/openheim/.openheim
      # Persistent workspace directory
      - openheim-workspace:/workspace

volumes:
  openheim-config:
    driver: local
  openheim-workspace:
    driver: local
```

```bash
# Start
ANTHROPIC_API_KEY=sk-ant-... docker compose up -d

# View logs
docker compose logs -f openheim
```

---

## systemd

Create `/etc/systemd/system/openheim.service`:

```ini
[Unit]
Description=Openheim LLM agent server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=openheim
Group=openheim
ExecStart=/usr/local/bin/openheim serve --host 127.0.0.1 --port 1217
Restart=on-failure
RestartSec=5s

# Secrets — set these in /etc/openheim/env or use a secret manager
EnvironmentFile=/etc/openheim/env

# Harden the process
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/home/openheim/.openheim

[Install]
WantedBy=multi-user.target
```

`/etc/openheim/env`:
```
ANTHROPIC_API_KEY=sk-ant-...
RUST_LOG=warn
```

```bash
# Create the service user
useradd --system --home /home/openheim --create-home openheim

# Copy config
mkdir -p /home/openheim/.openheim
cp config.toml /home/openheim/.openheim/
chown -R openheim:openheim /home/openheim/.openheim

# Enable and start
systemctl daemon-reload
systemctl enable openheim
systemctl start openheim
systemctl status openheim
```

The service runs as the `openheim` user, so `~/.openheim` is that user's `/home/openheim/.openheim`, not root's or yours. To keep the config in `/etc` and the data in `/var/lib` instead, set `data_dir = "/var/lib/openheim"` in `/etc/openheim/config.toml`, add `--config /etc/openheim/config.toml` to `ExecStart`, and set `ReadWritePaths=/var/lib/openheim`.

---

## Reverse proxy

The WebSocket endpoint requires a proxy that supports connection upgrades. Both nginx and Caddy handle this without extra plugins.

### nginx

```nginx
server {
    listen 443 ssl;
    server_name openheim.example.com;

    ssl_certificate     /etc/letsencrypt/live/openheim.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/openheim.example.com/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:1217;
        proxy_http_version 1.1;

        # WebSocket upgrade headers
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Keep long-running agent sessions alive
        proxy_read_timeout  3600s;
        proxy_send_timeout  3600s;
    }
}
```

### Caddy

```caddyfile
openheim.example.com {
    reverse_proxy localhost:1217
}
```

Caddy handles TLS and WebSocket upgrades automatically.

---

## Enterprise considerations

### Authentication

Openheim does not implement authentication itself. Add it at the reverse proxy layer:

- **Basic auth** (simple deployments): nginx `auth_basic` directive
- **OAuth2 / OIDC**: [oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) in front of nginx/Caddy
- **mTLS**: configure client certificate verification in your proxy for service-to-service use

### Network isolation

For environments where the agent should not reach arbitrary external hosts:

- Run in a network namespace or Docker network with explicit egress rules
- Allow only the LLM provider API endpoints and any MCP server URLs
- The agent calls outbound to: the configured LLM provider API, and any MCP server URLs defined in config

### Multi-tenancy

Openheim is designed as a single-user or small-team server. For multi-tenant deployments, run a separate instance per tenant (separate config, history directory, and API key) behind a routing layer. Sharing a single instance across tenants is not supported — conversation history and skills are not access-controlled at the application level.

### Data residency

Conversation history is written to `history/` in the data directory (`~/.openheim/history/` by default; see [Config and data locations](#config-and-data-locations)) as a pair of files per conversation: `{uuid}.json` (metadata, rewritten wholesale on each change) and `{uuid}.jsonl` (messages, appended one per line as the conversation grows). Back up the whole directory, not just `*.json` — the message content lives in the `.jsonl` files. For enterprises with strict data residency requirements, use a provider endpoint that runs in your region (Anthropic Bedrock, Azure OpenAI, self-hosted Ollama, etc.) and point openheim at it via an `openai_compatible` provider entry.
