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

Server mode is the primary deployment target for hosted and enterprise use.

---

## Interactive TUI

Running `openheim` with no subcommand opens the interactive terminal UI. Type a message and press Enter to chat. Type `:help` to see all commands.

| Command | Description |
|---|---|
| `:help` | Show all commands |
| `:q` / `:quit` | Exit |
| `:sessions` | Browse and restore saved sessions (interactive picker) |
| `:config` | Show current configuration |
| `:models` | Open model picker (arrow keys + Enter to switch mid-session) |
| `:models <name>` | Switch to a model directly without opening the picker |
| `:mcp` | Show MCP server statuses |
| `:skills` | List available skills |
| `:theme` | Open theme color picker |
| `:theme <name>` | Apply a theme color directly and save it to `config.toml` |

**Keyboard shortcuts:** `↑`/`↓` scroll · `PgUp`/`PgDn` page · `Ctrl+C` quit · `Esc` close any overlay

**Theme colors:** `white`, `gray`, `blue`, `cyan`, `magenta`, `green`, `yellow`, `red`, `pink`

The selected theme is persisted to `~/.openheim/config.toml` as `theme_color`.

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
- `http://{host}:{port}/api/*` — REST endpoints (config, models, skills, tools, sessions)

---

## Configuration

Openheim reads `~/.openheim/config.toml` at startup. In server deployments, place the config at the path the process user can read. See [configuration.md](./configuration.md) for the full reference.

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

---

## Docker

### Dockerfile

```dockerfile
FROM rust:1.85-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin openheim

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/openheim /usr/local/bin/openheim

# Config and history live here — mount a volume in production
RUN mkdir -p /root/.openheim
COPY config.toml /root/.openheim/config.toml

EXPOSE 1217
CMD ["openheim", "serve", "--host", "0.0.0.0", "--port", "1217"]
```

### docker-compose.yml

```yaml
services:
  openheim:
    build: .
    ports:
      - "1217:1217"
    volumes:
      - openheim-data:/root/.openheim
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RUST_LOG=warn
    restart: unless-stopped

volumes:
  openheim-data:
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

Conversation history is written to `~/.openheim/history/` as JSON files. Mount this path on network storage or back it up with standard file-system tooling. For enterprises with strict data residency requirements, use a provider endpoint that runs in your region (Anthropic Bedrock, Azure OpenAI, self-hosted Ollama, etc.) and point openheim at it via an `openai_compatible` provider entry.
