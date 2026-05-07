# Build stage
FROM rust:1.92-slim-bookworm as builder

WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifest files
COPY Cargo.toml Cargo.lock* ./

# Copy source code
COPY src ./src

# Build the application in release mode
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /workspace

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary from builder
COPY --from=builder /workspace/target/release/openheim /usr/local/bin/openheim

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create a non-root user with home directory for config
RUN useradd -m -u 1000 openheim && \
    chown -R openheim:openheim /workspace

USER openheim

# Create config directory (will be used if no volume mounted)
RUN mkdir -p /home/openheim/.openheim

COPY config.toml /home/openheim/.openheim/config.toml

# Expose the API port
EXPOSE 1217

# Set environment variables
ENV RUST_LOG=info

ENTRYPOINT ["docker-entrypoint.sh"]

# Default to API mode
CMD ["openheim", "serve"]
