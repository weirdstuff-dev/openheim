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

# Create a non-root user
RUN useradd -m -u 1000 openheim && \
    chown -R openheim:openheim /workspace

USER openheim

# Expose the API port
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info

# Default to API mode
CMD ["openheim", "--api-mode"]
