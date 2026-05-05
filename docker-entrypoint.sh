#!/bin/bash
set -e

CONFIG_DIR="$HOME/.openheim"
CONFIG_FILE="$CONFIG_DIR/config.toml"

# Initialize config if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found, initializing default config..."
    openheim init
    echo "Default config created at $CONFIG_FILE"
    echo "Edit this file or mount your own config to customize providers."
fi

# Execute the main command
exec "$@"
