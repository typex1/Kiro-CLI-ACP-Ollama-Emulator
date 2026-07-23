#!/usr/bin/env bash
# Start the Kiro CLI ACP Ollama Emulator server.
# The server listens on port 11435 by default (configurable via .env or env vars).

set -e

cd "$(dirname "$0")"

# Install dependencies if node_modules is missing
if [ ! -d node_modules ]; then
  echo "📦 Installing dependencies..."
  npm install
fi

echo "🚀 Starting Kiro ACP Ollama Emulator..."
npm start
