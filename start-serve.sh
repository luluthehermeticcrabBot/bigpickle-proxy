#!/usr/bin/env bash
# Start opencode serve + proxy together (serve mode)
set -euo pipefail

PROXY_PORT="${1:-8000}"
SERVE_PORT="${2:-4096}"

echo "=== Starting OpenCode serve on port $SERVE_PORT ==="
opencode serve --port "$SERVE_PORT" --hostname 127.0.0.1 &
SERVE_PID=$!

# Wait for serve to be ready
echo "Waiting for serve to be ready..."
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:$SERVE_PORT/global/health" > /dev/null 2>&1; then
        echo "✓ OpenCode serve ready"
        break
    fi
    sleep 1
done

if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "ERROR: OpenCode serve failed to start"
    exit 1
fi

echo "=== Starting proxy on port $PROXY_PORT ==="
python3 proxy.py --port "$PROXY_PORT" --mode serve --serve-port "$SERVE_PORT" &
PROXY_PID=$!

echo ""
echo "Ready! Proxy: http://127.0.0.1:$PROXY_PORT"
echo "Press Ctrl+C to stop"

cleanup() {
    echo "Shutting down..."
    kill "$PROXY_PID" 2>/dev/null || true
    kill "$SERVE_PID" 2>/dev/null || true
    wait
}
trap cleanup EXIT INT TERM

wait
