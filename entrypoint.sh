#!/usr/bin/env bash
# Start opencode serve + proxy inside container
set -e

MODE="${1:-serve}"
SERVE_PORT="${2:-4096}"
PROXY_PORT="${3:-8000}"

if [ "$MODE" = "cli" ]; then
    echo "=== Starting proxy in CLI mode ==="
    exec python3 /app/proxy.py --host 0.0.0.0 --port "$PROXY_PORT" --mode cli
fi

echo "=== Starting OpenCode serve on port $SERVE_PORT ==="
opencode serve --port "$SERVE_PORT" --hostname 127.0.0.1 &
SERVE_PID=$!

# Wait for serve to be ready
echo "Waiting for OpenCode serve..."
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:$SERVE_PORT/global/health" > /dev/null 2>&1; then
        echo "✓ OpenCode serve ready"
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "✗ OpenCode serve failed to start"
        exit 1
    fi
    sleep 2
done

echo "=== Starting proxy on port $PROXY_PORT ==="
exec python3 /app/proxy.py --host 0.0.0.0 --port "$PROXY_PORT" --mode serve --serve-port "$SERVE_PORT"
