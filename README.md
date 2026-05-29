# Big Pickle Proxy v0.3.0

OpenAI-compatible API proxy that forwards inference calls to OpenCode's cloud API
using UUID-based authentication. **No API key needed.** No local OpenCode
installation. Just the proxy and your agent harness.

## Quick Start

```bash
pip install -r requirements.txt
python proxy.py --port 8000

# Default mode is "cloud" — connects directly to OpenCode Zen API
```

Test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "messages": [{"role": "user", "content": "Say hello in 3 words"}]
  }'
```

List available models:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Modes

### Cloud Mode (`--mode cloud`) — Default, Recommended

Forwards requests directly to `https://opencode.ai/zen/v1/chat/completions` with
per-request UUID headers (`x-opencode-project`, `x-opencode-session`,
`x-opencode-request`, `x-opencode-client`). No Bearer token — the UUIDs
authenticate access to OpenCode's free models.

- **No local OpenCode** required
- **No session management** — stateless like OpenAI API
- **Tool calls** passed through natively (OpenAI format in → OpenAI format out)
- **Streaming** via SSE pass-through from cloud API
- **Model discovery** — `/v1/models` fetches live from Zen API

### CLI Mode (`--mode cli`)

Spawns `opencode -p "..." -q -f json -m <model>` per request. Requires OpenCode
installed locally. Text-only, no tool calls, ~2s cold start per request.

### Serve Mode (`--mode serve`)

Connects to a local `opencode serve` HTTP API. Requires OpenCode installed and
`opencode serve` running on the configured port. Supports tool calls via
ToolPart parsing.

## Free Models

The following models are available through the Zen API **without any API key**:

| Model | Backend | Notes |
|---|---|---|
| `big-pickle` | DeepSeek V3 | Reasoning model, verbose thinking |
| `deepseek-v4-flash-free` | DeepSeek V4 Flash | Fast, lightweight |
| `nemotron-3-super-free` | NVIDIA Nemotron 3 | Balanced performance |
| `mimo-v2.5-free` | Mimo V2.5 | Multilingual |

Run `curl http://127.0.0.1:8000/v1/models` for the live list — it queries the
Zen API directly.

**Formerly free, now require OpenCode Go subscription:**
`minimax-m2.5-free`, `qwen3.6-plus-free`

## Tool Calling (Cloud Mode)

Tool calls are forwarded natively — the proxy passes your `tools` array to the
OpenCode Zen API and returns the `tool_calls` response in standard OpenAI format.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "messages": [{"role": "user", "content": "What is 2+2? Use the calculator."}],
    "max_tokens": 2000,
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Performs arithmetic",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string", "description": "Math expression"}
          },
          "required": ["expression"]
        }
      }
    }]
  }'
```

**Multi-turn caveat:** Big Pickle (DeepSeek backend) requires `reasoning_content`
to be preserved across turns. When your agent receives an assistant message with
`reasoning_content`, pass it back unchanged in subsequent requests. Most agent
harnesses (Hermes, OpenAI SDK) handle this automatically.

## Use With Agent Harnesses

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

# List available models
models = client.models.list()
for m in models.data:
    print(m.id)

# Chat completion
response = client.chat.completions.create(
    model="big-pickle",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        }
    }],
)
print(response.choices[0].message.content)
```

## Container Deployment

```bash
docker-compose up -d
```

## Performance

| Mode | Cold Start | Per-Request | Tool Calls | Streaming | Local OpenCode |
|------|-----------|-------------|------------|-----------|----------------|
| Cloud | None | ~1-5s | ✅ | ✅ (real SSE) | Not needed |
| Serve | ~5s (once) | ~0.2-2s | ✅ | ✅ (fake) | Required |
| CLI | ~2s | ~2-5s | ❌ | ❌ | Required |

## Architecture

```
Your Agent (OpenAI SDK)
    │
    │  POST /v1/chat/completions
    │  (standard OpenAI format)
    ▼
Big Pickle Proxy (FastAPI)
    │
    │  POST https://opencode.ai/zen/v1/chat/completions
    │  + UUID headers (project, session, request, client)
    ▼
OpenCode Zen Cloud API
    │
    │  Returns OpenAI-format response
    ▼
Your Agent ← proxy passes through with proxy IDs
```

## Limitations

- Free models only — paid models require an API key (not yet supported)
- Big Pickle's `reasoning_content` must be preserved across multi-turn conversations
- Token counts and costs are reported as returned by the Zen API
- Cloud mode rate limits are determined by OpenCode (no control from proxy side)
- Previously-free models may have their promotions ended without notice — use `/v1/models` to check

## License

MIT
