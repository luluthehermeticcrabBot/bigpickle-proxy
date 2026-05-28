# Big Pickle Proxy v0.2.0

OpenAI-compatible API that forwards inference calls to OpenCode, giving your agent harness access to Big Pickle (Claude) through a legitimate OpenCode instance.

**New in v0.2.0: Tool-call forwarding in serve mode.**

## Quick Start

```bash
pip install -r requirements.txt

# CLI mode (text-only, no tool calls)
python proxy.py --port 8000 --mode cli

# Serve mode (faster, tool-call support)
python proxy.py --port 8000 --mode serve --serve-port 4096
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

## Modes

### CLI Mode (`--mode cli`)
- Spawns `opencode -p "..." -q -f json -m <model>` per request
- Text-only, no tool call support
- Each request runs in an isolated temp directory
- ~2s cold start, good for low-volume use

### Serve Mode (`--mode serve`) — Recommended
- Uses `opencode serve` HTTP API directly
- **Tool-call forwarding:** captures tool_call events and returns them in OpenAI format
- Lower latency (~200ms), session reuse
- Supports streaming (`stream: true`)

## Tool Call Flow (Serve Mode)

```
Your Agent                    Big Pickle Proxy              OpenCode Serve
    │                              │                             │
    │  POST /v1/chat/completions   │                             │
    │  {tools: [read_file, ...]}   │                             │
    │─────────────────────────────>│                             │
    │                              │  POST /session              │
    │                              │────────────────────────────>│
    │                              │  POST /session/:id/message  │
    │                              │  (prompt + tool definitions)│
    │                              │────────────────────────────>│
    │                              │                             │
    │                              │  Big Pickle sees tools,     │
    │                              │  generates tool_call events │
    │                              │  OpenCode DENIES execution  │
    │                              │  (permissions: deny-all)    │
    │                              │<────────────────────────────│
    │                              │                             │
    │  {                           │                             │
    │    choices: [{               │                             │
    │      finish_reason:          │                             │
    │        "tool_calls",         │                             │
    │      message: {              │                             │
    │        tool_calls: [{        │                             │
    │          function: {         │                             │
    │            name: "read_file",│                             │
    │            arguments: "..."  │                             │
    │          }                   │                             │
    │        }]                    │                             │
    │      }                       │                             │
    │    }]                        │                             │
    │  }                           │                             │
    │<─────────────────────────────│                             │
    │                              │                             │
    │  (Agent executes read_file)  │                             │
    │                              │                             │
    │  POST /v1/chat/completions   │                             │
    │  {messages: [                │                             │
    │    {role: "assistant",       │                             │
    │     tool_calls: [...]},      │                             │
    │    {role: "tool",            │                             │
    │     tool_call_id: "...",     │                             │
    │     content: "file contents"}│                             │
    │  ]}                          │                             │
    │─────────────────────────────>│                             │
    │                              │  Tool results formatted     │
    │                              │  inline in prompt           │
    │                              │  POST /session/:id/message  │
    │                              │────────────────────────────>│
    │                              │                             │
    │                              │  Big Pickle continues with  │
    │                              │  tool results in context    │
    │                              │<────────────────────────────│
    │                              │                             │
    │  {choices: [{message:        │                             │
    │    {content: "Final answer"}}]}                            │
    │<─────────────────────────────│                             │
```

### Required: Set OpenCode Permissions to Deny

For tool-call forwarding to work (OpenCode captures but doesn't execute), configure `~/.opencode.json`:

```json
{
  "permissions": {
    "*": "deny"
  }
}
```

Or via the serve API (check `/doc` for the exact endpoint — typically `PATCH /session/:id` with permission settings).

**If permissions aren't denied:** OpenCode will execute tools locally AND the proxy returns them to your agent → double execution.

**Testing tool calls:**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "messages": [
      {"role": "user", "content": "Read the file README.md"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a file from disk",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {
                "type": "string",
                "description": "Path to the file"
              }
            }
          }
        }
      }
    ]
  }'
```

Expected: Response contains `"finish_reason": "tool_calls"` with a `read_file` tool call.

## Model Aliases

| Alias | OpenCode Model String |
|-------|----------------------|
| `big-pickle` | `anthropic/claude-sonnet-4` |
| `claude-sonnet-4` | `anthropic/claude-sonnet-4` |
| `claude-opus-4` | `anthropic/claude-opus-4-20250514` |
| `claude-haiku` | `anthropic/claude-3.5-haiku` |
| `gpt-4o` | `openai/gpt-4o` |
| `gpt-4.1` | `openai/gpt-4.1` |
| `gemini-flash` | `google/gemini-2.5-flash` |

## Use With Agent Harnesses

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

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
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }
    }],
)
```

## Performance Notes

| Mode | Cold Start | Per-Request | Tool Calls | Streaming |
|------|-----------|-------------|------------|-----------|
| CLI  | N/A       | ~2-5s       | ❌         | ❌         |
| Serve | ~5s (once) | ~0.2-2s    | ✅         | ✅ (fake)  |

## Container Deployment

```bash
docker-compose up -d bigpickle-proxy-cli
```

Or build manually:

```bash
docker build -t bigpickle-proxy .
docker run -p 8000:8000 bigpickle-proxy --mode cli
```

## Limitations

- CLI mode: text-only, no tool calls, no streaming
- Serve mode streaming is fake (buffers full response, sends as one chunk)
- Token counts may be 0 (OpenCode doesn't expose them in all modes)
- Tool-call argument parsing depends on OpenCode's internal ToolPart format — test with your OpenCode version
- Session management: each request is a new OpenCode session (stateless like OpenAI API)
