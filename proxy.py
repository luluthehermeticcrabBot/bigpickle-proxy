#!/usr/bin/env python3
"""
Big Pickle Proxy — OpenAI-compatible API that forwards to OpenCode.

Modes:
  cli    — uses `opencode -p` (simpler, stateless, ~2s cold start per request)
  serve  — uses `opencode serve` HTTP API (faster, supports tool calls)

Usage:
  python proxy.py --port 8000 --mode cli
  python proxy.py --port 8000 --mode serve --serve-port 4096

Tool Calls (serve mode only):
  Set session permissions to deny-all so OpenCode doesn't execute tools locally.
  The proxy captures tool_call events and returns them in OpenAI format.
  Your calling agent executes the tools and sends results in the next request.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Model mapping ──────────────────────────────────────────────────────────

MODEL_ALIASES = {
    # Primary OpenCode-built-in (free) models
    "big-pickle":        "opencode/big-pickle",
    "claude-sonnet-4":   "opencode/claude-sonnet-4",
    "claude-sonnet":     "opencode/claude-sonnet-4",
    "claude-opus-4":     "opencode/claude-opus-4-1",
    "claude-opus":       "opencode/claude-opus-4-1",
    "claude-haiku":      "opencode/claude-haiku-4-5",
    "gpt-5":             "opencode/gpt-5.1",
    "gpt-5-codex":       "opencode/gpt-5.1-codex",
    "gpt-5.2":           "opencode/gpt-5.2",
    "gpt-5.3-codex":     "opencode/gpt-5.3-codex",
    "gpt-5.4":           "opencode/gpt-5.4",
    "gemini-flash":      "opencode/gemini-3-flash",
    "gemini-pro":        "opencode/gemini-3.1-pro",
    # OpenCode Go (bring-your-own-key) models
    "deepseek-v4-flash": "opencode-go/deepseek-v4-flash",
    "deepseek-v4-pro":   "opencode-go/deepseek-v4-pro",
    "kimi-k2.5":         "opencode-go/kimi-k2.5",
    "kimi-k2.6":         "opencode-go/kimi-k2.6",
    "qwen3.7-max":       "opencode-go/qwen3.7-max",
    "glm-5":             "opencode-go/glm-5",
    "mimo-v2.5":         "opencode-go/mimo-v2.5",
}

DEFAULT_MODEL = "opencode/big-pickle"


def resolve_model(requested: str) -> str:
    """Map user-facing model name to OpenCode provider/model string."""
    return MODEL_ALIASES.get(requested, requested)


def list_known_models() -> list[dict]:
    """Return OpenAI-format model list."""
    models = []
    for alias, full in MODEL_ALIASES.items():
        models.append({
            "id": alias,
            "object": "model",
            "created": 1718400000,
            "owned_by": "opencode-proxy",
        })
    return models


# ── Prompt conversion ──────────────────────────────────────────────────────

def messages_to_prompt(messages: list[dict]) -> str:
    """
    Convert OpenAI messages array to a single prompt string.
    Includes tool calls and tool results inline so OpenCode sees full context.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Handle multimodal content arrays
        if isinstance(content, list):
            texts = [p["text"] for p in content if p.get("type") == "text"]
            content = "\n".join(texts)

        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")

        elif role == "user":
            parts.append(content or "")

        elif role == "assistant":
            text = content or ""
            # Include tool calls in the prompt so OpenCode sees them
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    text += (
                        f"\n\n[Assistant called tool: {name} "
                        f"with arguments: {args}]"
                    )
            parts.append(text)

        elif role == "tool":
            tool_id = msg.get("tool_call_id", "unknown")
            result = content or ""
            parts.append(f"[Tool result for {tool_id}:\n{result}]")

        elif role == "function":
            # Legacy function role — treat like tool
            name = msg.get("name", "unknown")
            result = content or ""
            parts.append(f"[Function result for {name}:\n{result}]")

    prompt = "\n\n".join(p for p in parts if p)
    return prompt


# ── Tool-call parsing ──────────────────────────────────────────────────────

def parse_opencode_parts(parts: list[dict]) -> tuple[str, list[dict], dict]:
    """
    Parse OpenCode message parts into:
      - text_content: concatenated text from all text/reasoning parts
      - tool_calls:   list of OpenAI-format tool calls
      - usage:        token/cost info from step-finish parts

    ToolPart fields (best-effort — test with real OpenCode to confirm):
      {type: "tool", callID, tool (name), state, args?, input?, ...}
    """
    text_parts = []
    tool_calls = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for part in parts:
        ptype = part.get("type", "")

        if ptype == "text":
            text = part.get("text", "")
            if text and not part.get("ignored"):
                text_parts.append(text)

        elif ptype == "reasoning":
            text = part.get("text", "")
            if text:
                text_parts.append(f"<reasoning>\n{text}\n</reasoning>")

        elif ptype == "tool":
            call_id = part.get("callID", str(uuid.uuid4().hex[:12]))
            tool_name = part.get("tool", "unknown")

            # Try multiple possible argument field names
            args = part.get("args") or part.get("arguments") or \
                   part.get("input") or part.get("parameters") or {}
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = str(args)

            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_str,
                },
            })

        elif ptype == "step-finish":
            tokens = part.get("tokens", {})
            if isinstance(tokens, dict):
                usage["prompt_tokens"] = tokens.get("input", 0)
                usage["completion_tokens"] = tokens.get("output", 0)
                usage["total_tokens"] = tokens.get("total", 0)
            cost = part.get("cost", 0)
            if cost:
                usage.setdefault("cost", cost)

    return "\n".join(text_parts), tool_calls, usage


# ── CLI backend ─────────────────────────────────────────────────────────────

def run_opencode_cli(prompt: str, model: str, timeout: int = 120) -> dict:
    """
    Run opencode in non-interactive CLI mode.
    Returns {"response": "...", "model": "..."} on success.
    CLI mode does NOT support tool calls — text-only.
    """
    sandbox = tempfile.mkdtemp(prefix="opencode-sandbox-")

    try:
        cmd = [
            "opencode",
            "-p", prompt,
            "-q",
            "-f", "json",
            "-m", model,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=sandbox,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() or "(no stderr)"
            raise RuntimeError(
                f"OpenCode exited with code {result.returncode}: {stderr}"
            )

        output = result.stdout.strip()
        if not output:
            raise RuntimeError("OpenCode produced empty output")

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                raise RuntimeError(
                    f"Could not parse OpenCode output as JSON: {output[:500]}"
                )

        response_text = data.get("response", "")
        if not response_text:
            raise RuntimeError(
                f"OpenCode returned empty response. Output: {output[:500]}"
            )

        response_text = strip_tool_artifacts(response_text)

        return {"response": response_text, "model": model}

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"OpenCode timed out after {timeout}s")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def strip_tool_artifacts(text: str) -> str:
    """Remove tool call blocks from OpenCode response text."""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<function_calls>.*?</function_calls>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<invoke.*?>.*?</invoke>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<antml:function_calls>.*?</antml:function_calls>",
        "", text, flags=re.DOTALL,
    )
    return text.strip()


# ── Serve backend ───────────────────────────────────────────────────────────

class OpenCodeServeClient:
    """
    HTTP client for opencode serve API.
    Parses the streamed MessageV2.WithParts JSON response.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:4096"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def health(self) -> bool:
        client = await self._get_client()
        try:
            r = await client.get(f"{self.base_url}/global/health")
            return r.status_code == 200
        except Exception:
            return False

    async def create_session(self) -> str:
        client = await self._get_client()
        r = await client.post(f"{self.base_url}/session", json={})
        r.raise_for_status()
        return r.json()["id"]

    async def prompt(self, session_id: str, text: str,
                     model: str | None = None,
                     system: str | None = None,
                     agent: str | None = None) -> dict:
        """
        Send a prompt to OpenCode serve and return structured response.

        Returns:
          {
            "text": str,           # concatenated text parts
            "tool_calls": [...],   # OpenAI-format tool calls
            "usage": {...},        # token/cost info
            "raw_parts": [...]     # full parts array for debugging
          }
        """
        client = await self._get_client()

        body: dict = {
            "parts": [{"type": "text", "text": text}],
        }

        if model and "/" in model:
            provider_id, model_id = model.split("/", 1)
            body["model"] = {"providerID": provider_id, "modelID": model_id}
        elif model:
            body["model"] = {"providerID": "anthropic", "modelID": model}

        if system:
            body["system"] = system

        if agent:
            body["agent"] = agent

        r = await client.post(
            f"{self.base_url}/session/{session_id}/message",
            json=body,
            timeout=300.0,
        )
        if r.status_code >= 400:
            error_body = r.text[:1000] if r.text else "(empty body)"
            raise RuntimeError(
                f"OpenCode serve returned {r.status_code}: {error_body}"
            )
        r.raise_for_status()

        # Response is a JSON stream: one complete MessageV2.WithParts object.
        # Try parsing as a single JSON object, then fall back to line-by-line.
        raw = r.text.strip()

        parts = []

        # Case 1: entire response is one JSON object with {info, parts}
        try:
            msg = json.loads(raw)
            if isinstance(msg, dict) and "parts" in msg:
                parts = msg["parts"]
        except json.JSONDecodeError:
            pass

        # Case 2: newline-delimited JSON stream of events
        if not parts:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Direct part object
                if isinstance(event, dict) and "type" in event:
                    parts.append(event)
                # Nested: {type: "part", data: {...}}
                elif event.get("type") == "part" and "data" in event:
                    parts.append(event["data"])
                # Message wrapper
                elif "parts" in event:
                    parts.extend(event["parts"])

        text, tool_calls, usage = parse_opencode_parts(parts)

        # Debug: log raw parts when tool calls are detected
        if tool_calls:
            import sys
            print(f"\n[DEBUG] Tool calls detected — raw parts:", file=sys.stderr)
            for i, p in enumerate(parts):
                if p.get("type") == "tool":
                    print(f"  Part[{i}]: {json.dumps(p, indent=2)}", file=sys.stderr)
            print(file=sys.stderr)

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": usage,
            "raw_parts": parts,
        }


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Big Pickle Proxy",
    description="OpenAI-compatible API forwarding to OpenCode",
    version="0.2.0",
)

config: dict = {}


@app.get("/health")
async def health():
    return {"status": "ok", "mode": config.get("mode", "unknown")}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": list_known_models()}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    requested_model = body.get("model", DEFAULT_MODEL)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)
    openai_tools = body.get("tools", [])        # OpenAI-format tool definitions
    tool_choice = body.get("tool_choice", "auto")

    if not messages:
        raise HTTPException(status_code=400, detail="messages array is required")

    model = resolve_model(requested_model)

    # Extract system message if present (first message with role=system)
    system_msg = None
    if messages and messages[0].get("role") == "system":
        system_msg = messages[0].get("content", "")

    prompt = messages_to_prompt(messages)

    # Inject tool definitions into the prompt so Big Pickle knows what's available
    if openai_tools and config.get("mode") == "serve":
        tool_desc = _format_tools_for_prompt(openai_tools)
        prompt = (
            "IMPORTANT: You have been given a specific set of tools below. "
            "IGNORE any other tools you may know about (like read, write, edit, bash, grep, glob). "
            "ONLY use the tools listed here. "
            "Other tools do not work in this environment.\n\n"
            f"{tool_desc}\n\n"
            "When you need to use one of these tools, call it using the standard "
            "function calling format. The calling system will execute it and "
            "return the result for you to continue.\n\n"
            "---\n\n"
            f"{prompt}"
        )

    # Inject temperature hint
    if temperature is not None and temperature != 0.7:
        prompt = f"[temperature={temperature}] {prompt}"

    try:
        mode = config.get("mode", "cli")

        if mode == "serve":
            serve_port = config.get("serve_port", 4096)
            client = OpenCodeServeClient(f"http://127.0.0.1:{serve_port}")

            if not await client.health():
                raise HTTPException(
                    status_code=503,
                    detail=f"OpenCode serve not reachable at "
                           f"http://127.0.0.1:{serve_port}",
                )

            session_id = await client.create_session()
            result = await client.prompt(
                session_id, prompt, model,
                system=system_msg,
                agent=body.get("agent"),
            )

            response_text = result["text"]
            tool_calls = result["tool_calls"]
            usage = result["usage"]

            if not response_text and not tool_calls:
                raise RuntimeError("Empty response from serve API")

        else:  # cli mode
            timeout = config.get("timeout", 120)
            result = run_opencode_cli(prompt, model, timeout=timeout)
            response_text = result["response"]
            tool_calls = []
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Determine finish_reason
        if tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        # Build the response message
        message: dict = {"role": "assistant"}
        if response_text:
            message["content"] = response_text
        else:
            message["content"] = None

        if tool_calls:
            message["tool_calls"] = tool_calls

        response_body = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

        # Streaming support (serve mode only — fake streaming, sends all at once)
        if stream and mode == "serve":
            async def stream_gen():
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": message,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_gen(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        if stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming requires serve mode (--mode serve)",
            )

        return response_body

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _format_tools_for_prompt(tools: list[dict]) -> str:
    """Convert OpenAI tool definitions to a prompt-friendly format."""
    lines = ["Available tools:"]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        params = func.get("parameters", {})
        lines.append(f"\n- {name}: {desc}")
        if params.get("properties"):
            lines.append("  Parameters:")
            for prop_name, prop_info in params["properties"].items():
                ptype = prop_info.get("type", "any")
                pdesc = prop_info.get("description", "")
                lines.append(f"    {prop_name} ({ptype}): {pdesc}")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Big Pickle Proxy — OpenAI-compatible API → OpenCode"
    )
    parser.add_argument("--port", type=int, default=8000,
                        help="Proxy listen port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Proxy listen host (default: 127.0.0.1)")
    parser.add_argument("--mode", choices=["cli", "serve"], default="cli",
                        help="Backend mode: cli (subprocess) or serve (HTTP API)")
    parser.add_argument("--serve-port", type=int, default=4096,
                        help="OpenCode serve port (default: 4096, serve mode only)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="OpenCode timeout in seconds (default: 120, cli mode only)")

    args = parser.parse_args()

    config["mode"] = args.mode
    config["serve_port"] = args.serve_port
    config["timeout"] = args.timeout

    # Check opencode is available in CLI mode
    if args.mode == "cli":
        try:
            subprocess.run(
                ["opencode", "--version"],
                capture_output=True,
                timeout=5,
            )
        except FileNotFoundError:
            print("ERROR: 'opencode' not found in PATH. Is OpenCode installed?")
            print("Install: curl -fsSL https://opencode.ai/install | bash")
            sys.exit(1)
        except Exception as e:
            print(f"WARNING: Could not verify opencode: {e}")

    import uvicorn

    print(f"\n  Big Pickle Proxy v0.2.0")
    print(f"  Mode:      {args.mode}")
    print(f"  Listen:    http://{args.host}:{args.port}")
    print(f"  Models:    {', '.join(MODEL_ALIASES.keys())}")
    if args.mode == "serve":
        print(f"  OC Serve:  http://127.0.0.1:{args.serve_port}")
        print(f"  Tool calls: enabled (ensure OC permissions deny execution)")
    print(f"\n  Test:   curl http://{args.host}:{args.port}/v1/chat/completions \\")
    print(f'           -H "Content-Type: application/json" \\')
    print(f"           -d '{{\"model\":\"big-pickle\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
