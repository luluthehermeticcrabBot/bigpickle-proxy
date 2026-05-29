#!/usr/bin/env python3
"""
Big Pickle Proxy — OpenAI-compatible API that forwards to OpenCode.

Modes:
  cli    — uses `opencode -p` (simpler, stateless, ~2s cold start per request)
  serve  — uses `opencode serve` HTTP API (faster, supports tool calls)
  cloud  — forwards directly to OpenCode cloud API (no local OpenCode, UUID auth)

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

# ── Cloud API backend ──────────────────────────────────────────────────────

OPENCODE_CLOUD_URL = "https://opencode.ai/zen/v1/chat/completions"

# OpenWebUI rejects SSE chunks above ~16KB. Big Pickle's verbose reasoning
# can produce single chunks of 50KB+. Split at sentence boundaries.
SSE_MAX_CHUNK = 8192


def _split_sse_chunk(line: str) -> list[str]:
    """
    Split an oversized SSE data line into smaller valid chunks.
    Preserves the SSE format: yields 'data: {...}\\n\\n' lines.
    """
    prefix = ""
    if line.startswith("data: "):
        prefix = "data: "
        json_str = line[6:]
    elif line.startswith("data:"):
        prefix = "data:"
        json_str = line[5:]
    else:
        # Not a data line, pass through as-is
        return [f"{line}\\n\\n"]

    # Non-data lines (comments, [DONE]) pass through
    if not json_str.strip().startswith("{"):
        return [f"{line}\\n\\n"]

    # Check if it's small enough
    if len(line) <= SSE_MAX_CHUNK:
        return [f"{line}\\n\\n"]

    # Parse and split the content field(s)
    try:
        chunk = json.loads(json_str)
    except json.JSONDecodeError:
        return [f"{line}\\n\\n"]

    # Find the content-bearing field in the delta
    delta = chunk.get("choices", [{}])[0].get("delta", {})
    content = delta.get("content", "")
    reasoning = delta.get("reasoning_content", "")

    # If no oversized content, pass through
    if not content and not reasoning:
        return [f"{line}\\n\\n"]

    target_field = "reasoning_content" if reasoning else "content"
    text = reasoning or content

    if len(text) <= 2048:
        return [f"{line}\\n\\n"]

    # Split text at sentence boundaries (~1KB per chunk for streaming feel)
    sentences = _split_text(text, chunk_size=1024)
    if len(sentences) <= 1:
        return [f"{line}\\n\\n"]

    # Emit multiple chunks with progressive content
    result = []
    for i, sentence in enumerate(sentences):
        new_chunk = json.loads(json_str)  # deep copy
        new_chunk["choices"][0]["delta"] = {
            k: (sentence if k == target_field else v)
            for k, v in delta.items()
        }
        result.append(f"{prefix}{json.dumps(new_chunk)}\\n\\n")
    return result


def _split_text(text: str, chunk_size: int = 1024) -> list[str]:
    """Split text at sentence boundaries, keeping chunks under chunk_size."""
    import re
    parts = []
    current = ""

    # Split on sentence endings: . ! ? followed by space or newline
    tokens = re.split(r"(?<=[.!?])\\s+", text)

    for token in tokens:
        if len(current) + len(token) + 1 <= chunk_size:
            current = (current + " " + token).strip() if current else token
        else:
            if current:
                parts.append(current)
            # If a single sentence is still too big, force-split it
            if len(token) > chunk_size:
                for i in range(0, len(token), chunk_size):
                    parts.append(token[i:i + chunk_size])
                current = ""
            else:
                current = token

    if current:
        parts.append(current)

    return parts if len(parts) > 1 else [text]


def _make_cloud_headers() -> dict:
    """Generate UUID headers for OpenCode cloud API auth."""
    return {
        "Content-Type": "application/json",
        "User-Agent": "opencode/1.0.0",
        "x-opencode-project": str(uuid.uuid4()),
        "x-opencode-session": str(uuid.uuid4()),
        "x-opencode-request": str(uuid.uuid4()),
        "x-opencode-client": "opencode",
    }


def resolve_model(requested: str) -> str:
    """Map user-facing model name to OpenCode provider/model string."""
    return MODEL_ALIASES.get(requested, requested)


def list_known_models() -> list[dict]:
    """Return OpenAI-format model list (static fallback)."""
    models = []
    # Confirmed free models on OpenCode Zen (UUID auth, no API key)
    for model_id in [
        "big-pickle",
        "deepseek-v4-flash-free",
        "nemotron-3-super-free",
        "mimo-v2.5-free",
    ]:
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1718400000,
            "owned_by": "opencode",
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

            # Arguments live in state.input (confirmed from OpenCode ToolPart)
            state = part.get("state", {})
            args = state.get("input", {})
            # Fallbacks: top-level fields, string parsing
            if not args:
                args = part.get("args") or part.get("arguments") or \
                       part.get("input") or part.get("parameters") or {}
            if isinstance(args, dict):
                args_str = json.dumps(args)
            elif isinstance(args, str):
                try:
                    args_str = json.dumps(json.loads(args))
                except (json.JSONDecodeError, TypeError):
                    args_str = args
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
        """Create session with all tool permissions denied by default."""
        client = await self._get_client()
        body = {
            "permission": [
                {"permission": "*", "pattern": "*", "action": "deny"}
            ]
        }
        r = await client.post(f"{self.base_url}/session", json=body)
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
    version="0.3.0",
)

config: dict = {}


@app.get("/health")
async def health():
    return {"status": "ok", "mode": config.get("mode", "unknown")}


@app.get("/v1/models")
async def list_models(request: Request):
    """Return available models — fetched live from OpenCode Zen API."""
    mode = config.get("mode", "cloud")

    if mode == "cloud":
        import httpx as _httpx
        try:
            headers = _make_cloud_headers()
            async with _httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    "https://opencode.ai/zen/v1/models",
                    headers=headers,
                    timeout=10.0,
                )
                if r.status_code == 200:
                    return r.json()
        except Exception:
            pass  # Fall through to static list

    # Static fallback (used for cli/serve modes or if Zen is unreachable)
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

        if mode == "cloud":
            # ── Cloud mode: forward to OpenCode cloud API ──
            import httpx as _httpx

            # Cloud API uses plain model names — pass through as-is, no prefix
            model_resolved = requested_model

            # Always stream from upstream to avoid oversized single-line
            # responses that trigger OpenWebUI's aiohttp "Chunk too big" error.
            # For non-streaming clients, we buffer the stream and return JSON.
            headers = _make_cloud_headers()
            payload = {
                "model": model_resolved,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,  # Always stream from upstream
            }
            if openai_tools:
                payload["tools"] = openai_tools
            if tool_choice and tool_choice != "auto":
                payload["tool_choice"] = tool_choice

            if stream:
                # Client wants streaming — forward SSE with chunk splitting
                async def _cloud_stream():
                    async with _httpx.AsyncClient(timeout=300.0) as sc:
                        async with sc.stream(
                            "POST",
                            OPENCODE_CLOUD_URL,
                            json=payload,
                            headers=headers,
                            timeout=300.0,
                        ) as resp:
                            if resp.status_code >= 400:
                                body = await resp.aread()
                                yield f"data: {json.dumps({'error': f'Upstream {resp.status_code}: {body.decode()[:500]}'})}\\n\\n"
                                yield "data: [DONE]\\n\\n"
                                return
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                for chunk_line in _split_sse_chunk(line):
                                    yield chunk_line

                return StreamingResponse(
                    _cloud_stream(),
                    media_type="text/event-stream",
                )

            # Client wants non-streaming — buffer the SSE stream, return JSON
            chunks: list[dict] = []
            finish_reason = "stop"
            usage = {}
            merged_content = ""
            merged_reasoning = ""

            async with _httpx.AsyncClient(timeout=300.0) as sc:
                async with sc.stream(
                    "POST",
                    OPENCODE_CLOUD_URL,
                    json=payload,
                    headers=headers,
                    timeout=300.0,
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise RuntimeError(
                            f"OpenCode cloud API returned {resp.status_code}: "
                            f"{body.decode()[:500]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        chunks.append(chunk)

            # Merge all chunks into a single response
            tool_calls_map: dict[int, dict] = {}  # index -> tool_call dict
            for chunk in chunks:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        merged_content += delta["content"]
                    if delta.get("reasoning_content"):
                        merged_reasoning += delta["reasoning_content"]
                    # Accumulate tool calls from streaming deltas
                    tc_delta = delta.get("tool_calls", [])
                    for tc in tc_delta:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_map[idx]
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        func = tc.get("function", {})
                        if func.get("name"):
                            entry["function"]["name"] += func["name"]
                        if func.get("arguments"):
                            entry["function"]["arguments"] += func["arguments"]
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                if chunk.get("usage"):
                    usage = chunk["usage"]

            message = {"role": "assistant"}
            if merged_content:
                message["content"] = merged_content
            else:
                message["content"] = None
            if merged_reasoning:
                message["reasoning_content"] = merged_reasoning
            if tool_calls_map:
                message["tool_calls"] = [
                    tool_calls_map[i] for i in sorted(tool_calls_map)
                ]

            request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            result = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
            }
            # Pretty-print JSON to avoid OpenWebUI's aiohttp "Chunk too big"
            # error. aiohttp reads HTTP bodies line-by-line via readuntil();
            # a single 50KB unindented JSON line triggers its limit.
            # Indentation breaks response into short lines, each under 16KB.
            from fastapi.responses import Response as _Response
            pretty = json.dumps(result, indent=2, ensure_ascii=False)
            return _Response(
                content=pretty,
                media_type="application/json",
            )

        elif mode == "serve":
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
                detail="Streaming requires serve or cloud mode (--mode serve|cloud)",
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
    parser.add_argument("--mode", choices=["cli", "serve", "cloud"], default="cloud",
                        help="Backend mode: cloud (default, direct API), serve (local opencode), cli (subprocess)")
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

    print(f"\n  Big Pickle Proxy v0.3.0")
    print(f"  Mode:      {args.mode}")
    print(f"  Listen:    http://{args.host}:{args.port}")
    print(f"  Models:    {', '.join(MODEL_ALIASES.keys())}")
    if args.mode == "cloud":
        print(f"  Cloud API: {OPENCODE_CLOUD_URL}")
        free_models = [m["id"] for m in list_known_models()]
        print(f"  Free models: {', '.join(free_models)}")
        print(f"  Tool calls: enabled (forwarded to cloud API)")
        print(f"  Streaming:  enabled")
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
