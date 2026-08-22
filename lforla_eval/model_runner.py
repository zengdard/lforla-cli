"""Model runner — calls LLM APIs to evaluate benchmark samples.

Supports both single-turn evaluation (``run_sample``) and agentic oracle
evaluation with tool-calling (``run_oracle_sample``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx


def _post_with_retry(client: httpx.Client, url: str, *, json: dict, headers: dict,
                     attempts: int = 8) -> httpx.Response:
    """POST with exponential backoff on 429/5xx and error-bodied 200s
    (free-tier gateways sometimes answer HTTP 200 with {"error": ...})."""
    delay = 8.0

    def _body_has_error(resp: httpx.Response) -> bool:
        try:
            body = resp.json()
        except Exception:
            return False
        return isinstance(body, dict) and bool(body.get("error"))

    for attempt in range(attempts):
        r = client.post(url, json=json, headers=headers)
        if r.status_code == 200 and not _body_has_error(r):
            return r
        retryable = (
            r.status_code in (429, 500, 502, 503, 504)
            or (r.status_code == 200 and _body_has_error(r))
        )
        if retryable and attempt < attempts - 1:
            time.sleep(delay)
            delay = min(delay * 1.7, 60.0)
            continue
        r.raise_for_status()
    return r

# Optional per-1000-token pricing (USD) for cost-per-task metrics.
# Common pricing map for known model families (USD per 1K tokens).
# Overridable via LFORLA_PRICE_PER_1K_IN / LFORLA_PRICE_PER_1K_OUT env vars;
# unknown models default to None so cost metrics stay hidden.
PRICING: dict[str, tuple[float | None, float | None]] = {}


def load_pricing() -> None:
    """Populate the pricing table from environment overrides (input, output)."""
    import os as _os

    defaults = {
        # family: (input_per_1k, output_per_1k)
        "gpt-4o": (0.0025, 0.010),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4": (0.03, 0.06),
        "claude-3-opus": (0.015, 0.075),
        "claude-3-5-sonnet": (0.003, 0.015),
        "claude-3-5-haiku": (0.0008, 0.004),
    }
    for family, (pin, pout) in defaults.items():
        PRICING[family] = (pin, pout)

    def _num(v: str | None) -> float | None:
        if v in (None, ""):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    pin = _num(_os.getenv("LFORLA_PRICE_PER_1K_IN"))
    pout = _num(_os.getenv("LFORLA_PRICE_PER_1K_OUT"))
    if pin is not None or pout is not None:
        PRICING["*"] = (pin, pout)


load_pricing()


class ModelRunner:
    """Calls an LLM API for each sample and returns the result."""

    def __init__(self, model_id: str, provider: str = "openai", api_key: str | None = None):
        self.model_id = model_id
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY") or ""
        # Completion budget per call. Benchmarks with large structured outputs
        # (drone-build CAD sources) need far more than the old 2048 default.
        try:
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        except ValueError:
            self.max_tokens = 8192

    def estimate_cost_usd(self, tokens_input: int, tokens_output: int) -> float | None:
        """Estimate cost in USD for a call using the pricing table.

        Falls back: longest model-family prefix match, then the wildcard override.
        Returns None when no pricing is known so cost metrics stay hidden.
        """
        cur = PRICING.get(self.model_id)
        if cur is None:
            model = self.model_id.lower()
            cur = next(
                (PRICING[k] for k in PRICING if k != "*" and model.startswith(k.lower())),
                None,
            )
        if cur is None:
            cur = PRICING.get("*")
        if cur is None or (cur[0] is None and cur[1] is None):
            return None
        pin = cur[0] or 0.0
        pout = cur[1] or 0.0
        return (tokens_input / 1000 * pin) + (tokens_output / 1000 * pout)

    def _attach_cost(self, result: dict) -> dict:
        cost = self.estimate_cost_usd(int(result.get("tokens_input", 0)), int(result.get("tokens_output", 0)))
        if cost is not None:
            result["total_cost_usd"] = round(cost, 6)
        return result

    # ========================================================================
    # Single turn (legacy / non-oracle)
    # ========================================================================
    def run_sample(self, sample: dict) -> dict:
        """Run a single sample through the model."""
        input_data = sample.get("input_data", {})
        prompt = input_data.get("prompt") or input_data.get("instruction") or input_data.get("text", "")

        if self.provider == "mock":
            return self._attach_cost(self._run_mock(prompt))
        if not self.api_key and self.provider in ("openai", "anthropic"):
            raise RuntimeError(f"Missing API key. Set {self.provider.upper()}_API_KEY or pass --api-key")
        if self.provider == "openai":
            return self._attach_cost(self._run_openai(prompt))
        elif self.provider == "anthropic":
            return self._attach_cost(self._run_anthropic(prompt))
        elif self.provider == "ollama":
            return self._attach_cost(self._run_ollama(prompt))
        else:
            return self._attach_cost(self._run_generic(prompt))

    # ========================================================================
    # Oracle (agentic, tool-calling)
    # ========================================================================
    def run_oracle_sample(
        self,
        prompt: str,
        tool_specs: list[dict],
        handle_call: Any,
        *,
        max_iterations: int = 12,
        system: str | None = None,
    ) -> dict:
        """Run an agentic loop where the model may call oracle tools.

        ``tool_specs`` is the provider-specific tool list.
        ``handle_call(name, args)`` is invoked for every tool call and its
        return value is fed back to the model.

        Returns the same shape as ``run_sample`` (``output``, token counts,
        latency) plus ``tool_calls`` and ``iterations``.
        """
        start = time.time()
        total_in = 0
        total_out = 0
        calls_used: list[dict] = []

        if self.provider == "mock":
            content, tool_calls = self._oracle_mock(prompt, handle_call)
            calls_used = tool_calls
            tokens_in = len(prompt.split())
            tokens_out = len(content.split())
            elapsed = int((time.time() - start) * 1000)
            return self._attach_cost({
                "output": content,
                "execution_time_ms": elapsed,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "tool_calls": calls_used,
                "iterations": max(1, len(tool_calls)),
            })

        if not self.api_key and self.provider in ("openai", "anthropic"):
            raise RuntimeError(f"Missing API key. Set {self.provider.upper()}_API_KEY or pass --api-key")

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Native OpenAI-style tool-calling for the generic provider too
        # (opencode Zen and most gateways support it). Opt out with
        # LLM_NATIVE_TOOLS=0 for endpoints without tools support — the
        # text-based "TOOL:" protocol is used as fallback.
        native_generic = (
            self.provider == "generic" and os.getenv("LLM_NATIVE_TOOLS", "1") != "0"
        )
        if self.provider == "anthropic":
            runner_loop = self._oracle_loop_anthropic
        elif self.provider == "openai" or native_generic:
            runner_loop = self._oracle_loop_openai
        else:
            runner_loop = self._oracle_loop_generic

        content, tool_calls, usage_in, usage_out = runner_loop(
            messages, tool_specs, handle_call, max_iterations
        )
        elapsed = int((time.time() - start) * 1000)
        return self._attach_cost({
            "output": content,
            "execution_time_ms": elapsed,
            "tokens_input": usage_in,
            "tokens_output": usage_out,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
        })

    def _chat_openai(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> dict:
        # provider == "generic" reuses the native OpenAI tool-calling loop
        # against any OpenAI-compatible endpoint (LLM_ENDPOINT base URL).
        base_url = (
            "https://api.openai.com/v1"
            if self.provider == "openai"
            else os.getenv("LLM_ENDPOINT", "").rstrip("/")
        )
        if not base_url:
            raise RuntimeError("LLM_ENDPOINT environment variable required for generic provider")
        client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(900.0, connect=30.0))
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body: dict = {"model": self.model_id, "messages": messages, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        r = _post_with_retry(client, "/chat/completions", json=body, headers=headers)
        return r.json()

    def _oracle_loop_openai(self, messages, tool_specs, handle_call, max_iterations):
        tool_calls_log = []
        total_in = 0
        total_out = 0
        final_content = ""
        for _ in range(max_iterations):
            data = self._chat_openai(messages, tools=tool_specs or None)
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)

            tc = msg.get("tool_calls")
            if not tc:
                final_content = msg.get("content", "") or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": [
                        {
                            "id": t["id"],
                            "type": "function",
                            "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]},
                        }
                        for t in tc
                    ],
                }
            )
            for t in tc:
                name = t["function"]["name"]
                try:
                    args = json.loads(t["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = handle_call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = {"error": str(e)}
                tool_calls_log.append({"name": name, "args": args})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": t["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return final_content, tool_calls_log, total_in, total_out

    def _chat_anthropic(self, messages, tools=None) -> dict:
        client = httpx.Client(base_url="https://api.anthropic.com/v1", timeout=120)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body: dict = {"model": self.model_id, "max_tokens": self.max_tokens, "messages": messages}
        if tools:
            body["tools"] = tools
        r = client.post("/messages", json=body, headers=headers)
        r.raise_for_status()
        return r.json()

    def _oracle_loop_anthropic(self, messages, tool_specs, handle_call, max_iterations):
        tool_calls_log = []
        total_in = 0
        total_out = 0
        final_content = ""
        for _ in range(max_iterations):
            data = self._chat_anthropic(messages, tools=tool_specs or None)
            content_blocks = data.get("content", [])
            usage = data.get("usage", {})
            total_in += usage.get("input_tokens", 0)
            total_out += usage.get("output_tokens", 0)

            tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
            if not tool_uses:
                final_content = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                break

            assistant_block = {
                "role": "assistant",
                "content": content_blocks,
            }
            messages.append(assistant_block)
            tool_results = []
            for b in tool_uses:
                name = b["name"]
                args = b.get("input") or {}
                try:
                    result = handle_call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = {"error": str(e)}
                tool_calls_log.append({"name": name, "args": args})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        return final_content, tool_calls_log, total_in, total_out

    def _oracle_loop_generic(self, messages, tool_specs, handle_call, max_iterations):
        """Fallback for providers without native tool-calling (ollama/generic).

        Injects the available tools into the system prompt and instructs the
        model to emit tool calls as JSON lines prefixed with ``TOOL:``.
        """
        tool_desc = "\n".join(
            f"- {s.get('name')}: {s.get('description', '')}\n  input: {json.dumps(s.get('input_schema') or s.get('function', {}).get('parameters') or {}, ensure_ascii=False)}"
            for s in tool_specs
        )
        oracle_system = (
            f"You are an autonomous agent. You have access to these tools:\n{tool_desc}\n\n"
            "To call a tool, reply exactly with one JSON object on a single line prefixed by 'TOOL:' like:\n"
            'TOOL: {"name": "tool_name", "args": {...}}\n\n'
            "Keep calling tools until you have enough information, then emit your final answer "
            "as plain text (a JSON object on its own is also accepted)."
        )
        sys_msgs = [{"role": "system", "content": oracle_system}]
        for m in messages:
            if m.get("role") == "system":
                continue
            sys_msgs.append(m)
        msgs = sys_msgs

        tool_calls_log = []
        total_in = 0
        total_out = 0
        final_content = ""
        for _ in range(max_iterations):
            content, tin, tout = self._chat_text(msgs)
            total_in += tin
            total_out += tout
            lines = [l for l in content.splitlines() if l.strip().startswith("TOOL:")]
            if not lines:
                final_content = content
                break
            parsed_any = False
            for line in lines:
                try:
                    call = json.loads(line.split("TOOL:", 1)[1].strip())
                except json.JSONDecodeError:
                    continue
                name = call.get("name")
                args = call.get("args") or {}
                try:
                    result = handle_call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = {"error": str(e)}
                tool_calls_log.append({"name": name, "args": args})
                msgs.append({"role": "assistant", "content": line})
                msgs.append(
                    {"role": "user", "content": f"TOOL RESULT ({name}): {json.dumps(result, ensure_ascii=False)}"}
                )
                parsed_any = True
            if not parsed_any:
                final_content = content
                break
        return final_content, tool_calls_log, total_in, total_out

    def _chat_text(self, messages: list[dict]) -> tuple[str, int, int]:
        """Single text chat call for generic providers (ollama/generic). Returns (content, in, out)."""
        if self.provider == "ollama":
            base = os.getenv("OLLAMA_URL", "http://localhost:11434")
            client = httpx.Client(base_url=base, timeout=300)
            body = {"model": self.model_id, "messages": messages, "stream": False}
            r = client.post("/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "")
            return content, len(json.dumps(messages).split()), len(content.split())

        endpoint = os.getenv("LLM_ENDPOINT", "")
        if not endpoint:
            raise RuntimeError("LLM_ENDPOINT environment variable required for generic provider")
        client = httpx.Client(base_url=endpoint, timeout=120)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model_id, "messages": messages, "max_tokens": self.max_tokens}
        r = _post_with_retry(client, "/chat/completions", json=body, headers=headers)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _oracle_mock(self, prompt: str, handle_call) -> tuple[str, list[dict]]:
        """Deterministic mock for testing the oracle plumbing without a provider."""
        context = handle_call("get_context", {})
        candidates = handle_call("get_candidates", {}) or {"candidates": []}
        pool = candidates.get("candidates", [])
        # Pick up to num_seats candidates, one per needed role (best skill match).
        roles = context.get("roles", [])
        chosen = []
        for role in roles:
            rname = (role.get("role") or "").lower()
            for c in pool:
                if str(c.get("role") or "").lower() == rname:
                    chosen.append(c.get("id"))
                    break
        result = {
            "team": [{"id": cid} for cid in chosen],
            "justification": f"Mock oracle selection of {len(chosen)} candidates covering available roles.",
        }
        calls = [
            {"name": "get_context", "args": {}},
            {"name": "get_candidates", "args": {}},
        ]
        return json.dumps(result, ensure_ascii=False), calls

    # ========================================================================
    # Single-turn provider calls (legacy)
    # ========================================================================
    def _run_openai(self, prompt: str) -> dict:
        start = time.time()
        client = httpx.Client(base_url="https://api.openai.com/v1", timeout=120)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        r = _post_with_retry(client, "/chat/completions", json=body, headers=headers)
        data = r.json()
        elapsed = int((time.time() - start) * 1000)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return {
            "output": choice["message"]["content"],
            "execution_time_ms": elapsed,
            "tokens_input": usage.get("prompt_tokens", 0),
            "tokens_output": usage.get("completion_tokens", 0),
        }

    def _run_anthropic(self, prompt: str) -> dict:
        start = time.time()
        client = httpx.Client(base_url="https://api.anthropic.com/v1", timeout=120)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = client.post("/messages", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        elapsed = int((time.time() - start) * 1000)
        return {
            "output": data["content"][0]["text"],
            "execution_time_ms": elapsed,
            "tokens_input": data.get("usage", {}).get("input_tokens", 0),
            "tokens_output": data.get("usage", {}).get("output_tokens", 0),
        }

    def _run_ollama(self, prompt: str) -> dict:
        start = time.time()
        base = os.getenv("OLLAMA_URL", "http://localhost:11434")
        client = httpx.Client(base_url=base, timeout=300)
        body = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
        }
        r = client.post("/api/generate", json=body)
        r.raise_for_status()
        data = r.json()
        elapsed = int((time.time() - start) * 1000)
        return {
            "output": data.get("response", ""),
            "execution_time_ms": elapsed,
            "tokens_input": data.get("prompt_eval_count", 0),
            "tokens_output": data.get("eval_count", 0),
        }

    def _run_mock(self, prompt: str) -> dict:
        mock_response = f"Mock response for: {prompt[:50]}..."
        return {
            "output": mock_response,
            "execution_time_ms": 42,
            "tokens_input": len(prompt.split()),
            "tokens_output": len(mock_response.split()),
            "score": 100,
        }

    def _run_generic(self, prompt: str) -> dict:
        start = time.time()
        endpoint = os.getenv("LLM_ENDPOINT", "")
        if not endpoint:
            raise RuntimeError("LLM_ENDPOINT environment variable required for generic provider")
        client = httpx.Client(base_url=endpoint, timeout=120)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        r = _post_with_retry(client, "/chat/completions", json=body, headers=headers)
        data = r.json()
        elapsed = int((time.time() - start) * 1000)
        return {
            "output": data["choices"][0]["message"]["content"],
            "execution_time_ms": elapsed,
            "tokens_input": data.get("usage", {}).get("prompt_tokens", 0),
            "tokens_output": data.get("usage", {}).get("completion_tokens", 0),
        }
