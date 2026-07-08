"""Model runner — calls LLM APIs to evaluate benchmark samples."""

import json
import os
import time
from typing import Any

import httpx


class ModelRunner:
    """Calls an LLM API for each sample and returns the result."""

    def __init__(self, model_id: str, provider: str = "openai", api_key: str | None = None):
        self.model_id = model_id
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY") or ""

    def run_sample(self, sample: dict) -> dict:
        """Run a single sample through the model."""
        input_data = sample.get("input_data", {})
        prompt = input_data.get("prompt") or input_data.get("instruction") or input_data.get("text", "")

        if self.provider == "mock":
            return self._run_mock(prompt)
        if not self.api_key and self.provider in ("openai", "anthropic"):
            raise RuntimeError(f"Missing API key. Set {self.provider.upper()}_API_KEY or pass --api-key")
        if self.provider == "openai":
            return self._run_openai(prompt)
        elif self.provider == "anthropic":
            return self._run_anthropic(prompt)
        elif self.provider == "ollama":
            return self._run_ollama(prompt)
        else:
            return self._run_generic(prompt)

    def _run_openai(self, prompt: str) -> dict:
        start = time.time()
        client = httpx.Client(base_url="https://api.openai.com/v1", timeout=120)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }
        r = client.post("/chat/completions", json=body, headers=headers)
        r.raise_for_status()
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
            "max_tokens": 2048,
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
            "max_tokens": 2048,
        }
        r = client.post("/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        elapsed = int((time.time() - start) * 1000)
        return {
            "output": data["choices"][0]["message"]["content"],
            "execution_time_ms": elapsed,
            "tokens_input": data.get("usage", {}).get("prompt_tokens", 0),
            "tokens_output": data.get("usage", {}).get("completion_tokens", 0),
        }
