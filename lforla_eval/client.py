"""HTTP client for the LFORLA API."""

import json
import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "https://lforla.org/api/v1"
CONFIG_DIR = Path.home() / ".config" / "lforla"
CONFIG_FILE = CONFIG_DIR / "config.json"


class LforlaClient:
    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self.api_url = (api_url or os.getenv("LFORLA_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.getenv("LFORLA_API_KEY") or self._load_key()
        self._client = httpx.Client(base_url=self.api_url, timeout=60)
        if self.api_key:
            self._client.headers["X-API-Key"] = self.api_key

    def _load_key(self) -> str | None:
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                return data.get("api_key")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def save_key(self, api_key: str) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {"api_key": api_key}
        CONFIG_FILE.write_text(json.dumps(data, indent=2))

    def request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.api_url}{path}"
        r = self._client.request(method, url, **kwargs)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = r.text[:500]
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {detail}") from e
        if r.status_code == 204:
            return None
        return r.json()

    def get(self, path: str, params: dict | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, json_body: dict | None = None) -> Any:
        return self.request("POST", path, json=json_body)

    # ---- High-level API methods ----

    def get_benchmark(self, slug: str) -> dict:
        return self.get(f"/benchmarks/{slug}")

    def download_benchmark(self, slug: str) -> dict:
        return self.get(f"/benchmarks/{slug}/download")

    def get_dataset_samples(self, dataset_id: str) -> dict:
        return self.get(f"/datasets/{dataset_id}/samples")

    def submit_evaluation_run(self, data: dict) -> dict:
        return self.post("/evaluation-runs", data)

    def list_evaluation_runs(self, params: dict | None = None) -> list:
        return self.get("/evaluation-runs", params=params)

    def get_evaluation_run(self, run_id: str) -> dict:
        return self.get(f"/evaluation-runs/{run_id}")
