"""Where a model lives and what it costs.

An endpoint is an OpenAI-compatible base URL plus the environment variable that
holds its key. A model name resolves to an endpoint by prefix rule, or explicitly
through FX_MODELS (a JSON file) so a deployment can add models without code.

    resolve("gpt-5.6-sol")             -> Endpoint(openai)
    resolve("deepseek/deepseek-v4-pro") -> Endpoint(openrouter)
    resolve("openai/gpt-oss-20b")       -> Endpoint(local vLLM on FX_LOCAL_URL)

Prices are dollars per million tokens, (prompt, completion). Unknown models
are priced at the hosted default so a missing entry over-counts, never under.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOCAL_URL = os.environ.get("FX_LOCAL_URL", "http://localhost:8000/v1")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_URL = "https://api.openai.com/v1"

PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (2.0, 10.0), "gpt-5.6-luna": (0.2, 1.2), "gpt-5.6-terra": (2.0, 12.0),
    "deepseek/deepseek-v4-flash": (0.0886, 0.1772), "deepseek/deepseek-v4-flash-0731": (0.065, 0.18),
    "deepseek/deepseek-v4-pro": (0.9553, 1.9105), "qwen/qwen3.8-max-0902": (2.0, 6.0),
}
DEFAULT_PRICE = (2.0, 10.0)
LOCAL_PRICE = (0.0, 0.0)


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    key_env: str            # environment variable holding the API key; "" for a local server
    local: bool = False

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else "EMPTY"


OPENAI = Endpoint("openai", OPENAI_URL, "OPENAI_API_KEY")
OPENROUTER = Endpoint("openrouter", OPENROUTER_URL, "OPENROUTER_API_KEY")
LOCAL = Endpoint("local", LOCAL_URL, "", local=True)


def _custom() -> dict:
    p = os.environ.get("FX_MODELS")
    if p and Path(p).exists():
        return json.load(open(p))
    return {}


def resolve(model: str, base_url: Optional[str] = None) -> Endpoint:
    """base_url overrides everything (any OpenAI-compatible server, key from FX_API_KEY)."""
    if base_url:
        return Endpoint("custom", base_url, "FX_API_KEY", local=base_url.startswith("http://localhost") or base_url.startswith("http://127."))
    c = _custom().get(model)
    if c:
        return Endpoint(c.get("name", "custom"), c["base_url"], c.get("key_env", "FX_API_KEY"), bool(c.get("local")))
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return OPENAI
    if "/" in model and model.split("/", 1)[0] in ("deepseek", "qwen", "anthropic", "google", "meta-llama", "mistralai", "x-ai"):
        return OPENROUTER
    return LOCAL


def price(model: str, endpoint: Optional[Endpoint] = None) -> tuple[float, float]:
    c = _custom().get(model)
    if c and "price" in c:
        return tuple(c["price"])  # type: ignore[return-value]
    if model in PRICES:
        return PRICES[model]
    if endpoint is not None and endpoint.local:
        return LOCAL_PRICE
    return DEFAULT_PRICE


def cost(model: str, prompt_tokens: int, completion_tokens: int, endpoint: Optional[Endpoint] = None) -> float:
    pi, po = price(model, endpoint)
    return prompt_tokens * pi / 1e6 + completion_tokens * po / 1e6
