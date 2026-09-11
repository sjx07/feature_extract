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
DEFAULT_MODEL = os.environ.get("FX_MODEL", "deepseek/deepseek-v4-flash-0731")
LOCAL_EXAMPLES = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def models() -> list[dict]:
    """Every model the registry knows, with its endpoint and price; what the CLI and the site list."""
    names = list(PRICES) + [m for m in LOCAL_EXAMPLES if m not in PRICES] + [m for m in _custom() if m not in PRICES]
    out = []
    for m in names:
        ep = resolve(m); pi, po = price(m, ep)
        out.append({"model": m, "endpoint": ep.name, "price_in": pi, "price_out": po, "default": m == DEFAULT_MODEL})
    return out

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


def provider_body(model: str, base_url: Optional[str] = None) -> dict:
    """OpenRouter upstream routing. Default: sort candidates by throughput with fallbacks allowed (FACET measured
    20 tok/s upstreams turning a 15k-token reply into a 12-minute call). FX_PROVIDER_SORT=throughput|latency|price|''
    changes the sort, FX_PROVIDER_ORDER=A,B,C names upstreams to try first, FX_PROVIDER=Name pins one."""
    if resolve(model, base_url).name != "openrouter":
        return {}
    only = os.environ.get("FX_PROVIDER", "Wafer")
    if only:
        return {"provider": {"only": [only], "allow_fallbacks": False}}
    prov: dict = {"allow_fallbacks": True}
    # Upstreams serving this family at fp4 loop in their reasoning: Reka's DeepSeek v4 flash ran to the 32k ceiling with no
    # text on 4 of 6 probes of a 185-char prompt (2026-09-10), and a throughput sort lands there first. FX_PROVIDER_IGNORE
    # (comma-separated) replaces the list; empty string keeps every upstream.
    # OpenInference (fp8, the cheapest) looped the same way on 3 of 3 probes at 1,400 s each; DeepInfra returned 3 of 3 clean
    # but at 400-620 s a call; Wafer returned 5 of 6 clean at 50-86 s and 1,200 run calls at 127 s with 2% empty, reasoning
    # returned in the message and counted in usage. So Wafer only by default; FX_PROVIDER_ORDER=Wafer,DeepInfra allows fallback.
    ignore = os.environ.get("FX_PROVIDER_IGNORE", "Reka,OpenInference,Relace,Sail Research,Inceptron,GMICloud,AtlasCloud")
    if ignore:
        prov["ignore"] = [x.strip() for x in ignore.split(",") if x.strip()]
    sort = os.environ.get("FX_PROVIDER_SORT", "throughput")
    if sort:
        prov["sort"] = sort
    order = [p for p in os.environ.get("FX_PROVIDER_ORDER", "Wafer,DeepInfra").split(",") if p]
    if order:
        prov["order"] = order
    return {"provider": prov}


def reasoning_low(model: str, base_url: Optional[str] = None) -> dict:
    """The extra_body for low reasoning effort: the fallback when a reasoning reply exhausted its ceiling."""
    ep = resolve(model, base_url)
    if ep.name == "openrouter":
        return {"reasoning": {"effort": "low", "exclude": True}}
    return {"reasoning_effort": "low"}


def reasoning_off(model: str, base_url: Optional[str] = None) -> dict:
    """The extra_body that turns a model's hidden reasoning down or off on its endpoint. Measured on
    2026-09-10: vLLM's gpt-oss honours a top-level reasoning_effort (0.8 s, 37 tokens for a small
    JSON reply) and ignores OpenRouter's reasoning.enabled (136 tokens); Qwen3 on vLLM wants
    chat_template_kwargs.enable_thinking; OpenRouter wants reasoning.enabled; OpenAI reasoning
    models take reasoning_effort."""
    ep = resolve(model, base_url)
    if ep.name == "openrouter":
        return {"reasoning": {"enabled": False}}
    if ep.name == "openai":
        return {"reasoning_effort": "low"}
    return {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}
