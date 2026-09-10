"""One client for every OpenAI-compatible endpoint, returning a Reply, never raising
for a model-side failure.

    from fx.llm import Client
    c = Client(store, budget=50.0)
    r = c.complete("Say hi", model="openai/gpt-oss-20b", stage="probe")
    r.text, r.usage, r.cost, r.cached, r.error

What it handles, learned on the FACET runs:
- parameter shape: hosted models want max_completion_tokens and may reject a
  temperature; the shape is probed once per model and remembered.
- empty replies with no HTTP error (vLLM under load, a reasoning model that
  spent its budget thinking): retried a few times, then returned as error.
- transport failures the SDK does not retry: retried with backoff.
- denied or unreachable endpoints: returned as an error and never logged as a
  call with cost, so a resume refills the hole for free.
- streaming for long generations: a non-streamed 1M-context call can hang in
  CLOSE-WAIT; with stream=True chunks are collected as they arrive.
- the cache: a reply for the same (model, prompt, params) is served from the
  store at no cost; pass cache=False for sampling.
- the ledger: every non-cached call is priced and logged; the budget is checked
  before each call and BudgetExceeded stops the stage.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from ..store import Store, now
from .registry import Endpoint, cost as price_of, resolve

Messages = Union[str, Sequence[dict[str, str]]]

DENIED = ("401", "402", "403", "missing authentication", "invalid api key", "incorrect api key", "key limit exceeded",
          "insufficient credit", "out_of_credits", "unauthorized", "authenticationerror", "permissiondeniederror",
          "notfounderror", "error code: 404", "model_not_found")
UNREACHABLE = ("connection refused", "connection error", "connection reset", "connect error", "connecterror",
               "apiconnectionerror", "cannot connect to host", "no route to host", "network is unreachable",
               "name or service not known", "timed out", "timeout")


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    model: str
    prompt_sha: str
    usage: dict = field(default_factory=dict)      # prompt_tokens, completion_tokens (estimated when the server sends none)
    cost: float = 0.0
    latency: float = 0.0
    finish_reason: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None                     # "denied: ...", "unreachable: ...", "empty", "transport: ..."
    call_id: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


def sha_of(model: str, messages: Messages, params: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def _to_messages(m: Messages) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(m)}] if isinstance(m, str) else [dict(x) for x in m]


def _classify(err: Exception) -> Optional[str]:
    m = f"{type(err).__name__}: {err}".lower()
    if any(k in m for k in DENIED):
        return "denied"
    if any(k in m for k in UNREACHABLE):
        return "unreachable"
    return None


class Client:
    def __init__(self, store: Optional[Store] = None, *, budget: float = float("inf"), timeout: float = 300.0,
                 max_retries: int = 6, empty_retries: int = 3, max_connections: int = 256, base_url: Optional[str] = None,
                 log_replies: bool = True, stop: Optional[threading.Event] = None):
        self.store = store
        self.stop = stop                        # set it and every retry loop returns error 'stopped'; close() aborts in-flight calls
        self.budget = float(budget)
        self.timeout = timeout
        self.max_retries = max_retries
        self.empty_retries = empty_retries
        self.max_connections = max_connections
        self.base_url = base_url
        self.log_replies = log_replies
        self._clients: dict[str, Any] = {}
        self._shape: dict[str, dict] = {}       # per model: token_key, send_temp
        self._lock = threading.Lock()
        self.spent_session = 0.0

    # ---- endpoint plumbing
    def _sdk(self, ep: Endpoint):
        with self._lock:
            if ep.base_url not in self._clients:
                from openai import OpenAI
                import httpx
                http = httpx.Client(limits=httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=self.max_connections), timeout=self.timeout)
                c = OpenAI(base_url=ep.base_url, api_key=ep.api_key or "EMPTY", timeout=self.timeout, max_retries=0, http_client=http)
                _ = c.chat.completions   # warm the lazy property under one lock, not in a worker pool
                self._clients[ep.base_url] = c
            return self._clients[ep.base_url]

    def close(self) -> None:
        """Close every HTTP client so calls waiting on a server return at once (as transport errors)."""
        with self._lock:
            cs, self._clients = list(self._clients.values()), {}
        for c in cs:
            try:
                c.close()
            except Exception:
                pass

    def _stopped(self) -> bool:
        return self.stop is not None and self.stop.is_set()

    def spent(self) -> float:
        return self.store.spent() if self.store else self.spent_session

    def check_budget(self) -> None:
        s = self.spent()
        if s >= self.budget:
            raise BudgetExceeded(f"spent ${s:.2f} of the ${self.budget:.2f} budget")

    # ---- the call
    def complete(self, messages: Messages, *, model: str, max_tokens: Optional[int] = 8192, temperature: float = 0.0,
                 extra_body: Optional[dict] = None, stream: bool = False, cache: bool = True, stage: str = "", note: str = "",
                 system: Optional[str] = None, on_chunk=None, schema: Optional[dict] = None) -> Reply:
        """schema: a JSON schema the reply must satisfy; sent as response_format json_schema, which vLLM
        and OpenAI enforce at decoding time. A server that rejects response_format has it dropped for
        that model from then on, so a schema is a request, not a guarantee: parse the reply anyway."""
        msgs = _to_messages(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        params = {"max_tokens": max_tokens, "temperature": temperature, "extra_body": extra_body or None, "schema": schema}
        sha = sha_of(model, msgs, params)
        ep = resolve(model, self.base_url)
        if cache and self.store is not None:
            hit = self.store.cached_reply(model, sha)
            if hit is not None:
                r = Reply(hit, model, sha, cached=True)
                r.call_id = self._log(r, ep, stage, note, msgs)
                return r
        self.check_budget()
        t0 = time.time()
        text, usage, finish, err = self._call(ep, model, msgs, params, stream, on_chunk)
        latency = time.time() - t0
        if usage is None:
            usage = {"prompt_tokens": int(sum(len(m["content"]) for m in msgs) / 3.8), "completion_tokens": int(len(text) / 3.8), "estimated": True}
        r = Reply(text, model, sha, usage=usage, latency=round(latency, 2), finish_reason=finish, error=err)
        if err is None or err == "empty":
            r.cost = price_of(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), ep)
            self.spent_session += r.cost
            r.call_id = self._log(r, ep, stage, note, msgs)
        return r

    def _create(self, sdk, model: str, msgs, params: dict, stream: bool):
        shape = self._shape.setdefault(model, {"token_key": "max_tokens", "send_temp": True, "schema": True})
        tried_token = tried_temp = tried_schema = False
        while True:
            kw: dict[str, Any] = {"model": model, "messages": msgs}
            if params.get("extra_body"):
                kw["extra_body"] = params["extra_body"]
            if params.get("schema") and shape["schema"]:
                kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": params["schema"]}}
            if params.get("max_tokens") is not None:
                kw[shape["token_key"]] = params["max_tokens"]
            if shape["send_temp"]:
                kw["temperature"] = params.get("temperature", 0.0)
            if stream:
                kw["stream"] = True
                kw["stream_options"] = {"include_usage": True}
            try:
                return sdk.chat.completions.create(**kw)
            except Exception as e:
                m = str(e).lower()
                if "max_completion_tokens" in m and not tried_token:
                    shape["token_key"] = "max_completion_tokens"; tried_token = True; continue
                if "temperature" in m and not tried_temp and ("unsupported" in m or "does not support" in m or "only the default" in m):
                    shape["send_temp"] = False; tried_temp = True; continue
                if "response_format" in m and not tried_schema and shape["schema"]:
                    shape["schema"] = False; tried_schema = True; continue
                raise

    def _call(self, ep: Endpoint, model: str, msgs, params: dict, stream: bool, on_chunk):
        sdk = self._sdk(ep)
        text, usage, finish = "", None, None
        for attempt in range(self.empty_retries + 1):
            resp = None
            for t in range(self.max_retries + 1):
                if self._stopped():
                    return "", None, None, "stopped"
                try:
                    resp = self._create(sdk, model, msgs, params, stream)
                    break
                except Exception as e:
                    kind = _classify(e)
                    if kind == "denied":
                        return "", None, None, f"denied: {type(e).__name__}: {str(e)[:200]}"
                    if t >= self.max_retries:
                        return "", None, None, f"{kind or 'transport'}: {type(e).__name__}: {str(e)[:200]}"
                    time.sleep(min(30.0, 2.0 * (t + 1)) + random.uniform(0, 1.5))
            if stream:
                parts = []
                try:
                    for chunk in resp:
                        if getattr(chunk, "usage", None):
                            u = chunk.usage
                            usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
                        if chunk.choices:
                            d = chunk.choices[0].delta
                            if d and d.content:
                                parts.append(d.content)
                                if on_chunk:
                                    on_chunk(d.content)
                            if chunk.choices[0].finish_reason:
                                finish = chunk.choices[0].finish_reason
                except Exception as e:
                    return "".join(parts), usage, finish, f"transport: stream broke: {type(e).__name__}: {str(e)[:200]}"
                text = "".join(parts)
            else:
                u = getattr(resp, "usage", None)
                if u is not None:
                    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
                ch = resp.choices[0] if resp.choices else None
                text = (ch.message.content or "") if ch else ""
                finish = getattr(ch, "finish_reason", None) if ch else None
            if text.strip():
                return text, usage, finish, None
            if finish == "length":                 # the ceiling was spent before any text: the same request truncates the same way
                return text, usage, finish, "empty"
            if self._stopped():
                return text, usage, finish, "stopped"
            if attempt < self.empty_retries:
                time.sleep(min(10.0, 2.0 * (attempt + 1)) + random.uniform(0, 1.0))
        return text, usage, finish, "empty"

    def _log(self, r: Reply, ep: Endpoint, stage: str, note: str, msgs) -> Optional[int]:
        if self.store is None:
            return None
        return self.store.insert("call", {
            "at": now(), "stage": stage or None, "note": note or None, "model": r.model, "base_url": ep.base_url,
            "prompt_sha": r.prompt_sha, "prompt_chars": sum(len(m["content"]) for m in msgs), "reply_chars": len(r.text),
            "prompt_tokens": r.usage.get("prompt_tokens"), "completion_tokens": r.usage.get("completion_tokens"),
            "cost": 0.0 if r.cached else r.cost, "latency": r.latency, "finish_reason": r.finish_reason,
            "cached": int(r.cached), "error": r.error, "reply": r.text if self.log_replies else None})


def with_fallback(client: Client, primary: dict, *fallbacks: dict):
    """A callable messages -> Reply that tries the primary model settings, then each fallback on an empty or errored reply.
    Each dict holds complete() keyword arguments (model, extra_body, max_tokens ...)."""
    def call(messages: Messages, **kw) -> Reply:
        last = None
        for settings in (primary, *fallbacks):
            r = client.complete(messages, **{**settings, **kw})
            if r.ok:
                return r
            last = r
            if r.error and r.error.startswith("denied"):
                break
        return last  # type: ignore[return-value]
    return call
