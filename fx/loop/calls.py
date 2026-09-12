"""What every step shares to talk to the model: the token ceiling, the reply schemas, the streaming fan-out, and the
Stopped signal."""
from __future__ import annotations


from ..llm import Client
from ..llm.pool import _sem
from ..llm.registry import resolve
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_TOKENS = 32768
ASSIGN_SCHEMA = {"type": "object", "properties": {"assignments": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "feature": {"type": ["string", "null"]}, "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}, "required": ["id", "feature", "confidence"]}}}, "required": ["assignments"]}
NAME_SCHEMA = {"type": "object", "properties": {
    "decision": {"type": "string", "enum": ["variant", "feature", "same", "reject"]}, "why": {"type": "string"}, "parent": {"type": ["string", "null"]},
    "group": {"type": ["string", "null"]}, "aspect": {"type": ["string", "null"]},
    "name": {"type": "string"}, "definition": {"type": "string"}, "polarity": {"type": "string", "enum": ["require", "forbid"]},
    "examples": {"type": "array", "items": {"type": "string"}}, "members": {"type": "array", "items": {"type": "string"}}},
    "required": ["decision", "why", "parent", "group", "name", "definition", "polarity", "examples", "members"], "additionalProperties": False}


JOIN_SCHEMA = {"type": "object", "properties": {
    "proposals": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "verdict": {"type": "string", "enum": ["new", "existing", "duplicate"]},
                                                                             "feature": {"type": ["string", "null"]}, "of": {"type": ["string", "null"]}}, "required": ["id", "verdict"]}},
    "place": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "group": {"type": "string"}}, "required": ["id", "group"]}},
    "groups": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "aspect": {"type": "string"},
                                                                          "ids": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "definition", "aspect", "ids"]}},
    "pairs": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "verdict": {"type": "string", "enum": ["same", "two"]}}, "required": ["id", "verdict"]}}},
    "required": ["proposals"]}


class Stopped(Exception):
    pass


def _stream(client: Client, prompts: list[str], model: str, workers: int, note: str, system: str, schema=None, extra=None):
    """Yield (index, reply) as the calls land, under the endpoint's admission; a denial raises."""
    sem = _sem(resolve(model, client.base_url).base_url, max(workers, 1))

    def one(k):
        with sem:
            return k, client.complete(prompts[k], model=model, max_tokens=MAX_TOKENS, extra_body=extra, stage="library", note=note, system=system, schema=schema)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(one, k) for k in range(len(prompts))]):
            k, r = fut.result()
            if r.error and r.error.startswith("denied"):
                raise RuntimeError(r.error)
            yield k, r
