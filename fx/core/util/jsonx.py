"""Reading JSON out of model replies. Every stage that parses a reply uses this, so the
tolerance is in one place: prose before the JSON, a code fence around it, and braces inside
string values (a `{question}` slot, a fenced quote) do not break the parse."""
from __future__ import annotations

import json
from typing import Any, Optional


def extract_json(reply: str) -> Any:
    """The first JSON value in a reply, or None. The decoder is tried at the whole reply, then at
    every opening brace or bracket in order."""
    s = (reply or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def extract_object(reply: str, key: Optional[str] = None) -> Optional[dict]:
    """A JSON object from a reply; with `key`, one whose value at key is a list (a bare list is wrapped under key)."""
    obj = extract_json(reply)
    if key is not None:
        if isinstance(obj, list):
            obj = {key: obj}
        if not isinstance(obj, dict) or not isinstance(obj.get(key), list):
            return None
        return obj
    return obj if isinstance(obj, dict) else None


def as_str(v: Any) -> str:
    return str(v).strip() if v is not None else ""
