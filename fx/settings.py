"""Keys and endpoints by reference. Secrets never enter the store, a profile, a log or a page: they live in the shell's
environment or in one key file under the user's home that the site can write and never echoes.

    ~/.config/feature_extract/keys.env      KEY=value lines, mode 600 (FX_KEYS overrides the path)

`load()` puts the file's values into the environment where the environment has none (the shell wins), and runs when
`fx` is imported, so the CLI and the site read the same file. `status()` says which keys are set and where from, not
what they are. `probe()` asks an endpoint for its model list, the one call that costs nothing.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Optional

KEY_FILE = Path(os.environ.get("FX_KEYS", "~/.config/feature_extract/keys.env")).expanduser()
KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "FX_API_KEY")                 # the secrets: written blind, never shown
SETTINGS = ("FX_LOCAL_URL", "FX_PROVIDER", "FX_MODEL")                       # plain settings the same file may hold; shown
_from_file: set[str] = set()


def secret(name: str) -> bool:
    return name.endswith("_KEY") or name.endswith("_TOKEN") or name.endswith("_SECRET")


def read_file(path: Path = KEY_FILE) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def load(path: Path = KEY_FILE) -> list[str]:
    """The file's values into the environment where it has none. Returns the names taken from the file."""
    taken = []
    for k, v in read_file(path).items():
        if not os.environ.get(k):
            os.environ[k] = v; taken.append(k); _from_file.add(k)
    return taken


def save(name: str, value: str, path: Path = KEY_FILE) -> dict:
    """Write one KEY=value into the file (mode 600, parent 700) and into this process; an empty value removes it."""
    if not name or any(c in name for c in " =\n\"'") or name.startswith("#"):
        raise ValueError("not a variable name")
    have = read_file(path)
    if value:
        have[name] = value
    else:
        have.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(f"{k}={v}\n" for k, v in have.items()))
    tmp.chmod(0o600)
    tmp.replace(path)
    if value:
        os.environ[name] = value; _from_file.add(name)
    else:
        os.environ.pop(name, None); _from_file.discard(name)
    return {"name": name, "set": bool(value), "file": str(path)}


def status(path: Path = KEY_FILE) -> dict:
    """What is set and where from; a secret's value is never included, only whether it is set and its length."""
    from .llm.registry import LOCAL, OPENAI, OPENROUTER, local_url
    in_file = read_file(path)
    mode = None
    if path.exists():
        m = stat.S_IMODE(path.stat().st_mode); mode = f"{m:o}"
    def one(name: str) -> dict:
        v = os.environ.get(name, "")
        src = "file" if name in _from_file or (name in in_file and in_file[name] == v and v) else ("env" if v else None)
        d = {"name": name, "set": bool(v), "source": src, "secret": secret(name)}
        if secret(name):
            d["length"] = len(v)
        else:
            d["value"] = v
        return d
    endpoints = [{"name": OPENAI.name, "base_url": OPENAI.base_url, "key_env": OPENAI.key_env}, {"name": OPENROUTER.name, "base_url": OPENROUTER.base_url, "key_env": OPENROUTER.key_env},
                 {"name": LOCAL.name, "base_url": local_url(), "key_env": "", "local": True}]
    return {"file": str(path), "exists": path.exists(), "mode": mode, "loose": bool(mode) and mode not in ("600", "400"),
            "keys": [one(k) for k in KEYS], "settings": [one(k) for k in SETTINGS], "endpoints": endpoints,
            "extra": sorted(k for k in in_file if k not in KEYS and k not in SETTINGS)}


def probe(name: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 15.0) -> dict:
    """GET /models on an endpoint with its key: reachable, authorised, and how many models it lists. No cost."""
    import httpx
    from .llm.registry import LOCAL, OPENAI, OPENROUTER, local_url, resolve
    if base_url:
        ep = resolve("", base_url)
    else:
        ep = {"openai": OPENAI, "openrouter": OPENROUTER, "local": LOCAL}.get(name or "", LOCAL)
    url = (local_url() if ep.local and not base_url else ep.base_url).rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key and ep.api_key != "EMPTY" else {}
    t = time.time()
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
        ms = round(1000 * (time.time() - t))
        if r.status_code != 200:
            body = r.text[:200].replace(ep.api_key, "***") if ep.api_key else r.text[:200]
            return {"ok": False, "status": r.status_code, "ms": ms, "error": body, "url": url}
        try:
            data = r.json().get("data", [])
            names = [d.get("id", "") for d in data if isinstance(d, dict)][:400]
        except ValueError:
            names = []
        return {"ok": True, "status": 200, "ms": ms, "models": len(names), "sample": names[:12], "url": url}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": None, "ms": round(1000 * (time.time() - t)), "error": type(e).__name__ + ": " + str(e)[:200], "url": url}
