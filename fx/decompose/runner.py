"""The runner for stage 1: which prompts, what it will cost, and the run itself over the store.

    from fx.decompose import run, preview
    preview(store, corpus="text2sql", model="openai/gpt-oss-20b", workers=8)   -> the cost preview
    run(store, client, corpus="text2sql", model=..., workers=8, progress=cb)   -> per-prompt rows

Both the CLI and the GUI call these and nothing else calls the model for this stage. `run` is
resumable: a prompt with status done is skipped unless redo=True; a prompt that failed or was
interrupted is redone from scratch. Each finished prompt is written in one transaction: its
atoms, readings, gaps, and the decomp summary row. `progress(done, total, info)` fires after
every prompt; raising Stop from it ends the run cleanly.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ..llm import Client
from ..llm.pool import _sem
from ..llm.registry import price, reasoning_off, resolve
from ..store import Store, now
from .profile import metrics, profile
from .prompts import COMPONENTS_SCHEMA, SYSTEM

# defaults from the FACET v5 trees (2,740 prompts): calls ≈ 3 + 1.18 per 1k chars, 1,753 tokens in and 169 out per call
CALLS_BASE, CALLS_PER_KCHAR, TOKENS_IN, TOKENS_OUT = 3.0, 1.18, 1753, 169
MAX_TOKENS = 4096
REASONING = "off"      # the default: with reasoning on, gpt-oss-20b spent the whole 4,096-token reply budget thinking and returned a truncated reply after 74 s


class Stop(Exception):
    pass


def prompt_ids(store: Store, corpus: Optional[str] = None, ids: Optional[list[str]] = None, redo: bool = False, limit: int = 0) -> list[str]:
    if ids:
        rows = [{"id": i} for i in ids]
    elif corpus:
        rows = store.rows("SELECT p.id FROM prompt p JOIN corpus c ON c.id=p.corpus WHERE c.name=? ORDER BY p.rowid", (corpus,))
    else:
        rows = store.rows("SELECT id FROM prompt ORDER BY rowid")
    out = [r["id"] for r in rows]
    if not redo:
        done = {r["prompt"] for r in store.rows("SELECT prompt FROM decomp WHERE status='done'")}
        out = [i for i in out if i not in done]
    return out[:limit] if limit else out


def history(store: Store, model: str) -> dict:
    """What this store has seen for this model: calls per 1k chars and tokens per call, from finished prompts."""
    r = store.one("SELECT COUNT(*) n, SUM(calls) calls, SUM(chars) chars FROM decomp WHERE status='done' AND model=?", (model,))
    c = store.one("SELECT COUNT(*) n, AVG(prompt_tokens) tin, AVG(completion_tokens) tout, AVG(latency) lat FROM call WHERE stage='decompose' AND model=? AND cached=0 AND error IS NULL", (model,))
    out = {"prompts_seen": int(r["n"] or 0), "calls_seen": int(c["n"] or 0)}
    if r and (r["n"] or 0) >= 20 and r["chars"]:
        out["calls_per_kchar"] = round(1000 * r["calls"] / r["chars"], 3)
    if c and (c["n"] or 0) >= 50:
        out["tokens_in"], out["tokens_out"], out["seconds_per_call"] = round(c["tin"] or 0), round(c["tout"] or 0), round(c["lat"] or 0, 2)
    return out


def preview(store: Store, corpus: Optional[str] = None, model: str = "deepseek/deepseek-v4-flash", workers: int = 128, ids: Optional[list[str]] = None, redo: bool = False, limit: int = 0) -> dict:
    pids = prompt_ids(store, corpus, ids, redo, limit)
    if not pids:
        return {"prompts": 0, "note": "nothing to do"}
    q = ",".join("?" for _ in pids)
    chars = int(store.one(f"SELECT SUM(LENGTH(text)) c FROM prompt WHERE id IN ({q})", pids)["c"] or 0)
    h = history(store, model)
    per_k = h.get("calls_per_kchar")
    calls = int(round(len(pids) * CALLS_BASE + (per_k if per_k else CALLS_PER_KCHAR) * chars / 1000)) if not per_k else int(round(per_k * chars / 1000 + len(pids) * 1.0))
    tin, tout = h.get("tokens_in", TOKENS_IN), h.get("tokens_out", TOKENS_OUT)
    ep = resolve(model)
    pi, po = price(model, ep)
    dollars = calls * (tin * pi + tout * po) / 1e6
    spc = h.get("seconds_per_call")
    out = {"prompts": len(pids), "chars": chars, "calls": calls, "tokens_in": calls * tin, "tokens_out": calls * tout,
           "model": model, "endpoint": ep.name, "dollars": round(dollars, 2), "workers": workers, "basis": "this store" if per_k else "FACET v5 defaults", "history": h}
    if spc:
        out["seconds"] = round(calls * spc / max(workers, 1))
    else:
        out["seconds"] = None
        out["note"] = "no timing for this model yet: run a pilot to measure seconds per call"
    return out


def _clear(store: Store, pid: str) -> None:
    with store.lock:
        for t in ("atom", "reading", "gap"):
            store.con.execute(f"DELETE FROM {t} WHERE prompt=?", (pid,))
        store.con.execute("DELETE FROM decomp WHERE prompt=?", (pid,))
        store.con.commit()


def _write(store: Store, pid: str, text: str, tree, model: str) -> dict:
    m = metrics(tree, text)
    with store.lock:
        con = store.con
        for t in ("atom", "reading", "gap"):
            con.execute(f"DELETE FROM {t} WHERE prompt=?", (pid,))
        for part, depth, path in tree.walk():
            if part.span is None:
                continue
            cur = con.execute("INSERT INTO atom (prompt, path, lo, hi, kind, material, flags, start, end, depth) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (pid, path, part.span[0], part.span[1], "section" if (part.kind == "section" and part.children) else ("atom" if part.kind == "atom" else ("material" if part.kind == "material" else "unrefined")),
                               part.material, json.dumps(part.flags), part.start, part.end, depth))
            aid = cur.lastrowid
            if part.kind == "atom" and part.is_leaf:
                for f in part.facets:
                    con.execute("INSERT INTO reading (prompt, atom, verb, object, qualifier, polarity, condition, domain_terms, declaration) VALUES (?,?,?,?,?,?,?,?,?)",
                                (pid, aid, f.verb, f.object, f.qualifier, f.polarity, f.condition, json.dumps(f.domain_terms), f.declaration))
        for g in tree.gaps:
            con.execute("INSERT INTO gap (prompt, lo, hi, outcome, path) VALUES (?,?,?,?,?)", (pid, g["lo"], g["hi"], g["outcome"], g.get("path")))
        con.execute("INSERT OR REPLACE INTO decomp (prompt, status, model, chars, instruction_chars, covered_chars, coverage, material_share, n_atoms, n_material, n_readings, calls, seconds, reasks, gaps, flags, failures, error, at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, "done", model, m["chars"], m["instruction_chars"], m["covered_chars"], m["coverage"], m["material_share"], m["n_atoms"], m["n_material"], m["n_readings"],
                     m["calls"], m["seconds"], m["reasks"], json.dumps(m["gaps"]), json.dumps(tree.flags), json.dumps(tree.failures[:50]), None, now()))
        con.commit()
    return m


def decompose_one(store: Store, client: Client, pid: str, model: str, reasoning: str = REASONING, max_tokens: int = MAX_TOKENS) -> dict:
    row = store.one("SELECT text FROM prompt WHERE id=?", (pid,))
    if not row:
        raise KeyError(pid)
    text = row["text"]
    extra_body = reasoning_off(model, client.base_url) if reasoning == "off" else None
    calls = {"n": 0, "cost": 0.0, "errors": 0}

    def ask(prompt: str) -> str:
        r = client.complete(prompt, model=model, max_tokens=max_tokens, extra_body=extra_body, stage="decompose", note=pid, system=SYSTEM, schema=COMPONENTS_SCHEMA)
        calls["n"] += 1
        calls["cost"] += r.cost
        if r.error:
            calls["errors"] += 1
            if r.error.startswith("denied"):
                raise RuntimeError(r.error)
        return r.text

    try:
        tree = profile(text, ask)
    except Exception as e:
        _clear(store, pid)
        store.insert("decomp", {"prompt": pid, "status": "failed", "model": model, "error": f"{type(e).__name__}: {str(e)[:300]}", "at": now()})
        raise
    m = _write(store, pid, text, tree, model)
    return {"id": pid, **m, "cost": calls["cost"], "call_errors": calls["errors"]}


def run(store: Store, client: Client, corpus: Optional[str] = None, *, model: str = "deepseek/deepseek-v4-flash", workers: int = 128, ids: Optional[list[str]] = None,
        redo: bool = False, limit: int = 0, reasoning: str = REASONING, max_tokens: int = MAX_TOKENS,
        progress: Optional[Callable[[int, int, dict], None]] = None, max_inflight: Optional[int] = None) -> dict:
    pids = prompt_ids(store, corpus, ids, redo, limit)
    total = len(pids)
    summary = {"total": total, "done": 0, "failed": 0, "calls": 0, "cost": 0.0, "seconds": 0.0, "stopped": False}
    if not pids:
        return summary
    sem = _sem(resolve(model, client.base_url).base_url, max_inflight or max(workers, 1))
    stop = threading.Event()
    t0 = time.time()

    def one(pid):
        if stop.is_set():
            return pid, None, "stopped"
        with sem:
            if stop.is_set():
                return pid, None, "stopped"
            try:
                return pid, decompose_one(store, client, pid, model, reasoning, max_tokens), None
            except Exception as e:
                if "denied" in str(e):
                    stop.set()
                return pid, None, f"{type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, p) for p in pids]
        for f in as_completed(futs):
            pid, m, err = f.result()
            if err == "stopped":
                continue
            if err:
                summary["failed"] += 1
            else:
                summary["done"] += 1; summary["calls"] += m["calls"]; summary["cost"] += m["cost"]
            summary["seconds"] = round(time.time() - t0, 1)
            if progress:
                try:
                    progress(summary["done"] + summary["failed"], total, {"id": pid, "error": err, **(m or {})})
                except Stop:
                    stop.set()
                    summary["stopped"] = True
    return summary
