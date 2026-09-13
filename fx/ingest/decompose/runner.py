"""The runner for stage 1: which prompts, what it will cost, and the run itself over the store.

    from fx.ingest.decompose import run, preview
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

from fx.core.llm import Client
from fx.core.llm.pool import _sem
from fx.core.llm.registry import DEFAULT_MODEL, price, reasoning_low, resolve
from fx.core.store import Store, now
from fx.ingest.decompose.profile import metrics, profile
from fx.ingest.decompose.prompts import COMPONENTS_SCHEMA, SYSTEM

# defaults from the FACET v5 trees (2,740 prompts): calls ≈ 3 + 1.18 per 1k chars, 1,753 tokens in and 169 out per call
# The model reasons: FACET's pilot v9 read reasoning off as worse (content typed as guidance, sentences cut, polarity
# inverted) and the stage 1 comparison agreed (2 re-asks per prompt off against 0.3 on). Preview defaults from that pilot
# on DeepSeek v4 flash: 15.8k output tokens and about 100 s per call. Hidden reasoning counts against the reply ceiling.
CALLS_BASE, CALLS_PER_KCHAR, TOKENS_IN, TOKENS_OUT, SECONDS_PER_CALL = 3.0, 1.18, 1753, 15_800, 100.0
MAX_TOKENS = 32768
MAX_EXHAUSTED = 1          # a span whose ceiling is spent twice (primary, then low effort) stops the prompt's calls: 4ce1150110dbcb85 spent 6 x 8 min this way
MAX_CALLS = 40             # calls per prompt before it stops calling; what is left stays unrefined and shows in the queues
FANOUT = 4                 # sibling sections, gaps and leaf checks of one node refined at once (profile._parallel); 1 = one after another


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
    r = store.one("SELECT COUNT(*) n, SUM(d.calls) calls, SUM(LENGTH(p.text)) chars FROM decomp d JOIN prompt p ON p.id=d.prompt WHERE d.status='done' AND d.model=?", (model,))
    c = store.one("SELECT COUNT(*) n, AVG(prompt_tokens) tin, AVG(completion_tokens) tout, AVG(latency) lat FROM call WHERE stage='decompose' AND model=? AND cached=0 AND error IS NULL", (model,))
    out = {"prompts_seen": int(r["n"] or 0), "calls_seen": int(c["n"] or 0)}
    if r and (r["n"] or 0) >= 20 and r["chars"]:
        out["calls_per_kchar"] = round(1000 * r["calls"] / r["chars"], 3)
    if c and (c["n"] or 0) >= 50:
        out["tokens_in"], out["tokens_out"], out["seconds_per_call"] = round(c["tin"] or 0), round(c["tout"] or 0), round(c["lat"] or 0, 2)
    return out


def preview(store: Store, corpus: Optional[str] = None, model: str = DEFAULT_MODEL, workers: int = 512, ids: Optional[list[str]] = None, redo: bool = False, limit: int = 0) -> dict:
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
    spc = h.get("seconds_per_call") or SECONDS_PER_CALL
    out = {"prompts": len(pids), "chars": chars, "calls": calls, "tokens_in": calls * tin, "tokens_out": calls * tout,
           "model": model, "endpoint": ep.name, "dollars": round(dollars, 2), "workers": workers, "basis": "this store" if per_k else "FACET pilot v9", "history": h}
    out["seconds"] = round(calls * spc / max(workers, 1))
    return out


def _clear(store: Store, pid: str) -> None:
    with store.lock:
        for t in ("reading", "span", "decomp"):
            store.con.execute(f"DELETE FROM {t} WHERE prompt=?", (pid,))
        store.con.commit()


def _write(store: Store, pid: str, text: str, tree, model: str) -> dict:
    m = metrics(tree, text)
    with store.lock:
        con = store.con
        for t in ("reading", "span"):
            con.execute(f"DELETE FROM {t} WHERE prompt=?", (pid,))
        for part, depth, path in tree.walk():
            if part.span is None:
                continue
            kind = "section" if (part.kind == "section" and part.children) else ("atom" if part.kind == "atom" else ("material" if part.kind == "material" else "unrefined"))
            cur = con.execute("INSERT INTO span (prompt, path, lo, hi, kind, note, flags, start, end, depth) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (pid, path, part.span[0], part.span[1], kind, part.material, json.dumps(part.flags), part.start, part.end, depth))
            if part.is_leaf:
                for f in part.readings:
                    con.execute("INSERT INTO reading (prompt, span, verb, object, qualifier, polarity, condition, domain_terms, declaration) VALUES (?,?,?,?,?,?,?,?,?)",
                                (pid, cur.lastrowid, f.verb, f.object, f.qualifier, f.polarity, f.condition, json.dumps(f.domain_terms), f.declaration))
        for k, g in enumerate(x for x in tree.gaps if x["outcome"] == "declined"):
            con.execute("INSERT INTO span (prompt, path, lo, hi, kind, note, flags, depth) VALUES (?,?,?,?,?,?,?,?)", (pid, f"gap{k}", g["lo"], g["hi"], "gap", "declined", "[]", 0))
        con.execute("INSERT OR REPLACE INTO decomp (prompt, status, model, coverage, material_share, calls, seconds, reasks, flags, failures, error, at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, "done", model, m["coverage"], m["material_share"], m["calls"], m["seconds"], m["reasks"], json.dumps(tree.flags), json.dumps(tree.failures[:50]), None, now()))
        con.commit()
    return m


def decompose_one(store: Store, client: Client, pid: str, model: str, max_tokens: int = MAX_TOKENS, sem: Optional[threading.Semaphore] = None,
                  fanout: Optional[int] = None) -> dict:
    """`sem` bounds the calls in flight across every prompt of a run; `fanout` the siblings of one node refined at once."""
    row = store.one("SELECT text FROM prompt WHERE id=?", (pid,))
    if not row:
        raise KeyError(pid)
    text = row["text"]
    calls = {"n": 0, "cost": 0.0, "errors": 0, "fallbacks": 0, "exhausted": 0, "stopped": None}
    lock = threading.Lock()
    sem = sem or threading.Semaphore(FANOUT * 4)

    def call(prompt: str, extra: Optional[dict], note: str):
        with sem:
            r = client.complete(prompt, model=model, max_tokens=max_tokens, extra_body=extra or None, stage="decompose", note=note, system=SYSTEM, schema=COMPONENTS_SCHEMA)
        with lock:
            calls["n"] += 1
            calls["cost"] += r.cost
        return r

    def ask(prompt: str) -> str:
        with lock:
            if calls["stopped"]:
                return ""
            if calls["n"] >= MAX_CALLS:
                calls["stopped"] = f"calls_stopped:{MAX_CALLS} calls"
                return ""
        r = call(prompt, None, pid)
        if r.finish_reason == "length" and not r.text.strip():
            # the model spent the ceiling thinking: once more at low effort (FACET's fallback; 183 of 2,068 prompts needed it)
            with lock:
                calls["fallbacks"] += 1
            r = call(prompt, reasoning_low(model, client.base_url), pid + " fallback:low")
        if r.error:
            with lock:
                calls["errors"] += 1
            if r.error.startswith("denied"):
                raise RuntimeError(r.error)
            if r.error == "stopped":
                raise Stop()
        if r.finish_reason == "length" and not r.text.strip():
            with lock:
                calls["exhausted"] += 1
                if calls["exhausted"] >= MAX_EXHAUSTED:
                    calls["stopped"] = f"calls_stopped:{MAX_EXHAUSTED} replies spent the {max_tokens}-token ceiling"
        return r.text

    try:
        tree = profile(text, ask, fanout=FANOUT if fanout is None else fanout)
        if calls["stopped"]:
            tree.flags.append(calls["stopped"])
    except Stop:
        _clear(store, pid)                    # nothing written: the prompt stays to do and the next run resumes it
        raise
    except Exception as e:
        _clear(store, pid)
        store.insert("decomp", {"prompt": pid, "status": "failed", "model": model, "error": f"{type(e).__name__}: {str(e)[:300]}", "at": now()})
        raise
    m = _write(store, pid, text, tree, model)
    return {"id": pid, **m, "cost": calls["cost"], "call_errors": calls["errors"], "fallbacks": calls["fallbacks"]}


def run(store: Store, client: Client, corpus: Optional[str] = None, *, model: str = DEFAULT_MODEL, workers: int = 512, ids: Optional[list[str]] = None,
        redo: bool = False, limit: int = 0, max_tokens: int = MAX_TOKENS,
        progress: Optional[Callable[[int, int, dict], None]] = None, max_inflight: Optional[int] = None,
        stop: Optional[threading.Event] = None, fanout: Optional[int] = None) -> dict:
    """`workers` is the number of calls in flight across the run (the endpoint's admission); prompts in flight are as many
    as it takes to keep that busy. `stop`, once set (by the caller, or by `progress` raising Stop), cancels the prompts
    not yet started and aborts the calls in flight; prompts caught mid-way are left to do."""
    pids = prompt_ids(store, corpus, ids, redo, limit)
    total = len(pids)
    summary = {"total": total, "done": 0, "failed": 0, "calls": 0, "cost": 0.0, "seconds": 0.0, "stopped": False}
    if not pids:
        return summary
    sem = _sem(resolve(model, client.base_url).base_url, max_inflight or max(workers, 1))
    stop = stop if stop is not None else threading.Event()
    client.stop = stop
    t0 = time.time()

    def one(pid):
        if stop.is_set():
            return pid, None, "stopped"
        try:
            return pid, decompose_one(store, client, pid, model, max_tokens, sem=sem, fanout=fanout), None
        except Exception as e:
            if stop.is_set() or isinstance(e, Stop):
                return pid, None, "stopped"
            if "denied" in str(e):
                stop.set()
            return pid, None, f"{type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, p) for p in pids]
        for f in as_completed(futs):
            if stop.is_set() and not summary["stopped"]:
                summary["stopped"] = True
                ex.shutdown(wait=False, cancel_futures=True)   # nothing else starts
                client.close()                                 # calls in flight return now
            if f.cancelled():
                continue
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
    client.stop = None                                         # the client outlives the run
    return summary
