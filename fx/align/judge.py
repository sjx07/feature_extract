"""Read-only coherence of the seed: per global feature, a call sees the member features with a sample of the prompt
wordings each covers, and names the members that give a different instruction. Flags: feature = the global, other =
the member. A flag raised again after it was acted on is standing."""
from __future__ import annotations

from typing import Callable, Optional

from ..llm import Client
from ..llm.pool import run_many
from ..llm.registry import DEFAULT_MODEL, reasoning_low
from ..store import Store
from ..util.ids import parse_id
from ..util.jsonx import extract_object
from . import prompts as P
from .seed import globals_, seed_codebook

MAX_TOKENS = 32768


def samples(store: Store, fid: int, n: int = 6) -> list[str]:
    """A member feature's wordings by support, its variants' included."""
    return [r["declaration"] for r in store.rows("SELECT r.declaration FROM assignment a JOIN realization r ON r.id=a.realization WHERE a.feature IN (SELECT id FROM feature WHERE id=? OR parent=?) ORDER BY r.prompts DESC, r.n DESC LIMIT ?", (fid, fid, n))]


def judge(store: Store, client: Client, kind: str, model: str = DEFAULT_MODEL, workers: int = 64, effort: str = "low",
          progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    cb = seed_codebook(store, kind)
    tree = globals_(store, kind)
    gs = [s for g in tree for s in g["features"] if len(s["members"]) >= 2]
    prompts = [P.judge(s, s["members"], {m["id"]: samples(store, m["id"]) for m in s["members"]}) for s in gs]
    summary = {"codebook": cb, "calls": len(prompts), "misfits": 0, "new": 0, "standing": 0, "unparsed": 0}
    previous = {(int(r["feature"]), int(r["other"])) for r in store.rows("SELECT feature, other FROM flag WHERE codebook=? AND verdict='misfit'", (cb,))}
    with store.lock:
        store.con.execute("DELETE FROM flag WHERE codebook=?", (cb,)); store.con.commit()
    if not prompts:
        return summary
    replies = run_many(client, prompts, model=model, workers=workers, max_inflight=workers, stage="align", note=f"align:{kind}:judge", system=P.SYSTEM, max_tokens=MAX_TOKENS,
                       extra_body=reasoning_low(model, client.base_url) if effort == "low" else None)
    for k, (s, r) in enumerate(zip(gs, replies)):
        obj = extract_object(r.text) if r and r.text else None
        if not isinstance(obj, dict):
            summary["unparsed"] += 1; continue
        valid = {m["id"] for m in s["members"]}
        for m in obj.get("misfits") or []:
            fid = parse_id(m.get("id") if isinstance(m, dict) else m, valid)
            if fid is None:
                continue
            standing = int((s["id"], fid) in previous)
            store.insert("flag", {"codebook": cb, "feature": s["id"], "realization": None, "other": fid, "verdict": "misfit", "note": str((m.get("why") if isinstance(m, dict) else "") or "")[:300], "standing": standing})
            summary["misfits"] += 1; summary["standing" if standing else "new"] += 1
        if progress:
            progress(k + 1, len(gs), {"global": s["id"], "misfits": summary["misfits"]})
    return summary


def reopen(store: Store, kind: str) -> dict:
    """A first-time misfit member goes back to open with the judge's reason; standing ones stay."""
    from ..store import now
    cb = seed_codebook(store, kind)
    n = 0
    flags = store.rows("SELECT feature, other, note FROM flag WHERE codebook=? AND verdict='misfit' AND standing=0", (cb,))   # read before taking the lock: it is not reentrant
    with store.lock:
        for r in flags:
            n += store.con.execute("UPDATE alignment SET global=NULL, confidence='low', note=?, at=? WHERE feature=? AND global=?", (f"reopened:S{r['feature']}|{(r['note'] or '')[:200]}", now(), r["other"], r["feature"])).rowcount
        store.con.commit()
    return {"codebook": cb, "reopened": n}
