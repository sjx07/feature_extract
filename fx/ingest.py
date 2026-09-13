"""The Ingest side: what is in the library and how to add to it. A corpus is kept (renamed, retagged, added to, pruned,
deleted) and owns its codebook and its seed memberships; an import is an event that lands prompts in one. A profile is
the settings a run uses; a run brings one corpus current through the stages: decompose, codebook, align.

    corpora(store)            -> one row per corpus with its stage strip and last job
    profiles(store)           -> the saved profiles, the default first
    rename / retag / delete   -> the corpus edits; delete takes the corpus's decomposition, codebook and seed memberships with it
"""
from __future__ import annotations

import json
from typing import Optional

from .align.cards import SEED, seed_codebook_id
from .llm.registry import DEFAULT_MODEL
from .store import Store, now

DEFAULT_PROFILE = {"decompose_model": DEFAULT_MODEL, "codebook_model": "gpt-5.6-sol", "batch_model": DEFAULT_MODEL, "judge_model": "",
                   "embed_model": "BAAI/bge-large-en-v1.5", "budget": 30.0, "workers": 128, "effort": "low", "base_url": "", "kind": "guidance"}
KINDS = ("guidance", "material", "both")
ROLES = ("decompose", "codebook", "batch", "judge", "embed")


# ---- profiles
def profiles(store: Store) -> list[dict]:
    rows = [{"name": r["name"], "params": DEFAULT_PROFILE | json.loads(r["params"]), "at": r["at"]} for r in store.rows("SELECT * FROM profile ORDER BY name")]
    if not any(r["name"] == "default" for r in rows):
        rows.insert(0, {"name": "default", "params": dict(DEFAULT_PROFILE), "at": None})
    return sorted(rows, key=lambda r: (r["name"] != "default", r["name"]))


def profile(store: Store, name: str) -> dict:
    for p in profiles(store):
        if p["name"] == name:
            return p["params"]
    raise KeyError(name)


def save_profile(store: Store, name: str, params: dict) -> dict:
    name = name.strip()
    if not name or len(name) > 60:
        raise ValueError("a profile needs a short name")
    clean = {k: params.get(k, v) for k, v in DEFAULT_PROFILE.items()}
    clean["budget"] = float(clean["budget"] or 0) or None
    clean["workers"] = int(clean["workers"] or 128)
    clean["kind"] = clean["kind"] if clean.get("kind") in KINDS else "guidance"
    with store.lock:
        store.con.execute("INSERT INTO profile (name, params, at) VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET params=excluded.params, at=excluded.at", (name, json.dumps(clean), now()))
        store.con.commit()
    return {"name": name, "params": clean}


def delete_profile(store: Store, name: str) -> None:
    with store.lock:
        store.con.execute("DELETE FROM profile WHERE name=?", (name,)); store.con.commit()


# ---- jobs that are alive, and rows left 'running' by a process that died
STALE_MINUTES = 20


def running_jobs(store: Store, ws=None, stale_minutes: int = STALE_MINUTES) -> list[dict]:
    """Every job row marked running, with `live`: its log was written within the last stale_minutes (a step in flight writes
    a line per prompt or batch; a single long call can go quiet for a few minutes, a dead process for good)."""
    import os
    import time
    out = []
    for r in store.rows("SELECT id, kind, corpus, started FROM job WHERE status='running'"):
        d = dict(r); age = None
        if ws is not None:
            p = ws.job_log(int(r["id"]))
            if p.exists():
                age = (time.time() - os.path.getmtime(p)) / 60
        d["idle_minutes"] = round(age, 1) if age is not None else None
        d["live"] = age is not None and age < stale_minutes if ws is not None else True
        out.append(d)
    return out


def close_job(store: Store, jid: int, why: str = "closed by hand: no progress, the process that ran it is gone") -> dict:
    with store.lock:
        n = store.con.execute("UPDATE job SET status='stopped', error=?, finished=? WHERE id=? AND status='running'", (why, now(), jid)).rowcount
        store.con.commit()
    return {"id": jid, "closed": bool(n)}


# ---- corpora with their stage strip
def corpora(store: Store, kind: str = "guidance", ws=None) -> list[dict]:
    running = {j["corpus"]: j["kind"] for j in running_jobs(store, ws) if j["live"]}
    seed_id = seed_codebook_id(store, kind)
    seed = {"id": seed_id} if seed_id is not None else None
    n_libs = int(store.one("SELECT COUNT(DISTINCT corpus) n FROM codebook WHERE kind=? AND scope='corpus'", (kind,))["n"])   # alignment needs two
    out = []
    for c in store.rows("SELECT id, name, source, at FROM corpus ORDER BY name"):
        cid = int(c["id"])
        p = store.one("SELECT COUNT(*) n, COALESCE(SUM(LENGTH(text)),0) chars FROM prompt WHERE corpus=?", (cid,))
        dom = store.one("SELECT domain, COUNT(*) k FROM prompt WHERE corpus=? GROUP BY domain ORDER BY k DESC", (cid,))
        d = store.one("SELECT SUM(d.status='done') done, SUM(d.status='failed') failed, AVG(CASE WHEN d.status='done' THEN d.coverage END) cov FROM decomp d JOIN prompt p ON p.id=d.prompt WHERE p.corpus=?", (cid,))
        cb = store.one("SELECT id, version, round, at FROM codebook WHERE corpus=? AND kind=? ORDER BY id DESC", (cid, kind))
        row = {"id": cid, "name": c["name"], "source": c["source"], "at": c["at"], "prompts": int(p["n"]), "chars": int(p["chars"]), "domain": dom["domain"] if dom else None,
               "decomposed": int(d["done"] or 0), "failed": int(d["failed"] or 0), "coverage": round(d["cov"], 3) if d and d["cov"] is not None else None,
               "features": 0, "variants": 0, "open_wordings": 0, "unassigned": 0, "aligned": 0, "specific": 0, "open_cards": 0, "codebook": None}
        if cb:
            f = store.one("SELECT SUM(level='feature') f, SUM(level='variant') v FROM feature WHERE codebook=?", (cb["id"],))
            m = store.one("SELECT SUM(node IS NULL) open_w, COUNT(*) k FROM membership WHERE kind='realization' AND codebook=?", (cb["id"],))
            rz = store.one("SELECT COUNT(*) k FROM realization WHERE corpus=? AND kind=?", (cid, kind))
            row |= {"codebook": dict(cb), "features": int(f["f"] or 0), "variants": int(f["v"] or 0), "open_wordings": int(m["open_w"] or 0), "unassigned": int(rz["k"]) - int(m["k"] or 0)}
            if seed:
                a = store.one("SELECT SUM(m.node IS NOT NULL) aligned, SUM(m.node IS NULL AND m.note='specific') specific, COUNT(*) k FROM membership m JOIN feature f ON f.id=m.unit "
                              "WHERE m.kind='feature' AND m.codebook=? AND f.codebook=? AND f.level='feature'", (seed["id"], cb["id"]))
                row |= {"aligned": int(a["aligned"] or 0), "specific": int(a["specific"] or 0), "open_cards": row["features"] - int(a["k"] or 0)}
        row["tags"] = [{"field": r["field"], "values": int(r["k"])} for r in store.rows("SELECT t.field, COUNT(DISTINCT t.value) k FROM tag t JOIN prompt p ON p.id=t.prompt WHERE p.corpus=? GROUP BY t.field ORDER BY t.field", (cid,))]
        j = store.one("SELECT id, kind, status, spent, finished, started FROM job WHERE corpus=? ORDER BY id DESC", (c["name"],))
        row["last_job"] = dict(j) if j else None
        row["running"] = running.get(c["name"])
        # the strip: decomposed, codebook, aligned: 'done' | 'partial' | 'none' | 'running'
        st_d = "none" if not row["decomposed"] else ("done" if row["decomposed"] >= row["prompts"] else "partial")
        st_c = "none" if not cb else ("partial" if row["unassigned"] > 0 else "done")
        st_a = "none" if not (cb and seed) or not (row["aligned"] or row["specific"]) else ("done" if row["open_cards"] <= 0 else "partial")
        if row["running"]:
            k = row["running"]
            if k.startswith("decompose") or k == "profile" and st_d != "done":
                st_d = "running"
            elif k.startswith("library") or k == "profile" and st_c != "done":
                st_c = "running"
            elif k.startswith("align") or k == "profile":
                st_a = "running"
        row["stages"] = [st_d, st_c, st_a]
        row["alone"] = n_libs < 2
        row["pending"] = st_d != "done" or st_c != "done" or (st_a != "done" and n_libs >= 2)
        out.append(row)
    return out


# ---- corpus edits
def _cid(store: Store, name: str) -> int:
    r = store.one("SELECT id FROM corpus WHERE name=?", (name,))
    if not r:
        raise KeyError(name)
    return int(r["id"])


def rename(store: Store, name: str, new: str) -> dict:
    new = new.strip()
    if not new or new == SEED:
        raise ValueError("not a corpus name")
    cid = _cid(store, name)
    with store.lock:
        if store.con.execute("SELECT 1 FROM corpus WHERE name=?", (new,)).fetchone():
            raise ValueError(f"a corpus named {new} exists")
        store.con.execute("UPDATE corpus SET name=? WHERE id=?", (new, cid))
        store.con.execute("UPDATE job SET corpus=? WHERE corpus=?", (new, name))
        store.con.commit()
    return {"id": cid, "name": new}


def retag(store: Store, name: str, domain: Optional[str] = None, tags: Optional[dict] = None) -> dict:
    """Tags on every prompt of the corpus: the domain, and any field given (an empty value removes that field)."""
    from .tags import set_tags
    cid = _cid(store, name)
    all_tags = dict(tags or {})
    if domain is not None:
        all_tags["domain"] = domain.strip()
    n = 0
    if all_tags:
        with store.lock:
            for r in store.con.execute("SELECT id FROM prompt WHERE corpus=?", (cid,)).fetchall():
                set_tags(store, r["id"], all_tags, con=store.con); n += 1
            store.con.commit()
    return {"id": cid, "prompts": n, "fields": sorted(all_tags)}


def delete_prompts(store: Store, ids: list[str]) -> dict:
    """Prompts out of the store with their decomposition; the realizations keep their counts until the next collapse."""
    if not ids:
        return {"deleted": 0}
    q = ",".join("?" * len(ids))
    with store.lock:
        store.con.execute(f"DELETE FROM tag WHERE prompt IN ({q})", ids)
        store.con.execute(f"DELETE FROM reading WHERE prompt IN ({q})", ids)
        store.con.execute(f"DELETE FROM span WHERE prompt IN ({q})", ids)
        store.con.execute(f"DELETE FROM decomp WHERE prompt IN ({q})", ids)
        n = store.con.execute(f"DELETE FROM prompt WHERE id IN ({q})", ids).rowcount
        store.con.commit()
    return {"deleted": n}


def delete(store: Store, name: str) -> dict:
    """The corpus with its prompts, decomposition, realizations, codebooks (features, memberships, flags) and its features'
    seed memberships. The seed's globals stay; they lose these members."""
    cid = _cid(store, name)
    with store.lock:
        cbs = [int(r["id"]) for r in store.con.execute("SELECT id FROM codebook WHERE corpus=?", (cid,)).fetchall()]
        feats = [int(r["id"]) for r in store.con.execute(f"SELECT id FROM feature WHERE codebook IN ({','.join('?' * len(cbs)) or 'NULL'})", cbs).fetchall()] if cbs else []
        if feats:
            q = ",".join("?" * len(feats))
            store.con.execute(f"DELETE FROM membership WHERE kind='feature' AND unit IN ({q})", feats)
            store.con.execute(f"DELETE FROM embedding WHERE kind='feature' AND unit IN ({q})", feats)
            store.con.execute(f"DELETE FROM flag WHERE feature IN ({q}) OR other IN ({q})", feats + feats)
        if cbs:
            q = ",".join("?" * len(cbs))
            store.con.execute(f"DELETE FROM membership WHERE codebook IN ({q})", cbs)
            store.con.execute(f"DELETE FROM flag WHERE codebook IN ({q})", cbs)
            store.con.execute(f"DELETE FROM feature WHERE codebook IN ({q})", cbs)
            store.con.execute(f"DELETE FROM codebook WHERE id IN ({q})", cbs)
        rz = [int(r["id"]) for r in store.con.execute("SELECT id FROM realization WHERE corpus=?", (cid,)).fetchall()]
        if rz:
            q = ",".join("?" * len(rz))
            store.con.execute(f"DELETE FROM embedding WHERE kind='realization' AND unit IN ({q})", rz)
            store.con.execute("DELETE FROM realization WHERE corpus=?", (cid,))
        ids = [r["id"] for r in store.con.execute("SELECT id FROM prompt WHERE corpus=?", (cid,)).fetchall()]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]; q = ",".join("?" * len(chunk))
            store.con.execute(f"DELETE FROM tag WHERE prompt IN ({q})", chunk)
            store.con.execute(f"DELETE FROM reading WHERE prompt IN ({q})", chunk)
            store.con.execute(f"DELETE FROM span WHERE prompt IN ({q})", chunk)
            store.con.execute(f"DELETE FROM decomp WHERE prompt IN ({q})", chunk)
        store.con.execute("DELETE FROM prompt WHERE corpus=?", (cid,))
        store.con.execute("DELETE FROM import WHERE corpus=?", (cid,))
        store.con.execute("DELETE FROM corpus WHERE id=?", (cid,))
        store.con.commit()
    return {"id": cid, "name": name, "prompts": len(ids), "codebooks": len(cbs), "features": len(feats)}


# ---- what a job's stages look like now, for the job view
def stages(store: Store, corpus: Optional[str], kind: str = "guidance", min_coverage: float = 0.9) -> dict:
    """The three stage panels for one corpus: the decomposition with what needs a look, the codebook, the alignment."""
    from . import align as A
    from . import library as L
    out: dict = {"corpus": corpus, "kind": kind}
    if not corpus:
        return out
    try:
        cid = _cid(store, corpus)
    except KeyError:
        return out
    d = store.one("SELECT COUNT(*) n, SUM(d.status='done') done, SUM(d.status='failed') failed, AVG(CASE WHEN d.status='done' THEN d.coverage END) cov FROM prompt p LEFT JOIN decomp d ON d.prompt=p.id WHERE p.corpus=?", (cid,))
    low = [dict(r) for r in store.rows("SELECT p.id, d.coverage, SUBSTR(p.text,1,120) head FROM decomp d JOIN prompt p ON p.id=d.prompt WHERE p.corpus=? AND d.status='done' AND d.coverage < ? ORDER BY d.coverage LIMIT 100", (cid, min_coverage))]
    gaps = [dict(r) for r in store.rows("SELECT s.prompt id, SUBSTR(p.text, s.lo+1, MIN(s.hi-s.lo, 140)) text, s.note FROM span s JOIN prompt p ON p.id=s.prompt WHERE p.corpus=? AND s.kind='gap' ORDER BY s.hi-s.lo DESC LIMIT 100", (cid,))]
    failed = [dict(r) for r in store.rows("SELECT d.prompt id, d.error FROM decomp d JOIN prompt p ON p.id=d.prompt WHERE p.corpus=? AND d.status='failed' LIMIT 100", (cid,))]
    out["decomposition"] = {"prompts": int(d["n"]), "done": int(d["done"] or 0), "failed": int(d["failed"] or 0), "coverage": round(d["cov"], 3) if d["cov"] is not None else None,
                            "low_coverage": low, "gaps": gaps, "failures": failed, "todo": int(d["n"]) - int(d["done"] or 0)}
    st = L.status(store, corpus, kind)
    cb = L.latest(store, corpus, kind)
    out["codebook"] = {"versions": st["versions"], "realizations": st["realizations"], "readings": st["readings"], "current": dict(cb) if cb else None,
                       "groups": L.groups(store, int(cb["id"])) if cb else [], "flags": L.flags(store, int(cb["id"])) if cb else [], "leftover": L.leftovers(store, int(cb["id"]), corpus, kind)[:200] if cb else []}
    try:
        a = A.status(store, kind)
        per = a["per_corpus"].get(corpus, {})
        under = [g | {"features": [f | {"members": [m for m in f["members"] if m["corpus"] == corpus]} for f in g["features"] if any(m["corpus"] == corpus for m in f["members"])]} for g in A.globals_(store, kind)]
        under = [g for g in under if g["features"]]
        opens = [c for c in A.open_cards(store, kind) if c["corpus"] == corpus]
        out["alignment"] = {"seed": {k: a[k] for k in ("globals", "groups", "cards", "aligned", "domain_specific", "open", "flags", "standing", "rounds")}, "corpus": per, "under": under, "open": opens}
    except Exception as e:  # noqa: BLE001  a store without a seed yet
        out["alignment"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ---- what a run would cost, per corpus and stage, from the previews and the registry's prices
def estimate(store: Store, corpus: str, prof: dict, kind: str = "guidance") -> dict:
    """Dollars for bringing one corpus current under a profile: the decomposition left to do, the codebook steps of one round
    (cold start when there is no codebook, assign for the unplaced wordings, judge, name), and the alignment of its open cards.
    A guess from counts; the previews' own bases apply."""
    import math
    from . import decompose
    from . import library as L
    from .llm.registry import price, resolve
    kinds = ("guidance", "material") if kind == "both" else (kind,)
    out = {"corpus": corpus, "decompose": 0.0, "codebook": 0.0, "align": 0.0, "steps": []}
    try:
        d = decompose.preview(store, corpus, prof.get("decompose_model") or DEFAULT_MODEL, int(prof.get("workers") or 128))
        if d.get("prompts"):
            out["decompose"] = float(d["dollars"]); out["steps"].append({"stage": "decompose", "prompts": d["prompts"], "calls": d["calls"], "dollars": d["dollars"]})
    except Exception as e:  # noqa: BLE001
        out["steps"].append({"stage": "decompose", "error": str(e)[:120]})
    for k in kinds:
        try:
            has_cb = L.latest(store, corpus, k) is not None
            for step in (("assign", "judge", "name") if has_cb else ("coldstart", "assign", "judge", "name")):
                model = prof.get("codebook_model") if step in ("coldstart", "name") else prof.get("batch_model")
                r = L.preview(store, corpus, k, step, model or None)
                out["codebook"] += float(r["dollars"]); out["steps"].append({"stage": f"codebook {k}", "step": step, "calls": r["calls"], "dollars": r["dollars"]})
        except Exception as e:  # noqa: BLE001
            out["steps"].append({"stage": f"codebook {k}", "error": str(e)[:120]})
        # align: the corpus's cards not yet placed at the seed; a batch of 12 per assign call, a naming call per six
        row = next((c for c in corpora(store, k) if c["name"] == corpus), None)
        open_cards = (row["features"] - row["aligned"] - row["specific"]) if row and row["codebook"] else 0
        if open_cards > 0:
            bm, cm = prof.get("batch_model") or DEFAULT_MODEL, prof.get("codebook_model") or "gpt-5.6-sol"
            pa = price(bm, resolve(bm)); pn = price(cm, resolve(cm))
            a_calls, n_calls = math.ceil(open_cards / 12), max(1, open_cards // 6)
            dollars = a_calls * (2500 * pa[0] + 600 * pa[1]) / 1e6 + n_calls * (2000 * pn[0] + 400 * pn[1]) / 1e6
            out["align"] += dollars; out["steps"].append({"stage": f"align {k}", "cards": open_cards, "calls": a_calls + n_calls, "dollars": round(dollars, 3)})
    out["total"] = round(out["decompose"] + out["codebook"] + out["align"], 2)
    for k in ("decompose", "codebook", "align"):
        out[k] = round(out[k], 2)
    return out
