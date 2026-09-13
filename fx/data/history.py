"""History: git for the store, per corpus. A checkpoint is one corpus's rows (its prompts, decomposition, wordings, codebooks
with their features, memberships and flags) written as content-addressed blobs, one per table group, under
runs/history/objects and shared by every checkpoint and every branch: a codebook job's checkpoint adds only the codebook
blob and reuses the decomposition blob unchanged. The seed is its own line (its codebook, globals, memberships, flags).
Embeddings are not kept: they are recomputed for free by the next embed step.

    checkpoint(store, ws, "math", job=40)        -> a checkpoint row; blobs written once
    restore(store, ws, checkpoint_id)            -> the corpus as it was: its current rows go, the blobs come back with fresh
                                                    ids; seed rows that pointed at features now gone are pruned
    diff(store, checkpoint_id)                   -> counts then and now, features added and gone since
    branch(store, ws, checkpoint_id, "try2")     -> a new workspace runs/try2 with the store restored to that checkpoint

A restore is refused while a job is live on the corpus. Nothing here calls a model.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from fx.ingest.generalize.cards import SEED
from fx.core.store import Store, now

GROUPS = ("corpus", "prompts", "decomposition", "wordings", "codebook")


def objects_dir(ws) -> Path:
    d = Path(ws.root).parent / "history" / "objects"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- reading a corpus into groups
def _rows(store: Store, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in store.rows(sql, params)]


def _in(ids: list) -> str:
    return ",".join("?" * len(ids)) or "NULL"


def _chunks(ids: list, n: int = 500):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def groups(store: Store, corpus: str) -> dict[str, dict[str, list[dict]]]:
    """The corpus's rows by table group. For the seed: its codebook line only (memberships of every corpus's features)."""
    if corpus == SEED:
        out: dict[str, dict[str, list[dict]]] = {"corpus": {"corpus": []}}
        cbs = _rows(store, "SELECT * FROM codebook WHERE scope='seed' ORDER BY id")
        cb_ids = [int(x["id"]) for x in cbs]
        q = _in(cb_ids)
        out["codebook"] = {"codebook": cbs, "feature": _rows(store, f"SELECT * FROM feature WHERE codebook IN ({q}) ORDER BY id", cb_ids),
                           "membership": _rows(store, f"SELECT * FROM membership WHERE codebook IN ({q}) ORDER BY unit", cb_ids), "flag": _rows(store, f"SELECT * FROM flag WHERE codebook IN ({q}) ORDER BY id", cb_ids)}
        return out
    c = store.one("SELECT * FROM corpus WHERE name=?", (corpus,))
    if not c:
        raise KeyError(corpus)
    cid = int(c["id"])
    out = {"corpus": {"corpus": [dict(c)]}}
    if corpus != SEED:
        prompts = _rows(store, "SELECT * FROM prompt WHERE corpus=? ORDER BY rowid", (cid,))
        pids = [p["id"] for p in prompts]
        dec: dict[str, list[dict]] = {"span": [], "reading": [], "decomp": []}
        for ch in _chunks(pids):
            q = _in(ch)
            dec["span"] += _rows(store, f"SELECT * FROM span WHERE prompt IN ({q}) ORDER BY id", ch)
            dec["reading"] += _rows(store, f"SELECT * FROM reading WHERE prompt IN ({q}) ORDER BY id", ch)
            dec["decomp"] += _rows(store, f"SELECT * FROM decomp WHERE prompt IN ({q})", ch)
        tags: list[dict] = []
        for ch in _chunks(pids):
            tags += _rows(store, f"SELECT * FROM tag WHERE prompt IN ({_in(ch)}) ORDER BY prompt, field", ch)
        out["prompts"] = {"prompt": prompts, "import": _rows(store, "SELECT * FROM import WHERE corpus=? ORDER BY id", (cid,)), "tag": tags}
        out["decomposition"] = dec
        out["wordings"] = {"realization": _rows(store, "SELECT * FROM realization WHERE corpus=? ORDER BY id", (cid,))}
    cbs = _rows(store, "SELECT * FROM codebook WHERE corpus=? ORDER BY id", (cid,))
    cb_ids = [int(x["id"]) for x in cbs]
    q = _in(cb_ids)
    out["codebook"] = {"codebook": cbs,
                       "feature": _rows(store, f"SELECT * FROM feature WHERE codebook IN ({q}) ORDER BY id", cb_ids),
                       "membership": _rows(store, f"SELECT * FROM membership WHERE codebook IN ({q}) ORDER BY unit", cb_ids),
                       "flag": _rows(store, f"SELECT * FROM flag WHERE codebook IN ({q}) ORDER BY id", cb_ids)}
    return out


def counts_of(g: dict) -> dict:
    feats = g["codebook"]["feature"]
    ms = g["codebook"]["membership"]
    return {"prompts": len(g.get("prompts", {}).get("prompt", [])), "decomposed": sum(1 for d in g.get("decomposition", {}).get("decomp", []) if d["status"] == "done"),
            "readings": len(g.get("decomposition", {}).get("reading", [])), "realizations": len(g.get("wordings", {}).get("realization", [])),
            "codebooks": len(g["codebook"]["codebook"]), "groups": sum(1 for f in feats if f["level"] == "group"), "features": sum(1 for f in feats if f["level"] == "feature"),
            "variants": sum(1 for f in feats if f["level"] == "variant"), "placed": sum(1 for m in ms if m["node"] is not None), "open": sum(1 for m in ms if m["node"] is None),
            "flags": len(g["codebook"]["flag"])}


# ---- blobs
def _blob_bytes(tables: dict[str, list[dict]]) -> bytes:
    return json.dumps(tables, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_blob(ws, tables: dict[str, list[dict]]) -> tuple[str, int]:
    raw = _blob_bytes(tables)
    sha = hashlib.sha256(raw).hexdigest()
    p = objects_dir(ws) / f"{sha}.json.gz"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            fh.write(raw)
        tmp.replace(p)
    return sha, p.stat().st_size


def read_blob(ws, sha: str) -> dict[str, list[dict]]:
    p = objects_dir(ws) / f"{sha}.json.gz"
    with gzip.open(p, "rb") as fh:
        return json.loads(fh.read().decode())


# ---- checkpoints
def checkpoint(store: Store, ws, corpus: str, job: Optional[int] = None, note: Optional[str] = None) -> dict:
    """The corpus now, as a checkpoint: blobs written where new, one row in `checkpoint`. A checkpoint identical to the
    corpus's last one (same tree) is not repeated."""
    g = groups(store, corpus)
    tree: dict[str, str] = {}; size = 0
    for name, tables in g.items():
        sha, n = write_blob(ws, tables); tree[name] = sha; size += n
    counts = counts_of(g)
    last = store.one("SELECT id, tree FROM checkpoint WHERE corpus=? ORDER BY id DESC", (corpus,))
    if last and json.loads(last["tree"]) == tree:
        return {"id": int(last["id"]), "corpus": corpus, "tree": tree, "counts": counts, "bytes": size, "repeated": True}
    from fx.core.migrations import CURRENT
    cid = store.insert("checkpoint", {"corpus": corpus, "job": job, "at": now(), "note": note, "tree": tree, "counts": counts, "bytes": size, "schema": CURRENT})
    return {"id": cid, "corpus": corpus, "tree": tree, "counts": counts, "bytes": size, "repeated": False}


def checkpoints(store: Store, corpus: Optional[str] = None) -> list[dict]:
    rows = _rows(store, "SELECT c.*, j.kind job_kind, j.status job_status FROM checkpoint c LEFT JOIN job j ON j.id=c.job" + (" WHERE c.corpus=?" if corpus else "") + " ORDER BY c.id DESC", (corpus,) if corpus else ())
    for r in rows:
        r["tree"] = json.loads(r["tree"]); r["counts"] = json.loads(r["counts"])
    return rows


def translate(g: dict[str, dict[str, list[dict]]], schema: int) -> dict[str, dict[str, list[dict]]]:
    """Rows written under an older schema, brought to the current one: before version 3 a prompt's tags were its columns
    and meta keys (tag rows are derived); before 2 there were no import rows (a prompt's import is NULL); before 4 a codebook
    had no scope (restore sets it); 5 dropped tables no blob ever held."""
    from fx.data.tags import promotable
    if schema < 3 and "prompts" in g:
        tags = []
        for p in g["prompts"]["prompt"]:
            t = {k: p.get(k) for k in ("domain", "system", "task") if p.get(k) not in (None, "")}
            try:
                meta = json.loads(p.get("meta") or "{}")
            except ValueError:
                meta = {}
            t |= promotable(meta)
            p["meta"] = json.dumps({k: v for k, v in meta.items() if k not in t}, ensure_ascii=False)
            tags += [{"prompt": p["id"], "field": k, "value": v} for k, v in t.items()]
        g["prompts"]["tag"] = tags
        g["prompts"].setdefault("import", [])
    return g


def load(ws, ck: dict) -> dict[str, dict[str, list[dict]]]:
    from fx.core.migrations import CURRENT
    g = {name: read_blob(ws, sha) for name, sha in ck["tree"].items()}
    schema = int(ck.get("schema") or 1)
    if schema > CURRENT:
        raise RuntimeError(f"checkpoint #{ck.get('id')} was written under schema version {schema}; this code knows {CURRENT}")
    return translate(g, schema) if schema < CURRENT else g


# ---- restore, with fresh ids and the seed pruned
def _insert_rows(con: sqlite3.Connection, table: str, rows: list[dict], drop_id: bool = False) -> list[int]:
    ids = []
    for r in rows:
        r = dict(r)
        if drop_id:
            r.pop("id", None)
        keys = list(r)
        cur = con.execute(f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})", [r[k] for k in keys])
        ids.append(int(cur.lastrowid))
    return ids


def restore(store: Store, ws, checkpoint_id: int, live_corpora: Optional[set] = None) -> dict:
    """The corpus back to the checkpoint: its current rows are deleted (as `fx` deletes a corpus, seed memberships of its
    features included), the blobs are inserted with fresh ids (rewritten wherever they are referenced), and seed rows left
    pointing at features that no longer exist are pruned. For the seed line: its codebook rows are replaced; memberships
    whose feature no longer exists are dropped."""
    from fx.views import ingest
    ck = store.one("SELECT * FROM checkpoint WHERE id=?", (checkpoint_id,))
    if not ck:
        raise KeyError(checkpoint_id)
    ck = dict(ck) | {"tree": json.loads(ck["tree"])}
    corpus = ck["corpus"]
    _ = ck.get("schema")
    if live_corpora and corpus in live_corpora:
        raise RuntimeError(f"a job is running on {corpus}; stop it first")
    g = load(ws, ck)
    exists = store.one("SELECT id FROM corpus WHERE name=?", (corpus,)) if corpus != SEED else None
    seed_n0 = int(store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature'")["n"])
    if exists and corpus != SEED:
        ingest.delete(store, corpus)                                                       # takes the seed rows of its features with it
    with store.lock:
        con = store.con
        try:
            con.execute("BEGIN")
            if corpus == SEED:
                cid = None                                                                 # the seed's codebooks have no corpus; the current ones go
                for cb in [int(x["id"]) for x in con.execute("SELECT id FROM codebook WHERE scope='seed'").fetchall()]:
                    con.execute("DELETE FROM flag WHERE codebook=?", (cb,)); con.execute("DELETE FROM membership WHERE codebook=?", (cb,))
                    con.execute("DELETE FROM feature WHERE codebook=?", (cb,)); con.execute("DELETE FROM codebook WHERE id=?", (cb,))
            else:
                # corpus row: keep the id when free, else a fresh one
                crow = dict(g["corpus"]["corpus"][0])
                taken = con.execute("SELECT 1 FROM corpus WHERE id=?", (crow["id"],)).fetchone()
                if taken:
                    crow.pop("id")
                cid = _insert_rows(con, "corpus", [crow])[0]
            skipped = 0
            span_map: dict[int, int] = {}; rz_map: dict[int, int] = {}
            if corpus != SEED:
                imp_map: dict[int, int] = {}
                for im in g["prompts"].get("import", []):
                    imp_map[int(im["id"])] = _insert_rows(con, "import", [dict(im) | {"corpus": cid}], drop_id=True)[0]
                for p in g["prompts"]["prompt"]:
                    p = dict(p) | {"corpus": cid, "import": imp_map.get(int(p["import"])) if p.get("import") is not None else None}
                    if con.execute("SELECT 1 FROM prompt WHERE id=?", (p["id"],)).fetchone():
                        skipped += 1; continue
                    _insert_rows(con, "prompt", [p])
                have = {r[0] for r in con.execute("SELECT id FROM prompt WHERE corpus=?", (cid,)).fetchall()}
                for t in g["prompts"].get("tag", []):
                    if t["prompt"] in have:
                        con.execute("INSERT OR REPLACE INTO tag (prompt, field, value) VALUES (?, ?, ?)", (t["prompt"], t["field"], t["value"]))
                for s in g["decomposition"]["span"]:
                    if s["prompt"] in have:
                        span_map[int(s["id"])] = _insert_rows(con, "span", [s], drop_id=True)[0]
                for r in g["wordings"]["realization"]:
                    rz_map[int(r["id"])] = _insert_rows(con, "realization", [dict(r) | {"corpus": cid}], drop_id=True)[0]
                for r in g["decomposition"]["reading"]:
                    if r["prompt"] in have and int(r["span"]) in span_map:
                        rr = dict(r) | {"span": span_map[int(r["span"])], "realization": rz_map.get(int(r["realization"])) if r.get("realization") is not None else None}
                        _insert_rows(con, "reading", [rr], drop_id=True)
                for d in g["decomposition"]["decomp"]:
                    if d["prompt"] in have:
                        _insert_rows(con, "decomp", [d])
            cb_map: dict[int, int] = {}; f_map: dict[int, int] = {}
            for cb in g["codebook"]["codebook"]:
                cb_row = dict(cb) | {"corpus": cid, "scope": "seed" if corpus == SEED else "corpus"}
                cb_map[int(cb["id"])] = _insert_rows(con, "codebook", [cb_row], drop_id=True)[0]
            feats = sorted(g["codebook"]["feature"], key=lambda x: int(x["id"]))
            for f in feats:                                                                 # two passes: a group born later than its features, or a retired
                ex = json.loads(f.get("examples") or "[]")                                  # feature folded into a younger node, points at a higher id
                ff = dict(f) | {"codebook": cb_map[int(f["codebook"])], "parent": None, "prev": None, "examples": json.dumps([rz_map.get(int(e), int(e)) for e in ex])}
                f_map[int(f["id"])] = _insert_rows(con, "feature", [ff], drop_id=True)[0]
            for f in feats:
                if f.get("parent") is not None or f.get("prev") is not None:
                    con.execute("UPDATE feature SET parent=?, prev=? WHERE id=?", (f_map.get(int(f["parent"])) if f.get("parent") is not None else None,
                                                                                    f_map.get(int(f["prev"])) if f.get("prev") is not None else None, f_map[int(f["id"])]))
            for m in g["codebook"]["membership"]:
                mm = dict(m) | {"codebook": cb_map[int(m["codebook"])], "node": f_map.get(int(m["node"])) if m.get("node") is not None else None}
                if m["kind"] == "realization":
                    if int(m["unit"]) not in rz_map:
                        continue
                    mm["unit"] = rz_map[int(m["unit"])]
                else:                                                                       # the seed's members are other corpora's features
                    if not con.execute("SELECT 1 FROM feature WHERE id=?", (int(m["unit"]),)).fetchone():
                        skipped += 1; continue
                con.execute("INSERT OR REPLACE INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,?,?,?,?)",
                            (mm["kind"], mm["unit"], mm["codebook"], mm["node"], mm.get("confidence"), mm.get("note"), mm.get("at") or now()))
            for fl in g["codebook"]["flag"]:
                fr = dict(fl) | {"codebook": cb_map[int(fl["codebook"])], "feature": f_map.get(int(fl["feature"])), "other": f_map.get(int(fl["other"])) if fl.get("other") is not None else None,
                                 "realization": rz_map.get(int(fl["realization"])) if fl.get("realization") is not None else None}
                if fr["feature"] is None:
                    continue
                _insert_rows(con, "flag", [fr], drop_id=True)
            # prune what points at features that no longer exist (seed memberships of the replaced features, and the reverse)
            pruned = con.execute("DELETE FROM membership WHERE kind='feature' AND unit NOT IN (SELECT id FROM feature)").rowcount
            pruned += con.execute("DELETE FROM membership WHERE node IS NOT NULL AND node NOT IN (SELECT id FROM feature)").rowcount
            pruned += con.execute("DELETE FROM flag WHERE feature NOT IN (SELECT id FROM feature) OR (other IS NOT NULL AND other NOT IN (SELECT id FROM feature))").rowcount
            con.execute("DELETE FROM embedding WHERE kind='feature' AND unit NOT IN (SELECT id FROM feature)")
            con.execute("DELETE FROM embedding WHERE kind='realization' AND unit NOT IN (SELECT id FROM realization)")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK"); raise
    seed_n1 = int(store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature'")["n"])
    return {"checkpoint": checkpoint_id, "corpus": corpus, "restored": counts_of(g), "features_remapped": len(f_map),
            "seed_rows_pruned": (seed_n0 - seed_n1) if corpus != SEED else pruned + skipped, "skipped": skipped}


# ---- what changed since a checkpoint
def diff(store: Store, checkpoint_id: int, ws=None) -> dict:
    ck = store.one("SELECT * FROM checkpoint WHERE id=?", (checkpoint_id,))
    if not ck:
        raise KeyError(checkpoint_id)
    then = json.loads(ck["counts"]); corpus = ck["corpus"]
    try:
        g = groups(store, corpus); now_ = counts_of(g)
    except KeyError:
        return {"checkpoint": checkpoint_id, "corpus": corpus, "then": then, "now": None, "gone": True}
    out = {"checkpoint": checkpoint_id, "corpus": corpus, "then": then, "now": now_, "delta": {k: now_.get(k, 0) - then.get(k, 0) for k in then}}
    if ws is not None:
        old = load(ws, dict(ck) | {"tree": json.loads(ck["tree"])})
        names_then = {(f["level"], f["name"]) for f in old["codebook"]["feature"] if f["level"] in ("feature", "variant")}
        names_now = {(f["level"], f["name"]) for f in g["codebook"]["feature"] if f["level"] in ("feature", "variant")}
        out["features_added"] = sorted(n for l, n in names_now - names_then)[:200]
        out["features_gone"] = sorted(n for l, n in names_then - names_now)[:200]
    return out


# ---- a branch: a new workspace restored to a checkpoint
def branch(store: Store, ws, checkpoint_id: int, name: str) -> dict:
    from fx.core.paths import Workspace
    name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())
    if not name:
        raise ValueError("a branch needs a name")
    root = Path(ws.root).parent / name
    if root.exists():
        raise ValueError(f"{root} exists")
    root.mkdir(parents=True)
    with store.lock:
        store.con.execute("VACUUM INTO ?", (str(root / "store.db"),))
    new_ws = Workspace(root)
    new_store = Store(new_ws.store_path)
    r = restore(new_store, new_ws, checkpoint_id)
    new_store.con.close()
    return {"workspace": str(root), "serve": f"fx -w {root} serve --port 8782", "restored": r}
