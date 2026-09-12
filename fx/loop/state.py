"""What the loop reads and writes in the store: vectors, the tree and its nodes, memberships, anchors, the judge's
reasons. No model calls."""
from __future__ import annotations

import json
import numpy as np
from ..store import Store, now
from .level import Level
from collections import defaultdict
from typing import Optional

# ---- vectors
def embed(store: Store, lv: Level, enc=None, model: Optional[str] = None, batch: int = 512) -> dict:
    from ..library.encoders import EMBEDDER, encoder
    model = model or EMBEDDER
    have = {int(r["unit"]) for r in store.rows("SELECT unit FROM embedding WHERE kind=?", (lv.kind,))}
    todo = [u for u in lv.units() if u["id"] not in have]
    if not todo:
        return {"embedded": 0, "model": model}
    enc = enc or encoder(model)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        vecs = enc([u["text"] for u in chunk])
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        with store.lock:
            store.con.executemany("INSERT OR REPLACE INTO embedding (kind, unit, model, dim, vec) VALUES (?,?,?,?,?)",
                                  [(lv.kind, u["id"], model, int(vecs.shape[1]), vecs[k].astype(np.float32).tobytes()) for k, u in enumerate(chunk)])
            store.con.commit()
    return {"embedded": len(todo), "model": model}


def vectors(store: Store, kind: str, ids: list[int]) -> tuple[list[int], np.ndarray]:
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    out_ids, vecs = [], []
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in store.rows(f"SELECT unit, dim, vec FROM embedding WHERE kind=? AND unit IN ({','.join('?' * len(chunk))})", [kind] + chunk):
            out_ids.append(int(r["unit"])); vecs.append(np.frombuffer(r["vec"], dtype=np.float32, count=int(r["dim"])))
    return out_ids, (np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))


# ---- the tree and membership
def tree(store: Store, lv: Level) -> list[dict]:
    """Groups, features under them, variants under features, each node with support (its own and its children's), its
    anchors' texts and its member count."""
    units = {u["id"]: u for u in lv.units()}
    feats = [dict(r) | {"examples": json.loads(r["examples"] or "[]")} for r in store.rows("SELECT * FROM feature WHERE codebook=? ORDER BY id", (lv.codebook,))]
    mem: dict[int, list[int]] = defaultdict(list)
    for r in store.rows("SELECT unit, node FROM membership WHERE kind=? AND codebook=? AND node IS NOT NULL", (lv.kind, lv.codebook)):
        mem[int(r["node"])].append(int(r["unit"]))
    by_parent: dict = defaultdict(list)
    for f in feats:
        by_parent[f["parent"]].append(f)

    def fill(node):
        ms = [units[u] for u in mem.get(node["id"], []) if u in units]
        node["members_n"], node["own"] = len(ms), sum(u["support"] for u in ms)
        node["support"], node["groups_n"] = node["own"], len(set().union(*(u["groups"] for u in ms)) if ms else set())
        node["anchors"] = [units[e]["label"] for e in node["examples"] if e in units]
    out = []
    for g in by_parent.get(None, []):
        g["features"] = []
        for f in [x for x in by_parent.get(g["id"], []) if x["level"] == "feature"]:
            fill(f); f["variants"] = []
            for v in [x for x in by_parent.get(f["id"], []) if x["level"] == "variant"]:
                fill(v); f["variants"].append(v); f["support"] += v["support"]; f["members_n"] += v["members_n"]
            g["features"].append(f)
        g["support"] = sum(f["support"] for f in g["features"])
        out.append(g)
    return out


def nodes(tr: list[dict]) -> list[dict]:
    return [n for g in tr for f in g["features"] for n in [f] + f.get("variants", [])]


def members(store: Store, lv: Level, node: int, limit: int = 60) -> list[dict]:
    units = {u["id"]: u for u in lv.units()}
    rows = store.rows("SELECT unit, confidence, note FROM membership WHERE kind=? AND codebook=? AND node=?", (lv.kind, lv.codebook, node))
    out = [units[int(r["unit"])] | {"confidence": r["confidence"], "note": r["note"]} for r in rows if int(r["unit"]) in units]
    return sorted(out, key=lambda u: -u["support"])[:limit]


def open_units(store: Store, lv: Level) -> list[dict]:
    have = {int(r["unit"]): r for r in store.rows("SELECT unit, node, note FROM membership WHERE kind=? AND codebook=?", (lv.kind, lv.codebook))}
    return [u | {"note": have[u["id"]]["note"] if u["id"] in have else None} for u in lv.units() if u["id"] not in have or have[u["id"]]["node"] is None]


def anchors(store: Store, lv: Level) -> Optional[float]:
    """Share of the nodes' own example units that membership put back on them; None until placed."""
    hits = total = 0
    placed = {int(r["unit"]): r["node"] for r in store.rows("SELECT unit, node FROM membership WHERE kind=? AND codebook=?", (lv.kind, lv.codebook))}
    for f in store.rows("SELECT id, examples FROM feature WHERE codebook=? AND level IN ('feature','variant')", (lv.codebook,)):
        for e in json.loads(f["examples"] or "[]"):
            if e in placed:
                total += 1; hits += int(placed[e] == f["id"])
    agreement = round(hits / total, 3) if total else None
    with store.lock:
        store.con.execute("UPDATE codebook SET anchor_agreement=? WHERE id=?", (agreement, lv.codebook)); store.con.commit()
    return agreement


def _place(store: Store, lv: Level, unit: int, node: Optional[int], confidence: str, note: Optional[str] = None, keep_note_if_open: bool = True) -> None:
    store.con.execute("INSERT INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(kind, unit, codebook) DO UPDATE SET "
                      "node=excluded.node, confidence=excluded.confidence, at=excluded.at, note=CASE WHEN excluded.node IS NULL AND ? THEN membership.note ELSE excluded.note END",
                      (lv.kind, unit, lv.codebook, node, confidence, note, now(), int(keep_note_if_open)))


def judged(store: Store, lv: Level) -> dict[int, tuple[int, str]]:
    """unit -> (the node the judge removed it from, the reason)."""
    out = {}
    for r in store.rows("SELECT unit, note FROM membership WHERE kind=? AND codebook=? AND node IS NULL AND note LIKE 'reopened:%'", (lv.kind, lv.codebook)):
        body = r["note"].split(":", 1)[1]
        nid, _, why = body.partition("|")
        if nid.isdigit():
            out[int(r["unit"])] = (int(nid), why)
    return out
