"""The cube: what the Library side of the site reads. Facts are readings; the dimensions are the prompts' fields (corpus,
domain and the tags a source carried) and the seed's hierarchy (aspect group → global feature → per-corpus feature →
wording); the measure is distinct prompts. Selecting field values gives a slice; the slice shows which global features
its prompts carry and with what support, and drills down to wordings and prompts.

The seed is the spine. A per-corpus feature no global has absorbed yet (open, or domain-specific) is shown under its own
library, so a workspace with one library and no seed sees that library's tree: the one-library seed is the tree itself.
Nothing here writes; the induction (stage 2, stage 3) is what fills the tables this reads.

    facts(store, kind)                      -> one row per placed reading: prompt, realization, node, feature, global
    slice(store, kind, {"corpus": {"math"}}) -> facets, measures, the hierarchy present in the slice
    node(store, kind, filters, id)           -> a global or a per-corpus feature drilled to its wordings and prompts
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

from fx.ingest.generalize.cards import SEED, seed_codebook_id
from fx.core.store import Store

from fx.data.tags import field_order

RESERVED = {"kind", "view", "limit", "polarity", "text"}                    # query keys that are not tag fields
_cache: dict = {}


def _version(store: Store) -> tuple:
    """Changes whenever the induction wrote: the cube's cache key."""
    r = store.one("SELECT (SELECT COUNT(*) FROM membership) m, (SELECT MAX(id) FROM feature) f, (SELECT COUNT(*) FROM feature) fn, (SELECT COUNT(*) FROM reading) r, (SELECT COUNT(*) FROM prompt) p, "
                  "(SELECT MAX(at) FROM membership) a, (SELECT COUNT(*) FROM codebook) c, (SELECT MAX(at) FROM codebook) ca, (SELECT COUNT(*) FROM tag) t")      # a seed reset deletes and recreates: counts, ids and times all enter
    return (str(store.path), tuple(r))


def prompt_fields(store: Store) -> dict[str, dict[str, str]]:
    """Every prompt's field values: corpus, the columns, the meta tags. A missing value is absent, not 'None'."""
    key = ("prompts",) + _version(store)
    if key in _cache:
        return _cache[key]
    out: dict[str, dict[str, str]] = {r["id"]: {"corpus": r["corpus"]} for r in store.rows("SELECT p.id, k.name corpus FROM prompt p JOIN corpus k ON k.id=p.corpus")}
    for r in store.rows("SELECT prompt, field, value FROM tag"):
        f = out.get(r["prompt"])
        if f is not None and r["field"] != "corpus":
            f[r["field"]] = r["value"]
    _cache.clear(); _cache[key] = out
    return out


def field_names(fields: dict[str, dict[str, str]]) -> list[str]:
    """The fields in use across the prompts, corpus first, known ones in their order, the rest by name."""
    return field_order(k for f in fields.values() for k in f)


def libraries(store: Store, kind: str) -> dict[int, int]:
    """corpus id -> its latest codebook of the kind, the seed excluded."""
    return {int(r["corpus"]): int(r["id"]) for r in store.rows("SELECT c.id, c.corpus FROM codebook c WHERE c.kind=? AND c.scope='corpus' "
                                                              "AND c.id=(SELECT MAX(id) FROM codebook x WHERE x.corpus=c.corpus AND x.kind=c.kind)", (kind,))}


def seed_codebook(store: Store, kind: str) -> Optional[int]:
    return seed_codebook_id(store, kind)


def hierarchy(store: Store, kind: str) -> dict:
    """The feature dimension: every node of the latest library per corpus and of the seed, with feature -> global."""
    libs = libraries(store, kind)
    seed = seed_codebook(store, kind)
    cbs = list(libs.values()) + ([seed] if seed else [])
    if not cbs:
        return {"nodes": {}, "to_feature": {}, "to_global": {}, "libs": libs, "seed": seed, "corpus_of": {}}
    q = ",".join("?" * len(cbs))
    nodes = {int(r["id"]): dict(r) for r in store.rows(f"SELECT f.id, f.codebook, f.level, f.parent, f.aspect, f.name, f.definition, f.polarity, f.round, k.name corpus FROM feature f JOIN codebook c ON c.id=f.codebook LEFT JOIN corpus k ON k.id=c.corpus WHERE f.codebook IN ({q}) AND f.level IN ('group','feature','variant')", cbs)}      # a retired row is not a node; the seed has no corpus
    to_feature = {i: (n["parent"] if n["level"] == "variant" else i) for i, n in nodes.items() if n["level"] in ("feature", "variant")}
    to_global: dict[int, int] = {}
    if seed:
        to_global = {int(r["unit"]): int(r["node"]) for r in store.rows("SELECT unit, node FROM membership WHERE kind='feature' AND codebook=? AND node IS NOT NULL", (seed,))}
    return {"nodes": nodes, "to_feature": to_feature, "to_global": to_global, "libs": libs, "seed": seed}


def facts(store: Store, kind: str) -> tuple[list[tuple], dict]:
    """One row per reading placed on a node of its corpus's latest library: (prompt, realization, node, feature, global or None)."""
    key = ("facts", kind) + _version(store)
    if key in _cache:
        return _cache[key]
    h = hierarchy(store, kind)
    rows: list[tuple] = []
    if h["libs"]:
        # two indexed reads joined here: SQLite planned the three-way join as a scan of every reading per membership row
        q = ",".join("?" * len(h["libs"]))
        placed = {int(r["unit"]): int(r["node"]) for r in store.rows(f"SELECT unit, node FROM membership WHERE kind='realization' AND codebook IN ({q}) AND node IS NOT NULL", list(h["libs"].values()))}
        for r in store.rows("SELECT r.prompt, r.realization FROM reading r JOIN realization x ON x.id=r.realization WHERE x.kind=? AND r.realization IS NOT NULL", (kind,)):
            node = placed.get(int(r["realization"]))
            feat = h["to_feature"].get(node) if node is not None else None
            if feat is None:
                continue
            rows.append((r["prompt"], int(r["realization"]), node, feat, h["to_global"].get(feat)))
    _cache[key] = (rows, h)
    return rows, h


def parse_filters(params: dict) -> dict[str, set[str]]:
    """Query parameters to filters: a field's values comma-separated; values within a field are OR, fields AND. `text` is
    the search box: prompts whose text, or one of whose wordings, contains it."""
    out = {k: {v for v in str(vs).split(",") if v} for k, vs in params.items() if k not in RESERVED and vs and not k.startswith("_")}
    if str(params.get("polarity") or "").strip():
        out["polarity"] = {v for v in str(params["polarity"]).split(",") if v}
    if str(params.get("text") or "").strip():
        out["text"] = {str(params["text"]).strip()}
    return out


def text_matches(store: Store, text: str) -> set[str]:
    """Prompts matching the search box: the words in the prompt's text, or a wording (a reading's declaration) containing it."""
    key = ("text", text.lower()) + _version(store)
    if key in _cache:
        return _cache[key]
    like = f"%{text}%"
    ids = {r["id"] for r in store.rows("SELECT id FROM prompt WHERE text LIKE ?", (like,))}
    ids |= {r["prompt"] for r in store.rows("SELECT DISTINCT prompt FROM reading WHERE declaration LIKE ?", (like,))}
    _cache[key] = ids
    return ids


def select(fields: dict[str, dict[str, str]], filters: dict[str, set[str]], skip: Optional[str] = None, store: Optional[Store] = None) -> set[str]:
    out = set()
    for pid, f in fields.items():
        if all(k == skip or f.get(k) in vs for k, vs in filters.items() if k not in ("polarity", "text")):
            out.add(pid)
    if "text" in filters and store is not None:
        for t in filters["text"]:
            out &= text_matches(store, t)
    return out


def slice(store: Store, kind: str, filters: dict[str, set[str]]) -> dict:
    """The slice: facets (each field's values with the prompts they would keep, the field's own filter lifted), the measures,
    and the hierarchy present in it."""
    fields = prompt_fields(store)
    rows, h = facts(store, kind)
    selected = select(fields, filters, store=store)
    pol = filters.get("polarity")
    decomposed = {r["prompt"] for r in store.rows("SELECT prompt FROM decomp WHERE status='done'")} & selected
    # facets
    facets = []
    for name in field_names(fields):
        base = select(fields, filters, skip=name, store=store) if name in filters else selected
        counts: dict[str, int] = defaultdict(int)
        for pid in base:
            v = fields[pid].get(name)
            if v is not None:
                counts[v] += 1
        if counts:
            facets.append({"field": name, "values": sorted(({"value": v, "prompts": n, "on": v in filters.get(name, ())} for v, n in counts.items()), key=lambda d: (-d["prompts"], d["value"]))})
    # aggregate
    g_prompts: dict[int, set] = defaultdict(set); g_read: dict[int, int] = defaultdict(int)
    f_prompts: dict[int, set] = defaultdict(set); f_read: dict[int, int] = defaultdict(int)
    covered, on_global = set(), set()
    for pid, rid, node, feat, glob in rows:
        if pid not in selected:
            continue
        if pol and h["nodes"][feat].get("polarity", "require") not in pol:
            continue
        f_prompts[feat].add(pid); f_read[feat] += 1; covered.add(pid)
        if glob is not None:
            g_prompts[glob].add(pid); g_read[glob] += 1; on_global.add(pid)
    n = max(len(decomposed), 1)
    nodes = h["nodes"]

    def feat_row(fid: int) -> dict:
        f = nodes[fid]
        return {"id": fid, "corpus": f["corpus"], "name": f["name"], "definition": f["definition"] or "", "polarity": f["polarity"] or "require", "group": nodes.get(f["parent"], {}).get("name"),
                "prompts": len(f_prompts[fid]), "readings": f_read[fid], "share": round(len(f_prompts[fid]) / n, 4)}

    groups: dict[int, dict] = {}
    for gid, ps in g_prompts.items():
        g = nodes[gid]; grp = nodes.get(g["parent"]) or {"id": 0, "name": f"unplaced · {g.get('aspect') or 'other'}", "aspect": g.get("aspect") or "other", "definition": "features not yet under a group"}
        members = sorted((feat_row(f) for f, gl in h["to_global"].items() if gl == gid and f in f_prompts), key=lambda d: -d["prompts"])
        row = {"id": gid, "name": g["name"], "definition": g["definition"] or "", "polarity": g["polarity"] or "require", "round": g["round"], "prompts": len(ps), "readings": g_read[gid], "share": round(len(ps) / n, 4),
               "corpora": sorted({m["corpus"] for m in members}), "members": members}
        groups.setdefault(grp["id"], {"id": grp["id"], "name": grp["name"], "aspect": grp.get("aspect"), "definition": grp.get("definition") or "", "prompts": set(), "features": []})
        groups[grp["id"]]["features"].append(row); groups[grp["id"]]["prompts"] |= ps
    tree = sorted(({**g, "prompts": len(g["prompts"]), "features": sorted(g["features"], key=lambda d: -d["prompts"])} for g in groups.values()), key=lambda d: -d["prompts"])
    # per-corpus features the seed has not absorbed: under their own library and group
    local: dict[str, dict] = {}
    for fid, ps in f_prompts.items():
        if fid in h["to_global"]:
            continue
        f = nodes[fid]; grp = nodes.get(f["parent"]) or {"name": f"unplaced · {f.get('aspect') or 'other'}", "aspect": f.get("aspect") or "other"}
        lib = local.setdefault(f["corpus"], {"corpus": f["corpus"], "prompts": set(), "groups": {}})
        lib["prompts"] |= ps
        gr = lib["groups"].setdefault(grp["name"], {"name": grp["name"], "aspect": grp.get("aspect"), "prompts": set(), "features": []})
        gr["features"].append(feat_row(fid)); gr["prompts"] |= ps
    unaligned = []
    for lib in sorted(local.values(), key=lambda d: -len(d["prompts"])):
        gs = sorted(({**g, "prompts": len(g["prompts"]), "features": sorted(g["features"], key=lambda d: -d["prompts"])} for g in lib["groups"].values()), key=lambda d: -d["prompts"])
        unaligned.append({"corpus": lib["corpus"], "prompts": len(lib["prompts"]), "features": sum(len(g["features"]) for g in gs), "groups": gs})
    return {"kind": kind, "filters": {k: sorted(v) for k, v in filters.items()}, "facets": facets, "seed": h["seed"] is not None and bool(h["to_global"]),
            "prompts": len(selected), "decomposed": len(decomposed), "covered": len(covered), "on_global": len(on_global),
            "globals": sum(len(g["features"]) for g in tree), "groups": tree, "unaligned": unaligned, "libraries": len(h["libs"])}


def node(store: Store, kind: str, filters: dict[str, set[str]], fid: int, limit: int = 400) -> dict:
    """A node of the hierarchy in the slice: a global (its members per corpus, then their wordings) or a per-corpus feature
    (its wordings), each wording with the selected prompts it occurs in; and the prompts themselves."""
    fields = prompt_fields(store)
    rows, h = facts(store, kind)
    nodes = h["nodes"]
    if fid not in nodes:
        return {}
    selected = select(fields, filters, store=store)
    f = nodes[fid]; is_global = f["codebook"] == h["seed"]
    feats = {x for x, g in h["to_global"].items() if g == fid} if is_global else {fid}
    by_feat: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set)); prompts: dict[str, int] = defaultdict(int)
    for pid, rid, nd, feat, glob in rows:
        if feat in feats and pid in selected:
            by_feat[feat][rid].add(pid); prompts[pid] += 1
    rids = sorted({r for d in by_feat.values() for r in d})
    text = {}
    for i in range(0, len(rids), 500):
        chunk = rids[i:i + 500]
        text |= {int(r["id"]): dict(r) for r in store.rows(f"SELECT id, declaration, polarity, prompts, n FROM realization WHERE id IN ({','.join('?' * len(chunk))})", chunk)}
    # the quotes: each wording's span in the selected prompts, a dozen per wording
    quotes: dict[int, list[dict]] = defaultdict(list)
    for i in range(0, len(rids), 400):
        chunk = rids[i:i + 400]
        for r in store.rows(f"SELECT r.realization, r.prompt, SUBSTR(p.text, s.lo+1, MIN(s.hi-s.lo, 320)) text, s.hi-s.lo chars FROM reading r JOIN span s ON s.id=r.span JOIN prompt p ON p.id=r.prompt "
                            f"WHERE r.realization IN ({','.join('?' * len(chunk))}) ORDER BY r.prompt", chunk):
            if r["prompt"] in selected and len(quotes[int(r["realization"])]) < 12:
                quotes[int(r["realization"])].append({"prompt": r["prompt"], "text": r["text"], "chars": int(r["chars"])})
    members = []
    for feat, d in sorted(by_feat.items(), key=lambda kv: -len(set().union(*kv[1].values()))):
        x = nodes[feat]
        ws = sorted(({"id": r, "declaration": text[r]["declaration"], "polarity": text[r]["polarity"], "prompts": len(ps), "corpus_prompts": text[r]["prompts"], "quotes": quotes.get(r, [])} for r, ps in d.items() if r in text), key=lambda w: -w["prompts"])
        members.append({"id": feat, "corpus": x["corpus"], "name": x["name"], "definition": x["definition"] or "", "polarity": x["polarity"] or "require", "group": nodes.get(x["parent"], {}).get("name"),
                        "prompts": len(set().union(*d.values())), "wordings": ws})
    ps = sorted(prompts, key=lambda p: -prompts[p])[:limit]
    plist = []
    for i in range(0, len(ps), 400):
        chunk = ps[i:i + 400]
        plist += [dict(r) | {"readings": prompts[r["id"]], "fields": fields.get(r["id"], {})} for r in store.rows(f"SELECT p.id, k.name corpus, SUBSTR(p.text, 1, 160) head, LENGTH(p.text) chars FROM prompt p JOIN corpus k ON k.id=p.corpus WHERE p.id IN ({','.join('?' * len(chunk))})", chunk)]
    plist.sort(key=lambda p: -p["readings"])
    parent = nodes.get(f["parent"])
    return {"id": fid, "kind": kind, "global": is_global, "name": f["name"], "definition": f["definition"] or "", "polarity": f["polarity"] or "require", "corpus": None if is_global else f["corpus"],
            "group": {"name": parent["name"], "aspect": parent.get("aspect"), "definition": parent.get("definition") or ""} if parent else None, "round": f["round"],
            "global_of": h["to_global"].get(fid) if not is_global else None, "global_name": nodes.get(h["to_global"].get(fid), {}).get("name") if not is_global else None,
            "filters": {k: sorted(v) for k, v in filters.items()}, "selected": len(selected), "prompts": len(prompts), "readings": sum(prompts.values()),
            "members": members, "prompt_list": plist}


def prompts(store: Store, kind: str, filters: dict[str, set[str]], limit: int = 300) -> dict:
    """The slice projected on prompts: each with its fields, its readings placed on a feature, and the features it carries
    (globals by name, else the corpus feature), by readings."""
    fields = prompt_fields(store)
    rows, h = facts(store, kind)
    selected = select(fields, filters, store=store)
    pol = filters.get("polarity")
    per: dict[str, dict] = {}
    for pid, rid, node, feat, glob in rows:
        if pid not in selected or (pol and h["nodes"][feat].get("polarity", "require") not in pol):
            continue
        d = per.setdefault(pid, {"readings": 0, "features": defaultdict(int)})
        d["readings"] += 1
        d["features"][glob if glob is not None else feat] += 1
    ids = sorted(selected, key=lambda p: (-per.get(p, {}).get("readings", 0), p))[:limit]
    out = []
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        text = {r["id"]: dict(r) for r in store.rows(f"SELECT p.id, k.name corpus, SUBSTR(p.text, 1, 160) head, LENGTH(p.text) chars, d.status, d.coverage FROM prompt p JOIN corpus k ON k.id=p.corpus LEFT JOIN decomp d ON d.prompt=p.id WHERE p.id IN ({','.join('?' * len(chunk))})", chunk)}
        for pid in chunk:
            if pid not in text:
                continue
            d = per.get(pid, {"readings": 0, "features": {}})
            feats = sorted(d["features"].items(), key=lambda kv: -kv[1])[:6]
            out.append(text[pid] | {"fields": fields.get(pid, {}), "readings": d["readings"],
                                    "features": [{"id": f, "name": h["nodes"][f]["name"], "global": h["nodes"][f]["codebook"] == h["seed"], "n": n} for f, n in feats if f in h["nodes"]]})
    return {"kind": kind, "filters": {k: sorted(v) for k, v in filters.items()}, "prompts": len(selected), "listed": len(out), "list": out}
