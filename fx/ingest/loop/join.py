"""Step: join. The naming calls of a round ran in parallel, each over one neighbourhood, and could not see each other; the
join is one call that sees every proposal beside the tree and decides what the round adds: a proposal is new, or the same
instruction as an existing feature (its members go there), or a duplicate of another proposal (the two become one node).
It also places what has no group and founds a group where three unplaced features share a purpose, and it rules on the
pairs the sibling judge reported as indistinct: "same" folds the younger node into the older (members move, the younger
is retired with a pointer to where it went), "two" dismisses the report. The cold start's own pairs are never folded: they
are shown to a person. Then the round writes, once. Definitions never change."""
from __future__ import annotations

from typing import Callable, Optional

from fx.core.llm import Client
from fx.core.llm.registry import reasoning_low
from fx.core.store import Store, now
from fx.core.util.ids import parse_id, parse_ids
from fx.core.util.jsonx import extract_object
from fx.ingest.loop.calls import JOIN_SCHEMA, _stream
from fx.ingest.loop.level import Level
from fx.ingest.loop.state import nodes, tree


def indistinct_pairs(store: Store, lv: Level, features: dict) -> list[dict]:
    """The sibling judge's current indistinct reports, as pairs the join may rule on: at least one node of the pair was born
    in the loop (round > 0); a pair of the cold start's own features is trusted and left to a person."""
    out = []
    for r in store.rows("SELECT feature a, other b, note FROM flag WHERE codebook=? AND verdict='indistinct' AND standing=0", (lv.codebook,)):   # a pair ruled 'two' and raised again is standing: a report, not a question
        a, b = features.get(int(r["a"])), features.get(int(r["b"]))
        if a and b and a["polarity"] == b["polarity"] and max(a.get("round") or 0, b.get("round") or 0) > 0:
            older, younger = sorted([a, b], key=lambda f: (f.get("round") or 0, f["id"]))
            out.append({"older": older, "younger": younger, "why": r["note"] or ""})
    return out


def render_pairs(pairs: list[dict], node_prefix: str) -> str:
    out = []
    for q, p in enumerate(pairs, 1):
        o, y = p["older"], p["younger"]
        out.append(f"Q{q}: {node_prefix}{o['id']} (round {o.get('round') or 0}) {o['name']}: {o.get('definition') or ''}\n     ~ {node_prefix}{y['id']} (round {y.get('round') or 0}) {y['name']}: {y.get('definition') or ''}\n     the judge: {p['why'][:200]}")
    return "\n".join(out) if out else "(none)"


def _fold(store: Store, lv: Level, younger: int, older: int) -> int:
    """Fold node `younger` into `older`: its members and variants move, its flags go, and it is retired with a pointer."""
    n = store.con.execute("UPDATE membership SET node=? WHERE kind=? AND codebook=? AND node=?", (older, lv.kind, lv.codebook, younger)).rowcount
    store.con.execute("UPDATE feature SET parent=? WHERE parent=? AND level='variant'", (older, younger))
    store.con.execute("DELETE FROM flag WHERE codebook=? AND (feature=? OR other=?)", (lv.codebook, younger, younger))
    store.con.execute("UPDATE feature SET level='retired', prev=? WHERE id=?", (older, younger))
    return n


def _parse(text: str) -> Optional[dict]:
    """The join's JSON, even when a model let its reasoning run into the reply before the object: the usual extractor first,
    then the last object that starts at "proposals"."""
    obj = extract_object(text)
    if isinstance(obj, dict) and "proposals" in obj:
        return obj
    import json, re
    for m in reversed(list(re.finditer(r'\{\s*"proposals"', text))):
        try:
            return json.loads(text[m.start():text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            continue
    return obj if isinstance(obj, dict) else None


def join(store: Store, client: Client, lv: Level, proposals: list[dict], round_: int, model: str, effort: str = "low",
         progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    features = {n["id"]: n for n in nodes(tr) if n["level"] == "feature"}
    groups = {g["id"]: g for g in tr if g["id"] is not None}
    unplaced = [f for g in tr if g["id"] is None for f in g["features"]]
    pairs = indistinct_pairs(store, lv, features) if lv.prompt_join else []
    summary = {"codebook": lv.codebook, "round": round_, "proposals": len(proposals), "calls": 0, "unparsed": 0, "variants": 0, "features": 0,
               "into_existing": 0, "duplicates": 0, "topics": 0, "assigned": 0, "unplaced": 0, "placed": 0, "groups": 0, "still_unplaced": 0,
               "pairs": len(pairs), "folded": 0, "narrowed": 0, "kept_apart": 0}
    verdict = {k: ("new", None) for k in range(len(proposals))}
    place: dict = {}
    founded: list = []
    folds: list[tuple[int, int]] = []
    narrows: list[tuple[int, int]] = []
    dismissed: list[tuple[int, int]] = []
    cost = 0.0
    if lv.prompt_join and (proposals or pairs or (unplaced and not lv.groups_fixed)):
        # one call per aspect, and per slice of join_batch proposals within an aspect: a single call over a whole round
        # (447 proposals on text2sql) answered for 200 of them, all "new", and nothing else. Duplicates live within an
        # aspect almost always; a leak across chunks is caught by the sibling judge and folded by the next join.
        aspect_of_node = {n["id"]: g.get("aspect") or "other" for g in tr for n in g["features"] for n in [n] + n.get("variants", [])}
        aspect_of_group = {g["id"]: g.get("aspect") or "other" for g in tr if g["id"] is not None}
        def aspect_of(p: dict) -> str:
            if p["parent"] is not None:
                return aspect_of_node.get(p["parent"], "other")
            if p["group"] is not None:
                return aspect_of_group.get(p["group"], "other")
            return p["aspect"] or "other"
        chunks: list[dict] = []
        by_aspect: dict = {}
        for k, p in enumerate(proposals):
            by_aspect.setdefault(aspect_of(p), []).append(k)
        for f in unplaced:
            by_aspect.setdefault(f.get("aspect") or "other", [])
        for q, pr in enumerate(pairs):
            by_aspect.setdefault(aspect_of_node.get(pr["older"]["id"], "other"), [])
        for asp, ks in by_aspect.items():
            slices = [ks[i:i + lv.join_batch] for i in range(0, len(ks), lv.join_batch)] or [[]]
            for j, sl in enumerate(slices):
                chunks.append({"aspect": asp, "proposals": sl,
                               "unplaced": [f for f in unplaced if (f.get("aspect") or "other") == asp] if j == 0 and not lv.groups_fixed else [],
                               "pairs": [q for q, pr in enumerate(pairs) if aspect_of_node.get(pr["older"]["id"], "other") == asp] if j == 0 else []})
        chunks = [c for c in chunks if c["proposals"] or c["unplaced"] or c["pairs"]]
        prompts = [lv.prompt_join(tr, [proposals[k] for k in c["proposals"]], c["unplaced"], render_pairs([pairs[q] for q in c["pairs"]], lv.node_prefix)) for c in chunks]
        extra = reasoning_low(model, client.base_url) if effort == "low" else None
        for i, r in _stream(client, prompts, model, max(1, min(len(prompts), 16)), f"{lv.label}:join:r{round_}", lv.system, schema=JOIN_SCHEMA, extra=extra):
            c = chunks[i]
            summary["calls"] += 1; cost += r.cost or 0.0
            obj = _parse(r.text) if r and r.text else None
            if not isinstance(obj, dict):
                summary["unparsed"] += 1; continue
            local = c["proposals"]                                   # local P<j> (1-based) -> global index local[j-1]
            pids = set(range(1, len(local) + 1))
            def gkey(x) -> Optional[str]:                            # a local "P3" or a global "F41" -> the pending key
                if not isinstance(x, str):
                    return None
                j = parse_id(x, pids) if x.startswith("P") else None
                return f"P{local[j - 1] + 1}" if j is not None else (x if x.startswith("F") else None)
            for v in obj.get("proposals") or []:
                if not isinstance(v, dict):
                    continue
                j = parse_id(v.get("id"), pids)
                if j is None:
                    continue
                k = local[j - 1]
                if v.get("verdict") == "existing":
                    fid = parse_id(v.get("feature"), set(features))
                    if fid is not None and features[fid]["polarity"] == proposals[k]["polarity"]:
                        verdict[k] = ("existing", fid)
                elif v.get("verdict") == "duplicate":
                    of = parse_id(v.get("of"), pids)
                    if of is not None and of != j and proposals[local[of - 1]]["polarity"] == proposals[k]["polarity"]:
                        verdict[k] = ("duplicate", local[of - 1])
                elif v.get("verdict") == "topic":
                    verdict[k] = ("topic", None)                     # a topic, not an instruction: dropped, members stay open
            for pl in obj.get("place") or []:
                if isinstance(pl, dict) and gkey(pl.get("id")):
                    place[gkey(pl.get("id"))] = parse_id(pl.get("group"), set(groups))
            for g in obj.get("groups") or []:
                if isinstance(g, dict) and str(g.get("name") or "").strip():
                    founded.append(g | {"ids": [gkey(x) for x in (g.get("ids") or []) if gkey(x)]})
            qlocal = c["pairs"]; qids = set(range(1, len(qlocal) + 1))
            for v in obj.get("pairs") or []:
                j = parse_id(v.get("id"), qids) if isinstance(v, dict) else None
                if j is None:
                    continue
                pr = pairs[qlocal[j - 1]]
                if v.get("verdict") == "same":
                    folds.append((pr["younger"]["id"], pr["older"]["id"]))
                elif v.get("verdict") == "narrower" and lv.allow_variant and pr["younger"]["level"] == "feature" and pr["older"]["level"] == "feature":
                    narrows.append((pr["younger"]["id"], pr["older"]["id"]))
                elif v.get("verdict") == "two":
                    summary["kept_apart"] += 1; dismissed.append((pr["older"]["id"], pr["younger"]["id"]))
    # duplicates chain to a surviving proposal that is itself new
    def survivor(k: int) -> int:
        seen = set()
        while verdict[k][0] == "duplicate" and k not in seen:
            seen.add(k); k = verdict[k][1]
        return k
    for k in range(len(proposals)):
        if verdict[k][0] == "duplicate" and verdict[survivor(k)][0] != "new":
            verdict[k] = ("new", None)
    node_of: dict[int, int] = {}                                  # proposal index -> node id written or joined
    placed_units: set[int] = set()
    with store.lock:
        for k, p in enumerate(proposals):
            kind, target = verdict[k]
            if kind == "existing":
                node_of[k] = target; summary["into_existing"] += 1
            elif kind == "duplicate":
                continue
            elif kind == "topic":
                summary["topics"] += 1; continue
            else:
                row = {"codebook": lv.codebook, "prev": None, "aspect": None, "name": p["name"], "definition": p["definition"], "polarity": p["polarity"],
                       "examples": p["examples"], "round": round_}
                if p["parent"] is not None:
                    row |= {"level": "variant", "parent": p["parent"]}; summary["variants"] += 1
                else:
                    gid = p["group"]
                    if gid is None and lv.groups_fixed:              # the seed: the group of the aspect, else 'other'
                        gid = next((i for i, g in groups.items() if g.get("aspect") == p["aspect"]), None) or next((i for i, g in groups.items() if g.get("aspect") == "other"), None)
                    row |= {"level": "feature", "parent": gid, "aspect": p["aspect"] if gid is None else None}; summary["features"] += 1
                    if gid is None:
                        summary["unplaced"] += 1
                keys = list(row)
                cur = store.con.execute(f"INSERT INTO feature ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})", [__import__('json').dumps(row[x]) if isinstance(row[x], (list, dict)) else row[x] for x in keys])
                node_of[k] = int(cur.lastrowid)
        for k, p in enumerate(proposals):
            nid = node_of.get(k if verdict[k][0] != "duplicate" else survivor(k))
            if nid is None:
                continue
            if verdict[k][0] == "duplicate":
                summary["duplicates"] += 1
            for u in p["members"]:
                store.con.execute("INSERT INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,?,'high','named',?) ON CONFLICT(kind, unit, codebook) DO UPDATE SET node=excluded.node, confidence='high', note='named', at=excluded.at WHERE membership.node IS NULL",
                                  (lv.kind, u["id"], lv.codebook, nid, now()))
                placed_units.add(u["id"]); summary["assigned"] += 1
        # groups: place, then found
        pending = {f"F{f['id']}": f["id"] for f in unplaced} | {f"P{k + 1}": node_of[k] for k in range(len(proposals)) if verdict[k][0] == "new" and proposals[k]["parent"] is None and proposals[k]["group"] is None and k in node_of}
        taken: set[int] = set()
        for key, gid in place.items():
            fid = pending.get(key)
            if fid is not None and gid is not None and fid not in taken:
                taken.add(fid); summary["placed"] += store.con.execute("UPDATE feature SET parent=?, aspect=NULL WHERE id=? AND parent IS NULL", (gid, fid)).rowcount
        for g in founded:
            ids = [pending[x] for x in (g.get("ids") or []) if isinstance(x, str) and x in pending and pending[x] not in taken]
            if len(ids) < lv.group_min_features:
                continue
            aspect = str(g.get("aspect") or "other").lower()
            cur = store.con.execute("INSERT INTO feature (codebook, level, parent, prev, aspect, name, definition, polarity, examples, round) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                    (lv.codebook, "group", None, None, aspect if aspect in lv.aspects else "other", str(g["name"]).strip(), str(g.get("definition") or "").strip(), None, "[]", round_))
            for fid in ids:
                store.con.execute("UPDATE feature SET parent=?, aspect=NULL WHERE id=? AND parent IS NULL", (cur.lastrowid, fid))
            taken.update(ids); summary["groups"] += 1
        summary["still_unplaced"] = len(pending) - len(taken) if not lv.groups_fixed else 0
        for a, b in dismissed:                                    # a dismissed pair raised again is the report, not a question
            store.con.execute("UPDATE flag SET standing=1 WHERE codebook=? AND verdict='indistinct' AND ((feature=? AND other=?) OR (feature=? AND other=?))", (lv.codebook, a, b, b, a))
        # the judge's indistinct pairs the join called the same: the younger folds into the older
        retired: set[int] = set()
        for younger, older in folds:
            if younger in retired or older in retired or younger == older:
                continue
            _fold(store, lv, younger, older); retired.add(younger); summary["folded"] += 1
        # the younger of a pair that gives the older's instruction plus a rule becomes its variant: the hierarchy the tree has for that
        for younger, older in narrows:
            if younger in retired or older in retired:
                continue
            store.con.execute("UPDATE feature SET level='variant', parent=?, aspect=NULL WHERE id=? AND level='feature'", (older, younger))
            store.con.execute("UPDATE feature SET parent=? WHERE parent=? AND level='variant' AND id!=?", (older, younger, younger))   # a variant's variants move up
            store.con.execute("DELETE FROM flag WHERE codebook=? AND verdict='indistinct' AND (feature=? OR other=?)", (lv.codebook, younger, younger))
            summary["narrowed"] += 1
        # a neighbourhood whose seed was not placed has had its look
        for p in proposals:
            if p.get("seed") is not None and p["seed"] not in placed_units:
                store.con.execute("UPDATE membership SET note='specific' WHERE kind=? AND codebook=? AND unit=? AND node IS NULL AND (note IS NULL OR note='specific')", (lv.kind, lv.codebook, p["seed"]))
        store.con.commit()
    if progress and summary["calls"]:
        progress(1, 1, {"join": True, "features": summary["features"], "variants": summary["variants"], "into_existing": summary["into_existing"], "duplicates": summary["duplicates"], "groups": summary["groups"], "folded": summary["folded"], "narrowed": summary["narrowed"], "cost": cost, "calls": summary["calls"]})
    return summary
