"""Stage 3 on two seeded corpus libraries and the scripted server: cards, vectors, cluster across corpora, name a global,
assign against it, judge and reopen, the loop. No model, no network."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_server import FakeServer, reply  # noqa: E402
from test_library import fake_encoder  # noqa: E402

from fx import align as A  # noqa: E402
from fx.corpus import import_text  # noqa: E402
from fx.llm import Client  # noqa: E402
from fx.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def seed_library(store, corpus, feats):
    """A finished per-corpus library: a codebook with one group and the given features, each with one anchor wording assigned to it."""
    import_text(store, f"a prompt of {corpus}", name=corpus)
    cid = store.one("SELECT id FROM corpus WHERE name=?", (corpus,))["id"]
    pid = store.one("SELECT id FROM prompt WHERE corpus=?", (cid,))["id"]
    cb = store.insert("codebook", {"corpus": cid, "kind": "guidance", "version": 1, "model": "m", "round": 0, "notes": "", "at": "now"})
    gid = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": None, "aspect": "format", "name": "output", "definition": "the answer", "polarity": None, "examples": [], "round": 0})
    ids = {}
    for name, definition, polarity, wording in feats:
        rid = store.insert("realization", {"corpus": cid, "kind": "guidance", "key": f"{polarity}|{wording}", "polarity": polarity, "declaration": wording, "n": 3, "prompts": 3, "conditions": "[]", "head": wording.split()[0]})
        sid = store.insert("span", {"prompt": pid, "path": "0", "lo": 0, "hi": 1, "kind": "atom", "note": None, "flags": "[]", "start": "", "end": "", "depth": 0})
        store.insert("reading", {"prompt": pid, "span": sid, "verb": wording.split()[0], "object": " ".join(wording.split()[1:]), "qualifier": "", "polarity": polarity, "condition": "always", "domain_terms": "[]", "declaration": wording, "realization": rid})
        fid = store.insert("feature", {"codebook": cb, "level": "feature", "parent": gid, "prev": None, "aspect": None, "name": name, "definition": definition, "polarity": polarity, "examples": [rid], "round": 0})
        store.insert("membership", {"kind": "realization", "unit": rid, "codebook": cb, "node": fid, "confidence": "high", "at": "now"})
        ids[name] = fid
    return ids


SQL = [("return SQL only", "the answer is the SQL query and nothing else", "require", "return only the SQL query"),
       ("think step by step", "asks for stepwise reasoning", "require", "think step by step"),
       ("use table aliases", "alias every table in a join", "require", "use aliases for the tables")]
CYPHER = [("return Cypher only", "the answer is the Cypher query and nothing else", "require", "return only the Cypher query"),
          ("think step by step", "asks for stepwise reasoning", "require", "think step by step"),
          ("use MERGE for upserts", "prefer MERGE over CREATE", "require", "use MERGE")]


def test_cards_vectors_and_cross_corpus_candidates(store):
    a, b = seed_library(store, "sql", SQL), seed_library(store, "cypher", CYPHER)
    cs = A.cards(store, "guidance")
    assert len(cs) == 6 and {c["corpus"] for c in cs} == {"sql", "cypher"} and all(c["anchors"] and c["support"] == 3 for c in cs)
    assert A.embed(store, "guidance", enc=fake_encoder)["embedded"] == 6 and A.embed(store, "guidance", enc=fake_encoder)["embedded"] == 0
    c = A.candidates(store, "guidance")
    # neighbourhoods, no threshold: a seed card and its nearest cards from the other corpus, each card in one cluster a round
    assert c["seeds"] == 6 and c["specific"] == 0 and c["clusters"]
    for cl in c["clusters"]:
        assert cl["members"][0]["id"] == cl["seed"] and cl["groups"] == 2 and len(cl["members"]) <= 6
        assert all(m["corpus"] != cl["members"][0]["corpus"] for m in cl["members"][1:])
    assert len({m["id"] for cl in c["clusters"] for m in cl["members"]}) == c["in_clusters"]
    st = A.status(store, "guidance")
    assert st["cards"] == 6 and st["domain_specific"] == 0 and st["globals"] == 0 and st["groups"] == 8          # born with the aspect groups


def test_name_assign_judge_reopen_and_the_loop(store):
    a, b = seed_library(store, "sql", SQL), seed_library(store, "cypher", CYPHER)
    A.embed(store, "guidance", enc=fake_encoder)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        def router(body):
            text = body["messages"][-1]["content"]
            if "# CLUSTER" in text:
                # the namer reads a neighbourhood: the card under study first, then its neighbours; it names only the cards
                # that give the seed's instruction under other nouns, and rejects when none does
                rows = re.findall(r"^F(\d+) \[(\w+)\] \(require\) (.+?):", text, re.M)
                key = lambda nm: nm.replace("SQL", "query").replace("Cypher", "query")
                same = [(i, corp) for i, corp, nm in rows if key(nm) == key(rows[0][2])]
                if len({corp for _, corp in same}) < 2:
                    return reply(json.dumps({"decision": "reject", "why": "different instructions"}))
                nm = key(rows[0][2]).replace("return query only", "return the query only")
                gid = re.findall(r"^G(\d+) format", text, re.M)[0]
                return reply(json.dumps({"decision": "same", "why": "", "group": f"G{gid}", "name": nm, "definition": "one instruction across domains", "polarity": "require", "members": [f"F{i}" for i, _ in same]}))
            if "# GLOBAL FEATURES" in text:
                ids = re.findall(r"^F(\d+) ", text, re.M); sids = re.findall(r"^  S(\d+) ", text, re.M)
                return reply(json.dumps({"assignments": [{"id": f"F{i}", "feature": None, "confidence": "high"} for i in ids]}))
            if "# GLOBAL\n" in text:
                return reply(json.dumps({"misfits": []}))
            return reply("{}")
        srv.router = router
        r = A.run_round(store, c, "guidance", batch_model="m", codebook_model="m", workers=1, rounds=8, encoder=fake_encoder)
        st = r["status"]
        assert st["globals"] == 2 and st["aligned"] == 4 and st["domain_specific"] == 2 and r["stopped_because"].startswith("settled")
        tree = A.globals_(store, "guidance")
        g = next(s for grp in tree for s in grp["features"] if s["name"] == "return the query only")
        assert g["corpora"] == 2 and g["support"] == 6 and {m["corpus"] for m in g["members"]} == {"sql", "cypher"}
        # a global's founding members are its anchors and are never reopened; a later member is. Put 'use table aliases' on the
        # global, flag it: it is reopened with the reason; the assigner puts it back: the flag stands
        late = a["use table aliases"]
        with store.lock:
            store.con.execute("UPDATE membership SET node=?, note=NULL WHERE kind='feature' AND unit=?", (g["id"], late)); store.con.commit()
        store.insert("flag", {"codebook": st["codebook"], "feature": g["id"], "realization": None, "other": late, "verdict": "misfit", "note": "different", "standing": 0})
        assert A.reopen(store, "guidance")["reopened_misfits"] == 1
        row = store.one("SELECT node, note FROM membership WHERE kind='feature' AND unit=?", (late,))
        assert row["node"] is None and row["note"].startswith(f"reopened:{g['id']}|different")
        seen = []
        def router2(body):
            text = body["messages"][-1]["content"]; seen.append(text)
            ids = re.findall(r"^F(\d+) ", text, re.M)
            return reply(json.dumps({"assignments": [{"id": f"F{i}", "feature": f"S{g['id']}", "confidence": "high"} for i in ids]}))
        srv.router = router2
        s2 = A.assign(store, c, "guidance", model="m", workers=1)
        assert s2["settled"] == 1 and any("the judge removed this from S" in t for t in seen)
        assert store.one("SELECT node FROM membership WHERE kind='feature' AND unit=?", (late,))["node"] == g["id"]
        assert store.one("SELECT standing FROM flag WHERE other=?", (late,))["standing"] == 1


def test_site_seed_endpoint_and_cli_status(tmp_path):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    from fx.paths import Workspace
    from fx.cli import main
    ws = Workspace(tmp_path / "w"); store = Store(ws.store_path)
    seed_library(store, "sql", SQL); seed_library(store, "cypher", CYPHER)
    c = TestClient(make_app(ws, store))
    s = c.get("/api/seed?kind=guidance").json()
    assert s["status"]["cards"] == 6 and s["status"]["globals"] == 0 and set(s["status"]["per_corpus"]) == {"sql", "cypher"} and len(s["libraries"]) == 2 and isinstance(s["status"]["open"], int)
    assert main(["-w", str(ws.root), "align", "status"]) == 0


def test_regroup_moves_invented_groups_under_the_aspects(store):
    seed_library(store, "sql", SQL); seed_library(store, "cypher", CYPHER)
    cb = A.seed_codebook(store, "guidance")
    g = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": None, "aspect": "format", "name": "answer form", "definition": "x", "polarity": None, "examples": [], "round": 1})
    store.insert("feature", {"codebook": cb, "level": "feature", "parent": g, "prev": None, "aspect": None, "name": "return the query only", "definition": "x", "polarity": "require", "examples": [], "round": 1})
    r = A.regroup_by_aspect(store, "guidance")
    assert r["moved"] == 1 and r["groups_dropped"] == 1 and r["groups"] == 8
    fmt = store.one("SELECT id FROM feature WHERE codebook=? AND level='group' AND name='format'", (cb,))["id"]
    assert store.one("SELECT parent FROM feature WHERE name='return the query only'")["parent"] == fmt


def test_reset_deletes_the_seed_and_keeps_the_cards(store):
    seed_library(store, "sql", SQL); seed_library(store, "cypher", CYPHER)
    A.embed(store, "guidance", enc=fake_encoder)
    cb = A.seed_codebook(store, "guidance")
    A.candidates(store, "guidance")                                   # writes open memberships on the seed
    r = A.reset(store, "guidance")
    assert r["codebook"] == cb and r["deleted"]["nodes"] == 8 and r["deleted"]["memberships"] == 6
    assert store.one("SELECT COUNT(*) k FROM codebook WHERE id=?", (cb,))["k"] == 0
    assert store.one("SELECT COUNT(*) k FROM embedding WHERE kind='feature'")["k"] == 6          # the cards' vectors survive
    st = A.status(store, "guidance")                                  # a fresh seed, born with its aspect groups again
    assert st["globals"] == 0 and st["groups"] == 8 and st["aligned"] == 0 and st["cards"] == 6
    assert A.reset(store, "material")["codebook"] is None

