"""Stage 2 on a seeded store and the scripted server: collapse, a fake encoder, cold start, assign with the retrieval
shortlist, judge, clustering with specific marking, naming, and the round loop. No model and no network."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_server import FakeServer, reply  # noqa: E402

from fx import library as L  # noqa: E402
from fx.corpus import import_text  # noqa: E402
from fx.llm import Client  # noqa: E402
from fx.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def fake_encoder(texts):
    """Bag of words hashed into 128 dims (a stable hash): wordings that share words are near, others are not."""
    import zlib
    out = np.zeros((len(texts), 128), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            if w != "require" and w != "forbid":
                out[i, zlib.crc32(w.encode()) % 128] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# six prompts; two share the step-by-step wording, two the reason wording, and every prompt has one output atom and one material leaf
ATOMS = [("think", "step by step"), ("reason", "step by step"), ("think", "step by step"), ("reason", "step by step"), ("keep", "the answer short"), ("return", "JSON only")]


def seed(store, atoms=ATOMS):
    for k, (verb, obj) in enumerate(atoms):
        import_text(store, f"Prompt {k}. {verb} {obj}. Example: x", name="c")
    for k, p in enumerate(store.rows("SELECT id FROM prompt ORDER BY at, id")):
        verb, obj = atoms[k]
        for i, (kind, note, v, o, pol) in enumerate([("atom", None, verb, obj, "require"), ("atom", None, "return", "prose" if k % 2 else "JSON only", "forbid" if k % 2 else "require"), ("material", "example", "provide", "a worked example", "require")]):
            sid = store.insert("span", {"prompt": p["id"], "path": str(i), "lo": 0, "hi": 5, "kind": kind, "note": note, "flags": "[]", "start": "", "end": "", "depth": 0})
            store.insert("reading", {"prompt": p["id"], "span": sid, "verb": v, "object": o, "qualifier": "", "polarity": pol, "condition": "always", "domain_terms": "[]", "declaration": f"{v} {o}"})


def ids_of(store):
    return {r["declaration"]: r["id"] for r in L.realizations(store, "c", "guidance")}


def cb_reply(store, names=("think step by step", "return JSON only")):
    ids = ids_of(store)
    return reply(json.dumps({"groups": [
        {"name": "reasoning", "definition": "how the model reasons", "aspect": "reasoning", "features": [{"name": names[0], "definition": "asks for stepwise reasoning", "polarity": "require", "examples": [f"R{ids[names[0]]}", "R999"]}]},
        {"name": "output", "definition": "the shape of the answer", "aspect": "format", "features": [{"name": names[1], "definition": "the answer is JSON and nothing else", "polarity": "require", "examples": [f"R{ids[names[1]]}"]},
                                                                                                      {"name": "return prose", "definition": "prose answers are prohibited", "polarity": "forbid", "examples": [f"R{ids['return prose']}"]}]}]}))


def test_collapse_and_embed(store):
    seed(store)
    r = L.collapse(store, "c", "guidance")
    assert r["readings"] == 12 and r["realizations"] == 5                       # think/reason step by step, keep short, return JSON only (require), return prose (forbid)
    rz = L.realizations(store, "c", "guidance")
    assert rz[0]["n"] >= 2 and rz[0]["head"] in ("think", "reason", "return") and rz[0]["sample"]
    e = L.embed(store, "c", "guidance", enc=fake_encoder)
    assert e["embedded"] == 5 and L.embed(store, "c", "guidance", enc=fake_encoder)["embedded"] == 0      # idempotent
    ids, m = L.vectors(store, [d["id"] for d in rz])
    assert m.shape == (5, 128) and abs(float(m[0] @ m[0]) - 1) < 1e-5
    assert L.collapse(store, "c", "material")["realizations"] == 1


def test_coldstart_assign_judge(store):
    seed(store); L.collapse(store, "c", "guidance"); L.embed(store, "c", "guidance", enc=fake_encoder)
    ids = ids_of(store)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        srv.calls = 0; srv.script = [cb_reply(store)]
        r = L.coldstart(store, c, "c", "guidance", model="m")
        tree = L.groups(store, r["codebook"])
        assert r["groups"] == 2 and r["features"] == 3 and tree[0]["features"][0]["examples"] == [ids["think step by step"]]     # R999 dropped
        f_think, f_json, f_prose = tree[0]["features"][0], tree[1]["features"][0], tree[1]["features"][1]
        # first pass: everything but 'reason step by step' and 'keep the answer short' lands; the shortlist pass then recovers 'reason step by step'
        def router(body):
            text = body["messages"][-1]["content"]
            got = re.findall(r"^R(\d+) \|", text, re.M); feats = re.findall(r"^  F(\d+) \(", text, re.M)
            shortlist = "nearest" in text
            out = []
            for g in got:
                g = int(g)
                if g == ids["think step by step"]: out.append((g, f_think["id"]))
                elif g == ids["return JSON only"]: out.append((g, f_json["id"]))
                elif g == ids["return prose"]: out.append((g, f_prose["id"]))
                elif g == ids["reason step by step"] and shortlist: out.append((g, f_think["id"]))
                else: out.append((g, None))
            return reply(json.dumps({"assignments": [{"id": f"R{g}", "feature": f"F{f}" if f else None, "confidence": "high"} for g, f in out]}))
        srv.router = router
        a = L.assign(store, c, "c", "guidance", model="m", workers=1, batch=3)
        assert a["assigned"] == 4 and a["leftover"] == 1 and a["second_pass"] >= 1 and a["second_pass_assigned"] == 1 and a["anchor_agreement"] == 1.0
        assert L.assign(store, c, "c", "guidance", model="m")["batches"] == 0                                            # nothing new: no batches
        st = L.status(store, "c", "guidance")["versions"][0]
        assert st["assigned"] == 4 and st["leftover"] == 1 and st["reading_coverage"] == round(11 / 12, 3)
        srv.router = lambda body: reply(json.dumps({"misfits": [{"id": f"R{ids['reason step by step']}", "why": "x"}], "split": None})) if "# MEMBERS" in body["messages"][-1]["content"] else reply(json.dumps({"indistinct": [{"a": f"F{f_json['id']}", "b": f"F{f_prose['id']}", "why": "same"}]}))
        j = L.judge(store, c, "c", "guidance", model="m", workers=1)
        assert j["calls"] == 2 and j["misfits"] == 1 and j["indistinct"] == 1


def test_cluster_marks_specific_and_name_grows_the_tree(store):
    # four prompts say 'keep the answer short' in three wordings, one prompt says something nobody echoes
    atoms = [("keep", "the answer short"), ("keep", "the answer brief"), ("keep", "your answer short"), ("keep", "the answer short"), ("think", "step by step"), ("cite", "the source table")]
    seed(store, atoms); L.collapse(store, "c", "guidance"); L.embed(store, "c", "guidance", enc=fake_encoder)
    ids = ids_of(store)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        srv.calls = 0; srv.script = [cb_reply(store)]
        r = L.coldstart(store, c, "c", "guidance", model="m"); cb = r["codebook"]
        tree = L.groups(store, cb); f_think = tree[0]["features"][0]
        srv.router = lambda body: reply(json.dumps({"assignments": [{"id": f"R{g}", "feature": f"F{f_think['id']}" if int(g) == ids["think step by step"] else None, "confidence": "high"} for g in re.findall(r"^R(\d+) \|", body["messages"][-1]["content"], re.M)]}))
        L.assign(store, c, "c", "guidance", model="m", workers=1, shortlist=False)
        cands = L.candidates(store, cb, "c", "guidance", tau=0.5)
        assert cands["open"] >= 5 and len(cands["clusters"]) == 1 and cands["specific"] >= 1
        members = {d["declaration"] for d in cands["clusters"][0]["members"]}
        assert {"keep the answer short", "keep the answer brief", "keep your answer short"} <= members
        assert store.one("SELECT note FROM membership WHERE kind='realization' AND unit=?", (ids["cite the source table"],))["note"] == "specific"
        m = cands["clusters"][0]["members"]
        srv.router = lambda body: reply(json.dumps({"decision": "feature", "why": "", "parent": None, "group": {"name": "answer length", "definition": "how long the answer is", "aspect": "answer"},
                                                   "name": "keep the answer short", "definition": "the answer is brief", "polarity": "require", "examples": [f"R{m[0]['id']}"], "members": [f"R{d['id']}" for d in m]}))
        n = L.name(store, c, "c", "guidance", cb, cands["clusters"], round_=1, model="m", workers=1)
        assert n["features"] == 1 and n["groups"] == 1 and n["assigned"] == len(m)
        tree = L.groups(store, cb)
        assert tree[-1]["name"] == "answer length" and tree[-1]["features"][0]["round"] == 1 and tree[-1]["features"][0]["support"] >= 3
        assert store.one("SELECT note FROM membership WHERE kind='realization' AND unit=?", (m[0]["id"],))["note"] == "named"
        assert L.candidates(store, cb, "c", "guidance", tau=0.5)["clusters"] == []                                    # nothing left together
        # a variant: the naming call may narrow an existing feature instead
        srv.router = lambda body: reply(json.dumps({"decision": "variant", "why": "", "parent": f"F{f_think['id']}", "group": None, "name": "briefly", "definition": "stepwise but brief", "polarity": "require", "examples": [], "members": [f"R{d['id']}" for d in m]}))
        with store.lock:
            store.con.execute("UPDATE membership SET node=NULL, note=NULL WHERE kind='realization' AND codebook=? AND note='named'", (cb,)); store.con.commit()
        n = L.name(store, c, "c", "guidance", cb, [{"members": m, "prompts": 4}], round_=2, model="m", workers=1)
        assert n["variants"] == 1
        tree = L.groups(store, cb)
        assert tree[0]["features"][0]["variants"][0]["name"] == "briefly" and tree[0]["features"][0]["support"] == tree[0]["features"][0]["own"] + tree[0]["features"][0]["variants"][0]["support"]


def test_round_runs_the_growing_loop(store):
    atoms = [("keep", "the answer short"), ("keep", "the answer brief"), ("keep", "your answer short"), ("think", "step by step"), ("think", "step by step"), ("cite", "the source table")]
    seed(store, atoms); L.collapse(store, "c", "guidance")
    ids = ids_of(store)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        def router(body):
            text = body["messages"][-1]["content"]
            if "# CLUSTER" in text:
                m = re.findall(r"^R(\d+) \|", text, re.M)
                return reply(json.dumps({"decision": "feature", "why": "", "parent": None, "group": "G1", "name": "keep the answer short", "definition": "brief", "polarity": "require", "examples": [f"R{m[0]}"], "members": [f"R{x}" for x in m]}))
            if "# DECLARATIONS" in text and "# CODEBOOK" in text:
                feats = re.findall(r"^  F(\d+) \(", text, re.M)
                return reply(json.dumps({"assignments": [{"id": f"R{g}", "feature": f"F{feats[0]}" if int(g) == ids["think step by step"] else None, "confidence": "high"} for g in re.findall(r"^R(\d+) \|", text, re.M)]}))
            if "# MEMBERS" in text: return reply(json.dumps({"misfits": [], "split": None}))
            if "# GROUP" in text: return reply(json.dumps({"indistinct": []}))
            return cb_reply(store)
        srv.router = router
        logged = []
        r = L.run_round(store, c, "c", "guidance", batch_model="m", codebook_model="m", workers=1, rounds=3, tau=0.5, min_yield=1, encoder=fake_encoder, log=lambda n, res: logged.append(n))
        assert logged[:7] == ["collapse", "coldstart", "embed", "assign", "judge", "reopen", "cluster"] and "name" in logged and "reopen" != logged[-1]        # the loop never ends on a reopen
        assert r["stopped_because"].startswith("settled")
        v = r["versions"][0]
        assert v["features"] == 4 and v["rounds"] == 1 and v["specific"] >= 1 and v["assigned"] >= 4


def test_site_library_endpoints_and_job(tmp_path):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    from fx.paths import Workspace
    ws = Workspace(tmp_path / "w"); store = Store(ws.store_path); seed(store)
    with FakeServer() as srv:
        import fx.gui.server as srvmod
        srvmod.Client = lambda store, **kw: Client(store, base_url=srv.url)
        sys.modules["fx.library.encoders"].encoder = lambda model=None: fake_encoder      # the package attribute `embed` is the function; the module is in sys.modules
        c = TestClient(make_app(ws, store))
        L.collapse(store, "c", "guidance"); srv.script = [cb_reply(store)]
        r = c.post("/api/library/jobs", json={"corpus": "c", "kind": "guidance", "step": "coldstart", "model": "m"}).json()
        import time
        for _ in range(400):
            j = c.get(f"/api/jobs/{r['id']}").json()
            if j["status"] != "running":
                break
            time.sleep(0.05)
        assert j["status"] == "done", (j, open(ws.job_log(r['id'])).read()[-800:])
        lib = c.get("/api/library?corpus=c&kind=guidance").json()
        assert lib["codebook"]["version"] == 1 and len(lib["groups"]) == 2 and lib["realizations"] == 5
        fid = lib["groups"][0]["features"][0]["id"]
        assert c.get(f"/api/feature/{fid}").json()["feature"]["name"] == "think step by step"
        assert c.get("/api/library/preview?corpus=c&kind=guidance&step=assign&model=m").json()["calls"] == 1
        assert c.get("/api/library/preview?corpus=c&kind=guidance&step=cluster").json()["dollars"] == 0


def test_reopen_sends_the_judgement_to_the_assigner_who_adjudicates(store):
    seed(store); L.collapse(store, "c", "guidance"); L.embed(store, "c", "guidance", enc=fake_encoder)
    ids = ids_of(store)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        srv.calls = 0; srv.script = [cb_reply(store)]
        cb = L.coldstart(store, c, "c", "guidance", model="m")["codebook"]
        f_think = L.groups(store, cb)[0]["features"][0]
        srv.router = lambda body: reply(json.dumps({"assignments": [{"id": f"R{g}", "feature": f"F{f_think['id']}", "confidence": "high"} for g in re.findall(r"^R(\d+) \|", body["messages"][-1]["content"], re.M)]}))
        L.assign(store, c, "c", "guidance", model="m", workers=1, shortlist=False)          # the fake puts everything on 'think step by step'
        store.insert("flag", {"codebook": cb, "feature": f_think["id"], "realization": ids["return prose"], "other": None, "verdict": "misfit", "note": "not reasoning", "standing": 0})
        r = L.reopen(store, cb)
        assert r["reopened_misfits"] == 1
        a = store.one("SELECT node feature, note FROM membership WHERE kind='realization' AND unit=?", (ids["return prose"],))
        assert a["feature"] is None and a["note"] == f"reopened:{f_think['id']}|not reasoning"
        seen = []
        def router(body):
            text = body["messages"][-1]["content"]; seen.append(text)
            return reply(json.dumps({"assignments": [{"id": f"R{g}", "feature": f"F{f_think['id']}", "confidence": "high"} for g in re.findall(r"^R(\d+) \|", text, re.M)]}))
        srv.router = router
        a2 = L.assign(store, c, "c", "guidance", model="m", workers=1, only_open=True, shortlist=False)   # the assigner sees the reason and puts it back: the flag is standing
        assert any("the judge removed this from F" in t and "not reasoning" in t for t in seen)
        assert a2.get("settled") == 1
        assert store.one("SELECT node feature FROM membership WHERE kind='realization' AND unit=?", (ids["return prose"],))["feature"] == f_think["id"]
        assert store.one("SELECT standing FROM flag WHERE realization=?", (ids["return prose"],))["standing"] == 1
        assert L.reopen(store, cb)["reopened_misfits"] == 0


def test_a_repeated_flag_is_standing_and_not_reopened(store):
    seed(store); L.collapse(store, "c", "guidance"); L.embed(store, "c", "guidance", enc=fake_encoder)
    ids = ids_of(store)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        srv.calls = 0; srv.script = [cb_reply(store)]
        cb = L.coldstart(store, c, "c", "guidance", model="m")["codebook"]
        f_think = L.groups(store, cb)[0]["features"][0]
        srv.router = lambda body: reply(json.dumps({"assignments": [{"id": f"R{g}", "feature": f"F{f_think['id']}", "confidence": "high"} for g in re.findall(r"^R(\d+) \|", body["messages"][-1]["content"], re.M)]}))
        L.assign(store, c, "c", "guidance", model="m", workers=1, shortlist=False)
        judge_reply = lambda body: reply(json.dumps({"misfits": [{"id": f"R{ids['return prose']}", "why": "x"}], "split": None})) if "# MEMBERS" in body["messages"][-1]["content"] else reply(json.dumps({"indistinct": []}))
        srv.router = judge_reply
        j1 = L.judge(store, c, "c", "guidance", model="m", workers=1)
        assert j1["new"] == 1 and j1["standing"] == 0
        assert L.reopen(store, cb)["reopened_misfits"] == 1                              # first time: acted on
        with store.lock:                                                                  # nothing else fits: it goes back where it was
            store.con.execute("UPDATE membership SET node=?, note=NULL WHERE kind='realization' AND unit=?", (f_think["id"], ids["return prose"])); store.con.commit()
        j2 = L.judge(store, c, "c", "guidance", model="m", workers=1)
        assert j2["new"] == 0 and j2["standing"] == 1
        assert L.reopen(store, cb)["reopened_misfits"] == 0                              # second time: standing, the member stays
        assert store.one("SELECT node feature FROM membership WHERE kind='realization' AND unit=?", (ids["return prose"],))["feature"] == f_think["id"]
        assert L.status(store, "c", "guidance")["versions"][0]["standing"] == 1
