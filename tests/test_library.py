import json

import pytest

from fx import library as L
from fx.corpus import import_text
from fx.llm import Client
from fx.store import Store
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_server import FakeServer, reply  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def seed(store, n_prompts=6):
    """A corpus whose readings are already in the store: per prompt two guidance atoms and one material leaf."""
    cid = None
    for k in range(n_prompts):
        import_text(store, f"Prompt {k}. Think step by step. Return JSON only. Example: x", name="c")
    for k, p in enumerate(store.rows("SELECT id FROM prompt ORDER BY at, id")):
        for i, (kind, note, verb, obj, pol) in enumerate([("atom", None, "think" if k % 2 else "reason", "step by step", "require"), ("atom", None, "return", "JSON only" if k % 2 else "prose", "require" if k % 2 else "forbid"), ("material", "example", "provide", "a worked example", "require")]):
            sid = store.insert("span", {"prompt": p["id"], "path": str(i), "lo": 0, "hi": 5, "kind": kind, "note": note, "flags": "[]", "start": "", "end": "", "depth": 0})
            store.insert("reading", {"prompt": p["id"], "span": sid, "verb": verb, "object": obj, "qualifier": "", "polarity": pol, "condition": "always", "domain_terms": "[]", "declaration": f"{verb} {obj}"})
    return cid


def test_collapse_groups_identical_declarations(store):
    seed(store)
    r = L.collapse(store, "c", "guidance")
    assert r["readings"] == 12 and r["realizations"] == 4            # think/reason step by step, return JSON only, forbid return prose, 3 each
    rz = L.realizations(store, "c", "guidance")
    assert rz[0]["n"] == 3 and rz[0]["prompts"] == 3
    assert store.one("SELECT COUNT(*) k FROM reading WHERE realization IS NULL")["k"] == 6   # the material readings, not collapsed yet
    assert L.collapse(store, "c", "guidance")["new"] == 0                                    # idempotent
    assert L.collapse(store, "c", "material")["realizations"] == 1


def _cb_reply(rz, old=None):
    ids = {r["declaration"]: r["id"] for r in rz}
    return reply(json.dumps({"notes": "first", "groups": [
        {"id": old and f"G{old[0]}", "name": "reasoning", "definition": "how the model reasons", "aspect": "reasoning",
         "features": [{"id": old and f"F{old[1]}", "name": "think step by step", "definition": "asks for stepwise reasoning", "polarity": "require", "examples": [f"R{ids['think step by step']}", "R999"]}]},
        {"id": None, "name": "output", "definition": "the shape of the answer", "aspect": "format",
         "features": [{"id": None, "name": "return JSON only", "definition": "the answer is JSON and nothing else", "polarity": "require", "examples": [f"R{ids['return JSON only']}"]},
                      {"id": None, "name": "return prose", "definition": "prose answers are prohibited", "polarity": "forbid", "examples": [f"R{ids['return prose']}"]}]}]}))


def test_coldstart_assign_judge_revise(store):
    seed(store)
    L.collapse(store, "c", "guidance")
    rz = L.realizations(store, "c", "guidance")
    ids = {r["declaration"]: r["id"] for r in rz}
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        srv.calls = 0; srv.script = [_cb_reply(rz)]
        r = L.coldstart(store, c, "c", "guidance", model="m")
        assert r["version"] == 1 and r["groups"] == 2 and r["features"] == 3
        tree = L.groups(store, r["codebook"])
        f_think = tree[0]["features"][0]; f_json, f_prose = tree[1]["features"]
        assert f_think["examples"] == [ids["think step by step"]] and f_think["parent"] == tree[0]["id"]     # R999 dropped
        assert L.preview(store, "c", "guidance", "assign", model="m")["calls"] == 1

        srv.calls = 0; srv.script = [reply(json.dumps({"assignments": [{"id": f"R{ids['think step by step']}", "feature": f"F{f_think['id']}", "confidence": "high"},
                                                        {"id": f"R{ids['return JSON only']}", "feature": f"F{f_json['id']}", "confidence": "medium"},
                                                        {"id": f"R{ids['reason step by step']}", "feature": f"F{f_think['id']}", "confidence": "low"},
                                                        {"id": f"R{ids['return prose']}", "feature": None, "confidence": "high"}]}))]
        a = L.assign(store, c, "c", "guidance", model="m", workers=1)
        assert a["assigned"] == 3 and a["leftover"] == 1 and a["anchor_agreement"] == round(2 / 3, 3)     # prose's own example went to none
        assert L.assign(store, c, "c", "guidance", model="m")["batches"] == 0                                  # resumable: nothing left
        tree = L.groups(store, r["codebook"])
        assert tree[0]["features"][0]["support"] == 6 and tree[0]["support"] == 6 and tree[0]["features"][0]["realizations"] == 2
        st = L.status(store, "c", "guidance")
        assert st["versions"][0]["leftover"] == 1 and st["versions"][0]["low"] == 1 and st["versions"][0]["reading_coverage"] == round(9 / 12, 3)

        srv.calls = 0; srv.script = [reply(json.dumps({"misfits": [{"id": f"R{ids['reason step by step']}", "why": "x"}], "split": None})),          # f_think's two members
                      reply(json.dumps({"indistinct": [{"a": f"F{f_json['id']}", "b": f"F{f_prose['id']}", "why": "same thing"}]}))]        # the output group's siblings
        j = L.judge(store, c, "c", "guidance", model="m", workers=1)
        assert j["calls"] == 2 and j["misfits"] == 1 and j["indistinct"] == 1
        fl = L.flags(store, r["codebook"])
        assert sorted(x["verdict"] for x in fl) == ["indistinct", "misfit"]

        srv.calls = 0; srv.script = [_cb_reply(rz, old=(tree[0]["id"], f_think["id"]))]
        v2 = L.revise(store, c, "c", "guidance", model="m")
        assert v2["version"] == 2 and v2["kept"] == 1 and v2["leftover_seen"] == 2 and v2["flags_seen"] == 2            # leftover: none + low
        t2 = L.groups(store, v2["codebook"])
        assert t2[0]["prev"] == tree[0]["id"] and t2[0]["features"][0]["prev"] == f_think["id"] and t2[1]["features"][0]["prev"] is None
        assert L.latest(store, "c", "guidance")["version"] == 2 and L.latest(store, "c", "guidance", 1)["id"] == r["codebook"]


def test_unparsed_assign_batch_is_left_for_next_run(store):
    seed(store)
    L.collapse(store, "c", "guidance")
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        srv.calls = 0; srv.script = [_cb_reply(L.realizations(store, "c", "guidance")), reply("no json")]
        L.coldstart(store, c, "c", "guidance", model="m")
        a = L.assign(store, c, "c", "guidance", model="m", workers=1)
        assert a["unparsed"] == 1 and a["assigned"] == 0
        assert store.one("SELECT COUNT(*) k FROM assignment")["k"] == 0


def test_site_library_endpoints_and_job(tmp_path):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    from fx.paths import Workspace
    ws = Workspace(tmp_path / "w"); store = Store(ws.store_path); seed(store)
    with FakeServer() as srv:
        srv.script = [_cb_reply(sorted([{"declaration": d, "id": i} for i, d in enumerate(["think step by step", "reason step by step", "return JSON only", "return prose"], 1)], key=lambda x: x["id"]))]
        import fx.llm.registry as reg
        c = TestClient(make_app(ws, store))
        import fx.gui.server as srvmod
        srvmod.Client = lambda store, **kw: Client(store, base_url=srv.url)          # the job's client talks to the fake server
        r = c.post("/api/library/jobs", json={"corpus": "c", "kind": "guidance", "step": "coldstart", "model": "m"}).json()
        import time
        for _ in range(100):
            j = c.get(f"/api/jobs/{r['id']}").json()
            if j["status"] != "running":
                break
            time.sleep(0.05)
        assert j["status"] == "done", j
        lib = c.get("/api/library?corpus=c&kind=guidance").json()
        assert lib["codebook"]["version"] == 1 and len(lib["groups"]) == 2 and lib["realizations"] == 4
        fid = lib["groups"][0]["features"][0]["id"]
        f = c.get(f"/api/feature/{fid}").json()
        assert f["feature"]["name"] == "think step by step" and f["group"]["name"] == "reasoning"
        assert c.get("/api/library/preview?corpus=c&kind=guidance&step=assign&model=m").json()["calls"] == 1


def test_round_runs_the_sequence_and_stops_on_no_gain(store):
    seed(store)
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        L.collapse(store, "c", "guidance")
        rz = L.realizations(store, "c", "guidance"); ids = {r["declaration"]: r["id"] for r in rz}
        cbreply = _cb_reply(rz)
        def assign_reply(body):
            import re
            got = re.findall(r"^R(\d+) \|", body["messages"][-1]["content"], re.M)
            feats = re.findall(r"^  F(\d+) \(", body["messages"][-1]["content"], re.M)
            return reply(json.dumps({"assignments": [{"id": f"R{r}", "feature": f"F{feats[0]}", "confidence": "high"} for r in got]}))
        def router(body):
            text = body["messages"][-1]["content"]
            if "# DECLARATIONS" in text and "# CODEBOOK" in text and "Revise" not in text and "# LEFTOVER" not in text:
                return assign_reply(body)
            if "# LEFTOVER" in text:
                return _cb_reply(rz, old=(L.groups(store, L.latest(store, "c", "guidance")["id"])[0]["id"], L.groups(store, L.latest(store, "c", "guidance")["id"])[0]["features"][0]["id"]))
            if "# MEMBERS" in text:
                return reply(json.dumps({"misfits": [], "split": None}))
            if "# GROUP" in text:
                return reply(json.dumps({"indistinct": []}))
            return cbreply
        srv.router = router
        logged = []
        r = L.run_round(store, c, "c", "guidance", batch_model="m", codebook_model="m", workers=1, rounds=3, log=lambda n, res: logged.append(n))
        assert logged[:5] == ["collapse", "coldstart", "assign", "judge", "revise"] and logged[5] == "assign"
        # the fake puts every declaration on the first feature, so the other features' anchors fail: the anchor rule ends the loop after one round
        assert r["stopped_because"].startswith("anchor agreement fell") and len(r["versions"]) == 2
        v2 = L.status(store, "c", "guidance")["versions"][1]
        assert v2["assigned"] == 4 and v2["anchor_agreement"] < 0.9 and v2["notes"].startswith("[anchors")


def test_coldstart_takes_the_head_that_fits_and_records_the_rest(store):
    from fx.library.codebook import fit
    seed(store)
    L.collapse(store, "c", "guidance")
    rz = L.realizations(store, "c", "guidance")
    kept, dropped = fit(rz, budget_tokens=3000 + 40)              # room for about two lines after the overhead
    assert len(kept) >= 1 and dropped == len(rz) - len(kept) and kept[0]["prompts"] >= kept[-1]["prompts"]
    with FakeServer() as srv:
        srv.script = [_cb_reply(rz)]
        r = L.coldstart(store, Client(store, base_url=srv.url), "c", "guidance", model="m", context_tokens=3000 + 40)
        assert r["shown"] == len(kept) and r["waiting"] == dropped
        assert "waited for the loop" in L.latest(store, "c", "guidance")["notes"]
        assert f"\nR{rz[-1]['id']} |" not in srv.requests[0]["messages"][-1]["content"]   # the tail was not sent


def test_codebook_prompts_group_declarations_by_head():
    from fx.library.prompts import render_blocks
    rows = [{"id": 1, "polarity": "require", "head": "use", "declaration": "use the schema", "prompts": 5, "n": 5, "conditions": ["always"]},
            {"id": 2, "polarity": "require", "head": "use", "declaration": "use step-by-step reasoning", "prompts": 3, "n": 4, "conditions": ["if the question is hard"]},
            {"id": 3, "polarity": "forbid", "head": "use", "declaration": "use tools", "prompts": 1, "n": 1, "conditions": []}]
    text = render_blocks(rows, "guidance")
    assert text.startswith("## require · use (8 prompts, 2 wordings)\nR1 | the schema | 5\nR2 | step-by-step reasoning | 3 | when: if the question is hard\n## forbid · use (1 prompts, 1 wordings)\nR3 | tools | 1")
    mat = render_blocks([{"id": 9, "polarity": "require", "head": "example", "declaration": "provide a worked example", "prompts": 2, "n": 2, "conditions": []}], "material")
    assert mat == "## material kind example (2 prompts, 1 wordings)\nR9 | provide a worked example | 2"


def test_assign_prompt_carries_anchors_quotes_and_domain_terms(store):
    seed(store)
    L.collapse(store, "c", "guidance")
    rz = L.realizations(store, "c", "guidance")
    assert rz[0]["sample"] and rz[0]["head"] in ("think", "reason", "return")
    with FakeServer() as srv:
        srv.script = [_cb_reply(rz)]
        r = L.coldstart(store, Client(store, base_url=srv.url), "c", "guidance", model="m")
    from fx.library import prompts as P
    text = P.assign("guidance", L.groups(store, r["codebook"]), rz[:2])
    assert "e.g. think step by step" in text and 'quote: "' in text and "Identity is the instruction" in text
