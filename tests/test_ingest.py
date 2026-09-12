"""The Ingest side: corpora with their stage strip, the edits, profiles, one run through the stages, and the endpoints.
No model, no network: the stages are stand-ins that record their order."""
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cube import two_libraries_and_a_global  # noqa: E402

from fx import ingest as I  # noqa: E402
from fx import jobs  # noqa: E402
from fx.corpus import import_text  # noqa: E402
from fx.paths import Workspace  # noqa: E402
from fx.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def test_corpora_carry_their_stage_strip(store):
    a, b, g = two_libraries_and_a_global(store)
    import_text(store, "a fresh prompt", name="fresh")
    cs = {c["name"]: c for c in I.corpora(store)}
    assert cs["sql"]["prompts"] == 1 and cs["sql"]["decomposed"] == 1 and cs["sql"]["features"] == 3 and cs["sql"]["aligned"] == 1 and cs["sql"]["specific"] == 1
    assert cs["sql"]["stages"] == ["done", "done", "partial"] and cs["sql"]["pending"]           # one feature still open at the seed
    assert cs["fresh"]["stages"] == ["none", "none", "none"] and cs["fresh"]["pending"] and cs["fresh"]["codebook"] is None
    assert "seed" not in cs


def test_rename_retag_delete(store):
    a, b, g = two_libraries_and_a_global(store)
    I.rename(store, "sql", "text2sql")
    assert store.one("SELECT name FROM corpus WHERE name='text2sql'") and not store.one("SELECT 1 FROM corpus WHERE name='sql'")
    with pytest.raises(ValueError):
        I.rename(store, "text2sql", "cypher")
    I.retag(store, "text2sql", domain="sql", tags={"role": "staged"})
    p = store.one("SELECT domain, meta FROM prompt p JOIN corpus c ON c.id=p.corpus WHERE c.name='text2sql'")
    assert p["domain"] == "sql" and json.loads(p["meta"])["role"] == "staged"
    before = store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature' AND node IS NOT NULL")["n"]
    r = I.delete(store, "text2sql")
    assert r["prompts"] == 1 and r["codebooks"] == 1 and r["features"] == 4      # three features and their group
    assert not store.one("SELECT 1 FROM corpus WHERE name='text2sql'") and not store.one("SELECT 1 FROM prompt WHERE corpus=?", (r["id"],))
    assert store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature' AND node IS NOT NULL")["n"] == before - 1      # the global lost the sql member, kept the cypher one
    assert store.one("SELECT COUNT(*) n FROM feature WHERE id=?", (g,))["n"] == 1
    assert {c["name"] for c in I.corpora(store)} == {"cypher"}
    pid = store.one("SELECT id FROM prompt")["id"]
    assert I.delete_prompts(store, [pid])["deleted"] == 1 and not store.one("SELECT 1 FROM reading WHERE prompt=?", (pid,))


def test_profiles_default_and_saved(store):
    ps = I.profiles(store)
    assert ps[0]["name"] == "default" and ps[0]["params"]["codebook_model"] == "gpt-5.6-sol"
    I.save_profile(store, "cheap", {"batch_model": "openai/gpt-oss-120b", "budget": "5", "workers": "32"})
    p = I.profile(store, "cheap")
    assert p["batch_model"] == "openai/gpt-oss-120b" and p["budget"] == 5.0 and p["workers"] == 32 and p["decompose_model"] == I.DEFAULT_PROFILE["decompose_model"]
    assert [x["name"] for x in I.profiles(store)] == ["default", "cheap"]
    with pytest.raises(ValueError):
        I.save_profile(store, "", {})
    I.delete_profile(store, "cheap")
    assert [x["name"] for x in I.profiles(store)] == ["default"]


def test_run_profile_runs_the_stages_in_order_and_stops(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws"); store = Store(ws.store_path)
    import_text(store, "a prompt", name="c1")
    order = []
    import fx.decompose as D
    import fx.library as L
    import fx.align as A
    monkeypatch.setattr(D, "prompt_ids", lambda *a, **k: ["p1"])
    monkeypatch.setattr(D, "run", lambda *a, **k: order.append("decompose") or {"stopped": False, "failed": []})
    monkeypatch.setattr(L, "run_round", lambda *a, **k: order.append("codebook") or {"steps": [], "stopped_because": "settled"})
    monkeypatch.setattr(A, "run_round", lambda *a, **k: order.append("align") or {"steps": [], "stopped_because": "settled"})
    jid = jobs.start(store, ws, "profile", "c1", "m", {"profile": "default", "stage": "queued"}, 0)
    assert jobs.run_profile(store, ws, None, jid, "c1", I.DEFAULT_PROFILE) == "done"
    assert order == ["decompose", "codebook", "align"]
    j = store.one("SELECT status, params FROM job WHERE id=?", (jid,))
    assert j["status"] == "done" and json.loads(j["params"])["stage"] == "done"
    log = ws.job_log(jid).read_text()
    assert "stage decompose result" in log and "stage codebook guidance result" in log and "stage align guidance result" in log
    # a stop inside the codebook stage ends the run there
    order.clear()
    monkeypatch.setattr(L, "run_round", lambda *a, **k: order.append("codebook") or {"steps": [], "stopped_because": "stopped"})
    jid2 = jobs.start(store, ws, "profile", "c1", "m", {"profile": "default"}, 0)
    assert jobs.run_profile(store, ws, None, jid2, "c1", I.DEFAULT_PROFILE, stop=threading.Event()) == "stopped" and order == ["decompose", "codebook"]


def test_site_ingest_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    ws = Workspace(tmp_path / "ws"); store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    c = TestClient(make_app(ws, store))
    d = c.get("/api/ingest").json()
    assert {x["name"] for x in d["corpora"]} == {"sql", "cypher"} and d["profiles"][0]["name"] == "default" and d["models"]
    assert c.post("/api/profiles", json={"name": "cheap", "params": {"budget": 2}}).json()["params"]["budget"] == 2.0
    assert c.post("/api/corpus/sql/rename", json={"name": "cypher"}).status_code == 400
    assert c.post("/api/corpus/sql/retag", json={"domain": "d"}).json()["prompts"] == 1
    assert c.post("/api/corpus/nope/retag", json={"domain": "d"}).status_code == 404
    p = c.get("/api/cube?view=prompts&corpus=sql").json()
    assert p["prompts"] == 1 and p["list"][0]["corpus"] == "sql" and p["list"][0]["readings"] == 3 and {f["name"] for f in p["list"][0]["features"]} >= {"reason stepwise"}
    import fx.decompose as D
    import fx.library as L
    import fx.align as A
    monkeypatch.setattr(D, "prompt_ids", lambda *a, **k: [])
    monkeypatch.setattr(L, "run_round", lambda *a, **k: {"steps": [], "stopped_because": "settled"})
    monkeypatch.setattr(A, "run_round", lambda *a, **k: {"steps": [], "stopped_because": "settled"})
    r = c.post("/api/ingest/run", json={"corpora": ["sql"], "profile": "cheap"}).json()
    assert r["corpora"] == ["sql"] and len(r["ids"]) == 1
    import time
    for _ in range(50):
        j = c.get(f"/api/jobs/{r['ids'][0]}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done" and j["params"]["stage"] == "done" and j["params"]["profile"] == "cheap"
    st = c.get(f"/api/jobs/{r['ids'][0]}/stages").json()
    assert st["decomposition"]["done"] == 1 and st["codebook"]["current"] and len(st["codebook"]["groups"]) == 1
    assert st["alignment"]["corpus"]["aligned"] == 1 and st["alignment"]["under"][0]["features"][0]["name"] == "reason stepwise"
    assert c.post("/api/ingest/run", json={"corpora": [], "profile": "cheap"}).status_code == 400
    assert c.delete("/api/corpus/sql").json()["prompts"] == 1 and c.delete("/api/corpus/sql").status_code == 404
    assert c.post("/api/prompts/delete", json={"ids": [store.one("SELECT id FROM prompt")["id"]]}).json()["deleted"] == 1


def test_a_running_row_whose_log_went_quiet_is_stale_not_running(tmp_path):
    import os
    import time
    ws = Workspace(tmp_path / "ws"); store = Store(ws.store_path)
    import_text(store, "a prompt", name="c1")
    jid = jobs.start(store, ws, "decompose", "c1", "m", {}, 1)
    assert I.running_jobs(store, ws)[0]["live"] and I.corpora(store, ws=ws)[0]["running"] == "decompose"
    old = time.time() - 3600
    os.utime(ws.job_log(jid), (old, old))
    j = I.running_jobs(store, ws)[0]
    assert not j["live"] and j["idle_minutes"] > 50 and I.corpora(store, ws=ws)[0]["running"] is None
    assert I.close_job(store, jid)["closed"] and store.one("SELECT status FROM job WHERE id=?", (jid,))["status"] == "stopped" and not I.close_job(store, jid)["closed"]


def test_estimate_sums_the_stages_and_a_profile_kind_runs_both_kinds(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws"); store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    import_text(store, "a fresh prompt to decompose", name="fresh")
    e = I.estimate(store, "fresh", I.DEFAULT_PROFILE)
    assert e["decompose"] > 0 and e["total"] >= e["decompose"] and any(s["stage"] == "decompose" for s in e["steps"])
    e2 = I.estimate(store, "sql", I.DEFAULT_PROFILE)
    assert e2["decompose"] == 0 and e2["codebook"] >= 0 and e2["align"] > 0 and e2["total"] == round(e2["decompose"] + e2["codebook"] + e2["align"], 2)
    order = []
    import fx.decompose as D
    import fx.library as L
    import fx.align as A
    monkeypatch.setattr(D, "prompt_ids", lambda *a, **k: [])
    monkeypatch.setattr(L, "run_round", lambda st, cl, corpus, kind, **k: order.append(("codebook", kind)) or {"steps": [], "stopped_because": "settled"})
    monkeypatch.setattr(A, "run_round", lambda st, cl, kind, **k: order.append(("align", kind)) or {"steps": [], "stopped_because": "settled"})
    jid = jobs.start(store, ws, "profile", "sql", "m", {"profile": "default"}, 0)
    assert jobs.run_profile(store, ws, None, jid, "sql", I.DEFAULT_PROFILE | {"kind": "both"}) == "done"
    assert order == [("codebook", "guidance"), ("codebook", "material"), ("align", "guidance"), ("align", "material")]
    assert I.save_profile(store, "mat", {"kind": "material"})["params"]["kind"] == "material" and I.save_profile(store, "bad", {"kind": "nope"})["params"]["kind"] == "guidance"
