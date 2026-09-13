"""History: checkpoints as shared blobs, restore with fresh ids and the seed pruned, diff, branch. No model, no network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cube import two_libraries_and_a_global  # noqa: E402

from fx import history as H  # noqa: E402
from fx.paths import Workspace  # noqa: E402
from fx.store import Store  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "runs" / "dev")


def test_checkpoint_writes_shared_blobs_and_does_not_repeat(ws):
    store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    c1 = H.checkpoint(store, ws, "sql", note="first")
    assert c1["counts"] == {"prompts": 1, "decomposed": 1, "readings": 3, "realizations": 3, "codebooks": 1, "groups": 1, "features": 3, "variants": 0, "placed": 3, "open": 0, "flags": 0}
    objs = sorted(p.name for p in H.objects_dir(ws).iterdir())
    assert len(objs) == 5 and all(n.endswith(".json.gz") for n in objs)
    c2 = H.checkpoint(store, ws, "sql")
    assert c2["repeated"] and c2["id"] == c1["id"]
    c3 = H.checkpoint(store, ws, "cypher")
    assert not c3["repeated"] and c3["tree"]["corpus"] != c1["tree"]["corpus"]
    assert len(list(H.objects_dir(ws).iterdir())) == 10                                    # two corpora, five groups each, nothing shared between them; c2 added none
    seed = H.checkpoint(store, ws, "seed")
    assert seed["counts"]["features"] == 1 and seed["counts"]["placed"] == 2 and "prompts" not in seed["tree"]
    assert [c["corpus"] for c in H.checkpoints(store)] == ["seed", "cypher", "sql"]


def test_restore_brings_the_corpus_back_and_prunes_the_seed(ws):
    store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    ck = H.checkpoint(store, ws, "sql", note="before")
    cb = store.one("SELECT codebook FROM feature WHERE id=?", (a["think step by step"],))["codebook"]
    grp = store.one("SELECT parent FROM feature WHERE id=?", (a["think step by step"],))["parent"]
    new = store.insert("feature", {"codebook": cb, "level": "feature", "parent": grp, "prev": None, "aspect": None, "name": "a feature added later", "definition": "d", "polarity": "require", "examples": [], "round": 2})
    rid = store.one("SELECT id FROM realization WHERE declaration='use aliases for the tables'")["id"]
    with store.lock:
        store.con.execute("UPDATE membership SET node=? WHERE kind='realization' AND unit=?", (new, rid)); store.con.commit()
    d = H.diff(store, ck["id"], ws)
    assert d["delta"]["features"] == 1 and d["features_added"] == ["a feature added later"] and d["features_gone"] == []
    seed_before = store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature'")["n"]
    r = H.restore(store, ws, ck["id"])
    assert r["corpus"] == "sql" and r["restored"]["features"] == 3 and r["features_remapped"] == 4
    assert not store.one("SELECT 1 FROM feature WHERE name='a feature added later'")
    names = sorted(x["name"] for x in store.rows("SELECT f.name FROM feature f JOIN codebook c ON c.id=f.codebook JOIN corpus k ON k.id=c.corpus WHERE k.name='sql' AND f.level='feature'"))
    assert names == ["return SQL only", "think step by step", "use table aliases"]
    m = store.one("SELECT m.node FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND r.declaration='use aliases for the tables'")
    assert store.one("SELECT name FROM feature WHERE id=?", (m["node"],))["name"] == "use table aliases"          # back on its own feature, under a fresh id
    # the seed's rows for sql's replaced features are pruned; cypher's member of the global stays; the global stays
    assert store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature'")["n"] == seed_before - 2 and r["seed_rows_pruned"] >= 2
    assert store.one("SELECT COUNT(*) n FROM membership WHERE kind='feature' AND node=?", (g,))["n"] == 1 and store.one("SELECT 1 FROM feature WHERE id=?", (g,))
    assert store.one("SELECT COUNT(*) n FROM prompt p JOIN corpus k ON k.id=p.corpus WHERE k.name='sql'")["n"] == 1
    assert store.one("SELECT COUNT(*) n FROM reading r JOIN prompt p ON p.id=r.prompt JOIN corpus k ON k.id=p.corpus WHERE k.name='sql'")["n"] == 3
    with pytest.raises(RuntimeError):
        H.restore(store, ws, ck["id"], live_corpora={"sql"})
    # the seed line restores too: its memberships that point at features now gone are dropped
    sk = H.checkpoint(store, ws, "seed")
    r2 = H.restore(store, ws, sk["id"])
    assert r2["corpus"] == "seed" and store.one("SELECT COUNT(*) n FROM feature f JOIN codebook c ON c.id=f.codebook WHERE c.scope='seed' AND f.level='feature'")["n"] == 1


def test_branch_is_a_workspace_restored_to_the_checkpoint(ws):
    store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    ck = H.checkpoint(store, ws, "cypher")
    with store.lock:
        store.con.execute("DELETE FROM feature WHERE id=?", (b["use MERGE for upserts"],)); store.con.commit()
    r = H.branch(store, ws, ck["id"], "try2")
    root = Path(r["workspace"])
    assert root.name == "try2" and (root / "store.db").exists() and "serve" in r
    s2 = Store(root / "store.db")
    assert s2.one("SELECT COUNT(*) n FROM feature f JOIN codebook c ON c.id=f.codebook JOIN corpus k ON k.id=c.corpus WHERE k.name='cypher' AND f.level='feature'")["n"] == 3
    assert store.one("SELECT COUNT(*) n FROM feature f JOIN codebook c ON c.id=f.codebook JOIN corpus k ON k.id=c.corpus WHERE k.name='cypher' AND f.level='feature'")["n"] == 2
    with pytest.raises(ValueError):
        H.branch(store, ws, ck["id"], "try2")


def test_site_history_endpoints(ws, tmp_path):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    c = TestClient(make_app(ws, store))
    r = c.post("/api/history/checkpoint", json={"corpus": "sql"}).json()
    assert r["id"] == 1 and c.post("/api/history/checkpoint", json={"corpus": "nope"}).status_code == 404
    h = c.get("/api/history").json()
    assert len(h["checkpoints"]) == 1 and h["checkpoints"][0]["counts"]["features"] == 3
    assert c.get("/api/history/1/diff").json()["delta"]["features"] == 0
    assert c.post("/api/history/1/restore", json={}).json()["restored"]["features"] == 3
    assert c.post("/api/history/9/restore", json={}).status_code == 404
    b2 = c.post("/api/history/1/branch", json={"name": "b1"}).json()
    assert Path(b2["workspace"]).name == "b1" and c.post("/api/history/1/branch", json={"name": "b1"}).status_code == 400
