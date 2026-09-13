"""The cube over two seeded libraries and a seed with one global; the settings file. No model, no network."""
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_align import CYPHER, SQL, seed_library  # noqa: E402

from fx.views import library as cube
from fx.core import settings  # noqa: E402
from fx.ingest.generalize.seed import seed_codebook  # noqa: E402
from fx.core.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def two_libraries_and_a_global(store):
    a, b = seed_library(store, "sql", SQL), seed_library(store, "cypher", CYPHER)
    cb = seed_codebook(store, "guidance")
    grp = store.one("SELECT id FROM feature WHERE codebook=? AND level='group' AND name='reasoning'", (cb,))["id"]
    g = store.insert("feature", {"codebook": cb, "level": "feature", "parent": grp, "prev": None, "aspect": None, "name": "reason stepwise", "definition": "asks for stepwise reasoning", "polarity": "require", "examples": [], "round": 1})
    for fid in (a["think step by step"], b["think step by step"]):
        store.insert("membership", {"kind": "feature", "unit": fid, "codebook": cb, "node": g, "confidence": "high", "note": "named", "at": "now"})
    store.insert("membership", {"kind": "feature", "unit": a["use table aliases"], "codebook": cb, "node": None, "confidence": None, "note": "specific", "at": "now"})
    for pid in [r["id"] for r in store.rows("SELECT id FROM prompt")]:
        store.insert("decomp", {"prompt": pid, "status": "done", "model": "m", "coverage": 1.0, "material_share": 0, "calls": 1, "seconds": 1, "reasks": 0, "flags": "[]", "failures": "[]", "error": None, "at": "now"})
    return a, b, g


def test_slice_shows_the_global_and_each_library_s_own_features(store):
    a, b, g = two_libraries_and_a_global(store)
    s = cube.slice(store, "guidance", {})
    assert s["prompts"] == 2 and s["decomposed"] == 2 and s["seed"] and s["libraries"] == 2
    assert s["globals"] == 1 and s["groups"][0]["name"] == "reasoning"
    glob = s["groups"][0]["features"][0]
    assert glob["id"] == g and glob["prompts"] == 2 and sorted(glob["corpora"]) == ["cypher", "sql"] and len(glob["members"]) == 2
    assert s["covered"] == 2 and s["on_global"] == 2
    own = {u["corpus"]: u for u in s["unaligned"]}
    assert own["sql"]["features"] == 2 and own["cypher"]["features"] == 2                 # the two features per library the seed has not absorbed
    assert [f["field"] for f in s["facets"]][:2] == ["corpus", "domain"] or s["facets"][0]["field"] == "corpus"
    corpus = next(f for f in s["facets"] if f["field"] == "corpus")
    assert {v["value"]: v["prompts"] for v in corpus["values"]} == {"sql": 1, "cypher": 1}


def test_a_filter_narrows_the_slice_and_the_facet_keeps_the_alternatives(store):
    a, b, g = two_libraries_and_a_global(store)
    s = cube.slice(store, "guidance", cube.parse_filters({"corpus": "sql", "kind": "guidance"}))
    assert s["prompts"] == 1 and s["filters"] == {"corpus": ["sql"]}
    glob = s["groups"][0]["features"][0]
    assert glob["prompts"] == 1 and glob["corpora"] == ["sql"] and [m["corpus"] for m in glob["members"]] == ["sql"]
    assert [u["corpus"] for u in s["unaligned"]] == ["sql"]
    corpus = next(f for f in s["facets"] if f["field"] == "corpus")
    assert {v["value"]: v["on"] for v in corpus["values"]} == {"sql": True, "cypher": False}       # the field's own filter is lifted for its facet
    assert cube.slice(store, "guidance", {"polarity": {"forbid"}})["globals"] == 0


def test_a_node_drills_to_wordings_and_prompts(store):
    a, b, g = two_libraries_and_a_global(store)
    d = cube.node(store, "guidance", {}, g)
    assert d["global"] and d["name"] == "reason stepwise" and d["prompts"] == 2 and len(d["members"]) == 2
    m = {x["corpus"]: x for x in d["members"]}
    assert m["sql"]["wordings"][0]["declaration"] == "think step by step" and m["sql"]["wordings"][0]["prompts"] == 1
    assert len(d["prompt_list"]) == 2 and all(p["readings"] == 1 for p in d["prompt_list"])
    f = cube.node(store, "guidance", {"corpus": {"cypher"}}, b["use MERGE for upserts"])
    assert not f["global"] and f["corpus"] == "cypher" and f["global_of"] is None and f["prompts"] == 1 and f["members"][0]["wordings"][0]["declaration"] == "use MERGE"
    f2 = cube.node(store, "guidance", {}, a["think step by step"])
    assert f2["global_of"] == g and f2["global_name"] == "reason stepwise"
    assert cube.node(store, "guidance", {}, 10 ** 6) == {}


def test_settings_file_is_written_blind_and_loaded_where_the_shell_has_nothing(tmp_path, monkeypatch):
    p = tmp_path / "keys.env"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False); monkeypatch.delenv("FX_LOCAL_URL", raising=False)
    r = settings.save("OPENROUTER_API_KEY", "sk-or-testvalue", path=p)
    assert r["set"] and p.exists() and oct(p.stat().st_mode & 0o777) == "0o600" and os.environ["OPENROUTER_API_KEY"] == "sk-or-testvalue"
    settings.save("FX_LOCAL_URL", "http://localhost:8002/v1", path=p)
    assert settings.read_file(p) == {"OPENROUTER_API_KEY": "sk-or-testvalue", "FX_LOCAL_URL": "http://localhost:8002/v1"}
    st = settings.status(path=p)
    k = next(x for x in st["keys"] if x["name"] == "OPENROUTER_API_KEY")
    assert k["set"] and k["secret"] and k["length"] == 15 and "value" not in k and k["source"] == "file"
    assert next(x for x in st["settings"] if x["name"] == "FX_LOCAL_URL")["value"] == "http://localhost:8002/v1"
    assert next(e for e in st["endpoints"] if e["name"] == "local")["base_url"] == "http://localhost:8002/v1"
    from fx.core.llm.registry import resolve
    assert resolve("openai/gpt-oss-20b").base_url == "http://localhost:8002/v1"
    settings.save("OPENROUTER_API_KEY", "", path=p)
    assert "OPENROUTER_API_KEY" not in os.environ and "OPENROUTER_API_KEY" not in settings.read_file(p)
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    (tmp_path / "k2.env").write_text("OPENAI_API_KEY=from-file\nFX_PROVIDER=Wafer\n")
    assert settings.load(tmp_path / "k2.env") == ["FX_PROVIDER"] and os.environ["OPENAI_API_KEY"] == "from-shell"
    with pytest.raises(ValueError):
        settings.save("BAD NAME", "x", path=p)
    bad = settings.probe(base_url="http://127.0.0.1:9/v1", timeout=2)
    assert not bad["ok"] and bad["error"]


def test_site_cube_and_settings_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    from fx.core.paths import Workspace
    monkeypatch.setattr(settings, "KEY_FILE", tmp_path / "keys.env")
    ws = Workspace(tmp_path / "ws"); store = Store(ws.store_path)
    a, b, g = two_libraries_and_a_global(store)
    c = TestClient(make_app(ws, store))
    s = c.get("/api/cube?kind=guidance&corpus=sql,cypher&role=").json()
    assert s["prompts"] == 2 and s["globals"] == 1 and s["filters"] == {"corpus": ["cypher", "sql"]}
    d = c.get(f"/api/cube/node/{g}?corpus=sql").json()
    assert d["global"] and d["prompts"] == 1
    assert c.get("/api/cube/node/999999").status_code == 404
    st = c.get("/api/settings").json()
    assert {k["name"] for k in st["keys"]} == set(settings.KEYS) and st["models"] and "value" not in st["keys"][0]
    assert c.post("/api/settings/key", json={"name": "NOT_A_KEY", "value": "x"}).status_code == 400
    monkeypatch.delenv("FX_API_KEY", raising=False)
    assert c.post("/api/settings/key", json={"name": "FX_API_KEY", "value": "abc"}).json()["set"] and os.environ["FX_API_KEY"] == "abc"
    assert next(k for k in c.get("/api/settings").json()["keys"] if k["name"] == "FX_API_KEY")["length"] == 3


def test_the_search_box_narrows_the_slice_by_prompt_text_or_wording(store):
    a, b, g = two_libraries_and_a_global(store)
    f = cube.parse_filters({"text": " MERGE ", "corpus": ""})
    assert f == {"text": {"MERGE"}}
    s = cube.slice(store, "guidance", f)
    assert s["prompts"] == 1 and s["filters"] == {"text": ["MERGE"]}                      # the cypher prompt: its wording "use MERGE"
    assert [u["corpus"] for u in s["unaligned"]] == ["cypher"]
    assert cube.slice(store, "guidance", {"text": {"prompt of sql"}})["prompts"] == 1      # the prompt's own text
    assert cube.slice(store, "guidance", {"text": {"zzz"}})["prompts"] == 0
    p = cube.prompts(store, "guidance", {"text": {"step by step"}})
    assert p["prompts"] == 2


def test_the_map_places_globals_by_corpus_share_and_the_trees_are_per_corpus(store):
    a, b, g = two_libraries_and_a_global(store)
    from test_library import fake_encoder
    from fx.ingest.generalize import embed
    embed(store, "guidance", enc=fake_encoder)
    m = cube.feature_map(store, "guidance", {})
    assert [c["name"] for c in m["corpora"]] == ["cypher", "sql"] and m["prompts"] == 2
    assert len(m["globals"]) == 1 and m["globals"][0]["n_corpora"] == 2 and abs(m["globals"][0]["x"]) < 1 and abs(m["globals"][0]["y"]) < 1 and m["globals"][0]["rank"] == 0 and m["groups"] == ["reasoning"]
    assert {f["corpus"] for f in m["local"]} == {"cypher", "sql"} and len(m["local"]) == 4 and all(abs(math.hypot(f["x"], f["y"]) - cube.R * 1.06) < 1 for f in m["local"])
    one = cube.feature_map(store, "guidance", {"corpus": {"sql"}})
    assert one["globals"][0]["n_corpora"] == 1 and abs(math.hypot(one["globals"][0]["x"], one["globals"][0]["y"]) - cube.R) < 1           # one corpus: the rim
    t = cube.corpus_trees(store, "guidance", {})
    assert [x["corpus"] for x in t["trees"]] == ["cypher", "sql"] and t["trees"][1]["features"] == 3 and t["trees"][1]["aligned"] == 1
    sql = t["trees"][1]["groups"][0]
    assert sql["name"] == "output" and {f["name"] for f in sql["features"]} == {"return SQL only", "think step by step", "use table aliases"}
    assert next(f for f in sql["features"] if f["name"] == "think step by step")["global_name"] == "reason stepwise"
    assert [x["corpus"] for x in cube.corpus_trees(store, "guidance", {}, only="sql")["trees"]] == ["sql"]
