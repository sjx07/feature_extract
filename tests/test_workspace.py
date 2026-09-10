import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_server import FakeServer, reply  # noqa: E402

from fx.corpus import import_path  # noqa: E402
from fx.paths import Workspace  # noqa: E402
from fx.store import Store  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_workspace_layout(tmp_path):
    ws = Workspace(tmp_path / "w")
    assert ws.store_path == tmp_path / "w" / "store.db"
    assert ws.logs.is_dir() and ws.uploads.is_dir() and ws.exports.is_dir()
    assert ws.job_log(3).name == "job-3.log" and ws.upload_dir("a b/c").name == "a_b_c"


def test_facet_corpus_imports(tmp_path):
    store = Store(tmp_path / "s.db")
    r = import_path(store, ROOT / "data" / "corpora" / "facet_prompts.jsonl", name="facet", domain="text2cypher")
    assert r["added"] == 145
    p = store.one("SELECT * FROM prompt")
    assert json.loads(p["meta"])["provenance"] and p["system"] and p["domain"] == "text2cypher"
    assert {x["domain"] for x in store.rows("SELECT DISTINCT domain FROM prompt")} == {"text2cypher"}
    r2 = import_path(store, ROOT / "data" / "corpora" / "plain", name="plain")
    assert r2["added"] == 2 and r2["duplicates_elsewhere"] == 2        # both are text2cypher prompts of the corpus


def test_cli_decompose_is_a_job_with_a_log(tmp_path):
    from fx.cli import main
    ws = str(tmp_path / "ws")
    assert main(["-w", ws, "import", str(ROOT / "data" / "corpora" / "plain"), "--name", "plain"]) == 0
    assert (Path(ws) / "uploads").is_dir()
    with FakeServer() as srv:
        srv.script = [reply('{"components":[]}')]
        assert main(["-w", ws, "decompose", "--corpus", "plain", "--model", "m", "--base-url", srv.url, "--workers", "2", "--limit", "1"]) == 0
    store = Store(Path(ws) / "store.db")
    job = store.one("SELECT * FROM job")
    assert job["kind"] == "decompose" and job["status"] == "done" and job["done"] == 1 and json.loads(job["params"])["from"] == "cli"
    log = (Path(ws) / "logs" / "job-1.log").read_text()
    assert "job 1 decompose" in log and "coverage=" in log and log.strip().endswith("done")
