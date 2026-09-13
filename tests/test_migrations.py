"""The schema version: a new store is at the current version; an old store runs the steps once; a store ahead refuses."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fx.core import migrations  # noqa: E402
from fx.core.store import Store  # noqa: E402


def test_new_store_is_current_and_reopening_applies_nothing(tmp_path):
    s = Store(tmp_path / "s.db")
    assert s.version == migrations.CURRENT and s.applied == list(range(1, migrations.CURRENT + 1))
    s2 = Store(tmp_path / "s.db")
    assert s2.applied == [] and s2.version == migrations.CURRENT


def test_an_old_store_runs_the_steps_once(tmp_path):
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.executescript("""CREATE TABLE corpus (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, source TEXT, at TEXT NOT NULL);
        CREATE TABLE call (id INTEGER PRIMARY KEY, at TEXT NOT NULL, stage TEXT, note TEXT, model TEXT NOT NULL, base_url TEXT, prompt_sha TEXT NOT NULL, prompt_chars INTEGER, reply_chars INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, cost REAL NOT NULL DEFAULT 0, latency REAL, finish_reason TEXT, cached INTEGER NOT NULL DEFAULT 0, error TEXT, reply TEXT);
        CREATE TABLE assignment (realization INTEGER NOT NULL, codebook INTEGER NOT NULL, feature INTEGER, confidence TEXT, at TEXT NOT NULL, PRIMARY KEY (realization, codebook));
        INSERT INTO assignment VALUES (7, 1, 3, 'high', 'then');""")
    con.commit(); con.close()
    s = Store(p)
    assert s.applied and s.version == migrations.CURRENT
    assert {r[1] for r in s.rows("PRAGMA table_info(call)")} >= {"provider", "billed"}
    assert s.one("SELECT node FROM membership WHERE kind='realization' AND unit=7")["node"] == 3          # the legacy copy
    assert Store(p).applied == []


def test_a_store_ahead_of_the_code_refuses(tmp_path):
    p = tmp_path / "ahead.db"
    Store(p).con.close()
    con = sqlite3.connect(p); con.execute(f"PRAGMA user_version = {migrations.CURRENT + 50}"); con.commit(); con.close()
    with pytest.raises(migrations.StoreAhead):
        Store(p)


def test_imports_are_events_and_old_prompts_get_a_synthetic_one(tmp_path):
    from fx.data.corpus import import_text, imports
    s = Store(tmp_path / "s.db")
    r = import_text(s, "a pasted prompt", name="c1", domain="d")
    assert r["import"] == 1 and r["added"] == 1
    im = imports(s)[0]
    assert im["kind"] == "paste" and im["added"] == 1 and im["corpus_name"] == "c1" and im["domain"] == "d"
    assert s.one("SELECT import, domain FROM prompt")["import"] == 1 and s.one("SELECT domain FROM prompt")["domain"] == "d"
    # a prompt row from before imports were recorded gets a synthetic import when the step runs again on an old store
    con = sqlite3.connect(tmp_path / "s.db")
    con.execute("INSERT INTO prompt (id, corpus, sha, text, at) VALUES ('old', 1, 'x', 'old text', 'then')"); con.execute("PRAGMA user_version = 1"); con.commit(); con.close()
    s2 = Store(tmp_path / "s.db")
    assert s2.applied == list(range(2, migrations.CURRENT + 1))
    old = s2.one("SELECT i.kind, i.added FROM prompt p JOIN import i ON i.id=p.import WHERE p.id='old'")
    assert old["kind"] == "unrecorded" and old["added"] == 1


def test_tags_are_rows_and_the_columns_a_cache(tmp_path):
    from fx.data import tags
    from fx.data.corpus import add_prompts, get_corpus
    s = Store(tmp_path / "s.db")
    cid = get_corpus(s, "c1")
    add_prompts(s, cid, [{"id": "p1", "text": "one", "domain": "d", "system": "sys", "meta": {"role": "staged", "use_case": {"x": 1}, "pasted": True, "n": 3}}])
    t = tags.tags_of(s, ["p1"])["p1"]
    assert t == {"domain": "d", "system": "sys", "role": "staged", "n": "3"}
    assert s.one("SELECT meta FROM prompt WHERE id='p1'")["meta"] == '{"use_case": {"x": 1}, "pasted": true}'
    tags.set_tags(s, "p1", {"domain": "e", "role": "", "family": "qa"})
    assert tags.tags_of(s, ["p1"])["p1"] == {"domain": "e", "system": "sys", "n": "3", "family": "qa"} and s.one("SELECT domain FROM prompt WHERE id='p1'")["domain"] == "e"
    assert [f["field"] for f in tags.fields(s)] == ["domain", "family", "system", "n"]
    # an old store's columns and meta keys become rows when step 3 runs
    con = sqlite3.connect(tmp_path / "s.db")
    con.execute("INSERT INTO prompt (id, corpus, sha, text, at, task, meta) VALUES ('old', 1, 'x', 'old', 'then', 'qa', '{\"stage\": \"verify\", \"family\": null, \"provenance\": {\"a\": 1}}')")
    con.execute("PRAGMA user_version = 2"); con.commit(); con.close()
    s2 = Store(tmp_path / "s.db")
    assert s2.applied == list(range(3, migrations.CURRENT + 1)) and tags.tags_of(s2, ["old"])["old"] == {"task": "qa", "stage": "verify"} and s2.one("SELECT meta FROM prompt WHERE id='old'")["meta"] == '{"provenance": {"a": 1}}'
    from fx.views import library as cube
    f = cube.prompt_fields(s2)
    assert f["p1"] == {"corpus": "c1", "domain": "e", "system": "sys", "n": "3", "family": "qa"} and cube.field_names(f)[:3] == ["corpus", "domain", "task"]
    assert cube.parse_filters({"family": "qa", "kind": "guidance", "n": "3", "_x": "1"}) == {"family": {"qa"}, "n": {"3"}}


def test_the_seed_is_a_scope_and_an_old_seed_corpus_migrates(tmp_path):
    from test_align import SQL, seed_library
    from fx.ingest.generalize.seed import seed_codebook
    s = Store(tmp_path / "s.db")
    seed_library(s, "sql", SQL)
    cb = seed_codebook(s, "guidance")
    r = s.one("SELECT corpus, scope FROM codebook WHERE id=?", (cb,))
    assert r["corpus"] is None and r["scope"] == "seed" and seed_codebook(s, "guidance") == cb and not s.one("SELECT 1 FROM corpus WHERE name='seed'")
    # an old store: a corpus named seed with the seed codebook on it, and a NOT NULL corpus column
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.executescript("""CREATE TABLE corpus (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, source TEXT, at TEXT NOT NULL);
        CREATE TABLE codebook (id INTEGER PRIMARY KEY, corpus INTEGER NOT NULL REFERENCES corpus(id), kind TEXT NOT NULL, version INTEGER NOT NULL, model TEXT, round INTEGER NOT NULL, notes TEXT, anchor_agreement REAL, at TEXT NOT NULL);
        CREATE UNIQUE INDEX codebook_version ON codebook(corpus, kind, version);
        CREATE TABLE feature (id INTEGER PRIMARY KEY, codebook INTEGER NOT NULL REFERENCES codebook(id), level TEXT NOT NULL, parent INTEGER, prev INTEGER, aspect TEXT, name TEXT NOT NULL, definition TEXT, polarity TEXT, examples TEXT, round INTEGER);
        INSERT INTO corpus VALUES (1, 'sql', 'x', 'then'), (2, 'seed', 'align', 'then');
        INSERT INTO codebook VALUES (1, 1, 'guidance', 1, 'm', 0, '', NULL, 'then'), (2, 2, 'guidance', 1, NULL, 0, 'seed', NULL, 'then');
        INSERT INTO feature VALUES (5, 2, 'feature', NULL, NULL, NULL, 'a global', 'd', 'require', '[]', 1);""")
    con.commit(); con.close()
    s2 = Store(p)
    assert 4 in s2.applied
    rows = {int(r["id"]): (r["corpus"], r["scope"]) for r in s2.rows("SELECT id, corpus, scope FROM codebook")}
    assert rows == {1: (1, "corpus"), 2: (None, "seed")} and not s2.one("SELECT 1 FROM corpus WHERE name='seed'")
    assert s2.one("SELECT codebook FROM feature WHERE id=5")["codebook"] == 2 and seed_codebook(s2, "guidance") == 2


def test_legacy_tables_are_copied_once_more_and_dropped(tmp_path):
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.executescript("""CREATE TABLE assignment (realization INTEGER NOT NULL, codebook INTEGER NOT NULL, feature INTEGER, confidence TEXT, at TEXT NOT NULL, note TEXT, PRIMARY KEY (realization, codebook));
        CREATE TABLE vector (realization INTEGER PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE TABLE fvector (feature INTEGER PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE TABLE alignment (feature INTEGER PRIMARY KEY, global INTEGER, confidence TEXT, note TEXT, at TEXT NOT NULL);
        INSERT INTO assignment VALUES (7, 1, 3, 'high', 'then', 'named');
        INSERT INTO vector VALUES (7, 'm', 2, X'0000');
        INSERT INTO fvector VALUES (3, 'm', 2, X'0000');""")
    con.commit(); con.close()
    s = Store(p)
    assert s.version == migrations.CURRENT
    tables = {r["name"] for r in s.rows("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not tables & {"assignment", "vector", "fvector", "alignment"}
    assert s.one("SELECT node, note FROM membership WHERE kind='realization' AND unit=7")["note"] == "named"
    assert {(r["kind"], r["unit"]) for r in s.rows("SELECT kind, unit FROM embedding")} == {("realization", 7), ("feature", 3)}
    assert Store(tmp_path / "new.db").rows("SELECT name FROM sqlite_master WHERE name='assignment'") == []
