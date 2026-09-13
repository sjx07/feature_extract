"""The schema version: a new store is at the current version; an old store runs the steps once; a store ahead refuses."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fx import migrations  # noqa: E402
from fx.store import Store  # noqa: E402


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
    from fx.corpus import import_text, imports
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
    assert s2.applied == [2]
    old = s2.one("SELECT i.kind, i.added FROM prompt p JOIN import i ON i.id=p.import WHERE p.id='old'")
    assert old["kind"] == "unrecorded" and old["added"] == 1
