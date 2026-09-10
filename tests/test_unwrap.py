import json
from pathlib import Path

from fx.corpus import import_path, import_text
from fx.store import Store
from fx.unwrap import unwrap

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "You are a SQLite expert tasked with writing SQL for a given natural language user query. Your task is to write valid SQLite to answer the user questions for the tables provided."


def test_shapes():
    assert unwrap(f'parser.add_argument("--system_prompt", type=str, default="{PROMPT}", required=False)') == (PROMPT, "python string literal")
    assert unwrap(f'PROMPT = """{PROMPT}\nUse the schema."""') == (PROMPT + "\nUse the schema.", "python string literal")
    assert unwrap(f'prompt = f"""{PROMPT}\n{{question}}"""') == (PROMPT + "\n{question}", "python f-string")
    assert unwrap(f'instruction = (\n    "{PROMPT[:60]}"\n    "{PROMPT[60:]}"\n)') == (PROMPT, "python string literal")
    assert unwrap(f'class Out(BaseModel):\n    """{PROMPT}"""\n    x: int') == (PROMPT, "python string literal")
    assert unwrap(f'let p1 := s!"{PROMPT} {{stmt}}"') == (PROMPT + " {stmt}", "lean s! literal")
    assert unwrap(json.dumps({"system_description": PROMPT, "n": 3})) == (PROMPT, "json field system_description")
    assert unwrap("### Task\\nAnswer with SQL only.\\nNo explanation.\\n") == ("### Task\nAnswer with SQL only.\nNo explanation.", "escaped one-line string")


def test_leaves_real_prompts_alone():
    for t in (PROMPT, "Solve the problem.\n```python\nprint(\"hi\")\n```\nReturn code only.", "x = 1", "# Task\nAnswer briefly."):
        assert unwrap(t) == (t, None)


def test_import_keeps_the_original(tmp_path):
    store = Store(tmp_path / "s.db")
    r = import_text(store, f'sql_prompt = "{PROMPT}"', name="t")
    assert r["added"] == 1 and r["unwrapped"] == 1
    p = store.one("SELECT text, meta FROM prompt")
    assert p["text"] == PROMPT and json.loads(p["meta"])["harvest"]["wrapped"] == "python string literal"
    assert import_text(store, PROMPT, name="t")["skipped"] == 1                                   # the clean copy is the same prompt


def test_corpus_unwrap_count(tmp_path):
    store = Store(tmp_path / "s.db")
    r = import_path(store, ROOT / "data" / "corpora" / "facet" / "text2sql.jsonl", name="text2sql")
    assert 20 <= r["unwrapped"] <= 60
    hows = [json.loads(p["meta"])["harvest"]["wrapped"] for p in store.rows("SELECT meta FROM prompt WHERE meta LIKE '%\"wrapped\":%'")]
    assert "python string literal" in hows and "escaped one-line string" in hows
