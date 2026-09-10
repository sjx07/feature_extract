import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_server import FakeServer, reply  # noqa: E402

from fx.corpus import import_path, import_text, corpora  # noqa: E402
from fx.decompose import Tree, metrics, parse_components, profile  # noqa: E402
from fx.decompose import runner as stage  # noqa: E402
from fx.decompose.locate import validate_components  # noqa: E402
from fx.llm import Client  # noqa: E402
from fx.store import Store  # noqa: E402

PROMPT = """You are a query planning optimizer. Your task is to break down complex questions into retrieval steps.
Key Requirements:
- Group queries that can be executed in parallel into the same list
- Order groups based on data dependencies
Example 1:
User: "What was the impact of the crisis on the bank?"
Plan: [["impact of the crisis on the bank"]]
```python
subquery_template = ChatPromptTemplate.from_messages(msgs)
```
For simple questions, return a single query in a single group."""


def comp(start, end, kind="atom", facets=None, material=None):
    d = {"start": start, "end": end, "kind": kind}
    if kind == "atom":
        d["facets"] = facets or [{"verb": "do", "object": "the thing", "polarity": "require", "condition": "always", "domain_terms": []}]
    if material:
        d["material"] = material
    return d


def router(body):
    """A scripted refiner: answers by which SPAN it is asked about."""
    span = body["messages"][-1]["content"].rsplit("# SPAN\n<<<\n", 1)[1].rstrip(">\n")
    if span.startswith("You are a query planning optimizer"):            # the root: one section for the requirements, examples as material, the last sentence an atom
        return reply(json.dumps({"components": [
            comp("Your task is", "into retrieval steps.", facets=[{"verb": "break down", "object": "complex questions", "qualifier": "into retrieval steps", "polarity": "require"}]),
            comp("Key Requirements: -", "on data dependencies", kind="section"),
            comp("Example 1: User:", "the crisis on the bank\"]]", kind="material", material="example"),
            comp("```python subquery_template =", "from_messages(msgs) ```", kind="section"),          # the model is asked again about it, and calls it code
            comp("For simple questions,", "a single group.", facets=[{"verb": "return", "object": "a single query in a single group", "polarity": "require", "condition": "for simple questions"}])]}))
    if span.startswith("Key Requirements"):
        return reply(json.dumps({"components": [
            comp("- Group queries that", "the same list", facets=[{"verb": "group", "object": "queries that can be executed in parallel", "qualifier": "into the same list"}]),
            comp("- Order groups based", "on data dependencies", facets=[{"verb": "order", "object": "groups", "qualifier": "based on data dependencies"}])]}))
    if span.startswith("```python"):
        return reply(json.dumps({"components": [comp("```python subquery_template =", "from_messages(msgs) ```", kind="material", material="code")]}))
    return reply(json.dumps({"components": []}))


def test_parse_kinds_and_tolerance():
    out = parse_components('{"components":[{"start":"a b c","end":"x y z","kind":"material","material":"weird"},{"start":"a","end":"b","atomic":true,"facets":[{"verb":"use","object":"x","polarity":"maybe"}]},{"start":"q","end":"r","kind":"section"}]}')
    assert [c.kind for c in out] == ["material", "atom", "section"]
    assert out[0].material == "other" and "bad_material:weird" in out[0].flags
    assert out[1].facets[0].polarity == "require" and "bad_polarity:maybe" in out[1].flags
    assert parse_components("no json here") is None
    assert parse_components('```json\n{"components": []}\n```') == []


def test_validate_locates_and_names_failures():
    text = "Use the schema. Return JSON only. Use the schema."
    cs = parse_components('{"components":[{"start":"Use the schema.","end":"Use the schema.","kind":"atom","facets":[{"verb":"use","object":"the schema"}]},{"start":"Return JSON only.","end":"Return JSON only.","kind":"atom"},{"start":"nowhere","end":"here","kind":"section"}]}')
    fails = validate_components(cs, text, 0, len(text))
    assert cs[0].span == (0, 15) and any("more than once" in f for f in fails)
    assert cs[1].span == (16, 33) and any("no facets" in f for f in fails)
    assert cs[2].span is None and any("not found" in f for f in fails)


def test_profile_builds_tree_material_and_checks():
    with FakeServer() as srv:
        srv.router = router
        c = Client(base_url=srv.url)
        tree = profile(PROMPT, lambda p: c.complete(p, model="m", cache=False).text)
        atoms = {PROMPT[p.span[0]:p.span[1]] for p, _ in tree.atoms()}
        assert "Your task is to break down complex questions into retrieval steps." in atoms
        assert "- Group queries that can be executed in parallel into the same list" in atoms
        assert "For simple questions, return a single query in a single group." in atoms
        # the fenced block came back as a section, was refined in its own call, and the model called it code
        mats = {(PROMPT[p.span[0]:p.span[1]][:9], p.material) for p, _ in tree.material()}
        assert ("Example 1", "example") in mats and ("```python", "code") in mats
        # the role sentence the model dropped is not fabricated: it stays uncovered, and coverage says so
        assert "You are a query planning optimizer." not in atoms
        m = metrics(tree, PROMPT)
        assert m["n_atoms"] == 4 and m["n_material"] == 2 and m["calls"] >= 3
        assert 0.8 < m["coverage"] < 1.0 and 0 < m["material_share"] < 0.6


def test_gap_policy_skips_bare_slots_but_refines_prose_anywhere():
    from fx.decompose.profile import gap_worth_refining
    text = "```\ncode with several words in it\n```\n{question}\nPlease answer with care and cite the source."
    assert gap_worth_refining(text, 0, 34)
    assert not gap_worth_refining(text, 35, 45)
    assert gap_worth_refining(text, 46, len(text))


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store.db")


def test_import_jsonl_folder_and_text(store, tmp_path):
    j = tmp_path / "prompts.jsonl"
    rows = [{"prompt_id": "a:1", "text": "Solve it.", "domain": "math", "system_id": "s1", "is_prompt": True},
            {"prompt_id": "a:2", "text": "Write SQL.", "domain": "text2sql", "system_id": "s2"},
            {"prompt_id": "a:3", "text": "describes a prompt", "domain": "math", "is_prompt": False}]
    j.write_text("\n".join(json.dumps(r) for r in rows))
    r = import_path(store, j, name="facet", domain="math")
    assert r["added"] == 1 and store.one("SELECT domain, system FROM prompt WHERE id='a:1'")["system"] == "s1"
    assert import_path(store, j, name="facet", domain="math")["added"] == 0              # idempotent
    d = tmp_path / "folder"; d.mkdir(); (d / "x.txt").write_text("Solve it."); (d / "y.md").write_text("Another one.")
    r2 = import_path(store, d, name="files")
    assert r2["added"] == 2 and r2["duplicates_elsewhere"] == 1
    assert import_text(store, "pasted prompt")["added"] == 1
    names = {c["name"]: c["n_prompts"] for c in corpora(store)}
    assert names == {"facet": 1, "files": 2, "scratch": 1}


def test_stage_run_writes_rows_resumes_and_previews(store):
    import_text(store, PROMPT, name="demo")
    pid = store.one("SELECT id FROM prompt")["id"]
    with FakeServer() as srv:
        srv.router = router
        c = Client(store, base_url=srv.url)
        seen = []
        s = stage.run(store, c, "demo", model="m", workers=2, progress=lambda d, t, i: seen.append((d, t, i["coverage"])))
        assert s["done"] == 1 and s["failed"] == 0 and seen[-1][0] == 1 and seen[-1][2] > 0.8
        d = store.one("SELECT * FROM decomp WHERE prompt=?", (pid,))
        kinds = {r["kind"]: r["n"] for r in store.rows("SELECT kind, COUNT(*) n FROM span WHERE prompt=? GROUP BY kind", (pid,))}
        assert d["status"] == "done" and kinds["atom"] == 4 and kinds["material"] == 2
        assert kinds["section"] == 2                                     # requirements, and the fenced block the model refined into a material child
        assert store.one("SELECT COUNT(*) n FROM reading WHERE prompt=?", (pid,))["n"] == 4
        assert store.one("SELECT COUNT(*) n FROM reading r JOIN span s ON s.id=r.span WHERE r.prompt=? AND s.kind='atom'", (pid,))["n"] == 4
        calls_before = srv.calls
        s2 = stage.run(store, c, "demo", model="m", workers=2)                       # resume: nothing to do
        assert s2["total"] == 0 and srv.calls == calls_before
        s3 = stage.run(store, c, "demo", model="m", workers=2, redo=True)            # redo: served from the cache
        assert s3["done"] == 1 and srv.calls == calls_before
    pv = stage.preview(store, "demo", model="m", workers=4, redo=True)
    assert pv["prompts"] == 1 and pv["calls"] >= 3 and pv["dollars"] == 0.0 and pv["basis"] == "FACET v5 defaults"
    assert stage.preview(store, "demo", model="m")["prompts"] == 0


def test_stage_records_failure_and_stop(store):
    for k in range(6):
        import_text(store, f"Prompt number {k}. Do the thing carefully.", name="many")
    with FakeServer() as srv:
        srv.script = [reply("not json at all")]
        c = Client(store, base_url=srv.url, empty_retries=0)
        import fx.llm.client as cl
        cl.time.sleep = lambda s: None
        s = stage.run(store, c, "many", model="m", workers=2, limit=2)
        assert s["done"] == 2                                                # unparsable replies leave an unrefined leaf, not a failure
        d = store.rows("SELECT * FROM decomp")
        assert len(d) == 2 and all(r["coverage"] == 0 for r in d)
        assert store.one("SELECT COUNT(*) n FROM span WHERE kind='unrefined'")["n"] >= 2

        def stop_after_one(done, total, info):
            if done >= 1:
                raise stage.Stop()
        s2 = stage.run(store, c, "many", model="m", workers=1, progress=stop_after_one)
        assert s2["stopped"] and s2["done"] >= 1


def test_gui_api_end_to_end(store, tmp_path):
    from fastapi.testclient import TestClient
    from fx.gui.server import make_app
    from fx.paths import Workspace
    ws = Workspace(tmp_path / "ws")
    app = make_app(ws, store)
    t = TestClient(app)
    r = t.post("/api/import", data={"name": "demo"}, files={"file": ("p.txt", PROMPT.encode())})
    assert r.status_code == 200 and r.json()["added"] == 1
    assert t.post("/api/import", data={"name": "demo", "text": "pasted"}).json()["added"] == 1
    cs = t.get("/api/corpora").json()
    assert cs[0]["name"] == "demo" and cs[0]["n_prompts"] == 2
    pv = t.get("/api/preview?corpus=demo&model=m&workers=2").json()
    assert pv["prompts"] == 2 and "note" in pv
    with FakeServer() as srv:
        srv.router = router
        j = t.post("/api/jobs", json={"corpus": "demo", "model": "m", "workers": 2, "base_url": srv.url}).json()
        import time
        for _ in range(100):
            job = t.get(f"/api/jobs/{j['id']}").json()
            if job["status"] != "running":
                break
            time.sleep(0.1)
        assert job["status"] == "done" and job["done"] == 2 and job["calls"] >= 3
        assert (ws.uploads / "demo" / "p.txt").read_bytes() == PROMPT.encode()                     # the dropped file is kept
        log = t.get(f"/api/jobs/{j['id']}/log").json()
        assert ws.job_log(j["id"]).exists() and any("coverage=" in l for l in log["lines"]) and log["lines"][-1].endswith("done")
    ps = t.get("/api/prompts?corpus=demo&status=done").json()
    assert ps["total"] == 2 and len(ps["prompts"]) == 2
    pid = [p for p in ps["prompts"] if p["n_atoms"] == 4][0]["id"]
    p = t.get(f"/api/prompt/{pid}").json()
    assert p["decomp"]["status"] == "done" and len(p["spans"]) >= 7 and any(a["kind"] == "material" and a["note"] == "example" for a in p["spans"])
    q = t.get("/api/queues?corpus=demo").json()
    assert set(q) == {"low_coverage", "gaps", "unrefined", "failed"}
    assert t.get("/").status_code == 200 and t.get("/api/spend").json()["total"] == 0.0


def test_reask_never_replaces_a_parsed_reply_with_a_refusal():
    from fx.decompose.profile import _better
    from fx.decompose.contract import Component
    good = [Component("a", "b", "atom")]; good[0].span = (0, 3)
    assert not _better(None, ["not json"], good, ["component[1]: not found", "ambiguous"])
    assert _better(good, [], None, ["not json"])
    two = [Component("a", "b", "atom"), Component("c", "d", "atom")]; two[0].span = (0, 1); two[1].span = (2, 3)
    assert _better(two, ["x"], good, [])                                         # more located components wins over fewer failures


def test_locate_matches_recased_quotes():
    from fx.decompose.locate import resolve
    assert resolve("I will ask you a question.", "I Will ask", "a question.") == (0, 26)


def test_schema_is_sent_and_dropped_when_rejected(store):
    from fx.decompose.prompts import COMPONENTS_SCHEMA
    with FakeServer() as srv:
        srv.script = [reply("ok")]
        c = Client(store, base_url=srv.url)
        c.complete("q", model="m", schema=COMPONENTS_SCHEMA)
        assert srv.requests[0]["response_format"]["type"] == "json_schema"
        srv.script = [reply("a", reject="response_format"), reply("b"), reply("c")]; srv.calls = 0
        import fx.llm.client as cl; cl.time.sleep = lambda s: None
        r = c.complete("q2", model="m2", schema=COMPONENTS_SCHEMA)
        assert r.text == "b" and "response_format" not in srv.requests[-1]
        c.complete("q3", model="m2", schema=COMPONENTS_SCHEMA)
        assert "response_format" not in srv.requests[-1]                          # remembered for the model


def test_length_exhausted_reply_is_not_retried_and_falls_back_to_low_effort(store):
    import_text(store, "Answer briefly. Do not guess.", name="one")
    pid = store.one("SELECT id FROM prompt")["id"]
    with FakeServer() as srv:
        srv.script = [reply("", finish="length", completion_tokens=4096),                              # reasoning spent the ceiling
                      reply('{"components":[{"start":"Answer briefly.","end":"Do not guess.","kind":"atom","facets":[{"verb":"answer","object":"briefly","polarity":"require"}]}]}')]
        c = Client(store, base_url=srv.url, empty_retries=3)
        m = stage.decompose_one(store, c, pid, "m", reasoning="on")
        assert m["fallbacks"] == 1 and m["calls"] == 2                                              # one call, no empty retries, then the low-effort fallback
        rows = store.rows("SELECT note, error, finish_reason FROM call ORDER BY id")
        assert rows[0]["error"] == "empty" and rows[0]["finish_reason"] == "length" and rows[1]["note"].endswith("fallback:low")


def test_atom_without_facets_gets_one_call_for_them(store):
    import_text(store, "Answer briefly. Do not guess.", name="one")
    pid = store.one("SELECT id FROM prompt")["id"]
    with FakeServer() as srv:
        srv.script = [reply('{"components":[{"start":"Answer briefly.","end":"Do not guess.","kind":"atom"}]}'),          # root: an atom, no facets
                      reply('{"components":[{"start":"Answer briefly.","end":"Do not guess.","kind":"atom"}]}'),          # re-ask: same
                      reply('{"components":[{"start":"Answer briefly.","end":"Do not guess.","kind":"atom","facets":[{"verb":"answer","object":"briefly","polarity":"require"},{"verb":"guess","object":"","polarity":"forbid"}]}]}')]
        c = Client(store, base_url=srv.url)
        m = stage.decompose_one(store, c, pid, "m")
        assert store.one("SELECT COUNT(*) n FROM reading")["n"] == 2
        assert "facets_asked:0" in json.loads(store.one("SELECT flags FROM decomp")["flags"])


def test_stop_cancels_pending_prompts_and_leaves_them_to_do(store):
    import threading
    for k in range(8):
        import_text(store, f"Prompt number {k}. Do the thing carefully.", name="many")
    with FakeServer() as srv:
        srv.script = [reply('{"components":[{"start":"Prompt number","end":"carefully.","kind":"atom","facets":[{"verb":"do","object":"the thing","polarity":"require"}]}]}')] * 100
        c = Client(store, base_url=srv.url)
        stop = threading.Event()

        def after_two(done, total, info):
            if done >= 2:
                stop.set()
        s = stage.run(store, c, "many", model="m", workers=1, progress=after_two, stop=stop)
        assert s["stopped"] and s["done"] < 8
        assert store.one("SELECT COUNT(*) n FROM decomp")["n"] == s["done"]                    # cancelled prompts wrote nothing
        assert c.complete("x", model="m").error is None                                         # the client reopens after close()
