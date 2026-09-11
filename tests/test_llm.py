import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_server import FakeServer, reply  # noqa: E402

from fx.llm import BudgetExceeded, Client, cost, price, resolve, run_many, with_fallback  # noqa: E402
from fx.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store.db")


def fast(client):
    """no backoff sleeps in tests"""
    import fx.llm.client as c
    c.time.sleep = lambda s: None
    return client


def test_reply_usage_cost_and_log(store):
    with FakeServer() as srv:
        srv.script = [reply("hello world", 100, 50)]
        c = Client(store, base_url=srv.url)
        r = c.complete("hi", model="deepseek/deepseek-v4-pro", stage="t")
        assert r.ok and r.text == "hello world"
        assert r.usage == {"prompt_tokens": 100, "completion_tokens": 50}
        assert r.cost == pytest.approx(100 * 0.9553 / 1e6 + 50 * 1.9105 / 1e6)
        assert r.finish_reason == "stop" and r.call_id == 1
        row = store.one("SELECT * FROM call")
        assert row["model"] == "deepseek/deepseek-v4-pro" and row["stage"] == "t" and row["cached"] == 0 and row["reply"] == "hello world"
        assert store.spent() == pytest.approx(r.cost)


def test_local_endpoint_is_free_and_estimates_tokens_when_absent(store):
    with FakeServer() as srv:
        srv.script = [{"text": "x" * 380, "usage": None, "status": 200}]
        c = Client(store, base_url=srv.url)
        r = c.complete("p" * 38, model="openai/gpt-oss-20b")
        assert r.ok and r.cost == 0.0
        assert r.usage["estimated"] and r.usage["prompt_tokens"] == 10 and r.usage["completion_tokens"] == 100


def test_cache_hit_costs_nothing_and_is_logged_as_cached(store):
    with FakeServer() as srv:
        srv.script = [reply("first"), reply("second")]
        c = Client(store, base_url=srv.url)
        a = c.complete("same", model="m")
        b = c.complete("same", model="m")
        assert a.text == b.text == "first" and b.cached and b.cost == 0.0
        assert srv.calls == 1
        assert store.rows("SELECT cached FROM call ORDER BY id")[1]["cached"] == 1
        d = c.complete("same", model="m", cache=False)
        assert d.text == "second" and srv.calls == 2
        e = c.complete("same", model="m", temperature=0.7)      # different params, different key
        assert e.text == "second" and srv.calls == 3


def test_empty_reply_is_retried_then_reported(store):
    with FakeServer() as srv:
        srv.script = [reply("", 5, 0), reply("", 5, 0), reply("late answer")]
        c = fast(Client(store, base_url=srv.url, empty_retries=3))
        r = c.complete("q", model="m")
        assert r.ok and r.text == "late answer" and srv.calls == 3
        srv.script = [reply("", 5, 0)]; srv.calls = 0
        r = c.complete("q2", model="m")
        assert not r.ok and r.error == "empty" and srv.calls == 4
        assert store.one("SELECT error FROM call ORDER BY id DESC")["error"] == "empty"


def test_hosted_parameter_shape_is_probed_once(store):
    with FakeServer() as srv:
        srv.script = [reply("a", reject="max_tokens"), reply("b", reject_temperature=True), reply("c"), reply("d")]
        c = fast(Client(store, base_url=srv.url))
        r = c.complete("q", model="gpt-x", max_tokens=100)
        assert r.ok and r.text == "c"
        bodies = srv.requests
        assert "max_tokens" in bodies[0] and "max_completion_tokens" in bodies[1] and "temperature" not in bodies[2]
        r2 = c.complete("q2", model="gpt-x", max_tokens=100)
        assert r2.text == "d" and "max_completion_tokens" in bodies[3] and "temperature" not in bodies[3]


def test_denied_is_not_logged_and_does_not_cost(store):
    with FakeServer() as srv:
        srv.script = [{"text": "", "status": 401, "message": "Incorrect API key provided"}]
        c = fast(Client(store, base_url=srv.url))
        r = c.complete("q", model="m")
        assert not r.ok and r.error.startswith("denied") and srv.calls == 1
        assert store.rows("SELECT * FROM call") == [] and store.spent() == 0.0


def test_unreachable_is_reported_not_raised(store):
    c = fast(Client(store, base_url="http://127.0.0.1:9/v1", max_retries=1, timeout=1.0))
    r = c.complete("q", model="m")
    assert not r.ok and r.error and (r.error.startswith("unreachable") or r.error.startswith("transport"))
    assert store.rows("SELECT * FROM call") == []


def test_budget_stops_before_the_call(store):
    with FakeServer() as srv:
        srv.script = [reply("a", 1_000_000, 0)]        # $2 at gpt-5.6-sol's prompt price; the price table wins over the local URL
        c = Client(store, base_url=srv.url, budget=3.0)
        c.complete("q", model="gpt-5.6-sol")
        assert store.spent() == pytest.approx(2.0)
        c.complete("q2", model="gpt-5.6-sol")
        with pytest.raises(BudgetExceeded):
            c.complete("q3", model="gpt-5.6-sol")
        assert srv.calls == 2
        cached = c.complete("q", model="gpt-5.6-sol")   # a cache hit needs no budget
        assert cached.cached


def test_streaming_collects_chunks_and_usage(store):
    with FakeServer() as srv:
        srv.script = [reply("streamed text here", 7, 4)]
        c = Client(store, base_url=srv.url)
        seen = []
        r = c.complete("q", model="m", stream=True, on_chunk=seen.append)
        assert r.ok and r.text == "streamed text here" and "".join(seen) == r.text
        assert r.usage == {"prompt_tokens": 7, "completion_tokens": 4} and r.finish_reason == "stop"
        assert srv.requests[0]["stream"] is True


def test_system_and_message_lists(store):
    with FakeServer() as srv:
        c = Client(store, base_url=srv.url)
        c.complete([{"role": "user", "content": "u"}], model="m", system="be terse")
        msgs = srv.requests[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "be terse"} and msgs[1]["content"] == "u"


def test_with_fallback_moves_on_from_empty(store):
    with FakeServer() as srv:
        srv.script = [reply("", 1, 0), reply("from fallback")]
        c = fast(Client(store, base_url=srv.url, empty_retries=0))
        call = with_fallback(c, {"model": "a"}, {"model": "b", "extra_body": {"reasoning": {"enabled": False}}})
        r = call("q")
        assert r.ok and r.model == "b" and srv.requests[1]["reasoning"] == {"enabled": False}


def test_run_many_keeps_order_and_reports_progress(store):
    with FakeServer() as srv:
        srv.script = [reply(f"r{i}") for i in range(1)]
        srv.script = [reply("r")]
        c = Client(store, base_url=srv.url)
        seen = []
        out = run_many(c, [f"job {i}" for i in range(20)], model="m", workers=5, progress=lambda d, t: seen.append((d, t)))
        assert len(out) == 20 and all(r.ok for r in out) and seen[-1] == (20, 20)
        assert srv.calls == 20
        # a second run is served from the cache
        out2 = run_many(c, [f"job {i}" for i in range(20)], model="m", workers=5)
        assert all(r.cached for r in out2) and srv.calls == 20


def test_run_many_stops_on_denial(store):
    with FakeServer() as srv:
        srv.script = [{"text": "", "status": 401}]
        c = fast(Client(store, base_url=srv.url))
        out = run_many(c, [f"job {i}" for i in range(40)], model="m", workers=4)
        assert any(r is not None and r.error.startswith("denied") for r in out)
        assert srv.calls < 40


def test_registry_resolution_and_prices():
    assert resolve("gpt-5.6-sol").name == "openai"
    assert resolve("deepseek/deepseek-v4-pro").name == "openrouter"
    assert resolve("openai/gpt-oss-20b").name == "local" and resolve("Qwen/Qwen2.5-7B-Instruct").local
    assert price("deepseek/deepseek-v4-flash") == (0.0886, 0.1772)
    assert price("nobody/knows", resolve("nobody/knows")) == (0.0, 0.0)          # local by default: free
    assert price("gpt-unknown", resolve("gpt-unknown")) == (2.0, 10.0)         # hosted unknown: counted at the top price
    assert cost("gpt-5.6-sol", 1_000_000, 100_000) == pytest.approx(2.0 + 1.0)


def test_custom_models_file(tmp_path, monkeypatch):
    p = tmp_path / "models.json"
    p.write_text('{"my/model": {"name": "lab", "base_url": "http://lab:8000/v1", "key_env": "LAB_KEY", "local": true, "price": [0.1, 0.2]}}')
    monkeypatch.setenv("FX_MODELS", str(p))
    ep = resolve("my/model")
    assert ep.name == "lab" and ep.base_url == "http://lab:8000/v1" and price("my/model", ep) == (0.1, 0.2)


def test_cli_spend_and_models(store, capsys, tmp_path):
    from fx.cli import main
    with FakeServer() as srv:
        srv.script = [reply("ready.", 3, 2)]
        assert main(["--workspace", str(tmp_path / "ws"), "llm", "probe", "--model", "m", "--base-url", srv.url]) == 0
    main(["--workspace", str(tmp_path / "ws"), "llm", "spend"])
    out = capsys.readouterr().out
    assert "total $" in out and "m " in out
    assert main(["llm", "models"]) == 0


def test_openrouter_routing_excludes_looping_upstreams(monkeypatch):
    from fx.llm.registry import provider_body
    monkeypatch.delenv("FX_PROVIDER_IGNORE", raising=False); monkeypatch.delenv("FX_PROVIDER_SORT", raising=False); monkeypatch.delenv("FX_PROVIDER", raising=False)
    assert provider_body("deepseek/deepseek-v4-flash-0731") == {"provider": {"only": ["Wafer"], "allow_fallbacks": False}}       # the default: one upstream
    monkeypatch.setenv("FX_PROVIDER", "")
    p = provider_body("deepseek/deepseek-v4-flash-0731")["provider"]
    assert p["sort"] == "throughput" and p["allow_fallbacks"] and "Reka" in p["ignore"] and p["order"] == ["Wafer", "DeepInfra"]
    assert provider_body("gpt-5.6-luna") == {} and provider_body("openai/gpt-oss-20b") == {}
    monkeypatch.setenv("FX_PROVIDER", "Baidu")
    assert provider_body("deepseek/deepseek-v4-flash-0731") == {"provider": {"only": ["Baidu"], "allow_fallbacks": False}}


def test_ledger_keeps_the_upstream_and_the_billed_cost(store):
    with FakeServer() as srv:
        srv.script = [reply("hi", provider="Wafer", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0042})]
        c = Client(store, base_url=srv.url)
        r = c.complete("x", model="deepseek/deepseek-v4-flash-0731", stage="t")
        assert r.provider == "Wafer" and r.billed == 0.0042
        row = store.one("SELECT provider, billed, cost FROM call")
        assert row["provider"] == "Wafer" and row["billed"] == 0.0042 and abs(store.spent() - 0.0042) < 1e-9        # spend is what was billed, not the list price
