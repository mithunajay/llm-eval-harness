from evalkit.tracing import Tracer


def test_span_records_latency_tokens_and_cost():
    t = Tracer("t1", rates=(3.0, 15.0))
    with t.span("model_call", provider="x") as s:
        s.set_tokens(1000, 200)
        s.set_output("hello")
    sp = t.spans[0]
    assert sp["span"] == "model_call" and sp["latency_ms"] >= 0 and sp["error"] is None
    assert sp["cost_usd"] == round((1000 * 3.0 + 200 * 15.0) / 1e6, 6)
    assert t.total_cost == sp["cost_usd"]


def test_error_is_recorded_and_reraised_and_step_is_found():
    t = Tracer("t2")
    with t.span("ok"):
        pass
    try:
        with t.span("boom"):
            raise ValueError("bad input")
    except ValueError:
        pass
    else:
        raise AssertionError("exception should propagate")
    assert t.failed_step() == "boom"
    assert "ValueError: bad input" in t.spans[1]["error"]


def test_long_values_are_truncated():
    t = Tracer("t3")
    with t.span("x", blob="a" * 5000):
        pass
    assert len(t.spans[0]["input"]) < 400


def test_dump_writes_one_json_line_per_span():
    import json, os, tempfile
    t = Tracer("t4")
    with t.span("a"):
        pass
    with t.span("b"):
        pass
    p = os.path.join(tempfile.mkdtemp(), "trace.jsonl")
    t.dump(p)
    lines = [json.loads(x) for x in open(p)]
    assert [l["span"] for l in lines] == ["a", "b"] and all(l["trace_id"] == "t4" for l in lines)
