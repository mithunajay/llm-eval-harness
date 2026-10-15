import asyncio, time
from helpers import DATASET, SCHEMA, PROMPT
from evalkit import schema
from evalkit.judge import Judge
from evalkit.providers import MockProvider, MockJudgeProvider, ProviderError
from evalkit.runner import run_eval

TEMPLATE = open(PROMPT).read()
DS = schema.load_dataset(DATASET, SCHEMA)


def run(provider, judge=True, **kw):
    j = Judge(MockJudgeProvider()) if judge else None
    return asyncio.run(run_eval(DS, provider, TEMPLATE, j, **kw))


def test_good_model_passes_all_cases():
    r = run(MockProvider("good"))
    assert r["summary"]["passed"] == 10 and r["summary"]["judge_mean"] == 5.0
    assert r["summary"]["failures_by_check"] == {}


def test_regressed_model_fails_the_expected_cases_with_reasons():
    r = run(MockProvider("regressed"))
    failed = {x.case_id for x in r["results"] if not x.passed}
    assert failed == {"c01", "c03", "c04", "c06", "c08", "c10"}
    by = r["summary"]["failures_by_check"]
    assert by["numbers_grounded"] == 4 and by["banned_phrases"] == 4 and by["word_limit"] == 2
    odd = next(x for x in r["results"] if x.case_id == "c01")
    assert any("not in facts: 300" in s for s in odd.failure_reasons)


def test_cases_run_concurrently():
    start = time.perf_counter()
    r = run(MockProvider("good", delay=0.1), judge=False, concurrency=10)
    elapsed = time.perf_counter() - start
    assert r["summary"]["passed"] == 10
    assert elapsed < 0.6, f"expected concurrent execution, took {elapsed:.2f}s (sequential would be ~1s)"


def test_concurrency_limit_is_respected():
    active, peak = 0, 0

    class Probe(MockProvider):
        async def complete(self, prompt, system=""):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            try:
                return await super().complete(prompt, system)
            finally:
                active -= 1

    run(Probe("good"), judge=False, concurrency=3)
    assert peak <= 3


def test_provider_error_is_captured_and_traced_to_the_step():
    class Boom:
        name = "boom"

        async def complete(self, prompt, system=""):
            raise ProviderError("rate limited")

    r = run(Boom(), judge=False)
    assert r["summary"]["errors"] == 10 and r["summary"]["passed"] == 0
    first = r["results"][0]
    assert first.failed_step == "model_call" and "rate limited" in first.error


def test_timeout_is_captured():
    r = run(MockProvider("good", delay=0.3), judge=False, timeout=0.05)
    assert r["summary"]["errors"] == 10
    assert all("TimeoutError" in x.error for x in r["results"])


def test_flaky_judge_makes_cases_fail_visibly():
    j = Judge(MockJudgeProvider(flaky=True))
    r = asyncio.run(run_eval(DS, MockProvider("good"), TEMPLATE, j, concurrency=1))
    assert r["summary"]["judge_invalid"] == 3 and r["summary"]["passed"] == 7
    assert any("judge invalid" in s for x in r["results"] for s in x.failure_reasons)


def test_traces_are_written_per_step():
    import json, os, tempfile
    d = tempfile.mkdtemp()
    run(MockProvider("good"), trace_dir=d, run_id="r1")
    spans = [json.loads(l) for l in open(os.path.join(d, "r1.jsonl"))]
    assert {s["span"] for s in spans} == {"build_prompt", "model_call", "checks", "judge"}
    assert len(spans) == 40 and all(s["trace_id"].startswith("r1:c") for s in spans)
