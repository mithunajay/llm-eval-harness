import asyncio
from helpers import DATASET, SCHEMA, PROMPT, temp_copy_of_dataset
from evalkit import schema, regression
from evalkit.judge import Judge
from evalkit.providers import MockProvider, MockJudgeProvider
from evalkit.runner import run_eval

TEMPLATE = open(PROMPT).read()


def run(mode, ds=None):
    ds = ds or schema.load_dataset(DATASET, SCHEMA)
    return asyncio.run(run_eval(ds, MockProvider(mode), TEMPLATE, Judge(MockJudgeProvider())))


def test_same_model_twice_is_not_a_regression():
    base = regression.to_baseline(run("good"))
    rep = regression.compare(run("good"), base)
    assert not rep.regressed and rep.newly_failing == []


def test_bad_change_is_flagged_with_the_exact_cases():
    base = regression.to_baseline(run("good"))
    rep = regression.compare(run("regressed"), base)
    assert rep.regressed
    assert rep.newly_failing == ["c01", "c03", "c04", "c06", "c08", "c10"]
    assert rep.pass_rate_before == 1.0 and rep.pass_rate_after == 0.4
    assert "REGRESSION" in rep.render()


def test_improvement_is_reported_as_fixed_not_regression():
    base = regression.to_baseline(run("regressed"))
    rep = regression.compare(run("good"), base)
    assert not rep.regressed and len(rep.fixed) == 6


def test_comparing_across_different_datasets_is_refused():
    base = regression.to_baseline(run("good"))
    d, ds_path, sc = temp_copy_of_dataset()
    lines = open(ds_path).read().splitlines()
    open(ds_path, "w").write("\n".join(lines[:-1]) + "\n")
    schema.freeze(ds_path, sc, "openers", "1.0.1")
    other = run("good", schema.load_dataset(ds_path, sc))
    try:
        regression.compare(other, base)
    except regression.BaselineError as e:
        assert "different dataset" in str(e)
    else:
        raise AssertionError("expected BaselineError")


def test_judge_mean_drop_alone_can_fail_the_gate():
    base = regression.to_baseline(run("good"))
    cur = run("good")
    cur["summary"]["judge_mean"] = 4.0  # simulated quality drop with the same pass/fail results
    rep = regression.compare(cur, base)
    assert rep.regressed and any("judge mean" in r for r in rep.reasons)
