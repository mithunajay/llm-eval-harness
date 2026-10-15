import asyncio, json
from helpers import build_fixture_dataset, PUBMEDQA_PROMPT, PUBMEDQA_SCHEMA
from evalkit import regression, schema, tasks
from evalkit.judge import Judge
from evalkit.providers import ScriptedProvider
from evalkit.runner import run_eval

TEMPLATE = open(PUBMEDQA_PROMPT).read()
D, PATH = build_fixture_dataset(12)
DS = schema.load_dataset(PATH, PUBMEDQA_SCHEMA)
BY_Q = {c["question"]: c for c in DS.cases}


def answer_for(prompt, tweaks=None):
    question = next(l for l in prompt.splitlines() if l.startswith("QUESTION: "))[len("QUESTION: "):]
    case = BY_Q[question]
    answer = json.dumps({"answer": case["gold"], "rationale": case["contexts"][0]})
    return (tweaks or {}).get(case["id"], lambda c, a: a)(case, answer)


def go(tweaks=None, judge=None):
    provider = ScriptedProvider(lambda p: answer_for(p, tweaks))
    return asyncio.run(run_eval(DS, provider, TEMPLATE, judge, task=tasks.PUBMEDQA))


def wrong_label(case, _):
    other = "no" if case["gold"] != "no" else "yes"
    return json.dumps({"answer": other, "rationale": case["contexts"][0]})


def test_prompt_contains_question_and_abstract():
    case = DS.cases[0]
    prompt = tasks.PUBMEDQA.render(case, TEMPLATE)
    assert f"QUESTION: {case['question']}" in prompt and case["contexts"][0] in prompt and "{context}" not in prompt


def test_perfect_model_passes_everything():
    r = go()
    assert r["summary"]["passed"] == 12 and r["summary"]["check_pass_rates"]["correct"] == 1.0
    assert r["meta"]["task"] == "pubmedqa"


def test_mixed_failures_are_counted_per_check():
    ids = [c["id"] for c in DS.cases]
    tweaks = {
        ids[0]: wrong_label,
        ids[1]: lambda c, a: "I think the answer is yes.",
        ids[2]: lambda c, a: json.dumps({"answer": c["gold"], "rationale": c["contexts"][0] + " You should take it."}),
        ids[3]: lambda c, a: json.dumps({"answer": c["gold"], "rationale": "Cats generally prefer sleeping inside boxes."}),
    }
    s = go(tweaks)["summary"]
    assert s["passed"] == 8
    assert s["failures_by_check"]["correct"] == 2          # wrong label + unparseable
    assert s["failures_by_check"]["valid_json"] == 1
    assert s["failures_by_check"]["no_personal_advice"] == 1
    assert s["failures_by_check"]["rationale_grounded"] == 2  # unparseable + ungrounded
    assert s["check_pass_rates"]["correct"] == round(1 - 2 / 12, 4)


def test_regression_gate_uses_explicit_tolerances_for_noisy_models():
    base = regression.to_baseline(go())
    first = DS.cases[0]["id"]
    noisy = go({first: wrong_label})
    strict = regression.compare(noisy, base)
    assert strict.regressed and strict.newly_failing == [first]
    lenient = regression.compare(noisy, base, max_new_failures=1, pass_rate_tolerance=0.1)
    assert not lenient.regressed
    big = go({c["id"]: wrong_label for c in DS.cases[:6]})
    assert regression.compare(big, base, max_new_failures=1, pass_rate_tolerance=0.1).regressed


def test_judge_prompt_for_qa_includes_abstract_answer_and_rationale():
    seen = []

    def fake_judge(prompt):
        seen.append(prompt)
        return json.dumps({"score": 5, "reason": "faithful"})

    r = go(judge=Judge(ScriptedProvider(fake_judge), task=tasks.PUBMEDQA))
    assert r["summary"]["judge_mean"] == 5.0 and r["summary"]["passed"] == 12
    p = seen[0]
    assert "ABSTRACT:" in p and "ANSWER:" in p and "RATIONALE:" in p and "{" not in p.split("RATIONALE:")[0].replace('{"score"', "")


def test_unparseable_output_still_reaches_the_judge_as_rationale_text():
    seen = []
    tweaks = {DS.cases[0]["id"]: lambda c, a: "totally not json"}
    go(tweaks, judge=Judge(ScriptedProvider(lambda p: (seen.append(p), json.dumps({"score": 1, "reason": "x"}))[1]), task=tasks.PUBMEDQA))
    assert any("(unparseable)" in p and "totally not json" in p for p in seen)


def test_newly_failing_rule_works_even_when_the_pass_rate_is_unchanged():
    a, b = DS.cases[0]["id"], DS.cases[1]["id"]
    base = regression.to_baseline(go({a: wrong_label}))      # a fails in the baseline
    cur = go({b: wrong_label})                                # b fails now; a is fixed; pass rate identical
    rep = regression.compare(cur, base)
    assert rep.newly_failing == [b] and rep.fixed == [a]
    assert rep.pass_rate_before == rep.pass_rate_after
    assert rep.regressed and any("passed before now fail" in r for r in rep.reasons)
    assert not regression.compare(cur, base, max_new_failures=1).regressed


def test_pass_rate_rule_works_independently_of_the_flip_count():
    base = regression.to_baseline(go())
    noisy = go({DS.cases[0]["id"]: wrong_label})              # exactly one flip
    only_rate = regression.compare(noisy, base, max_new_failures=1)  # flip allowed, tolerance 0
    assert only_rate.regressed and only_rate.reasons == ["pass rate dropped"]
    assert not regression.compare(noisy, base, max_new_failures=1, pass_rate_tolerance=0.1).regressed
