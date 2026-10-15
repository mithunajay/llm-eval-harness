import json
from evalkit import qa_checks as Q

CASE = {"gold": "yes", "contexts": ["The intervention significantly reduced systolic pressure among participants in twelve weeks."]}
GOOD = json.dumps({"answer": "yes", "rationale": "The intervention significantly reduced systolic pressure among participants."})


def res(output, case=CASE):
    return {c.name: c for c in (f(case, output) for f in Q.QA_CHECKS)}


def test_good_answer_passes_all():
    r = res(GOOD)
    assert all(c.passed for c in r.values()), [(c.name, c.detail) for c in r.values() if not c.passed]


def test_json_wrapped_in_text_is_still_parsed():
    assert res("Sure, here you go: " + GOOD + " Hope that helps.")["correct"].passed


def test_malformed_output_fails_every_dependent_check_without_crashing():
    r = res("I think the answer is yes.")
    assert not r["valid_json"].passed and not r["correct"].passed and not r["label_valid"].passed
    assert not r["rationale_grounded"].passed and not r["rationale_length"].passed


def test_wrong_label_is_reported_with_detail():
    c = res(json.dumps({"answer": "no", "rationale": "The intervention significantly reduced systolic pressure."}))["correct"]
    assert not c.passed and "gold 'yes'" in c.detail


def test_invalid_label_value():
    r = res(json.dumps({"answer": "probably", "rationale": "The intervention significantly reduced systolic pressure."}))
    assert not r["label_valid"].passed and not r["correct"].passed


def test_answer_label_is_case_and_space_insensitive():
    assert res(json.dumps({"answer": " YES ", "rationale": "The intervention significantly reduced systolic pressure."}))["correct"].passed


def test_rationale_too_long():
    long = json.dumps({"answer": "yes", "rationale": "word " * 100})
    assert not res(long)["rationale_length"].passed


def test_personal_advice_is_flagged():
    out = json.dumps({"answer": "yes", "rationale": "The intervention significantly reduced systolic pressure, so you should take it."})
    r = res(out)["no_personal_advice"]
    assert not r.passed and "you should" in r.detail


def test_ungrounded_rationale_is_flagged():
    out = json.dumps({"answer": "yes", "rationale": "Cats generally prefer sleeping inside boxes."})
    assert not res(out)["rationale_grounded"].passed
