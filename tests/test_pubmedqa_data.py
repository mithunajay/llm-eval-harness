import json, os
from helpers import FIXTURE, build_fixture_dataset, raises, PUBMEDQA_SCHEMA
from evalkit import schema
from evalkit.pubmedqa_data import PubMedQAFormatError, clean, load_records, normalize, sample


def test_original_layout_is_normalized():
    cases = load_records(FIXTURE)
    assert len(cases) == 12 and cases[0]["id"].startswith("FIXTURE-")
    c = next(c for c in cases if c["id"] == "FIXTURE-001")
    assert c["gold"] == "yes" and len(c["contexts"]) == 2 and c["question"].endswith("?")


def test_huggingface_layout_is_normalized():
    rec = {"pubid": 123, "question": "Q?", "context": {"contexts": ["a", "b"], "labels": [], "meshes": []},
           "long_answer": "x", "final_decision": "Maybe"}
    c = normalize(rec)
    assert c == {"id": "123", "question": "Q?", "contexts": ["a", "b"], "gold": "maybe", "tags": ["pubmedqa", "pqa-l"]}


def test_jsonl_export_is_loaded():
    d = os.path.join(os.path.dirname(FIXTURE))
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "rows.jsonl")
    open(p, "w").write(json.dumps({"pubid": 1, "question": "Q?", "context": {"contexts": ["a"]}, "final_decision": "yes"}) + "\n")
    assert load_records(p)[0]["id"] == "1"


def test_unknown_layout_fails_clearly():
    e = raises(PubMedQAFormatError, normalize, {"foo": 1}, "x")
    assert "unrecognised" in str(e)
    e = raises(PubMedQAFormatError, normalize, {"QUESTION": "q", "CONTEXTS": "not a list", "final_decision": "yes"}, "x")
    assert "contexts" in str(e)


def test_clean_drops_unevaluable_records():
    cases = [{"id": "1", "question": "q", "contexts": ["c"], "gold": "yes"},
             {"id": "2", "question": "q", "contexts": ["c"], "gold": "unknown"},
             {"id": "3", "question": "q", "contexts": [], "gold": "no"}]
    kept, dropped = clean(cases)
    assert [c["id"] for c in kept] == ["1"] and dropped["bad_label"] == 1 and dropped["no_context"] == 1


def test_sample_is_deterministic_and_order_independent():
    cases = load_records(FIXTURE)
    a = sample(cases, 6, seed=7)
    b = sample(list(reversed(cases)), 6, seed=7)
    assert [c["id"] for c in a] == [c["id"] for c in b] and len(a) == 6
    assert [c["id"] for c in a] != [c["id"] for c in sample(cases, 6, seed=8)]


def test_sample_keeps_the_label_mix_or_balances_it():
    cases = load_records(FIXTURE)  # 6 yes, 4 no, 2 maybe
    prop = sample(cases, 6, seed=1)
    assert {l: sum(c["gold"] == l for c in prop) for l in ("yes", "no", "maybe")} == {"yes": 3, "no": 2, "maybe": 1}
    bal = sample(cases, 6, seed=1, balanced=True)
    assert {l: sum(c["gold"] == l for c in bal) for l in ("yes", "no", "maybe")} == {"yes": 2, "no": 2, "maybe": 2}


def test_sample_refuses_when_not_enough_cases():
    e = raises(PubMedQAFormatError, sample, load_records(FIXTURE), 12, 1, True)  # needs 4 maybe, only 2
    assert "maybe" in str(e)


def test_prepare_script_writes_a_frozen_valid_dataset():
    d, out = build_fixture_dataset(n=12)
    ds = schema.load_dataset(out, PUBMEDQA_SCHEMA)
    assert len(ds.cases) == 12 and ds.version == "1.0.0" and ds.name == "pubmedqa-pqal-sample"


def test_schema_rejects_a_bad_gold_label():
    case = {"id": "x", "question": "q", "contexts": ["c"], "gold": "probably"}
    errors = schema.validate_case(case, json.load(open(PUBMEDQA_SCHEMA)))
    assert any("gold: must be one of" in e for e in errors)
