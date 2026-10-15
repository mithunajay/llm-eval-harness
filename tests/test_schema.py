import json, os
from helpers import DATASET, SCHEMA, temp_copy_of_dataset
from evalkit import schema


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def test_frozen_dataset_loads():
    ds = schema.load_dataset(DATASET, SCHEMA)
    assert len(ds.cases) == 10 and ds.version == "1.0.0" and len(ds.sha256) == 64


def test_missing_and_unexpected_fields_are_reported():
    case = {"id": "x", "persona": "p", "max_words": 35, "must_mention": [],
            "facts": {"org": "o", "top_repo": "r", "active_repos": 1}, "surprise": 1}
    errors = schema.validate_case(case, json.load(open(SCHEMA)))
    assert "facts.total_stars: missing" in errors
    assert "surprise: unexpected field" in errors


def test_wrong_type_is_reported():
    case = {"id": "x", "persona": "p", "max_words": "35", "must_mention": [],
            "facts": {"org": "o", "top_repo": "r", "active_repos": True, "total_stars": 1}}
    errors = schema.validate_case(case, json.load(open(SCHEMA)))
    assert "max_words: expected integer" in errors
    assert "facts.active_repos: expected integer" in errors  # bool is not an int here


def test_duplicate_ids_fail_loading():
    d, ds, sc = temp_copy_of_dataset()
    lines = open(ds).read().splitlines()
    open(ds, "w").write("\n".join(lines + [lines[0]]) + "\n")
    e = raises(schema.DatasetError, schema.load_dataset, ds, sc, False)
    assert "duplicate id" in str(e)


def test_silent_dataset_change_is_caught_by_hash():
    d, ds, sc = temp_copy_of_dataset()
    schema.freeze(ds, sc, "openers", "1.0.0")
    assert len(schema.load_dataset(ds, sc).cases) == 10
    lines = open(ds).read().splitlines()
    open(ds, "w").write("\n".join(lines[:-1]) + "\n")  # drop a case without re-freezing
    e = raises(schema.DatasetError, schema.load_dataset, ds, sc)
    assert "hash mismatch" in str(e)


def test_unfrozen_dataset_is_rejected():
    d, ds, sc = temp_copy_of_dataset()
    e = raises(schema.DatasetError, schema.load_dataset, ds, sc)
    assert "freeze" in str(e)
