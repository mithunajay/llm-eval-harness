import csv, tempfile, os
from helpers import raises
from evalkit import calibration as cal


def test_kappa_perfect_chance_and_degenerate():
    assert cal.cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert abs(cal.cohens_kappa([True, True, False, False], [True, False, True, False])) < 1e-9  # chance level
    assert cal.cohens_kappa([True, True], [True, True]) == 1.0                                  # degenerate, identical
    assert cal.cohens_kappa([True, True], [False, False]) == 0.0


def rows(pairs):
    return [{"case_id": f"c{i}", "output": "x", "judge_score": j, "human_score": h} for i, (j, h) in enumerate(pairs)]


def test_agreement_numbers():
    rep = cal.agreement(rows([(5, 5), (4, 5), (2, 2), (5, 3), (1, 4)]))
    assert rep["n"] == 5 and rep["exact_agreement"] == 0.4 and rep["within_1"] == 0.6
    assert rep["judge_minus_human_mean"] == round((0 - 1 + 0 + 2 - 3) / 5, 3)
    assert -1 <= rep["kappa_pass_at_4"] <= 1


def test_blank_human_scores_are_skipped_and_none_is_an_error():
    rep = cal.agreement(rows([(5, 5), (4, "")]))
    assert rep["n"] == 1
    e = raises(ValueError, cal.agreement, rows([(5, "")]))
    assert "human_score" in str(e)
    e = raises(ValueError, cal.agreement, rows([(5, 9)]))
    assert "between 1 and 5" in str(e)


def fake_run():
    res = [{"case_id": f"c{i}", "output": f"out {i}\nline", "judge": {"score": 4, "valid": True}} for i in range(10)]
    res.append({"case_id": "bad", "output": "x", "judge": {"score": None, "valid": False}})
    res.append({"case_id": "nojudge", "output": "x", "judge": None})
    return {"results": res}


def test_label_sheet_samples_only_valid_judged_outputs_and_is_seeded():
    a = cal.make_label_sheet(fake_run(), 5, seed=1)
    b = cal.make_label_sheet(fake_run(), 5, seed=1)
    assert a == b and len(a) == 5
    assert all(r["case_id"].startswith("c") and r["human_score"] == "" and "\n" not in r["output"] for r in a)
    assert {r["case_id"] for r in a} != {r["case_id"] for r in cal.make_label_sheet(fake_run(), 5, seed=2)}


def test_sheet_round_trips_through_csv():
    p = os.path.join(tempfile.mkdtemp(), "labels.csv")
    cal.write_sheet(cal.make_label_sheet(fake_run(), 3), p)
    loaded = list(csv.DictReader(open(p)))
    assert list(loaded[0].keys()) == cal.SHEET_FIELDS and len(loaded) == 3
