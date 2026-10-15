import json, os, tempfile
from helpers import DATASET, SCHEMA, PROMPT
from evalkit.__main__ import main

BASE = ["run", "--dataset", DATASET, "--schema", SCHEMA, "--prompt", PROMPT]


def test_cli_baseline_then_clean_then_regression_exit_codes():
    d = tempfile.mkdtemp()
    base, traces = os.path.join(d, "b.json"), os.path.join(d, "t")
    assert main(BASE + ["--provider", "mock:good", "--save-baseline", base, "--traces", traces]) == 0
    assert json.load(open(base))["summary"]["passed"] == 10
    assert main(BASE + ["--provider", "mock:good", "--baseline", base, "--traces", traces]) == 0
    assert main(BASE + ["--provider", "mock:regressed", "--baseline", base, "--traces", traces]) == 1


def test_cli_returns_2_on_dataset_error():
    d = tempfile.mkdtemp()
    bad = os.path.join(d, "bad.jsonl")
    open(bad, "w").write('{"id": "x"}\n')
    assert main(["run", "--dataset", bad, "--schema", SCHEMA, "--prompt", PROMPT]) == 2


def test_cli_rejects_mock_provider_for_pubmedqa_with_exit_2():
    from helpers import PUBMEDQA_PROMPT, build_fixture_dataset
    d, ds = build_fixture_dataset(12)
    code = main(["run", "--task", "pubmedqa", "--dataset", ds, "--prompt", PUBMEDQA_PROMPT, "--provider", "mock:good"])
    assert code == 2


def test_cli_label_sheet_and_agreement_flow():
    import csv
    d = tempfile.mkdtemp()
    run_path, sheet = os.path.join(d, "run.json"), os.path.join(d, "labels.csv")
    assert main(BASE + ["--provider", "mock:good", "--save-run", run_path, "--traces", os.path.join(d, "t")]) == 0
    assert main(["label-sheet", "--run", run_path, "--out", sheet, "--n", "6"]) == 0
    rows = list(csv.DictReader(open(sheet)))
    assert len(rows) == 6
    assert main(["agreement", "--labels", sheet]) == 2          # nothing labelled yet
    for r in rows:
        r["human_score"] = "5"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    assert main(["agreement", "--labels", sheet]) == 0
