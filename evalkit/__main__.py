"""CLI.
  python -m evalkit freeze --dataset D --schema S --name N --version V
  python -m evalkit run --task openers|pubmedqa --dataset D --prompt P --provider mock:good|anthropic [--baseline B] [--save-baseline B]
  python -m evalkit label-sheet --run runs/x.json --out labels.csv     # export outputs for you to score by hand
  python -m evalkit agreement --run runs/x.json --labels labels.csv    # judge vs your labels
Exit codes: 0 ok, 1 regression or failing cases, 2 dataset/config error."""
import argparse, asyncio, csv, json, os, sys

from evalkit import calibration, regression, schema, tasks
from evalkit.judge import Judge
from evalkit.providers import AnthropicProvider, MockJudgeProvider, make_provider
from evalkit.runner import run_eval, run_to_dict


def _run(a):
    task = tasks.get_task(a.task)
    schema_path = a.schema or task.schema_path
    judge_choice = a.judge or ("mock" if task.name == "openers" else "none")
    if task.name != "openers" and (a.provider.startswith("mock:") or judge_choice == "mock"):
        print(f"CONFIG ERROR: mock providers only support the openers task; use --provider anthropic "
              f"(and --judge anthropic or none) for {task.name}", file=sys.stderr)
        return 2
    try:
        ds = schema.load_dataset(a.dataset, schema_path)
    except schema.DatasetError as e:
        print(f"DATASET ERROR: {e}", file=sys.stderr)
        return 2
    template = open(a.prompt).read()
    judge = None if judge_choice == "none" else Judge(MockJudgeProvider() if judge_choice == "mock" else AnthropicProvider(), task=task)
    run = asyncio.run(run_eval(ds, make_provider(a.provider), template, judge, a.concurrency, a.traces, task=task))
    s = run["summary"]
    print(f"run {run['run_id']}  task {task.name}  dataset {ds.name}@{ds.version}  prompt {run['meta']['prompt_fingerprint']}  {run['meta']['provider']}")
    print(f"fully passing {s['passed']}/{s['cases']}  judge_mean {s['judge_mean']}  errors {s['errors']}  avg_latency_ms {s['avg_latency_ms']}  est_cost_usd {s['est_cost_usd']}")
    print("per-check pass rate: " + ", ".join(f"{k} {v:.0%}" for k, v in s["check_pass_rates"].items()))
    for r in run["results"]:
        if not r.passed:
            print(f"  FAIL {r.case_id}: " + " | ".join(r.failure_reasons) + f"  (trace {r.trace_id})")
    if a.save_run:
        os.makedirs(os.path.dirname(a.save_run) or ".", exist_ok=True)
        json.dump(run_to_dict(run), open(a.save_run, "w"), indent=2)
        print(f"run saved to {a.save_run}")
    if a.save_baseline:
        regression.save_baseline(run, a.save_baseline)
        print(f"baseline saved to {a.save_baseline}")
    if a.baseline:
        try:
            rep = regression.compare(run, json.load(open(a.baseline)), max_new_failures=a.max_new_failures,
                                     pass_rate_tolerance=a.pass_rate_tolerance)
        except regression.BaselineError as e:
            print(f"BASELINE ERROR: {e}", file=sys.stderr)
            return 2
        print(rep.render())
        return 1 if rep.regressed else 0
    return 0 if s["passed"] == s["cases"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evalkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--dataset", required=True); fr.add_argument("--schema", required=True)
    fr.add_argument("--name", required=True); fr.add_argument("--version", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--task", default="openers", choices=list(tasks.TASKS))
    rn.add_argument("--dataset", required=True); rn.add_argument("--schema"); rn.add_argument("--prompt", required=True)
    rn.add_argument("--provider", default="mock:good")
    rn.add_argument("--judge", choices=["mock", "anthropic", "none"])
    rn.add_argument("--baseline"); rn.add_argument("--save-baseline"); rn.add_argument("--save-run")
    rn.add_argument("--max-new-failures", type=int, default=0, help="cases allowed to flip pass->fail")
    rn.add_argument("--pass-rate-tolerance", type=float, default=0.0, help="allowed pass-rate drop, e.g. 0.03")
    rn.add_argument("--traces", default="traces"); rn.add_argument("--concurrency", type=int, default=4)
    ls = sub.add_parser("label-sheet")
    ls.add_argument("--run", required=True); ls.add_argument("--out", required=True)
    ls.add_argument("--n", type=int, default=30); ls.add_argument("--seed", type=int, default=1)
    ag = sub.add_parser("agreement")
    ag.add_argument("--run"); ag.add_argument("--labels", required=True); ag.add_argument("--threshold", type=int, default=4)
    a = ap.parse_args(argv)

    if a.cmd == "freeze":
        try:
            print(json.dumps(schema.freeze(a.dataset, a.schema, a.name, a.version), indent=2))
            return 0
        except schema.DatasetError as e:
            print(f"DATASET ERROR: {e}", file=sys.stderr)
            return 2
    if a.cmd == "label-sheet":
        rows = calibration.make_label_sheet(json.load(open(a.run)), a.n, a.seed)
        calibration.write_sheet(rows, a.out)
        print(f"wrote {len(rows)} outputs to {a.out}; fill the human_score column (1-5) by hand, then run `agreement`")
        return 0
    if a.cmd == "agreement":
        try:
            rep = calibration.agreement(list(csv.DictReader(open(a.labels))), a.threshold)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(json.dumps(rep, indent=2))
        return 0
    return _run(a)


if __name__ == "__main__":
    sys.exit(main())
