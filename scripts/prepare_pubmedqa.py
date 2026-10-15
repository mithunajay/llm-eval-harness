"""Build a frozen, versioned evaluation set from the real PubMedQA labeled data.
  python scripts/prepare_pubmedqa.py --input ori_pqal.json --n 100 --seed 7 --version 1.0.0"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evalkit import schema
from evalkit.pubmedqa_data import PubMedQAFormatError, clean, load_records, sample, write_jsonl

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PubMedQA PQA-L file (ori_pqal.json or a JSON/JSONL export)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--balanced", action="store_true", help="equal yes/no/maybe instead of the natural mix")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "pubmedqa_sample.jsonl"))
    a = ap.parse_args()
    try:
        cases, dropped = clean(load_records(a.input))
        chosen = sample(cases, a.n, a.seed, a.balanced)
    except PubMedQAFormatError as e:
        print(f"FORMAT ERROR: {e}", file=sys.stderr)
        return 2
    write_jsonl(chosen, a.out)
    meta = schema.freeze(a.out, os.path.join(ROOT, "datasets", "schema_pubmedqa.json"), "pubmedqa-pqal-sample", a.version)
    mix = {l: sum(c["gold"] == l for c in chosen) for l in ("yes", "no", "maybe")}
    print(f"read {len(cases) + sum(dropped.values())} records, dropped {dropped}, wrote {len(chosen)} to {a.out}")
    print(f"label mix: {mix}  seed={a.seed}  balanced={a.balanced}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
