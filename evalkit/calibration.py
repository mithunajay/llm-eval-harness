"""Check an LLM judge against human labels. Export a sample for you to score by hand, then
compare. This is how you defend a judge: report how often it agrees with a person."""
import csv, random

SHEET_FIELDS = ["case_id", "output", "judge_score", "human_score"]


def make_label_sheet(run, n, seed=1):
    rows = [r for r in run["results"] if r.get("judge") and r["judge"]["valid"] and r.get("output")]
    rnd = random.Random(seed)
    rows.sort(key=lambda r: r["case_id"])
    rnd.shuffle(rows)
    return [{"case_id": r["case_id"], "output": r["output"].replace("\n", " "),
             "judge_score": r["judge"]["score"], "human_score": ""} for r in rows[:n]]


def write_sheet(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        w.writeheader()
        w.writerows(rows)


def cohens_kappa(a, b):
    n = len(a)
    if n == 0:
        raise ValueError("no labels to compare")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def agreement(rows, threshold=4):
    labelled = [r for r in rows if str(r.get("human_score", "")).strip() != ""]
    if not labelled:
        raise ValueError("no human_score values filled in")
    judge = [int(r["judge_score"]) for r in labelled]
    human = [int(float(r["human_score"])) for r in labelled]
    if any(not 1 <= h <= 5 for h in human):
        raise ValueError("human_score must be between 1 and 5")
    n = len(labelled)
    return {"n": n,
            "exact_agreement": round(sum(j == h for j, h in zip(judge, human)) / n, 3),
            "within_1": round(sum(abs(j - h) <= 1 for j, h in zip(judge, human)) / n, 3),
            "judge_minus_human_mean": round(sum(j - h for j, h in zip(judge, human)) / n, 3),
            f"kappa_pass_at_{threshold}": round(cohens_kappa([j >= threshold for j in judge], [h >= threshold for h in human]), 3)}
