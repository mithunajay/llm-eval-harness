"""Turn the public PubMedQA labeled set (PQA-L, 1,000 expert-annotated yes/no/maybe questions,
MIT licence, Jin et al. 2019) into this harness's dataset format.

Accepts either layout:
  * original release: a JSON object keyed by PubMed id, each value with QUESTION, CONTEXTS, final_decision
  * Hugging Face export: rows with pubid, question, context{contexts}, final_decision (JSON list or JSONL)
Unknown layouts fail with a clear message rather than guessing."""
import json, random

LABELS = ("yes", "no", "maybe")


class PubMedQAFormatError(Exception):
    pass


def normalize(record, pubid=None):
    if "QUESTION" in record:
        question, contexts, gold = record["QUESTION"], record.get("CONTEXTS"), record.get("final_decision")
    elif "question" in record:
        ctx = record.get("context")
        contexts = ctx.get("contexts") if isinstance(ctx, dict) else ctx
        question, gold = record["question"], record.get("final_decision")
        pubid = record.get("pubid", pubid)
    else:
        raise PubMedQAFormatError(f"unrecognised record layout; keys: {sorted(record)[:8]}")
    if pubid is None:
        raise PubMedQAFormatError("record has no PubMed id")
    gold = str(gold).strip().lower() if gold is not None else None
    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        raise PubMedQAFormatError(f"record {pubid}: contexts must be a list of strings")
    return {"id": str(pubid), "question": str(question).strip(), "contexts": [c.strip() for c in contexts if c.strip()],
            "gold": gold, "tags": ["pubmedqa", "pqa-l"]}


def load_records(path):
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        return [normalize(r) for r in rows]
    data = json.load(open(path))
    if isinstance(data, dict):
        return [normalize(v, pubid=k) for k, v in data.items()]
    if isinstance(data, list):
        return [normalize(r) for r in data]
    raise PubMedQAFormatError("top level must be an object keyed by id, or a list of records")


def clean(cases):
    """Drop records that cannot be evaluated; report how many and why."""
    kept, dropped = [], {"bad_label": 0, "no_context": 0, "no_question": 0}
    for c in cases:
        if c["gold"] not in LABELS:
            dropped["bad_label"] += 1
        elif not c["contexts"]:
            dropped["no_context"] += 1
        elif not c["question"]:
            dropped["no_question"] += 1
        else:
            kept.append(c)
    return kept, dropped


def sample(cases, n, seed=7, balanced=False):
    rnd = random.Random(seed)
    by = {l: sorted((c for c in cases if c["gold"] == l), key=lambda c: c["id"]) for l in LABELS}
    for v in by.values():
        rnd.shuffle(v)
    if balanced:
        counts = {l: n // 3 for l in LABELS}
        counts["yes"] += n - sum(counts.values())
    else:
        total = len(cases)
        raw = {l: n * len(by[l]) / total for l in LABELS}
        counts = {l: int(raw[l]) for l in LABELS}
        for l in sorted(LABELS, key=lambda l: raw[l] - counts[l], reverse=True)[: n - sum(counts.values())]:
            counts[l] += 1
    for l in LABELS:
        if counts[l] > len(by[l]):
            raise PubMedQAFormatError(f"need {counts[l]} {l!r} cases but only {len(by[l])} available")
    chosen = [c for l in LABELS for c in by[l][: counts[l]]]
    return sorted(chosen, key=lambda c: c["id"])


def write_jsonl(cases, path):
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
