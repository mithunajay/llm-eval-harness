"""Deterministic checks for yes/no/maybe research-QA outputs (PubMedQA-style).
The model must reply with JSON: {"answer": "yes|no|maybe", "rationale": "..."}"""
import json, re

from evalkit.checks import CheckResult

LABELS = ("yes", "no", "maybe")
ADVICE = ["you should", "i recommend", "guaranteed", "cures "]
MAX_RATIONALE_WORDS = 80
MIN_CONTEXT_OVERLAP = 3
WORD_RE = re.compile(r"[a-z]{5,}")


def parse_answer(output):
    if not output:
        return None
    start, end = output.find("{"), output.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(output[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or not isinstance(data.get("rationale"), str):
        return None
    return data


def valid_json(case, output):
    ok = parse_answer(output) is not None
    return CheckResult("valid_json", ok, "" if ok else "not JSON with string 'answer' and 'rationale'")


def label_valid(case, output):
    d = parse_answer(output)
    ok = bool(d) and d["answer"].strip().lower() in LABELS
    return CheckResult("label_valid", ok, "" if ok else "answer must be yes, no or maybe")


def correct(case, output):
    d = parse_answer(output)
    if not d:
        return CheckResult("correct", False, "unparseable output")
    got = d["answer"].strip().lower()
    return CheckResult("correct", got == case["gold"], f"answered {got!r}, gold {case['gold']!r}")


def rationale_length(case, output):
    d = parse_answer(output)
    if not d:
        return CheckResult("rationale_length", False, "unparseable output")
    n = len(d["rationale"].split())
    return CheckResult("rationale_length", 0 < n <= MAX_RATIONALE_WORDS, f"{n} words, limit {MAX_RATIONALE_WORDS}")


def no_personal_advice(case, output):
    d = parse_answer(output)
    text = (d["rationale"] if d else output or "").lower()
    hit = [a.strip() for a in ADVICE if a in text]
    return CheckResult("no_personal_advice", not hit, ", ".join(hit))


def rationale_grounded(case, output):
    """Heuristic: the rationale shares at least a few longer words with the abstract."""
    d = parse_answer(output)
    if not d:
        return CheckResult("rationale_grounded", False, "unparseable output")
    overlap = set(WORD_RE.findall(d["rationale"].lower())) & set(WORD_RE.findall(" ".join(case["contexts"]).lower()))
    return CheckResult("rationale_grounded", len(overlap) >= MIN_CONTEXT_OVERLAP,
                       f"{len(overlap)} shared words, need {MIN_CONTEXT_OVERLAP}")


QA_CHECKS = [valid_json, label_valid, correct, rationale_length, no_personal_advice, rationale_grounded]
