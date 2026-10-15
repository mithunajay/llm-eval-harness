"""Deterministic checks. Cheap, repeatable, and run before any LLM judge."""
import re
from dataclasses import dataclass

BANNED = ["hope this email finds you", "i came across your", "quick question", "synergy", "game-changer"]
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _numbers(text):
    return {n.replace(",", "") for n in NUM_RE.findall(text)}


def _fact_numbers(value):
    if isinstance(value, dict):
        out = set()
        for v in value.values():
            out |= _fact_numbers(v)
        return out
    return _numbers(str(value)) if not isinstance(value, bool) else set()


def non_empty(case, output):
    ok = bool(output and output.strip())
    return CheckResult("non_empty", ok, "" if ok else "empty output")


def word_limit(case, output):
    n = len((output or "").split())
    return CheckResult("word_limit", n <= case["max_words"], f"{n} words, limit {case['max_words']}")


def banned_phrases(case, output):
    hit = [b for b in BANNED if b in (output or "").lower()]
    return CheckResult("banned_phrases", not hit, ", ".join(hit))


def numbers_grounded(case, output):
    invented = sorted(_numbers(output or "") - _fact_numbers(case["facts"]))
    return CheckResult("numbers_grounded", not invented, "not in facts: " + ", ".join(invented) if invented else "")


def must_mention(case, output):
    low = (output or "").lower()
    missing = [m for m in case["must_mention"] if m.lower() not in low]
    return CheckResult("must_mention", not missing, "missing: " + ", ".join(missing) if missing else "")


ALL_CHECKS = [non_empty, word_limit, banned_phrases, numbers_grounded, must_mention]


def run_checks(case, output, checks=None):
    return [c(case, output) for c in (checks or ALL_CHECKS)]
