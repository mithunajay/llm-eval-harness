"""Compare a run to a saved baseline and fail on regressions, so a prompt or model change
cannot quietly make things worse."""
import json
from dataclasses import dataclass, field


class BaselineError(Exception):
    pass


def to_baseline(run):
    return {"meta": run["meta"], "summary": run["summary"],
            "cases": {r.case_id: {"passed": r.passed,
                                  "judge_score": (r.judge or {}).get("score")} for r in run["results"]}}


def save_baseline(run, path):
    with open(path, "w") as f:
        json.dump(to_baseline(run), f, indent=2)


@dataclass
class Report:
    newly_failing: list = field(default_factory=list)
    fixed: list = field(default_factory=list)
    pass_rate_before: float = 0.0
    pass_rate_after: float = 0.0
    judge_before: float | None = None
    judge_after: float | None = None
    judge_drop_limit: float = 0.3
    regressed: bool = False
    reasons: list = field(default_factory=list)

    def render(self):
        lines = [f"pass rate: {self.pass_rate_before:.0%} -> {self.pass_rate_after:.0%}"]
        if self.judge_before is not None and self.judge_after is not None:
            lines.append(f"judge mean: {self.judge_before} -> {self.judge_after}")
        if self.newly_failing:
            lines.append("newly failing: " + ", ".join(self.newly_failing))
        if self.fixed:
            lines.append("fixed: " + ", ".join(self.fixed))
        lines.append("RESULT: " + ("REGRESSION - " + "; ".join(self.reasons) if self.regressed else "no regression"))
        return "\n".join(lines)


def compare(run, baseline, judge_drop_limit=0.3, max_new_failures=0, pass_rate_tolerance=0.0):
    """Real models are not perfectly repeatable, so tolerances are explicit:
    max_new_failures = how many cases may flip from pass to fail; pass_rate_tolerance = allowed pass-rate drop."""
    if run["meta"]["dataset_sha256"] != baseline["meta"]["dataset_sha256"]:
        raise BaselineError("baseline was made on a different dataset; re-freeze and re-baseline before comparing")
    cur = to_baseline(run)
    rep = Report(pass_rate_before=baseline["summary"]["pass_rate"], pass_rate_after=cur["summary"]["pass_rate"],
                 judge_before=baseline["summary"]["judge_mean"], judge_after=cur["summary"]["judge_mean"],
                 judge_drop_limit=judge_drop_limit)
    for cid, was in baseline["cases"].items():
        now = cur["cases"].get(cid)
        if now is None:
            rep.reasons.append(f"case {cid} missing from current run")
            continue
        if was["passed"] and not now["passed"]:
            rep.newly_failing.append(cid)
        if not was["passed"] and now["passed"]:
            rep.fixed.append(cid)
    if len(rep.newly_failing) > max_new_failures:
        rep.reasons.append(f"{len(rep.newly_failing)} case(s) that passed before now fail (allowed {max_new_failures})")
    if rep.pass_rate_before - rep.pass_rate_after > pass_rate_tolerance + 1e-9:
        rep.reasons.append("pass rate dropped")
    if rep.judge_before is not None and rep.judge_after is not None and rep.judge_before - rep.judge_after > judge_drop_limit:
        rep.reasons.append(f"judge mean dropped by more than {judge_drop_limit}")
    rep.regressed = bool(rep.reasons)
    return rep
