"""Async eval runner. Each case runs through traced steps: build_prompt, model_call, checks, judge."""
import asyncio, hashlib, json, os, time
from dataclasses import dataclass, asdict, field

from evalkit import tasks as tasks_mod
from evalkit.checks import run_checks
from evalkit.providers import ProviderError
from evalkit.tracing import Tracer


@dataclass
class CaseResult:
    case_id: str
    output: str | None
    checks: list
    judge: dict | None
    passed: bool
    error: str | None
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    trace_id: str
    failed_step: str | None
    failure_reasons: list = field(default_factory=list)


def render_prompt(template, case, task=None):
    return (task or tasks_mod.OPENERS).render(case, template)


def prompt_fingerprint(template):
    return hashlib.sha256(template.encode()).hexdigest()[:8]


async def _run_case(case, provider, template, judge, run_id, sem, trace_dir, timeout, task):
    tracer = Tracer(f"{run_id}:{case['id']}")
    output, error, judge_result = None, None, None
    async with sem:
        try:
            with tracer.span("build_prompt", case_id=case["id"]) as s:
                prompt = render_prompt(template, case, task)
                s.set_output(prompt)
            with tracer.span("model_call", provider=provider.name) as s:
                comp = await asyncio.wait_for(provider.complete(prompt, task.system), timeout)
                output = comp.text
                s.set_tokens(comp.tokens_in, comp.tokens_out)
                s.set_output(output)
        except (ProviderError, asyncio.TimeoutError) as e:
            error = f"{type(e).__name__}: {e}"
        with tracer.span("checks") as s:
            checks = run_checks(case, output, task.checks)
            s.set_output([{"name": c.name, "passed": c.passed} for c in checks])
        if judge and error is None:
            try:
                with tracer.span("judge") as s:
                    judge_result = await judge.score(case, output, s)
            except (ProviderError, asyncio.TimeoutError) as e:
                error = f"judge {type(e).__name__}: {e}"
    reasons = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    if error:
        reasons.append(error)
    if judge:
        if judge_result is None or not judge_result.valid:
            reasons.append("judge invalid: " + (judge_result.error if judge_result else "no result"))
        elif judge_result.score < judge.threshold:
            reasons.append(f"judge score {judge_result.score} < {judge.threshold}")
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        tracer.dump(os.path.join(trace_dir, f"{run_id}.jsonl"))
    return CaseResult(
        case_id=case["id"], output=output, checks=[asdict(c) for c in checks],
        judge=asdict(judge_result) if judge_result else None, passed=not reasons, error=error,
        latency_ms=tracer.total_latency_ms, tokens_in=sum(s["tokens_in"] for s in tracer.spans),
        tokens_out=sum(s["tokens_out"] for s in tracer.spans), cost_usd=tracer.total_cost,
        trace_id=tracer.trace_id, failed_step=tracer.failed_step(), failure_reasons=reasons)


def summarize(results):
    n = len(results)
    scores = [r.judge["score"] for r in results if r.judge and r.judge["valid"]]
    by_check, totals = {}, {}
    for r in results:
        for c in r.checks:
            totals[c["name"]] = totals.get(c["name"], 0) + 1
            if not c["passed"]:
                by_check[c["name"]] = by_check.get(c["name"], 0) + 1
    return {"cases": n, "passed": sum(r.passed for r in results),
            "pass_rate": round(sum(r.passed for r in results) / n, 4) if n else 0.0,
            "judge_mean": round(sum(scores) / len(scores), 3) if scores else None,
            "judge_invalid": sum(1 for r in results if r.judge and not r.judge["valid"]),
            "errors": sum(1 for r in results if r.error),
            "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 2) if n else 0.0,
            "tokens_in": sum(r.tokens_in for r in results), "tokens_out": sum(r.tokens_out for r in results),
            "est_cost_usd": round(sum(r.cost_usd for r in results), 6), "failures_by_check": by_check,
            "check_pass_rates": {k: round(1 - by_check.get(k, 0) / v, 4) for k, v in totals.items()}}


async def run_eval(dataset, provider, template, judge=None, concurrency=4, trace_dir=None, timeout=30, run_id=None, task=None):
    task = task or tasks_mod.OPENERS
    run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*[
        _run_case(c, provider, template, judge, run_id, sem, trace_dir, timeout, task) for c in dataset.cases])
    return {"run_id": run_id,
            "meta": {"task": task.name, "dataset": dataset.name, "dataset_version": dataset.version, "dataset_sha256": dataset.sha256,
                     "prompt_fingerprint": prompt_fingerprint(template), "provider": provider.name},
            "summary": summarize(results), "results": results}


def run_to_dict(run):
    """JSON-serializable form of a run, for saving and later analysis (e.g. judge calibration)."""
    return {"run_id": run["run_id"], "meta": run["meta"], "summary": run["summary"],
            "results": [asdict(r) for r in run["results"]]}
