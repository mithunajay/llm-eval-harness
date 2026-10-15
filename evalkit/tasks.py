"""A Task bundles everything task-specific: schema, system prompt, prompt rendering,
deterministic checks, and the judge rubric. The runner and judge stay generic."""
import json, os
from dataclasses import dataclass
from typing import Callable

from evalkit import checks, qa_checks

ROOT = os.path.join(os.path.dirname(__file__), "..")


@dataclass
class Task:
    name: str
    schema_path: str
    system: str
    render: Callable
    checks: list
    judge_prompt: Callable


def _fill(template, values):
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


# ---- openers (cold email opener) ----
OPENER_RUBRIC = """You are grading a one-sentence cold email opener for a technical buyer.
Score 1-5. 5 = specific, grounded only in the facts given, plain tone. 1 = invented claims, flattery, or off-topic.
Respond with JSON only: {{"score": <1-5>, "reason": "<short reason>"}}
FACTS: {facts}
OUTPUT: {output}"""


def _opener_render(case, template):
    return _fill(template, {"facts_json": json.dumps(case["facts"]), "persona": case["persona"], "max_words": case["max_words"]})


def _opener_judge(case, output):
    return OPENER_RUBRIC.format(facts=json.dumps(case["facts"]), output=(output or "").replace("\n", " "))


# ---- pubmedqa (biomedical research question answering) ----
QA_RUBRIC = """You are grading the RATIONALE of an answer to a biomedical research question.
The answer must rely ONLY on the abstract provided. Score 1-5 for faithfulness:
5 = every claim in the rationale is supported by the abstract; 3 = partly supported or vague;
1 = contains claims that are not in the abstract, or contradicts it.
Respond with JSON only: {{"score": <1-5>, "reason": "<short reason>"}}
QUESTION: {question}
ABSTRACT:
{abstract}
ANSWER: {answer}
RATIONALE: {rationale}"""


def _qa_render(case, template):
    return _fill(template, {"question": case["question"], "context": "\n".join(case["contexts"])})


def _qa_judge(case, output):
    parsed = qa_checks.parse_answer(output)
    answer, rationale = (parsed["answer"], parsed["rationale"]) if parsed else ("(unparseable)", output or "")
    return QA_RUBRIC.format(question=case["question"], abstract="\n".join(case["contexts"]), answer=answer, rationale=rationale)


OPENERS = Task("openers", os.path.join(ROOT, "datasets", "schema_openers.json"),
               "You write one-sentence cold email openers. Use only the facts given. Do not invent numbers.",
               _opener_render, checks.ALL_CHECKS, _opener_judge)
PUBMEDQA = Task("pubmedqa", os.path.join(ROOT, "datasets", "schema_pubmedqa.json"),
                "You answer biomedical research questions using only the abstract provided. Reply with JSON only.",
                _qa_render, qa_checks.QA_CHECKS, _qa_judge)
TASKS = {t.name: t for t in (OPENERS, PUBMEDQA)}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASKS)}")
    return TASKS[name]
