"""LLM-as-judge scoring with a strict output contract. If the judge answers in an
unparseable way, the result is marked invalid and the case does not pass silently."""
import json
from dataclasses import dataclass

from evalkit import tasks


class JudgeParseError(Exception):
    pass


def parse_judge(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeParseError("no JSON object in judge reply")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"invalid JSON: {e.msg}")
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or int(score) != score or not 1 <= score <= 5:
        raise JudgeParseError(f"score must be an integer 1-5, got {score!r}")
    return int(score), str(data.get("reason", ""))


@dataclass
class JudgeResult:
    score: int | None
    reason: str
    valid: bool
    error: str = ""


class Judge:
    def __init__(self, provider, threshold=4, task=None):
        self.provider, self.threshold, self.task = provider, threshold, task or tasks.OPENERS

    async def score(self, case, output, span=None):
        comp = await self.provider.complete(self.task.judge_prompt(case, output))
        if span:
            span.set_tokens(comp.tokens_in, comp.tokens_out)
            span.set_output(comp.text)
        try:
            score, reason = parse_judge(comp.text)
        except JudgeParseError as e:
            return JudgeResult(None, "", False, str(e))
        return JudgeResult(score, reason, True)
