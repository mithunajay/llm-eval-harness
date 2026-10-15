import asyncio
from evalkit.judge import Judge, parse_judge, JudgeParseError
from evalkit.providers import MockJudgeProvider


def bad(text):
    try:
        parse_judge(text)
    except JudgeParseError:
        return True
    return False


def test_parse_valid_and_wrapped_json():
    assert parse_judge('{"score": 4, "reason": "ok"}') == (4, "ok")
    assert parse_judge('Here you go: {"score": 5, "reason": "great"} thanks') == (5, "great")


def test_parse_rejects_bad_replies():
    assert bad("Sure! Looks fine.")
    assert bad('{"score": 9, "reason": "x"}')
    assert bad('{"score": 3.5, "reason": "x"}')
    assert bad('{"score": true, "reason": "x"}')
    assert bad('{"reason": "no score"}')
    assert bad("{not json}")


CASE = {"facts": {"org": "acme", "top_repo": "core", "active_repos": 3, "total_stars": 10}}


def test_judge_scores_grounded_high_and_ungrounded_low():
    j = Judge(MockJudgeProvider())
    good = asyncio.run(j.score(CASE, "acme has 3 repos with recent commits, led by core."))
    badr = asyncio.run(j.score(CASE, "acme grew 300% and core is a game-changer."))
    assert good.valid and good.score == 5
    assert badr.valid and badr.score == 2


def test_flaky_judge_marks_unparseable_as_invalid_not_as_a_pass():
    j = Judge(MockJudgeProvider(flaky=True))
    results = [asyncio.run(j.score(CASE, "acme has 3 repos, led by core.")) for _ in range(3)]
    assert [r.valid for r in results] == [True, True, False]
    assert results[2].score is None and results[2].error
