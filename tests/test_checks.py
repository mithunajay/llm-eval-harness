from evalkit import checks

CASE = {"max_words": 10, "must_mention": ["acme", "core"],
        "facts": {"org": "acme", "top_repo": "core", "active_repos": 3, "total_stars": 12000}}


def by_name(output):
    return {c.name: c for c in checks.run_checks(CASE, output)}


def test_grounded_output_passes_everything():
    r = by_name("acme has 3 repos with recent commits, led by core.")
    assert all(c.passed for c in r.values()), [(c.name, c.detail) for c in r.values() if not c.passed]


def test_invented_number_is_caught():
    c = by_name("acme grew 300% and core is great")["numbers_grounded"]
    assert not c.passed and "300" in c.detail


def test_formatted_fact_number_is_still_grounded():
    assert by_name("acme core has 12,000 stars")["numbers_grounded"].passed


def test_banned_phrase_is_caught():
    assert not by_name("acme core is a game-changer")["banned_phrases"].passed


def test_word_limit_is_caught():
    c = by_name("acme core " + "word " * 20)["word_limit"]
    assert not c.passed and "limit 10" in c.detail


def test_must_mention_is_caught():
    c = by_name("acme is nice")["must_mention"]
    assert not c.passed and "core" in c.detail


def test_empty_output_fails_non_empty():
    assert not by_name("   ")["non_empty"].passed
    assert not checks.non_empty(CASE, None).passed
