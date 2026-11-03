import pytest

from spamdetector.scoring import (
    DEFAULT_TFIDF_WEIGHT, build_reasons, combine_scores, compute_risk, context_bonus,
    find_keywords, score_message,
)


def test_keywords_match_whole_words_only():
    assert find_keywords("A window to freedom") == []  # 'win' / 'free' are not whole words here
    assert "win" in find_keywords("You can win today")
    assert "rs." in find_keywords("Win Rs.50000 now")


def test_reasons_cover_the_main_signals():
    text = "URGENT!! CLAIM FREE PRIZE http://bit.ly/x 03001234567"
    reasons = " | ".join(build_reasons(text, True, "unknown"))
    for fragment in ("spam keywords", "URL", "long number", "capital letters", "exclamation", "forwarded", "unknown sender"):
        assert fragment in reasons


def test_friend_message_gets_no_context_bonus():
    assert context_bonus(False, "friend") == 0
    assert context_bonus(True, "unknown") == 15


@pytest.mark.parametrize("score,risk", [(0, "Low"), (39.9, "Low"), (40, "Medium"), (64.9, "Medium"), (65, "High"), (100, "High")])
def test_risk_tiers(score, risk):
    assert compute_risk(score)[0] == risk


def test_ensemble_is_weighted_average():
    assert combine_scores(80, 20, 0.65) == pytest.approx(0.65 * 80 + 0.35 * 20)
    assert combine_scores(80, 20, 1.0) == 80
    assert DEFAULT_TFIDF_WEIGHT == 0.65


def test_score_message_uses_both_models_and_clamps():
    v = score_message("hi", source="friend", tfidf_score=100, transformer_score=100)
    assert v.spam_score == 100 and v.label == "Spam" and "ensemble" in v.model_used
    low = score_message("hi", source="friend", tfidf_score=0, transformer_score=0)
    assert low.label == "Benign" and low.risk == "Low"


def test_single_model_and_rule_based_fallbacks():
    assert score_message("hi", source="friend", tfidf_score=10, tfidf_label="TF-IDF (RF)").model_used == "TF-IDF (RF)"
    assert score_message("hi", source="friend", transformer_score=10).model_used == "DistilBERT"
    v = score_message("Win free cash prize now, click http://bit.ly/x", source="friend")
    assert v.model_used == "Rule-based fallback" and v.label == "Spam"


def test_phishing_floor():
    text = "URGENT: your bank account will be blocked. Verify OTP immediately to claim money"
    v = score_message(text, source="unknown", tfidf_score=5, transformer_score=5)
    assert v.spam_score >= 55 and v.label == "Spam"
    assert score_message(text, source="friend", tfidf_score=5, transformer_score=5).label == "Benign"
