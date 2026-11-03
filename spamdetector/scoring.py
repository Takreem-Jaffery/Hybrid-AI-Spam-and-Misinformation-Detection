"""Rule-based signals, the model ensemble and the risk tiers.

Pure functions only (no model objects, no I/O) so the API and ``scripts/evaluate.py`` share
exactly the same scoring logic and the logic is trivial to unit-test.

Scores are *spam scores* on a 0-100 scale. They are NOT calibrated probabilities (the
original project called this "confidence", which is misleading, especially for a benign
verdict where a high number means "more likely spam").
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Default weight of the TF-IDF model in the ensemble; DistilBERT gets 1 - weight.
# 0.65 / 0.35 is the weighting from the original project. It was chosen a priori, not tuned;
# `scripts/evaluate.py` reports a sensitivity analysis so you can see how much it matters.
DEFAULT_TFIDF_WEIGHT = 0.65

SPAM_THRESHOLD = 50.0  # a score must EXCEED this to be Spam
HIGH_RISK = 65.0
MEDIUM_RISK = 40.0

# Carried over unchanged from the original main.py
SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize", "claim", "urgent",
    "congratulations", "credit", "loan", "offer", "discount", "click",
    "call now", "limited", "act now", "guaranteed", "reward", "selected",
    "verify", "account", "bank", "password", "otp", "lucky", "earn",
    "money", "rupees", "rs.", "lakh", "crore", "investment", "double",
    "triple", "profit", "scheme", "deal", "expire", "expires", "today only",
    "iphone", "gift", "voucher", "jackpot", "congratulation",
    "send this", "share this", "forward this", "chain message",
]


def _keyword_regex(keyword: str) -> re.Pattern[str]:
    # Whole-word match. The original used a plain substring test, so "win" matched
    # "window" and "free" matched "freedom". Keywords ending in punctuation ("rs.") only
    # get a leading boundary so that "Rs.50000" still matches.
    pattern = r"(?<!\w)" + re.escape(keyword)
    if keyword[-1].isalnum():
        pattern += r"(?!\w)"
    return re.compile(pattern)


_KEYWORD_RES = [(kw, _keyword_regex(kw)) for kw in SPAM_KEYWORDS]
_URL_RE = re.compile(r"https?://|www\.|bit\.ly|tinyurl|t\.me", re.I)
_LONG_NUMBER_RE = re.compile(r"\b\d{10,}\b")
_OTP_RE = re.compile(r"\botp\b|\bverif\w+\b|\bblock\w*\b", re.I)
_URGENCY_RE = re.compile(r"\burgent\b|\bimmediately\b|\bexpire\b|\bblock\b", re.I)


def find_keywords(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw, rx in _KEYWORD_RES if rx.search(lower)]


def build_reasons(text: str, is_forwarded: bool, source: str) -> list[str]:
    reasons: list[str] = []
    found = find_keywords(text)
    if len(found) >= 3:
        reasons.append(f"Contains {len(found)} spam keywords: {', '.join(found[:4])}")
    elif found:
        reasons.append(f"Contains suspicious words: {', '.join(found)}")

    if _URL_RE.search(text):
        reasons.append("Contains a suspicious URL or shortened link")
    if _LONG_NUMBER_RE.search(text):
        reasons.append("Contains a long number (possible phone / account)")

    no_space = re.sub(r"\s", "", text)
    caps_ratio = len(re.findall(r"[A-Z]", no_space)) / max(len(no_space), 1)
    if caps_ratio > 0.4:
        reasons.append("Excessive use of capital letters")
    if text.count("!") >= 2:
        reasons.append("Multiple exclamation marks")
    if is_forwarded:
        reasons.append("Message is marked as forwarded")
    if source == "unknown":
        reasons.append("Source is an unknown sender")
    return reasons


def context_bonus(is_forwarded: bool, source: str) -> float:
    """Metadata signals added on top of the model score (heuristic, not learned)."""
    bonus = 0.0
    if is_forwarded:
        bonus += 8.0
    if source == "unknown":
        bonus += 7.0
    return bonus


def compute_risk(score: float) -> tuple[str, str]:
    if score >= HIGH_RISK:
        return "High", "#ef4444"
    if score >= MEDIUM_RISK:
        return "Medium", "#f59e0b"
    return "Low", "#22c55e"


def rule_based_score(text: str) -> float:
    """Fallback used only when no ML model is available."""
    score = min(len(find_keywords(text)) * 12, 50)
    if _URL_RE.search(text):
        score += 20
    if _LONG_NUMBER_RE.search(text):
        score += 10
    return float(score)


def combine_scores(tfidf: float, transformer: float, tfidf_weight: float = DEFAULT_TFIDF_WEIGHT) -> float:
    """Weighted average of the two model spam scores (both 0-100)."""
    return tfidf_weight * tfidf + (1.0 - tfidf_weight) * transformer


@dataclass(frozen=True)
class Verdict:
    label: str  # "Spam" | "Benign"
    spam_score: float  # 0-100, rounded to 1 decimal
    risk: str  # "Low" | "Medium" | "High"
    risk_color: str
    model_used: str
    explanation: str
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def score_message(
    text: str,
    *,
    is_forwarded: bool = False,
    source: str = "unknown",
    tfidf_score: float | None = None,
    transformer_score: float | None = None,
    tfidf_label: str = "TF-IDF classifier",
    tfidf_weight: float = DEFAULT_TFIDF_WEIGHT,
) -> Verdict:
    """Combine model scores, metadata and rules into the final verdict.

    Case A: both models available  -> ``w * tfidf + (1 - w) * distilbert + context``
    Case B: only one model         -> that model's score + context
    Case C: no model at all        -> rule-based score + context
    """
    reasons = build_reasons(text, is_forwarded, source)
    ctx = context_bonus(is_forwarded, source)

    if tfidf_score is not None and transformer_score is not None:
        final = combine_scores(tfidf_score, transformer_score, tfidf_weight) + ctx
        model_used = f"{tfidf_label} + DistilBERT ensemble"
    elif tfidf_score is not None:
        final = tfidf_score + ctx
        model_used = tfidf_label
    elif transformer_score is not None:
        final = transformer_score + ctx
        model_used = "DistilBERT"
    else:
        final = rule_based_score(text) + ctx
        model_used = "Rule-based fallback"

    # Phishing floor: OTP/verification + urgency language + 4 or more spam keywords from an
    # unknown sender is never allowed to score below 55. Heuristic, kept from the original.
    if (
        len(find_keywords(text)) >= 4
        and _OTP_RE.search(text)
        and _URGENCY_RE.search(text)
        and source == "unknown"
    ):
        final = max(final, 55.0)

    final = min(max(final, 0.0), 100.0)
    label = "Spam" if final > SPAM_THRESHOLD else "Benign"  # strict: same rule as scripts/evaluate.py and sklearn predict()
    risk, color = compute_risk(final)

    if label == "Spam":
        main = reasons[0] if reasons else "suspicious patterns detected"
        explanation = f"{model_used} flagged spam (spam score {final:.0f}/100): {main[0].lower() + main[1:]}."
    else:
        explanation = f"No strong spam indicators found by {model_used} (spam score {final:.0f}/100)."

    return Verdict(
        label=label,
        spam_score=round(final, 1),
        risk=risk,
        risk_color=color,
        model_used=model_used,
        explanation=explanation,
        reasons=reasons,
    )
