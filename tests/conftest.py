"""Shared fixtures. The tiny corpus below is synthetic and only exists to exercise code paths;
real model quality is covered by the `slow` test and scripts/evaluate.py."""

import pickle

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from spamdetector.preprocess import preprocess

SPAM = [
    "Congratulations! You won a free prize. Click now to claim!",
    "FREE cash! Win Rs.50000 instantly. Call now, offer expires today!",
    "URGENT: your bank account is blocked. Verify OTP immediately at http://bit.ly/x",
    "You have been selected for a cash reward. Call to claim now.",
    "Winner! Claim your free iPhone voucher today only",
    "Limited offer: double your money with our investment scheme",
]
HAM = [
    "Hey, are we still meeting at 6 for dinner?",
    "Can you send me the assignment notes when you get a chance?",
    "I'll call you when I get home",
    "Thanks for the lift this morning, see you tomorrow",
    "Did you watch the match last night?",
    "Mum says dinner is ready, come downstairs",
]


@pytest.fixture()
def models_dir(tmp_path, monkeypatch):
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("model", LogisticRegression(max_iter=1000, random_state=0))])
    pipe.fit([preprocess(t) for t in SPAM + HAM], [1] * len(SPAM) + [0] * len(HAM))
    with open(tmp_path / "pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)
    (tmp_path / "pipeline_meta.json").write_text('{"model": "Logistic Regression"}')
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return tmp_path


@pytest.fixture()
def empty_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return tmp_path
