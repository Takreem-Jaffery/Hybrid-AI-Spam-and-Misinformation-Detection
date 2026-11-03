"""Trains on the real dataset and checks the published numbers. Run with: pytest -m slow"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from spamdetector import config
from spamdetector.data import load_dataset, split_dataset

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not config.data_path().exists(), reason="dataset not downloaded")]


def test_random_forest_reproduces_reported_numbers():
    s = split_dataset(load_dataset())
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=config.SEED))]).fit(s.X_train, s.y_train)
    pred = pipe.predict(s.X_test)
    # Current baseline is recorded in results/classical_training.json.
    assert accuracy_score(s.y_test, pred) * 100 == pytest.approx(98.21, abs=0.3)
    assert f1_score(s.y_test, pred) * 100 == pytest.approx(92.81, abs=0.3)
