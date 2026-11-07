import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from scripts.evaluate_grouped import classification_metrics, evaluate_grouped_folds, score_scenarios


@pytest.fixture
def grouped_messages():
    rows = []
    for label, marker in ((0, "ordinary"), (1, "offer")):
        for group_id in range(10):
            clean_text = f"{marker} message template{group_id}"
            for copy_id in range(2):
                rows.append(
                    {
                        "text": f"{marker} message template {group_id} copy {copy_id}",
                        "clean_text": clean_text,
                        "label_binary": label,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def small_pipelines():
    return {
        "Naive Bayes": Pipeline(
            [("tfidf", TfidfVectorizer()), ("model", MultinomialNB())]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
    }


def test_evaluates_five_outer_folds_with_independent_inner_selection(grouped_messages, small_pipelines):
    report = evaluate_grouped_folds(grouped_messages, small_pipelines, seed=42)

    assert len(report["folds"]) == 5
    assert sum(fold["split"]["test_rows"] for fold in report["folds"]) == len(grouped_messages)
    for fold in report["folds"]:
        assert fold["split"]["train_test_group_overlap"] == 0
        assert fold["model_selection"]["selected"] in small_pipelines
        assert set(fold["model_selection"]["models"]) == set(small_pipelines)
        assert len(
            fold["model_selection"]["models"][fold["model_selection"]["selected"]]["fold_spam_f1"]
        ) == 5


def test_reports_metric_mean_and_population_sd_across_outer_folds(grouped_messages, small_pipelines):
    report = evaluate_grouped_folds(grouped_messages, small_pipelines, seed=42)
    fold_f1 = [
        fold["outer_test"]["content_only"]["f1"]
        for fold in report["folds"]
    ]
    aggregate = report["aggregate_outer_test"]["scenarios"]["content_only"]["f1"]

    assert aggregate["mean"] == pytest.approx(np.mean(fold_f1), abs=0.01)
    assert aggregate["sd"] == pytest.approx(np.std(fold_f1), abs=0.01)


def test_scenarios_apply_the_documented_context_offsets():
    texts = ["A routine note"]          # no spam keywords, so only the context bonus differs
    model_scores = np.array([30.0])

    scores = score_scenarios(texts, model_scores)

    assert scores["content_only"][0] == pytest.approx(30.0)
    assert scores["default_unknown_sender"][0] == pytest.approx(37.0)       # +7
    assert scores["forwarded_known_sender"][0] == pytest.approx(38.0)       # +8
    assert scores["forwarded_unknown_sender"][0] == pytest.approx(45.0)     # +15


def test_grouped_metrics_include_fixed_threshold_and_both_classes():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([10, 70, 40, 90]))

    assert metrics["threshold"] == 50
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["support"] == {"ham": 2, "spam": 2}
    assert metrics["f1"] == 50
