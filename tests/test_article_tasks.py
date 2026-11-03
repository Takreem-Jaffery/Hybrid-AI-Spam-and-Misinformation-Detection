import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scripts.train_article_models import _read_article, load_clickbait_records, split_indices
from spamdetector.article_tasks import ArticleTaskModels


def test_article_task_models_return_independent_categories(tmp_path):
    models_dir = tmp_path / "models"
    veracity = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression())])
    clickbait = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression())])
    texts = ["fact checked report", "fabricated claim", "ordinary headline", "shocking secret revealed"]
    labels = ["mostly true", "mostly false", "not clickbait", "heavily clickbait"]
    for pipeline, task_texts, train_labels in (
        (veracity, texts[:2], labels[:2]),
        (clickbait, texts[2:], labels[2:]),
    ):
        pipeline.fit(task_texts, train_labels)
    for task, pipeline in (("article_veracity", veracity), ("clickbait", clickbait)):
        directory = models_dir / task
        directory.mkdir(parents=True)
        joblib.dump(pipeline, directory / f"{task}.joblib")
        (directory / f"{task}_metrics.json").write_text(
            json.dumps(
                {
                    "split_method": "publisher_group_holdout",
                    "test_accuracy": 0.6,
                    "test_macro_f1": 0.4,
                    "majority_baseline_accuracy": 0.5,
                    "majority_baseline_macro_f1": 0.2,
                }
            ),
            encoding="utf-8",
        )

    result = ArticleTaskModels(models_dir).review("shocking secret", "fabricated claim")
    assert result["veracity_category"] in {"mostly true", "mostly false"}
    assert result["clickbait_category"] in {"not clickbait", "heavily clickbait"}
    assert 0 <= result["veracity_model_score"] <= 1
    assert 0 <= result["clickbait_model_score"] <= 1
    assert result["evaluation"]["article_veracity"]["test_macro_f1"] == 0.4
    assert result["evaluation"]["clickbait"]["majority_baseline_macro_f1"] == 0.2


def test_clickbait_training_labels_follow_nearest_original_severity(tmp_path):
    directory = tmp_path / "clickbait17" / "train" / "clickbait17-train"
    directory.mkdir(parents=True)
    instances = [
        {"id": "1", "postText": ["A plain title"]},
        {"id": "2", "postText": ["A little tease"]},
        {"id": "3", "postText": ["A strong tease"]},
        {"id": "4", "postText": ["An extreme tease"]},
    ]
    truth = [
        {"id": "4", "truthMean": 1.0},
        {"id": "3", "truthMean": 0.66},
        {"id": "2", "truthMean": 0.33},
        {"id": "1", "truthMean": 0.0},
    ]
    (directory / "instances.jsonl").write_text(
        "\n".join(json.dumps(record) for record in instances), encoding="utf-8"
    )
    (directory / "truth.jsonl").write_text("\n".join(json.dumps(record) for record in truth), encoding="utf-8")
    texts, labels, groups = load_clickbait_records(tmp_path)
    assert len(texts) == 4
    assert labels == [
        "not clickbait",
        "slightly clickbait",
        "considerably clickbait",
        "heavily clickbait",
    ]
    assert groups == ["", "", "", ""]


def test_group_split_is_repeatable_and_keeps_publishers_separate():
    labels = ["mostly true", "mostly false"] * 12
    groups = [f"publisher-{index // 4}" for index in range(len(labels))]
    train_a, test_a, method = split_indices(labels, groups)
    train_b, test_b, _ = split_indices(labels, groups)
    assert method == "publisher_group_holdout"
    assert set(train_a) == set(train_b)
    assert set(test_a) == set(test_b)
    assert not set(groups[i] for i in train_a) & set(groups[i] for i in test_a)


def test_article_xml_extracts_title_and_body_paragraphs(tmp_path):
    article = tmp_path / "article.xml"
    article.write_text(
        "<article><title>Headline text</title><body><p>First paragraph.</p><p>Second paragraph.</p></body></article>",
        encoding="utf-8",
    )
    assert _read_article(article) == ("Headline text", "First paragraph. Second paragraph.")
