"""Independent article-veracity and headline clickbait classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from . import config


class ArticleTaskModels:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or config.article_models_dir()
        self.veracity = self._load("article_veracity")
        self.clickbait = self._load("clickbait")
        self.veracity_metrics = self._load_metrics("article_veracity")
        self.clickbait_metrics = self._load_metrics("clickbait")

    def _load(self, name: str):
        path = self.models_dir / name / f"{name}.joblib"
        return joblib.load(path) if path.exists() else None

    def _load_metrics(self, name: str) -> dict | None:
        path = self.models_dir / name / f"{name}_metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def status(self) -> dict[str, bool]:
        return {"article_veracity_available": self.veracity is not None, "clickbait_available": self.clickbait is not None}

    def review(self, title: str, body: str) -> dict:
        if self.veracity is None or self.clickbait is None:
            missing = []
            if self.veracity is None:
                missing.append("article veracity")
            if self.clickbait is None:
                missing.append("clickbait")
            raise FileNotFoundError(f"Missing trained {' and '.join(missing)} model(s). Run scripts/train_article_models.py.")

        article_scores = self.veracity.predict_proba([f"{title}\n\n{body}"])[0]
        headline_scores = self.clickbait.predict_proba([title])[0]
        article_index = int(article_scores.argmax())
        clickbait_index = int(headline_scores.argmax())
        return {
            "veracity_category": str(self.veracity.classes_[article_index]),
            "veracity_model_score": round(float(article_scores[article_index]), 4),
            "veracity_scores": {
                str(label): round(float(score), 4)
                for label, score in zip(self.veracity.classes_, article_scores)
            },
            "clickbait_category": str(self.clickbait.classes_[clickbait_index]),
            "clickbait_model_score": round(float(headline_scores[clickbait_index]), 4),
            "clickbait_scores": {
                str(label): round(float(score), 4)
                for label, score in zip(self.clickbait.classes_, headline_scores)
            },
            "evaluation": {
                name: {
                    key: metrics[key]
                    for key in (
                        "split_method",
                        "test_accuracy",
                        "test_macro_f1",
                        "majority_baseline_accuracy",
                        "majority_baseline_macro_f1",
                    )
                    if key in metrics
                }
                for name, metrics in (
                    ("article_veracity", self.veracity_metrics),
                    ("clickbait", self.clickbait_metrics),
                )
                if metrics is not None
            },
            "model_used": "Separate TF-IDF classifiers trained on BuzzFeed-Webis 2016 and Webis Clickbait-17",
        }
