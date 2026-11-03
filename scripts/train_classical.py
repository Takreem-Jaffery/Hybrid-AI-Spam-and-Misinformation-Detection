#!/usr/bin/env python
"""Train the three classical models (TF-IDF + Naive Bayes / Logistic Regression / Random Forest).

    python scripts/train_classical.py

Carried over from notebook cells 2-3 (same TF-IDF settings, same hyper-parameters, same
random_state=42 stratified 80/20 split). One deliberate change: the "best" model that ships in
models/pipeline.pkl is now chosen by 5-fold stratified cross-validation F1 on the TRAINING
split. The notebook picked it by test-set F1, which means the test set had already influenced
which model was reported. Pass --select-by test to reproduce the old behaviour.

Writes
    models/classical/<model>.pkl   one fitted pipeline per model
    models/pipeline.pkl            the selected pipeline used by the API
    models/pipeline_meta.json      which model was selected and why
    results/classical_training.json
"""

from __future__ import annotations

import argparse
import json
import warnings
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spamdetector import config  # noqa: E402
from spamdetector.data import load_dataset, split_dataset  # noqa: E402


def make_pipelines() -> dict[str, Pipeline]:
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("model", MultinomialNB())]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("model", LogisticRegression(max_iter=1000, random_state=config.SEED))]),
        "Random Forest": Pipeline([("tfidf", tfidf()), ("model", RandomForestClassifier(n_estimators=100, random_state=config.SEED))]),
    }


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    # Harmless scipy/sklearn version-mismatch noise from LogisticRegression
    warnings.filterwarnings("ignore", message="Unknown solver options")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--select-by", choices=["cv", "test"], default="cv")
    args = ap.parse_args()

    split = split_dataset(load_dataset(args.data))
    print(f"train={len(split.X_train)}  test={len(split.X_test)}  (spam share {split.y_train.mean():.1%})")

    out_dir = config.classical_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    report: dict[str, dict] = {}
    pipelines = make_pipelines()
    print(f"\n{'Model':<22} {'CV F1':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 64)
    for name, pipe in pipelines.items():
        t0 = time.time()
        cv_f1 = cross_val_score(pipe, split.X_train, split.y_train, cv=cv, scoring="f1")
        pipe.fit(split.X_train, split.y_train)
        pred = pipe.predict(split.X_test)
        m = {
            "cv_f1_mean": round(float(cv_f1.mean()) * 100, 2),
            "cv_f1_std": round(float(cv_f1.std()) * 100, 2),
            "test_accuracy": round(accuracy_score(split.y_test, pred) * 100, 2),
            "test_precision": round(precision_score(split.y_test, pred) * 100, 2),
            "test_recall": round(recall_score(split.y_test, pred) * 100, 2),
            "test_f1": round(f1_score(split.y_test, pred) * 100, 2),
            "confusion_matrix": confusion_matrix(split.y_test, pred).tolist(),
            "seconds": round(time.time() - t0, 1),
        }
        report[name] = m
        with open(out_dir / f"{slug(name)}.pkl", "wb") as f:
            pickle.dump(pipe, f)
        print(f"{name:<22} {m['cv_f1_mean']:>7}% {m['test_accuracy']:>6}% {m['test_precision']:>6}% {m['test_recall']:>6}% {m['test_f1']:>6}%")

    key = "cv_f1_mean" if args.select_by == "cv" else "test_f1"
    best = max(report, key=lambda n: report[n][key])
    print(f"\nSelected: {best}  (by {'cross-validated F1 on the training split' if args.select_by == 'cv' else 'TEST F1 - leaks the test set'})")

    with open(config.pipeline_path(), "wb") as f:
        pickle.dump(pipelines[best], f)
    config.pipeline_meta_path().write_text(json.dumps({"model": best, "selected_by": args.select_by, "seed": config.SEED}, indent=2) + "\n")

    config.results_dir().mkdir(parents=True, exist_ok=True)
    (config.results_dir() / "classical_training.json").write_text(
        json.dumps({"selected": best, "selected_by": args.select_by, "models": report}, indent=2) + "\n"
    )
    print(f"Saved {config.pipeline_path()}")
    print("Next: python scripts/train_distilbert.py   (GPU recommended)   or   python scripts/evaluate.py")


if __name__ == "__main__":
    np.random.seed(config.SEED)
    main()
