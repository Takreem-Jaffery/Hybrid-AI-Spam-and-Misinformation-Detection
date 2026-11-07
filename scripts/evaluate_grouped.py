#!/usr/bin/env python
"""Evaluate the SMS baseline with nested, exact-text-grouped five-fold validation.

    python scripts/evaluate_grouped.py

Each outer fold is held out from model selection. For every fold, candidate models are selected
using grouped inner cross-validation on that fold's training rows. This experiment is separate
from scripts/evaluate.py and does not overwrite its results or the deployed model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_classical import make_pipelines  # noqa: E402
from spamdetector import config  # noqa: E402
from spamdetector.data import load_dataset  # noqa: E402
from spamdetector.scoring import SPAM_THRESHOLD, score_message  # noqa: E402

OUTER_FOLDS = 5
INNER_FOLDS = 5
AGGREGATE_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
    "average_precision",
)


def classification_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | int | list]:
    predictions = (scores > SPAM_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": SPAM_THRESHOLD,
        "accuracy": round(float(accuracy_score(y, predictions) * 100), 2),
        "precision": round(float(precision_score(y, predictions, zero_division=0) * 100), 2),
        "recall": round(float(recall_score(y, predictions, zero_division=0) * 100), 2),
        "f1": round(float(f1_score(y, predictions, zero_division=0) * 100), 2),
        "macro_f1": round(float(f1_score(y, predictions, average="macro", zero_division=0) * 100), 2),
        "roc_auc": round(float(roc_auc_score(y, scores) * 100), 2),
        "average_precision": round(float(average_precision_score(y, scores) * 100), 2),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"ham": int((y == 0).sum()), "spam": int((y == 1).sum())},
    }


def score_scenarios(texts: list[str], model_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the shared scoring function to every content and metadata scenario."""
    scenarios = {
        "content_only": ("friend", False),
        "default_unknown_sender": ("unknown", False),
        "forwarded_known_sender": ("friend", True),
        "forwarded_unknown_sender": ("unknown", True),
    }
    return {
        name: np.asarray(
            [
                score_message(
                    text,
                    source=source,
                    is_forwarded=forwarded,
                    tfidf_score=float(model_score),
                ).spam_score
                for text, model_score in zip(texts, model_scores)
            ]
        )
        for name, (source, forwarded) in scenarios.items()
    }


def evaluate_grouped_folds(
    dataset: pd.DataFrame,
    pipelines: dict,
    seed: int,
) -> dict:
    """Run nested model selection and evaluation on each of five grouped outer folds."""
    texts = dataset["clean_text"].to_numpy()
    raw_texts = dataset["text"].to_numpy()
    labels = dataset["label_binary"].to_numpy()
    groups = dataset["clean_text"].to_numpy()
    outer_splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    fold_reports = []
    scenario_fold_metrics: dict[str, list[dict]] = {}
    held_out_indices = []

    for fold_number, (train_idx, test_idx) in enumerate(
        outer_splitter.split(texts, labels, groups), start=1
    ):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        group_overlap = len(train_groups.intersection(test_groups))
        if group_overlap:
            raise ValueError(f"Outer fold {fold_number} has {group_overlap} train/test groups in common.")

        x_train, y_train, g_train = texts[train_idx], labels[train_idx], groups[train_idx]
        x_test, y_test = texts[test_idx], labels[test_idx]
        inner_splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
        model_cv = {}

        for name, template in pipelines.items():
            fold_f1 = []
            for inner_train_idx, validation_idx in inner_splitter.split(x_train, y_train, g_train):
                pipeline = clone(template)
                pipeline.fit(x_train[inner_train_idx], y_train[inner_train_idx])
                scores = pipeline.predict_proba(x_train[validation_idx])[:, 1] * 100
                fold_f1.append(
                    float(
                        f1_score(
                            y_train[validation_idx],
                            scores > SPAM_THRESHOLD,
                            zero_division=0,
                        )
                        * 100
                    )
                )
            model_cv[name] = {
                "fold_spam_f1": [round(value, 2) for value in fold_f1],
                "mean_spam_f1": round(float(np.mean(fold_f1)), 2),
                "std_spam_f1": round(float(np.std(fold_f1)), 2),
            }

        selected = max(model_cv, key=lambda name: model_cv[name]["mean_spam_f1"])
        model = clone(pipelines[selected]).fit(x_train, y_train)
        model_scores = model.predict_proba(x_test)[:, 1] * 100
        scores_by_scenario = score_scenarios(raw_texts[test_idx].tolist(), model_scores)
        metrics_by_scenario = {
            scenario: classification_metrics(y_test, scores)
            for scenario, scores in scores_by_scenario.items()
        }
        for scenario, metrics in metrics_by_scenario.items():
            scenario_fold_metrics.setdefault(scenario, []).append(metrics)

        held_out_indices.extend(test_idx.tolist())
        fold_reports.append(
            {
                "fold": fold_number,
                "split": {
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "train_unique_groups": len(train_groups),
                    "test_unique_groups": len(test_groups),
                    "train_test_group_overlap": group_overlap,
                },
                "model_selection": {
                    "metric": "mean spam-class F1 at score > 50",
                    "models": model_cv,
                    "selected": selected,
                },
                "outer_test": {
                    "content_only": metrics_by_scenario["content_only"],
                    "metadata_scenarios": {
                        name: metrics
                        for name, metrics in metrics_by_scenario.items()
                        if name != "content_only"
                    },
                },
            }
        )

    if len(held_out_indices) != len(dataset) or len(set(held_out_indices)) != len(dataset):
        raise ValueError("Outer folds did not hold out every row exactly once.")

    aggregate = {
        scenario: {
            metric: {
                "mean": round(float(np.mean([fold[metric] for fold in fold_metrics])), 2),
                "sd": round(float(np.std([fold[metric] for fold in fold_metrics])), 2),
            }
            for metric in AGGREGATE_METRICS
        }
        for scenario, fold_metrics in scenario_fold_metrics.items()
    }
    return {
        "split": {
            "method": f"{OUTER_FOLDS}-fold StratifiedGroupKFold",
            "seed": seed,
            "inner_cv": f"{INNER_FOLDS}-fold StratifiedGroupKFold per outer training fold",
        },
        "folds": fold_reports,
        "aggregate_outer_test": {
            "summary": "mean and population standard deviation (sd) across outer folds",
            "scenarios": aggregate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "grouped_baseline.json")
    args = parser.parse_args()

    dataset_path = args.data or config.data_path()
    dataset = load_dataset(dataset_path)
    group_labels = dataset.groupby("clean_text", sort=False)["label_binary"].nunique()
    conflicts = int((group_labels > 1).sum())
    if conflicts:
        raise ValueError(f"Found {conflicts} normalized text groups with conflicting labels.")

    evaluation = evaluate_grouped_folds(dataset, make_pipelines(), config.SEED)
    clean_text_counts = dataset["clean_text"].value_counts()
    report = {
        "experiment": "nested_grouped_exact_text_five_fold",
        "dataset": {
            "file": dataset_path.name,
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "rows": int(len(dataset)),
            "unique_preprocessed_texts": int(len(clean_text_counts)),
            "repeated_text_groups": int((clean_text_counts > 1).sum()),
            "extra_duplicate_rows": int((clean_text_counts - 1).clip(lower=0).sum()),
            "conflicting_label_groups": conflicts,
        },
        **evaluation,
        "metadata_note": "Counterfactual only; the dataset has no sender or forwarded labels.",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for fold in report["folds"]:
        content = fold["outer_test"]["content_only"]
        print(
            f"Fold {fold['fold']}: selected {fold['model_selection']['selected']}; "
            f"F1 {content['f1']}%, precision {content['precision']}%, "
            f"recall {content['recall']}%, group overlap "
            f"{fold['split']['train_test_group_overlap']}"
        )
    print("Outer test mean ± sd:")
    for scenario, metrics in report["aggregate_outer_test"]["scenarios"].items():
        values = metrics["f1"]
        print(f"  {scenario}: F1 {values['mean']:.2f} ± {values['sd']:.2f}%")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
