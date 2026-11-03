#!/usr/bin/env python
"""Train separate CPU-friendly veracity and clickbait text classifiers.

    python scripts/train_article_models.py

Uses publisher-held-out evaluation where publisher groups can be identified. No article
text is copied into the repository or model reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spamdetector import config  # noqa: E402

SEED = config.SEED
VERACITY_LABELS = ("mostly true", "mixture of true and false", "mostly false", "no factual content")
CLICKBAIT_LABELS = (
    "not clickbait",
    "slightly clickbait",
    "considerably clickbait",
    "heavily clickbait",
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower().replace("_", "-")


def load_veracity_records(root: Path) -> tuple[list[str], list[str], list[str]]:
    data_dir = root / "buzzfeed_webis"
    metadata_path = data_dir / "overview.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}; run scripts/download_article_datasets.py first.")

    texts, labels, groups = [], [], []
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = (row.get("veracity") or "").strip().lower()
            if label not in VERACITY_LABELS:
                continue
            xml_path = data_dir / "articles" / (row.get("XML") or "")
            if not xml_path.is_file():
                raise FileNotFoundError(f"Missing article XML file: {xml_path}")
            title, body = _read_article(xml_path)
            if not title and not body:
                continue
            texts.append(f"{title}\n\n{body}".strip())
            labels.append(label)
            groups.append((row.get("portal") or "").strip().lower() or "unknown")
    if len(set(labels)) < 2:
        raise ValueError("The BuzzFeed-Webis dataset did not provide at least two usable veracity labels.")
    return texts, labels, groups


def _read_article(path: Path) -> tuple[str, str]:
    root = ET.parse(path).getroot()
    title = ""
    paragraphs = []
    for element in root.iter():
        name = _local_name(element.tag)
        text = " ".join(part.strip() for part in element.itertext() if part.strip())
        if not text:
            continue
        if not title and name in {"title", "headline", "article-title"}:
            title = text
        elif name in {"p", "paragraph"} and text not in paragraphs:
            paragraphs.append(text)
    if paragraphs:
        return title, " ".join(paragraphs)
    all_text = " ".join(part.strip() for part in root.itertext() if part.strip())
    return title, all_text


def load_clickbait_records(root: Path, split: str = "train") -> tuple[list[str], list[str], list[str]]:
    data_dir = root / "clickbait17" / split
    instance_files = sorted(data_dir.rglob("instances.jsonl"))
    truth_files = sorted(data_dir.rglob("truth.jsonl"))
    if len(instance_files) != 1 or len(truth_files) != 1:
        raise FileNotFoundError(
            f"Expected one instances.jsonl and truth.jsonl under {data_dir}; "
            "run scripts/download_article_datasets.py first."
        )

    instances = _read_jsonl_by_id(instance_files[0])
    truth = _read_jsonl_by_id(truth_files[0])
    shared_ids = instances.keys() & truth.keys()
    if not shared_ids or len(shared_ids) < min(len(instances), len(truth)) * 0.99:
        raise ValueError(
            f"Clickbait {split} instance/annotation IDs do not align: "
            f"{len(instances)} instances, {len(truth)} annotations, {len(shared_ids)} matches."
        )

    texts, labels, groups = [], [], []
    for identifier in sorted(shared_ids):
        record = instances[identifier]
        annotation = truth[identifier]
        headline = record.get("postText", "")
        if isinstance(headline, list):
            headline = " ".join(str(part) for part in headline)
        headline = str(headline).strip()
        if not headline:
            continue
        mean = annotation.get("truthMean")
        if mean is None:
            judgments = annotation.get("truthJudgments", [])
            mean = float(np.mean(judgments)) if judgments else None
        if mean is None:
            continue
        score = float(mean)
        if not 0 <= score <= 1:
            raise ValueError(f"Clickbait truthMean outside [0, 1] for item {identifier}: {score}")
        category = min(range(4), key=lambda index: abs(score - index / 3))
        texts.append(headline)
        labels.append(CLICKBAIT_LABELS[category])
        groups.append("")
    if len(set(labels)) < 2:
        raise ValueError(f"The Clickbait-17 {split} data did not provide at least two clickbait severity levels.")
    return texts, labels, groups


def _read_jsonl_by_id(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            identifier = str(record.get("id", "")).strip()
            if not identifier or identifier in records:
                raise ValueError(f"Missing or duplicate Clickbait-17 item ID at {path}:{line_number}")
            records[identifier] = record
    return records


def make_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=100_000, sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=600, class_weight="balanced", random_state=SEED)),
        ]
    )


def split_indices(labels: list[str], groups: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    x = np.arange(len(labels))
    classes = set(labels)
    unique_groups = set(groups)
    if len(unique_groups) >= 3 and all(group for group in groups):
        candidates = GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=SEED).split(x, labels, groups)
        valid = []
        for train_idx, test_idx in candidates:
            if set(np.asarray(labels)[train_idx]) == classes:
                ratio = abs(len(test_idx) / len(labels) - 0.2)
                valid.append((ratio, train_idx, test_idx))
        if valid:
            _, train_idx, test_idx = min(valid, key=lambda item: item[0])
            return train_idx, test_idx, "publisher_group_holdout"
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(split.split(x, labels))
    return train_idx, test_idx, "stratified_random_holdout"


def train_task(
    name: str,
    texts: list[str],
    labels: list[str],
    groups: list[str],
    output_dir: Path,
    eval_data: tuple[list[str], list[str], list[str]] | None = None,
) -> dict:
    x = np.asarray(texts, dtype=object)
    y = np.asarray(labels, dtype=object)
    if eval_data is None:
        train_idx, test_idx, split_method = split_indices(labels, groups)
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]
        test_groups = np.asarray(groups)[test_idx]
        all_labels = sorted(set(labels))
    else:
        train_x, train_y = x, y
        test_x, test_y, test_groups = (np.asarray(values, dtype=object) for values in eval_data)
        split_method = "official_validation_split"
        all_labels = sorted(set(labels) | set(test_y))
    pipeline = make_pipeline()
    started = time.monotonic()
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    baseline = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    baseline_predictions = baseline.predict(test_x)
    report = classification_report(
        test_y,
        predictions,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    metadata = {
        "task": name,
        "model": "TF-IDF word/character n-grams + class-weighted Logistic Regression",
        "seed": SEED,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_manifest_sha256": _manifest_sha256(config.article_data_dir() / "MANIFEST.json"),
        "model_sha256": _file_sha256(model_path),
        "split_method": split_method,
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "publisher_groups": len({group for group in groups if group}),
        "held_out_publishers": sorted({group for group in test_groups if group}),
        "test_accuracy": round(float(accuracy_score(test_y, predictions)), 4),
        "test_macro_f1": round(float(f1_score(test_y, predictions, average="macro")), 4),
        "majority_baseline_accuracy": round(float(accuracy_score(test_y, baseline_predictions)), 4),
        "majority_baseline_macro_f1": round(
            float(f1_score(test_y, baseline_predictions, average="macro")), 4
        ),
        "classification_report": report,
        "train_label_counts": dict(Counter(train_y)),
        "test_label_counts": dict(Counter(test_y)),
        "training_seconds": round(time.monotonic() - started, 2),
    }
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"{name}: {len(train_y)} train / {len(test_y)} test; "
        f"accuracy={metadata['test_accuracy']:.3f}; macro-F1={metadata['test_macro_f1']:.3f}; "
        f"split={split_method}; {metadata['training_seconds']:.1f}s"
    )
    return metadata


def _manifest_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("all", "article_veracity", "clickbait"), default="all")
    args = parser.parse_args()
    root = config.article_data_dir()
    models_dir = config.article_models_dir()
    results_dir = config.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "article_task_training.json"
    reports = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if args.task in ("all", "article_veracity"):
        veracity = load_veracity_records(root)
        reports["article_veracity"] = train_task(
            "article_veracity", *veracity, models_dir / "article_veracity"
        )
    if args.task in ("all", "clickbait"):
        clickbait_train = load_clickbait_records(root, "train")
        clickbait_validation = load_clickbait_records(root, "validation")
        reports["clickbait"] = train_task(
            "clickbait",
            *clickbait_train,
            models_dir / "clickbait",
            eval_data=clickbait_validation,
        )
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
