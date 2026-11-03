#!/usr/bin/env python
"""Evaluate every model on the SAME held-out test split and write results/.

    python scripts/evaluate.py

Models: the three classical pipelines (if trained), DistilBERT (if trained) and the
TF-IDF + DistilBERT ensemble exactly as the API computes it (spamdetector.scoring.combine_scores,
default weights 0.65 / 0.35, threshold 50). The API's metadata bonuses are not applied because the
dataset has no forwarded/sender fields.

Reported per model: accuracy, precision, recall, F1 (threshold 0.5), ROC-AUC, PR-AUC, and a 95 %
bootstrap interval for F1 (the test set has only 149 spam messages, so single numbers are noisy).
A weight sweep for the ensemble is a SENSITIVITY analysis on the test set, not a tuning step:
the shipped weight stays at its a-priori value.

Writes: results/metrics.json, results/metrics.md, and four PNG figures.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spamdetector import config  # noqa: E402
from spamdetector.data import load_dataset, split_dataset  # noqa: E402
from spamdetector.preprocess import preprocess  # noqa: E402
from spamdetector.scoring import DEFAULT_TFIDF_WEIGHT, SPAM_THRESHOLD, combine_scores  # noqa: E402

# Validated categorical slots 1-3 (blue, orange, aqua) from the dataviz reference palette.
C_TFIDF, C_BERT, C_ENS = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def metrics_at_threshold(y, score, threshold=SPAM_THRESHOLD) -> dict:
    # Strict ">" matches scikit-learn's predict(): a score of exactly 50 (common for a Random Forest
    # whose trees split 50/50) is called ham. This keeps the numbers identical to the notebook.
    pred = (score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred) * 100,
        "precision": precision_score(y, pred, zero_division=0) * 100,
        "recall": recall_score(y, pred, zero_division=0) * 100,
        "f1": f1_score(y, pred, zero_division=0) * 100,
        "roc_auc": roc_auc_score(y, score) * 100,
        "pr_auc": average_precision_score(y, score) * 100,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_f1(y, score, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pred = (score > SPAM_THRESHOLD).astype(int)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals.append(f1_score(y[s], pred[s], zero_division=0) * 100)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_scores(split) -> dict[str, np.ndarray]:
    """Spam scores (0-100) on the test set for every model that exists."""
    scores: dict[str, np.ndarray] = {}
    for name, fname in [("Naive Bayes", "naive_bayes"), ("Logistic Regression", "logistic_regression"), ("Random Forest", "random_forest")]:
        p = config.classical_dir() / f"{fname}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                scores[name] = pickle.load(f).predict_proba(split.X_test.tolist())[:, 1] * 100
    if config.distilbert_dir().joinpath("config.json").exists():
        try:
            from spamdetector.inference import TransformerScorer

            scores["DistilBERT"] = TransformerScorer.load().score_batch(split.X_test_raw.tolist())
        except ImportError:
            print("torch/transformers not installed: skipping DistilBERT (pip install -r requirements-transformer.txt)")
    # The ensemble uses whichever classical model is deployed (models/pipeline.pkl).
    if config.pipeline_path().exists() and "DistilBERT" in scores:
        with open(config.pipeline_path(), "rb") as f:
            deployed = pickle.load(f)
        tf = deployed.predict_proba(split.X_test.tolist())[:, 1] * 100
        scores["Ensemble"] = combine_scores(tf, scores["DistilBERT"], DEFAULT_TFIDF_WEIGHT)
        scores["_deployed_tfidf"] = tf
    return scores


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def color_for(name: str) -> str:
    return {"DistilBERT": C_BERT, "Ensemble": C_ENS}.get(name, C_TFIDF)


def make_figures(y, scores, table, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.labelcolor": INK2, "text.color": INK})
    names = [n for n in table if not n.startswith("_")]
    focus = [n for n in ("Random Forest", "DistilBERT", "Ensemble") if n in scores]

    # 1) grouped bars: precision / recall / F1
    fig, ax = plt.subplots(figsize=(9, 4.2))
    style(ax)
    keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        xs = np.arange(len(keys)) + (i - (len(names) - 1) / 2) * width
        vals = [table[n][k] for k, _ in keys]
        ax.bar(xs, vals, width * 0.88, color=color_for(n), alpha=1.0 if n in focus else 0.45, label=n,
               hatch="///" if n == "Logistic Regression" else None, edgecolor=SURFACE, linewidth=0)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(keys)), [l for _, l in keys])
    ax.set_ylim(60, 104)
    ax.set_ylabel("Score (%) on the 1,115-message test split")
    ax.set_title("Spam class: precision, recall and F1 (y-axis starts at 60 %)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    # 2) precision-recall curves
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for n in focus:
        p, r, _ = precision_recall_curve(y, scores[n])
        ax.plot(r, p, color=color_for(n), linewidth=2, label=f"{n}  (PR-AUC {table[n]['pr_auc']:.1f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Precision-recall (spam class)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    plt.close(fig)

    # 3) calibration
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.62, 0.55, "perfectly calibrated", color=INK2, fontsize=8, rotation=38)
    for n in [m for m in focus if m != "Ensemble"]:
        frac, mean = calibration_curve(y, scores[n] / 100, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, color=color_for(n), linewidth=2, marker="o", markersize=5, label=n)
    ax.set_xlabel("Predicted spam probability")
    ax.set_ylabel("Observed share of spam")
    ax.set_title("Calibration (quantile bins)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150)
    plt.close(fig)

    # 4) confusion matrices
    fig, axes = plt.subplots(1, len(focus), figsize=(3.6 * len(focus), 3.6))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, focus):
        t = table[n]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["Ham", "Spam"])
        ax.set_yticks([0, 1], ["Ham", "Spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(n, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for the F1 interval (0 = skip)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    split = split_dataset(load_dataset(args.data))
    y = split.y_test.to_numpy()
    scores = load_scores(split)
    names = [n for n in scores if not n.startswith("_")]
    if not names:
        sys.exit("No trained models found. Run scripts/train_classical.py (and scripts/train_distilbert.py) first.")

    table = {n: metrics_at_threshold(y, scores[n]) for n in names}
    if args.bootstrap:
        for n in names:
            lo, hi = bootstrap_f1(y, scores[n], args.bootstrap, config.SEED)
            table[n]["f1_ci95"] = [round(lo, 2), round(hi, 2)]

    sweep = {}
    if "Ensemble" in scores:
        for w in (0.0, 0.25, 0.5, DEFAULT_TFIDF_WEIGHT, 0.75, 1.0):
            s = combine_scores(scores["_deployed_tfidf"], scores["DistilBERT"], w)
            m = metrics_at_threshold(y, s)
            sweep[f"{w:.2f}"] = {k: round(m[k], 2) for k in ("precision", "recall", "f1", "roc_auc")}

    out = config.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    rounded = {n: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()} for n, m in table.items()}
    (out / "metrics.json").write_text(json.dumps({
        "test_size": int(len(y)), "spam_in_test": int(y.sum()), "threshold": SPAM_THRESHOLD,
        "ensemble_tfidf_weight": DEFAULT_TFIDF_WEIGHT, "models": rounded,
        "ensemble_weight_sensitivity_on_test": sweep,
    }, indent=2) + "\n")

    lines = ["| Model | Accuracy | Precision | Recall | F1 | F1 95% CI | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        m = rounded[n]
        ci = f"{m['f1_ci95'][0]:.1f}-{m['f1_ci95'][1]:.1f}" if "f1_ci95" in m else "-"
        lines.append(f"| {n} | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {ci} | {m['roc_auc']:.2f} | {m['pr_auc']:.2f} |")
    md = "\n".join(lines)
    (out / "metrics.md").write_text(md + "\n")
    print(f"\nTest split: {len(y)} messages ({int(y.sum())} spam)\n\n{md}")
    if sweep:
        print("\nEnsemble TF-IDF weight sensitivity (sensitivity only, not used for tuning):")
        for w, m in sweep.items():
            print(f"  w_tfidf={w}: precision={m['precision']} recall={m['recall']} F1={m['f1']} ROC-AUC={m['roc_auc']}")

    if not args.no_figures:
        make_figures(y, scores, table, out)
        print(f"\nFigures and metrics written to {out}")


if __name__ == "__main__":
    main()
