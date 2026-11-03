"""Dataset loading and the train/test split, defined once for every script.

Why 5572 rows and not 5574?
---------------------------
The UCI SMS Spam Collection file has 5574 lines (4827 ham / 747 spam). Some messages start
with a double quote that is never closed on the same line, so pandas' default CSV parser
(which treats ``"`` as a quote character) glues them to following lines and the result has
5572 rows (4825 ham / 747 spam): two ham messages are silently lost. The original notebook used that default parser. The current classical baseline also retains these
rows and the same seeded split; its measurements are recorded in
``results/classical_training.json``. Earlier notebook metrics differ and are not used as the
current test expectations. We keep the default parser so the dataset and split remain
comparable to the original experiment.
(Passing ``quoting=csv.QUOTE_NONE`` would load all 5574 messages but give a different split.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .preprocess import preprocess


@dataclass(frozen=True)
class Split:
    """The 80/20 stratified split. ``*_raw`` is the original text (used by DistilBERT);
    ``X_train`` / ``X_test`` are the preprocessed text (used by the TF-IDF models)."""

    X_train_raw: pd.Series
    X_test_raw: pd.Series
    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: Path | str | None = None, *, validate: bool = True) -> pd.DataFrame:
    path = Path(path) if path else config.data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    # NOTE: default quoting on purpose, see the module docstring.
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    if validate:
        n_ham = int((df["label"] == "ham").sum())
        n_spam = int((df["label"] == "spam").sum())
        if (len(df), n_ham, n_spam) != (config.EXPECTED_ROWS, config.EXPECTED_HAM, config.EXPECTED_SPAM):
            raise ValueError(
                f"Unexpected dataset: {len(df)} rows ({n_ham} ham / {n_spam} spam); expected "
                f"{config.EXPECTED_ROWS} ({config.EXPECTED_HAM} / {config.EXPECTED_SPAM}). "
                "Re-download with `python scripts/download_data.py --force`."
            )
    df["clean_text"] = df["text"].apply(preprocess)
    df["label_binary"] = (df["label"] == "spam").astype(int)
    return df


def split_dataset(df: pd.DataFrame) -> Split:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"],
        df["label_binary"],
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df["label_binary"],
    )
    return Split(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=df["clean_text"].loc[X_train_raw.index],
        X_test=df["clean_text"].loc[X_test_raw.index],
        y_train=y_train,
        y_test=y_test,
    )
