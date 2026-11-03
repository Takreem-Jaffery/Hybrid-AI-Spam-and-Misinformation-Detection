#!/usr/bin/env python
"""Fine-tune DistilBERT for spam / ham (notebook cells 5-6 as a script).

    pip install -r requirements-transformer.txt
    python scripts/train_distilbert.py              # ~8 min on a T4 GPU, hours on CPU

Same recipe as the notebook: distilbert-base-uncased, 6 epochs, batch 32, AdamW lr 2e-5,
weight decay 0.01, 15 % linear warm-up, max length 128, class-weighted cross-entropy
(weight = n_total / (2 * n_class), i.e. ham 0.577 / spam 3.727), gradient clipping at 1.0.
The model after the LAST epoch is saved (no checkpoint is picked using the test set).

Changes from the notebook
    * all random seeds are set (python, numpy, torch, DataLoader shuffling)
    * the loss was computed twice (once inside the model because ``labels=`` was passed, once
      weighted); the unweighted one is gone
    * after saving, the model is reloaded and its logits compared with the in-memory model,
      which catches the "untrained classification head was saved" bug of the first version
    * the saved model must pass the same startup sanity check the API uses
    * the model AND tokenizer go in one folder: models/distilbert/

GPU runs are not bit-for-bit deterministic; expect roughly +/-0.5 points between runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spamdetector import config  # noqa: E402
from spamdetector.data import load_dataset, split_dataset  # noqa: E402


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="distilbert-base-uncased", help="HF model id or a local folder")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.15)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--output", type=Path, default=None, help="default: models/distilbert")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None, help="debug/smoke tests only")
    ap.add_argument("--no-sanity-check", action="store_true", help="smoke tests only: skip the behavioural check")
    args = ap.parse_args()

    import torch
    import transformers
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from spamdetector.inference import MAX_LEN, TransformerScorer, run_sanity_check

    if args.max_len != MAX_LEN:
        print(f"WARNING: --max-len {args.max_len} differs from the {MAX_LEN} the API uses at inference time.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}  transformers={transformers.__version__}")
    if device == "cpu":
        print("WARNING: no GPU found. Full training on CPU takes hours. Use Colab (Runtime > T4 GPU) or --max-train-samples to try it out.")
    set_seed(args.seed)

    split = split_dataset(load_dataset(args.data))
    X, y = split.X_train_raw.tolist(), split.y_train.tolist()  # RAW text for DistilBERT
    if args.max_train_samples:
        X, y = X[: args.max_train_samples], y[: args.max_train_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    # The "MISSING classifier.*" warning above is expected: the 2-class head starts random and is trained below.

    labels = np.array(y)
    n_total, n_ham, n_spam = len(labels), int((labels == 0).sum()), int((labels == 1).sum())
    weights = torch.tensor([n_total / (2 * n_ham), n_total / (2 * n_spam)], dtype=torch.float, device=device)
    print(f"class weights: ham={weights[0]:.3f} spam={weights[1]:.3f}")

    def encode(texts: list[str]):
        return tokenizer(texts, max_length=args.max_len, padding="max_length", truncation=True, return_tensors="pt")

    enc = encode(X)
    train_ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y, dtype=torch.long))
    gen = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)

    total_steps = len(train_dl) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_frac), num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print(f"training {args.epochs} epochs, {len(train_dl)} batches/epoch, {total_steps} steps")
    epoch_losses: list[float] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for input_ids, attn, labels_b in train_dl:
            logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
            loss = loss_fn(logits, labels_b.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
        epoch_losses.append(running / len(train_dl))
        print(f"  epoch {epoch + 1}/{args.epochs}  avg_loss={epoch_losses[-1]:.4f}")
    print(f"training took {time.time() - t0:.0f}s")

    # ---- save, then prove the saved copy behaves like the trained one -------------------------
    out = args.output or config.distilbert_dir()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    scorer = TransformerScorer.load(out, device=device)
    probe = ["Congratulations! You won a free prize. Click now to claim!", "Hey, are we still meeting at 6?", "URGENT: verify your bank account now"]
    model.eval()
    with torch.inference_mode():
        ref = model(**tokenizer(probe, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(device)).logits
        got = scorer.model(**tokenizer(probe, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(device)).logits
    if not torch.allclose(ref, got, atol=1e-4):
        raise RuntimeError("Reloaded model disagrees with the trained model: the classification head did not save correctly.")
    print("save check: reloaded model matches the trained model")

    ok, spam_p, ham_p = run_sanity_check(scorer)
    print(f"sanity check: obvious spam -> {spam_p:.0%} spam, obvious ham -> {ham_p:.0%} spam  ({'PASS' if ok else 'FAIL'})")
    if not ok and not args.no_sanity_check:
        raise RuntimeError("The trained model failed the behavioural sanity check; not a usable model.")

    (out / "training_info.json").write_text(json.dumps({
        "base_model": args.model_name, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "warmup_frac": args.warmup_frac, "max_len": args.max_len, "seed": args.seed,
        "train_samples": len(X), "epoch_avg_loss": [round(v, 4) for v in epoch_losses],
        "class_weights": {"ham": round(float(weights[0]), 3), "spam": round(float(weights[1]), 3)},
        "torch": torch.__version__, "transformers": transformers.__version__, "device": device,
    }, indent=2) + "\n")
    print(f"Saved model + tokenizer to {out}\nNext: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
