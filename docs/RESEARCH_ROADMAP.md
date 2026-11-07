# Research roadmap for stronger findings

The question is not whether more data can be added; it is whether appropriately licensed,
representative data improves performance on a frozen target-domain test set. Mixing datasets can
help coverage, leave results unchanged, or make them worse. The current message model is trained
on the English SMS Spam Collection and has not been evaluated on real WhatsApp messages.

## 1. Keep tasks and claims separate

- Message spam/phishing classification and article veracity/clickbait are distinct tasks. Do not
  combine their labels or report them as one "misinformation detector."
- Define the intended message labels before collecting more data (for example, unsolicited
  marketing, scam/phishing, and legitimate). A message can be suspicious without being provably
  malicious; preserve ambiguous cases rather than forcing unreliable labels.
- Keep current-news verification out of scope unless the project adds a source-backed evidence
  retrieval and claim-checking evaluation.

## 2. Vet candidate datasets before training

For each candidate corpus, record the original source, license/terms, language, date range,
platform, label definitions, collection method, privacy constraints, and checksum. Do not assume a
dataset is reusable just because it is downloadable.

Message datasets from SMS, email, web comments, and social platforms may provide useful additional
language, but they have different conventions and class priors. Keep a `source_dataset` field and
measure each source separately. Email corpora such as Enron or SpamAssassin should be treated as
cross-domain experiments, not as WhatsApp-like data. Article datasets must remain in the separate
article pipelines.

Never put private messages, phone numbers, names, account identifiers, or live user-submitted text
into the repository. Any future user-data collection needs explicit informed consent, a retention
and deletion policy, redaction, and a documented annotation protocol.

## 3. Add data incrementally and prevent leakage

1. Reproduce the current SMS-only baseline from the pinned numerical stack.
2. Add one rights-cleared dataset at a time; normalize only labels with a defensible mapping.
3. Deduplicate exact and near-duplicate texts across every source before splitting.
4. Keep test examples grouped by source, sender/campaign where permitted, and time. Do not let
   copies or templates appear on both sides of a split.
5. Select models and thresholds using training/validation data only. Freeze the final test set
   before tuning.

A random split of a merged corpus can reward source-specific wording and duplicate templates,
rather than generalizable scam detection.

## 4. Evaluate the intended use, not just a convenient proxy

Keep the existing stratified SMS result as a reproducible baseline, then report:

- per-class precision, recall, F1, support, and confusion matrices;
- macro-F1 and precision-recall curves/AUPRC for imbalanced data;
- false-positive rate and operating-threshold tradeoffs, with thresholds selected on validation
  data;
- uncertainty intervals and a majority/keyword/simple-model baseline;
- per-dataset, language, time, and channel results; and
- an untouched, recent, independently labeled target-domain set before claiming WhatsApp-like
  usefulness.

If a target-domain test set cannot be collected with consent and privacy safeguards, state that
generalization to real user prompts is unknown. A larger training corpus alone cannot establish
real-world performance. Treat model scores as uncalibrated unless calibration is separately
measured and reported.

## 5. Regenerate the report from evidence

For every published number, link the exact dataset version/checksum, split, code revision,
environment, command, and saved evaluation output. Separate measured results from proposed work.
Do not retain earlier report claims about unimplemented datasets, transformer runs, WhatsApp
integration, or high hybrid accuracy unless those experiments are independently reproduced with
valid held-out evaluation.

The candidate-source audit and current exact-text leakage measurement are recorded in
[`DATASET_REVIEW.md`](DATASET_REVIEW.md). Run `python scripts/evaluate_grouped.py` for a separate
nested, exact-text-grouped SMS baseline; do not compare it as if it were the same split as the
historical random-split result.
