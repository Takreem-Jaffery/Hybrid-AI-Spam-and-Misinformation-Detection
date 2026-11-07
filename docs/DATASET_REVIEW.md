# Message dataset review

Reviewed 6 November 2026 for a reproducible, public project intended to analyze suspicious
messages, with eventual WhatsApp-like use. A recent upload date is not evidence that a corpus is
recently collected, representative, or safe to train on. A dataset is not admitted to training
until its source, collection/consent process, label instructions, privacy handling, license, and
deduplication/split method can be checked.

## Recommendation

**Do not merge any newly found corpus into the training data yet.** The broader multilingual
search found candidate language coverage, but no new corpus currently meets the provenance,
consent/privacy, label, and reuse requirements for training or a defensible held-out benchmark:

- The Cameroon corpus claims French, English, Pidgin, and local-dialect SMS and includes an Excel
  file. Its public README does not explain message sourcing, consent, label annotation, split
  construction, or a license. It is the strongest candidate to ask its authors about, not data to
  merge or redistribute yet.
- The Bengali corpus covers Bengali, English, Banglish, and code-mixed SMS, but its public card
  does not explain where messages came from, whether they are synthetic, how they were annotated,
  or how duplicate campaigns were handled. Treat it as unverified, not as ground-truth training
  data.
- A 21-language SMS corpus is explicitly machine-translated from the UCI collection, declares
  GPL, and does not provide independent real-world messages. Its translated copies can leak the
  same original messages across splits and cannot establish cross-language real-world accuracy.
  At most, consider it for a separately labeled translation/tokenizer smoke test after license
  compatibility review.
- The Farsi project describes collection of real Farsi SMS but says those data are not included
  because licensing/publication review is pending. Its synthetic examples are not real-message
  evaluation data; wait for an authorized release.
- MOZ-Smishing is described as crowd-sourced real messages and has a test file, but examples expose
  transaction/account-like details and OTPs. Its card does not document consent, redaction, or
  collection methodology; it declares the restrictive CreativeML OpenRAIL-M license. Do not copy
  it into the repository or use it for training without resolving those issues.
- The Korean corpus is marked CC BY-NC-SA 4.0, has no disclosed original source/consent process or
  fixed evaluation split, and is unsuitable for a broadly reusable/commercially adaptable model.
- The India SMS and OTP datasets have no adequate independent provenance for this project's
  evaluation. The OTP corpus explicitly says it is synthetic; it can test code paths, not establish
  real-world accuracy.

The existing UCI SMS Spam Collection remains the reproducible baseline, not a modern target-domain
benchmark. It is English, drawn from older UK/Singapore sources, and its original labels combine
different kinds of unwanted messages.

## Candidate comparison

| Dataset | Recency and scope | Labels / splits | License and provenance review | Decision |
|---|---|---|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Collected from older sources; English SMS, 5,574 source lines | ham/spam; project uses a stratified random split | UCI documents constituent sources and volunteer notice for one ham source; CC BY 4.0 | Keep as the historical baseline only |
| [SMS Spam Multilingual Collection](https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset) | Hugging Face card lists 21 languages; created Jan 2023; all entries are machine translations of UCI SMS examples | ham/spam inherited from the source; no independent source/time holdout | Card declares GPL and describes translations using M2M100; not independent samples and creates direct source-message leakage risk | Do not merge as real multilingual data; optional language-pipeline smoke test only after license review |
| [Cameroonian SMS spam repository](https://github.com/loicyoumbi9/Detection-of-SMS-SPAM-IN-CAMEROONIAN-CONTEXT) | Repository created May 2025; claims French, English, Pidgin, and local-dialect SMS; Excel corpus is present | Model notebooks are present; the README does not document a reusable benchmark split or operational labels | Repository has no declared license; public README gives no sourcing/consent, redaction, annotation, or collection-date details | Best lead for author follow-up; do not download, train, or redistribute until permission and methods are documented |
| [Bengali SMS Smishing Dataset](https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset), associated with [Islam's 2026 thesis](https://doi.org/10.60867/00000158) | Repository created Dec 2025, updated Jun 2026; 7,005 Bengali, English, Banglish, and code-mixed SMS | smish/promo/normal; provided train/validation/test files, 4,903/701/1,401; card says stratified random seed 42 | Card declares MIT and split counts, but does not document message origins, consent, synthetic-vs-real status, annotation protocol, or campaign-level deduplication | Candidate for exploratory domain testing only after author clarifies provenance and split leakage; not for training or published performance claims yet |
| [English/Farsi SMS spam research repository](https://github.com/Hosein-Ghojavand/multilingual-spam-detection) | Repository created Jun 2024; work describes English and Farsi data and synthetic augmentation | Separate English/Farsi original, synthetic, and combined experiments described | GitHub declares no license; README states collected real Farsi data are excluded pending university licensing/publication review; synthetic data may be shared on request | Do not use current contents as a licensed Farsi corpus; revisit only after an authorized release and methodology review |
| [MOZ-Smishing](https://huggingface.co/datasets/MOZNLP/MOZ-Smishing) | Repository created May 2025; 2,561 Portuguese messages described as crowd-sourced in Mozambique, focused on mobile-money scams | legitimate/smishing; repository contains `test.csv` only | Card declares CreativeML OpenRAIL-M but provides no study citation or visible consent/redaction details; sample messages include transaction identifiers and OTP-like values | Potential separate cross-language challenge set only after license, consent, and privacy review; do not redistribute |
| [Korean Smishing Message Dataset](https://huggingface.co/datasets/jmjmjm3/kor-smishing-message) | Repository created Jan 2026; Korean normal/smishing messages | normal/smishing plus rationale/type; no fixed split stated | CC BY-NC-SA 4.0; original source, consent, and annotation process are not documented on the card | Do not merge; non-commercial/share-alike terms and unclear provenance |
| [India Spam SMS](https://huggingface.co/datasets/CloveAI/india-spam-sms) | Repository created Sep 2025; 20,010 rows | integer labels; one train split only | Declares MIT, but the card has no source, label mapping, collection process, or test split | Reject pending provenance and label documentation |
| [SMS OTP Spam Dataset](https://huggingface.co/datasets/alusci/sms-otp-spam-dataset) | Repository created May 2025; 10,000 synthetic English OTP-style messages | valid/not-valid; synthetic timestamps and phone IDs | Declares MIT and explicitly says synthetic; labels include delivery-status values, not a clear scam taxonomy | Use, at most, for synthetic software smoke tests; never use as real-world accuracy evidence |

Repository metadata, license declarations, and dataset cards were checked on 6 November 2026.
Dataset-host declarations are not independent legal or scientific verification.

## Operational label proposal

Keep the source dataset's original label unchanged and add a separate, versioned project mapping:

| Project label | Operational definition |
|---|---|
| `fraud_or_phishing` | Deceptive attempt to obtain money, credentials, authentication codes, personal data, or to induce a harmful action |
| `unsolicited_promotion` | Unrequested marketing or bulk promotion without evidence of deception or credential/payment theft |
| `legitimate` | Expected personal, service, transactional, or authentication message with no observed deceptive intent |
| `uncertain` | Available text/context is insufficient or annotators disagree; do not silently force into a binary class |

The legacy UCI `spam` label is **not** equivalent to `fraud_or_phishing`; it may include legitimate
marketing cases. Keep UCI metrics under the original `ham`/`spam` task. For any third-party
corpus, publish its original labels, the exact crosswalk, and how unmapped/ambiguous labels are
handled. Compare a fraud-only model separately from promotion detection; do not count promotion as
phishing without evidence.

For new annotation, store `source_dataset`, `source_split`, `language`, `channel`, collection
period, label, annotator count/agreement, and a campaign/template group where permitted. Do not
store phone numbers, names, account identifiers, or raw user messages without documented consent
and a retention/deletion policy. Keep an untouched source/time/group holdout. A random split of
messages is not a target-domain test.

## Immediate reproducibility finding: duplicate leakage

The current 5,572-row SMS file has 5,094 distinct texts after the project's existing preprocessing:
478 extra rows are exact duplicates, forming 323 repeated-text groups. In the current random
80/20 split, **128 repeated-text groups appear in both train and test**. This can make the random
split optimistic. No normalized duplicate group in this corpus had conflicting labels.

The existing accuracy/F1 table is preserved as the historical random-split baseline for
comparability. `scripts/evaluate_grouped.py` adds a separate, group-held-out experiment: all
identical preprocessed texts stay on one side of the holdout and inner model-selection folds. It
does not replace the baseline or use outer test folds for model selection. The refreshed run uses
five outer folds with a separate five-fold grouped inner model-selection run per outer fold:

| Metric | Mean ± population SD across outer folds |
|---|---:|
| Accuracy | 97.74 ± 0.22% |
| Precision | 99.35 ± 1.00% |
| Recall | 83.49 ± 2.59% |
| Spam F1 | 90.70 ± 1.58% |

Inner grouped CV selected Random Forest in folds 1, 3, and 5, and Naive Bayes in folds 2 and 4.
Computed train/test exact-text group overlap was zero in every fold. This remains an evaluation
on the old English SMS corpus, not evidence of performance on current messaging apps. Full
per-fold results, model selection, group counts, and context-rule scenarios are in
`results/grouped_baseline.json`; its experiment identifier is
`nested_grouped_exact_text_five_fold`.

## Scoring-rule review

The model has no sender-known/forwarded annotations in its training corpus. The current API adds
7 score points to the default `unknown` source and 8 points to a forwarded message, and has a
hand-coded OTP/urgency score floor. These are unvalidated policy heuristics, not learned
probabilities.

On the existing random test split only, the default unknown-source bonus changed Random Forest
results from 0 to 1 false positive and from 20 to 15 false negatives (precision 100.0% to 99.3%;
recall 86.6% to 89.9%). This is a descriptive, single-dataset ablation—not evidence that the
bonus generalizes: the SMS corpus contains no sender-context labels and its split has duplicates.
The grouped evaluation records counterfactual context scenarios separately. Do not tune bonuses or
thresholds on either reported test split. Until a consented, annotated context dataset exists,
show context as context and do not describe the resulting score as a probability.

Across the five grouped outer folds, the model-only spam F1 was 90.70 ± 1.58%; the default
unknown-sender scenario was 92.19 ± 1.42%, and the forwarded+unknown scenario was
93.34 ± 0.96%. These are counterfactual results on a corpus with no sender/forwarded annotations;
they show only a threshold tradeoff on this corpus and do not establish that either policy improves
real-user decisions.

## Sources

- UCI collection and source history: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
- Bengali dataset card and files: <https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset>
- Bengali research repository record: <https://doi.org/10.60867/00000158>
- Machine-translated multilingual SMS dataset card: <https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset>
- Cameroon dataset repository: <https://github.com/loicyoumbi9/Detection-of-SMS-SPAM-IN-CAMEROONIAN-CONTEXT>
- English/Farsi research repository and data-availability note: <https://github.com/Hosein-Ghojavand/multilingual-spam-detection>
- Mozambique dataset card and files: <https://huggingface.co/datasets/MOZNLP/MOZ-Smishing>
- Korean dataset card and files: <https://huggingface.co/datasets/jmjmjm3/kor-smishing-message>
- India SMS dataset card and files: <https://huggingface.co/datasets/CloveAI/india-spam-sms>
- Synthetic OTP dataset card: <https://huggingface.co/datasets/alusci/sms-otp-spam-dataset>
