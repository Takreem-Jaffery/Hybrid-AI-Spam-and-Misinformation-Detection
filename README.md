# Message Risk Analyzer

Paste a message for a spam assessment, or review an article against separate historical veracity
and headline-clickbait models. The message scanner uses a TF-IDF classifier and optional
fine-tuned DistilBERT. Served by a FastAPI backend and React UI.

**Authors:** Takreem Jaffery and Aneeza Junaid.

![CI](https://github.com/Takreem-Jaffery/Hybrid-AI-Spam-and-Misinformation-Detection/actions/workflows/ci.yml/badge.svg?branch=main)

> **Scope.** The app has separate message spam and article review modes; it does not connect to messaging services or automatically fact-check current news. Spam models use the English SMS Spam Collection. Article veracity and headline clickbait are independent experimental models based on limited, older datasets. Their outputs are dataset-relative model signals, not determinations of truth. See [Limitations](#limitations).

## What is in the repo

| Path | Purpose |
|---|---|
| `spamdetector/` | Library: preprocessing, data split, scoring, inference, advisor, FastAPI app |
| `scripts/` | dataset downloaders, CPU training scripts, `train_distilbert.py`, and evaluation |
| `frontend/` | React 19 + Vite UI |
| `tests/` | Fast API/model tests plus a dataset-backed classical-model reproducibility test |
| `docs/` | Implementation history and document-sharing guidance |

## Quick start

Requires Python 3.10+ and Node.js 22.12+. The examples below use PowerShell; activate the
virtual environment with `source .venv/bin/activate` on macOS/Linux.

```powershell
git clone https://github.com/Takreem-Jaffery/Hybrid-AI-Spam-and-Misinformation-Detection.git
cd Hybrid-AI-Spam-and-Misinformation-Detection

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

python scripts/download_data.py
python scripts/train_classical.py
python scripts/download_article_datasets.py
python scripts/train_article_models.py
python -m pytest -q
```

The article archives total about 1.1 GB. Dataset files and trained model artifacts are generated
locally and intentionally excluded from Git. The tests use a small UCI dataset download for the
slow reproducibility check; the article archives are not required for CI.

### Run the app

```bash
# terminal 1: API (works with the TF-IDF model alone)
uvicorn spamdetector.api:app --port 8000

# terminal 2: UI
cd frontend
npm ci
npm run dev                                     # http://localhost:5173
```

Without trained article models, Article Review displays an explicit unavailable response; the message scanner continues to work independently. `/health` reports model availability. The web app's **How does the model work?** link explains each pipeline, data source, and limitation.

### Add DistilBERT (optional, GPU recommended)

```bash
pip install -r requirements-transformer.txt
python scripts/train_distilbert.py   # ~8 min on a Colab T4, hours on CPU
python scripts/evaluate.py           # all models, same test split -> results/
```

### Add the AI advisor (optional)

```bash
cp .env.example .env                 # PowerShell: Copy-Item .env.example .env
# put your key in GROQ_API_KEY=...   (get one at console.groq.com)
```

The key lives only on the server. The browser calls `POST /advice`, never Groq directly.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /predict` | `{message, is_forwarded, source}` (`source`: `unknown` / `friend` / `group`) | `label`, `spam_score` (0-100), `risk`, `riskColor`, `model_used`, `explanation`, `reasons` |
| `POST /article/review` | `{title, body}` | separate graded veracity category and headline clickbait category with model scores |
| `POST /advice` | same | `verdict`, `reasoning`, `actions`, `report_to` (503 without a key, 502 if Groq fails) |
| `GET /health` | | which models loaded |

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"message":"URGENT! You won a free prize, click now","is_forwarded":true,"source":"unknown"}'
```

## How scoring works

```
spam_score = 0.65 * TF-IDF probability + 0.35 * DistilBERT probability     (each 0-100)
           + 8 if forwarded + 7 if sender unknown
phishing floor: messages matching phishing patterns are raised to at least 55
Spam if score > 50 (strictly)   |   risk: High >= 65, Medium >= 40, Low otherwise
```

The weights, bonuses and thresholds are **hand-chosen heuristics**, not fitted. `evaluate.py` includes a weight sweep so you can see how sensitive the result is, but the sweep uses the test set and is a sensitivity analysis, not tuning.
The score is not a calibrated probability.

## Results

Same stratified 80/20 split everywhere (seed 42; 4,457 train / 1,115 test; 13.4 % spam).

| Model | Accuracy | Precision | Recall | F1 | Source |
|---|---|---|---|---|---|
| Naive Bayes | 97.85 % | 100.0 % | 83.89 % | 91.24 % | current saved run |
| Logistic Regression | 97.76 % | 95.59 % | 87.25 % | 91.23 % | current saved run |
| Random Forest | 98.21 % | 100.0 % | 86.58 % | 92.81 % | selected by training-only CV; current saved run |

These results use the same stratified 80/20 split (seed 42; 4,457 train / 1,115 test; 13.4%
spam) and are recorded in `results/classical_training.json`. The slow test checks the Random
Forest baseline against the current run. Older notebook and report values are not treated as
reproduced results. DistilBERT and ensemble figures are omitted until the current pipeline is
run and its outputs are saved and reviewed.

The first article-task CPU baseline (Python 3.10.9, scikit-learn 1.6.1, seed 42) achieved:

| Task | Evaluation split | Accuracy | Macro-F1 | Majority baseline macro-F1 |
|---|---|---:|---:|---:|
| Article veracity | Two publishers held out | 66.06 % | 26.32 % | 19.61 % |
| Headline clickbait severity | Provided validation split | 44.00 % | 35.58 % | 14.92 % |

These are initial research baselines, not deployment-quality results. In the veracity publisher
holdout, the model scored zero F1 for `mostly false` and `no factual content`; a majority-category
prediction would be misleading. The Article Review UI displays the measured task macro-F1 and
majority baseline alongside each result. See `results/article_task_training.json` for class-level
metrics, splits, dataset-manifest hash, and versions.

## Limitations

- **Dataset:** ~5.5k English SMS from a UK/Singapore collection, mostly 2000s-era. It does not contain Urdu or Roman-Urdu messages, platform-specific forwards, or modern scam wording, so performance may not generalize to other messaging apps.
- **Test set:** 1,115 messages with 149 spam. One message changes recall by about 0.7 points; differences of a point or two between models are within noise (`evaluate.py` reports bootstrap intervals).
- **Metadata rules:** "forwarded" and "unknown sender" add fixed points to the score. No data supports those exact weights.
- **The LLM advisor** can be wrong and is not part of the model's accuracy.
- **Reporting contacts** (`frontend/src/helplines.js`) were cross-checked on 2026-10-30 but can change; confirm on the official sites.
- **Dataset quirk:** UCI's file has 5,574 lines, but pandas' default parser yields 5,572 rows. We keep it so results match the original; see `spamdetector/data.py`.

## Reproducibility

Seed 42 everywhere; one shared `preprocess()`; the dataset is validated by row/class counts and its sha256 is written to `data/MANIFEST.json`. GPU training is not bit-for-bit deterministic. The model selected for `pipeline.pkl` is now chosen by cross-validation on the training split, not by test F1 (see `docs/CHANGES_FROM_ORIGINAL.md`).

## Data and licences

Dataset: Almeida, T. & Hidalgo, J., *SMS Spam Collection*, UCI Machine Learning Repository, CC BY 4.0, <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>. Not redistributed here; `download_data.py` fetches it.
Base model: `distilbert-base-uncased` (Apache-2.0). Code: MIT (see `LICENSE`).

Article datasets are downloaded by `scripts/download_article_datasets.py` into the ignored
`data/article_tasks/` directory, verified against the source MD5 checksums, and not committed:

- Potthast et al., *BuzzFeed-Webis Fake News Corpus 2016*, CC BY 4.0,
  <https://doi.org/10.5281/zenodo.1239675>. The training pipeline keeps the original graded
  `mostly true`, `mixture of true and false`, `mostly false`, and `no factual content` categories.
  It does not force these labels into a binary true/fake target.
- Potthast et al., *Webis Clickbait Corpus 2017*, CC BY 4.0,
  <https://doi.org/10.5281/zenodo.5530410>. The pipeline maps the original four-point judgments
  to the nearest of the four original severity levels, trains on the provided training split, and
  evaluates on the provided validation split. The official test archive is intentionally not
  downloaded; the corpus owners provide official test evaluation through TIRA. The validation
  download is large because its ZIP bundles media; the downloader extracts only the JSONL text and
  labels, not the images.

The models use word and character TF-IDF with class-weighted Logistic Regression. Evaluation
holds out publisher groups when the downloaded records expose enough groups, otherwise it uses
a deterministic stratified random split. Reports are written under `models/article_tasks/` and
`results/article_task_training.json`. A group holdout is not a substitute for testing on current
news, and scores are not calibrated probabilities of factual truth. Current evidence search is
not implemented; Article Review is local/offline.
