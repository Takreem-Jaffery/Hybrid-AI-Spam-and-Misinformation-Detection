# Changes from the original course project

What was carried over unchanged: the dataset and 80/20 stratified split (seed 42), the preprocessing function, the TF-IDF settings and the three classical models, the DistilBERT recipe (6 epochs, batch 32, lr 2e-5, weight decay 0.01, 15 % warm-up, max length 128, class-weighted loss), the 0.65/0.35 ensemble, the risk tiers, the advisor prompt, and the UI design and CSS.

## Security
- **The Groq API key was hardcoded in `App.jsx`** and shipped to every browser. It has been revoked. The key now lives only on the server (`.env`), and the UI calls `POST /advice`.
- Advisor prompt wraps the user's message in `<message>` delimiters as a basic prompt-injection guard; `report_to` is forced to null for benign messages in code, not just in the prompt.
- CORS restricted to the dev UI origin (was `*`); message length capped; `source` validated.
- Unverified phone numbers and e-mail addresses in the "report" popup replaced with cross-checked ones (FIA 1991 / complaint.fia.gov.pk, PTA 0800-55055 and SMS 9000).

## Correctness
- **Model selection no longer uses the test set.** The notebook picked the "best" classical model by test F1. Now 5-fold CV on the training split (`--select-by test` restores the old behaviour).
- **DistilBERT save bug guard.** After saving, the model is reloaded and its logits compared with the in-memory model, and it must pass the same sanity check the API runs at startup.
- Loss was computed twice per step (inside the model and weighted); now once, weighted.
- Seeds set for python, numpy, torch and the DataLoader.
- Tie handling made consistent: Spam means score **> 50** (strictly) in the API and in `evaluate.py`; the API used `>=` in an earlier draft.
- Keyword rule matches whole words (substring matching flagged "win" in "window").
- `preprocess()` defined once and imported everywhere (it was duplicated).
- The UI's silent fallback to a second, hand-written keyword scorer is gone. If the API is down, the UI says so instead of showing a different model's answer as if it were the real one.
- "Confidence" renamed **Spam score**: it is a weighted score, not a calibrated probability.
- Fake 800 ms loading delay removed; request timeout raised to 30 s (first DistilBERT call is slow on CPU).

## Reproducibility and engineering
- One config module, env-overridable paths; scripts replace notebook cells; dataset download with validation and sha256 manifest.
- Dataset quirk documented (5,574 lines vs 5,572 parsed rows).
- Test suite (fast synthetic/API checks and a dataset-backed reproducibility check), CI for backend and frontend.
- Evaluation script: all models on one split, bootstrap F1 intervals, PR/calibration plots, ensemble weight sensitivity sweep.

## Known gaps
- DistilBERT and ensemble metrics need re-running with `train_distilbert.py` + `evaluate.py`.
- The app has separate experimental article-veracity and clickbait baselines, not current-news fact-checking.
