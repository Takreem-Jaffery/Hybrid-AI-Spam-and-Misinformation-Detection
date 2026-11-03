# Contributing

## Development checks

Use Python 3.10+ and Node.js 22.13+. Install the backend development requirements with
`python -m pip install -r requirements-dev.txt`, then run:

```text
python -m pytest -q
cd frontend
npm ci
npm test
npm run lint
npm run build
```

The backend tests download only the small SMS Spam Collection for the reproducibility test.
The larger article corpora and trained models are generated locally and must not be committed.

## Pull requests

Create a short-lived feature branch from `main` and open a pull request. Keep changes focused,
describe user-visible behavior and limitations, and include test/evaluation evidence. Update
documentation when behavior or reported results change. Do not select models or tune thresholds
using the held-out test data.

For dataset or model changes, document the original source, license, checksum, label mapping,
split strategy, and per-class evaluation. Do not include private user messages or credentials.

Use Conventional Commit-style messages such as `fix: validate article input` or
`docs: clarify evaluation limits`.
