.PHONY: setup data train test api ui
setup:      ; pip install -r requirements-dev.txt && cd frontend && npm install
data:       ; python scripts/download_data.py
train:      ; python scripts/train_classical.py
test:       ; python -m pytest -q
api:        ; uvicorn spamdetector.api:app --reload --port 8000
ui:         ; cd frontend && npm run dev
