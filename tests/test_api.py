import pytest
from fastapi.testclient import TestClient

from spamdetector import advisor
from spamdetector.api import app


@pytest.fixture()
def client(models_dir):
    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["tfidf_available"] is True
    assert body["transformer_active"] is False and body["transformer_note"]
    assert body["article_veracity_available"] is False and body["clickbait_available"] is False


def test_predict_spam_and_ham(client):
    spam = client.post("/predict", json={"message": "Congratulations! You won a free prize. Click now to claim!", "source": "unknown"}).json()
    assert spam["label"] == "Spam" and spam["risk"] in ("Medium", "High") and 0 <= spam["spam_score"] <= 100
    assert spam["model_used"].startswith("TF-IDF (Logistic Regression)")
    ham = client.post("/predict", json={"message": "Hey, are we still meeting at 6 for dinner?", "source": "friend"}).json()
    assert ham["label"] == "Benign" and ham["spam_score"] < spam["spam_score"]
    assert set(spam) == {"label", "spam_score", "risk", "riskColor", "model_used", "explanation", "reasons"}


def test_predict_validation(client):
    assert client.post("/predict", json={"message": "   "}).status_code == 400
    assert client.post("/predict", json={"message": "hi", "source": "alien"}).status_code == 422
    assert client.post("/predict", json={"message": "x" * 2001}).status_code == 422


def test_article_review_requires_trained_models(client):
    response = client.post("/article/review", json={"title": "Headline", "body": "Article body"})
    assert response.status_code == 503
    assert "Run scripts/train_article_models.py" in response.json()["detail"]


def test_article_review_rejects_empty_fields(client):
    assert client.post("/article/review", json={"title": " ", "body": "Article body"}).status_code == 400
    assert client.post("/article/review", json={"title": "Headline", "body": " "}).status_code == 400


def test_rule_based_fallback_without_any_model(empty_models_dir):
    with TestClient(app) as c:
        r = c.post("/predict", json={"message": "Win free cash prize now, click http://bit.ly/x", "source": "unknown"}).json()
        assert r["model_used"] == "Rule-based fallback" and r["label"] == "Spam"
        assert c.get("/health").json()["tfidf_available"] is False


def test_advice_requires_server_side_key(client):
    r = client.post("/advice", json={"message": "free cash"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_advice_success_and_benign_never_gets_a_report_target(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = {}

    async def fake(message, verdict, **kw):
        seen.update(label=verdict.label, key=kw["api_key"])
        return advisor.Advice(verdict="ok", reasoning="because", actions=["a", "b"], report_to="FIA")

    monkeypatch.setattr(advisor, "request_advice", fake)
    r = client.post("/advice", json={"message": "Congratulations! You won a free prize. Click now to claim!"})
    assert r.status_code == 200 and r.json()["actions"] == ["a", "b"]
    assert seen == {"label": "Spam", "key": "test-key"}


def test_advice_upstream_failure_is_a_502_without_leaking_details(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def boom(*a, **k):
        raise advisor.AdvisorError("Groq API returned HTTP 401.")

    monkeypatch.setattr(advisor, "request_advice", boom)
    r = client.post("/advice", json={"message": "free cash"})
    assert r.status_code == 502 and "test-key" not in r.text
