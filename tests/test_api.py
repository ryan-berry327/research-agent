"""HTTP-layer tests via TestClient. Offline: with LLM_PROVIDER unset the service
uses the StubLLM. TestClient is a context manager so the lifespan runs."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.service import app


def test_health_returns_200():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["agent_ready"] is True


def test_research_returns_answer_citations_and_trail():
    with TestClient(app) as client:
        resp = client.post("/research",
                           json={"question": "How many departments are using LLMs?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "How many departments are using LLMs?"
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["citations"], list)
    assert isinstance(body["trail"], list) and body["trail"]
    # the trail should record the agent's phases
    phases = {step["phase"] for step in body["trail"]}
    assert {"plan", "route", "gather", "analyse", "synthesise"} <= phases


def test_research_rejects_too_short_question():
    with TestClient(app) as client:
        resp = client.post("/research", json={"question": "x"})  # < min_length=3
    assert resp.status_code == 422


def test_research_rejects_too_long_question():
    with TestClient(app) as client:
        resp = client.post("/research", json={"question": "a" * 501})  # > max_length=500
    assert resp.status_code == 422
