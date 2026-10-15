"""Tests for the Gemini report chat assistant. No network calls are made."""

from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import chat_assistant, llm
from backend.llm import ChatProviderError, ChatReply, GeminiChatProvider
from backend.main import app


def make_record(**overrides):
    base = dict(
        id="rec-1", created_at=datetime(2026, 10, 15), patient_age=58, eye_side="OD", status="completed",
        primary_disease="glaucoma",
        disease_findings=[
            {"disease_id": "normal", "confidence": 0.05, "is_detected": False},
            {"disease_id": "glaucoma", "disease_name": "Glaucoma", "confidence": 0.82, "is_detected": True,
             "test_sensitivity": 0.57, "test_specificity": 0.96, "test_positives": 128},
            {"disease_id": "amd", "disease_name": "AMD", "confidence": 0.03, "is_detected": False},
        ],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------- report context and prompt

def test_report_context_grounds_findings():
    ctx = chat_assistant.report_context(make_record())
    assert "Main finding: Glaucoma" in ctx
    assert "AI probability 82%" in ctx
    assert "tested on 128 eyes" in ctx
    assert "right eye" in ctx


def test_report_context_rejected_and_missing():
    assert "retaken" in chat_assistant.report_context(make_record(status="rejected_for_quality"))
    assert "No screening report is open" in chat_assistant.report_context(None)


def test_answer_sends_guidelines_and_trims_history(monkeypatch):
    seen = {}

    def fake_chat(messages, system, **_):
        seen["messages"], seen["system"] = messages, system
        return ChatReply(text="ok", model="gemini-test")

    monkeypatch.setattr(chat_assistant.chat_provider, "chat", fake_chat)
    history = [{"role": "assistant", "content": "hello"}] + [{"role": "user", "content": f"q{i}"} for i in range(20)]
    chat_assistant.answer(make_record(), history)
    assert "does not measure the stage" in seen["system"]
    assert "Main finding: Glaucoma" in seen["system"]
    assert len(seen["messages"]) == chat_assistant.MAX_TURNS
    assert seen["messages"][0]["role"] == "user"


# ---------------------------------------------------------------- API endpoints

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_chat_endpoint_success(client, monkeypatch):
    monkeypatch.setattr(chat_assistant.chat_provider, "chat",
                        lambda messages, system, **_: ChatReply(text="Glaucoma affects the optic nerve.", model="gemini-test"))
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "What is glaucoma?"}]})
    assert r.status_code == 200
    assert r.json() == {"reply": "Glaucoma affects the optic nerve.", "model": "gemini-test"}


def test_chat_endpoint_reports_provider_errors(client, monkeypatch):
    def fail(*_, **__):
        raise ChatProviderError("rate_limited", "Too many requests.", 429)

    monkeypatch.setattr(chat_assistant.chat_provider, "chat", fail)
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429
    assert r.json()["detail"]["code"] == "rate_limited"


def test_chat_endpoint_validates_input(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    assert client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "x"}]}).status_code == 400
    assert client.post("/api/chat", json={"screening_id": "missing", "messages": [{"role": "user", "content": "x"}]}).status_code == 404


def test_status_never_exposes_key(client, monkeypatch):
    monkeypatch.setenv(llm.API_KEY_ENV, "secret-value-123")
    body = client.get("/api/chat/status").json()
    assert body["configured"] is True and body["provider"] == "google-gemini" and body["model"] == llm.GEMINI_MODEL
    assert "secret-value-123" not in str(body)


# ---------------------------------------------------------------- Gemini provider

def _response(status, payload):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "https://example.invalid"))


def _ok(text="Python is a friendly programming language."):
    return _response(200, {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "(thinking)", "thought": True}, {"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8},
        "modelVersion": "gemini-test",
    })


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv(llm.API_KEY_ENV, "test-key")
    monkeypatch.setattr(llm, "RETRY_DELAYS_S", (0.0, 0.0))
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    return GeminiChatProvider(model="gemini-3.8-flash")


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv(llm.API_KEY_ENV, raising=False)
    with pytest.raises(ChatProviderError) as e:
        GeminiChatProvider().chat([{"role": "user", "content": "hi"}], "system")
    assert e.value.code == "not_configured"


def test_request_shape_and_key_header(provider, monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json)
        return _ok()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    reply = provider.chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                           {"role": "user", "content": "explain"}], "be kind")
    assert reply.text == "Python is a friendly programming language."  # thought parts are dropped
    assert seen["url"].endswith("/models/gemini-3.8-flash:generateContent")
    assert "key=" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "test-key"
    assert [c["role"] for c in seen["json"]["contents"]] == ["user", "model", "user"]
    assert seen["json"]["systemInstruction"]["parts"][0]["text"] == "be kind"
    assert seen["json"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_retries_on_high_demand_then_succeeds(provider, monkeypatch):
    responses = iter([_response(503, {"error": {"message": "high demand"}}), _ok("second try")])
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: next(responses))
    assert provider.chat([{"role": "user", "content": "hi"}], "s").text == "second try"


@pytest.mark.parametrize("status, payload, expected", [
    (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}, "auth_failed"),
    (403, {"error": {"message": "PERMISSION_DENIED"}}, "access_denied"),
    (404, {"error": {"message": "model not found"}}, "model_unavailable"),
    (429, {"error": {"message": "RESOURCE_EXHAUSTED"}}, "rate_limited"),
    (503, {"error": {"message": "high demand"}}, "service_unavailable"),
    (200, {"promptFeedback": {"blockReason": "SAFETY"}}, "blocked"),
    (200, {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}, "blocked"),
    (200, {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]}, "empty_reply"),
])
def test_error_mapping(provider, monkeypatch, status, payload, expected):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: _response(status, payload))
    with pytest.raises(ChatProviderError) as e:
        provider.chat([{"role": "user", "content": "hi"}], "s")
    assert e.value.code == expected


def test_timeout_and_network_errors(provider, monkeypatch):
    def timeout(*_, **__):
        raise httpx.ReadTimeout("slow")

    def offline(*_, **__):
        raise httpx.ConnectError("offline")

    for fn, code in ((timeout, "timeout"), (offline, "network")):
        monkeypatch.setattr(llm.httpx, "post", fn)
        with pytest.raises(ChatProviderError) as e:
            provider.chat([{"role": "user", "content": "hi"}], "s")
        assert e.value.code == code
