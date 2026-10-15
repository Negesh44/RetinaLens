"""Chat model provider: Google Gemini (Gemini Developer API).

Calls the generateContent REST endpoint with httpx. The API key is read from the GEMINI_API_KEY
environment variable (loaded from the project's .env by backend.config), sent in the
x-goog-api-key header (never in the URL), and never logged or returned to clients.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("retinalens.llm")

API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT_S = 90.0  # Flash answers have taken up to ~50 s under load
RETRY_DELAYS_S = (2.0, 5.0)  # retries for "high demand" (503) and transient server errors
BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class ChatProviderError(Exception):
    """A failure the user can understand; `code` is stable for the frontend, `status` is the HTTP status to return."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class ChatReply:
    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", ""))
    except ValueError:
        return response.text[:200]


class GeminiChatProvider:
    name = "google-gemini"

    def __init__(self, model: str = GEMINI_MODEL):
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(API_KEY_ENV))

    def _request_body(self, messages: List[Dict[str, str]], system: str, max_tokens: int, temperature: float) -> dict:
        generation = {"maxOutputTokens": max_tokens, "temperature": temperature}
        if self.model.startswith("gemini-3"):
            # Short, conversational answers do not need deep reasoning; this keeps replies fast.
            generation["thinkingConfig"] = {"thinkingLevel": "low"}
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": generation,
        }

    def _post(self, body: dict) -> httpx.Response:
        url = f"{GEMINI_BASE_URL}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": os.environ[API_KEY_ENV], "Content-Type": "application/json"}
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            try:
                response = httpx.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT_S)
            except httpx.TimeoutException:
                raise ChatProviderError("timeout", "The assistant took too long to answer. Please try again.", 504)
            except httpx.TransportError:
                raise ChatProviderError("network", "Could not reach the Gemini service. Check the internet connection and try again.", 502)
            if response.status_code in (500, 503) and attempt < len(RETRY_DELAYS_S):
                logger.info("Gemini returned %s; retrying", response.status_code)
                time.sleep(RETRY_DELAYS_S[attempt])
                continue
            return response
        return response

    def chat(self, messages: List[Dict[str, str]], system: str, max_tokens: int = 1024, temperature: float = 0.4) -> ChatReply:
        if not self.configured:
            raise ChatProviderError("not_configured", f"The chat assistant is not set up yet: the {API_KEY_ENV} environment variable is missing.", 503)

        response = self._post(self._request_body(messages, system, max_tokens, temperature))
        code = response.status_code
        if code != 200:
            detail = _error_message(response)
            if code == 400 and "API key" in detail:
                raise ChatProviderError("auth_failed", "Google rejected the Gemini API key. It may be wrong, expired or deleted; create a new key and update GEMINI_API_KEY.", 502)
            if code in (401, 403):
                raise ChatProviderError("access_denied", "This Gemini API key is not allowed to use the model (key restrictions or the API is disabled for its project).", 502)
            if code == 404:
                raise ChatProviderError("model_unavailable", f"The Gemini model {self.model} is not available for this key.", 502)
            if code == 429:
                raise ChatProviderError("rate_limited", "The assistant has reached its usage limit for now. Please wait a moment and try again.", 429)
            if code in (500, 503):
                raise ChatProviderError("service_unavailable", "The Gemini service is very busy right now. Please try again in a minute.", 503)
            logger.warning("Gemini rejected a chat request (HTTP %s): %s", code, detail[:200])
            raise ChatProviderError("bad_request", "The assistant could not process this message.", 502)

        data = response.json()
        block = (data.get("promptFeedback") or {}).get("blockReason")
        candidates = data.get("candidates") or []
        finish = candidates[0].get("finishReason") if candidates else None
        if block or finish in BLOCKED_FINISH_REASONS:
            raise ChatProviderError("blocked", "The assistant could not answer that question. Please ask it in a different way.", 422)

        parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not text:
            raise ChatProviderError("empty_reply", "The assistant returned an empty answer. Please try rephrasing.", 502)
        usage = data.get("usageMetadata") or {}
        return ChatReply(text=text, model=data.get("modelVersion") or self.model,
                         input_tokens=usage.get("promptTokenCount"), output_tokens=usage.get("candidatesTokenCount"))


chat_provider = GeminiChatProvider()
