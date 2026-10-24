import threading
import time

import httpx
import pytest
from google.genai import errors

from app.recognizer import GeminiRecognizer


class _Resp:
    text = '{"tiles": []}'
    usage_metadata = None
    candidates = []


class _Models:
    """요청마다 behaviors 를 하나씩 꺼내 쓴다: 예외면 던지고, 숫자면 그만큼 잠든 뒤 답한다."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0
        self.lock = threading.Lock()

    def generate_content(self, model, contents, config):
        with self.lock:
            self.calls += 1
            b = self.behaviors.pop(0) if self.behaviors else 0
        assert config.http_options is None  # Gemini 에 기한을 넘기지 않는다
        if isinstance(b, Exception):
            raise b
        time.sleep(b)
        return _Resp()


def _rec(behaviors, attempts=2, hedge_s=0.05):
    r = GeminiRecognizer.__new__(GeminiRecognizer)
    from google.genai import types

    r._types = types
    r.model = "fake"
    r.thinking = "default"
    r.locate_thinking = "default"
    r.hedge_s = hedge_s
    r.locate_hedge_s = 35.0
    r.attempts = attempts
    r.client = type("C", (), {"models": _Models(behaviors)})()
    return r


def _deadline():
    return errors.ServerError(504, {"error": {"code": 504, "message": "Deadline", "status": "DEADLINE_EXCEEDED"}})


def test_slow_first_request_is_hedged_and_fast_one_wins():
    r = _rec([1.0, 0])
    t = time.perf_counter()
    c = r.call(b"jpeg", "p", {"type": "object"})
    assert c.data == {"tiles": []}
    assert r.client.models.calls == 2
    assert time.perf_counter() - t < 0.5


def test_fast_answer_sends_only_one():
    r = _rec([0])
    r.call(b"jpeg", "p", {"type": "object"})
    assert r.client.models.calls == 1


def test_server_error_sends_another_at_once():
    r = _rec([_deadline(), 0], hedge_s=5)
    t = time.perf_counter()
    r.call(b"jpeg", "p", {"type": "object"})
    assert r.client.models.calls == 2
    assert time.perf_counter() - t < 1


def test_timeout_then_failure_raises_last_error():
    r = _rec([httpx.ReadTimeout("slow"), _deadline()])
    with pytest.raises(errors.ServerError):
        r.call(b"jpeg", "p", {"type": "object"})
    assert r.client.models.calls == 2


def test_client_error_is_not_retried():
    err = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    r = _rec([err])
    with pytest.raises(errors.ClientError):
        r.call(b"jpeg", "p", {"type": "object"})
    assert r.client.models.calls == 1


def test_locate_uses_longer_hedge():
    r = _rec([])
    seen = {}

    def fake_call(jpeg, prompt, schema, thinking=None, hedge_s=None):
        seen["hedge_s"] = hedge_s
        from app.recognizer import GeminiCall

        return GeminiCall({"hands": [], "rivers": [], "melds": [], "dead_wall": None, "center": None, "notes": ""},
                          1, 0, 0, "fake")

    r.call = fake_call
    r.locate(b"jpeg")
    assert seen["hedge_s"] == 35.0
