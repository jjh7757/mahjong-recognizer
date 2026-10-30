import threading
import time

import httpx
import pytest
from google.genai import errors

from app import recognizer as rec
from app.recognizer import GeminiRecognizer, pool_sizes, rate_limited, retryable


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


def _rec(behaviors, attempts=2, hedge_s=0.05, budget_s=60.0, rate_backoff_s=(0.05, 0.1)):
    r = GeminiRecognizer.__new__(GeminiRecognizer)
    from google.genai import types

    r._types = types
    r.model = "fake"
    r.thinking = "default"
    r.locate_thinking = "default"
    r.hedge_s = hedge_s
    r.locate_hedge_s = 35.0
    r.attempts = attempts
    r.budget_s = budget_s
    r.rate_backoff_s = rate_backoff_s
    r.client = type("C", (), {"models": _Models(behaviors)})()
    return r


def _deadline():
    return errors.ServerError(504, {"error": {"code": 504, "message": "Deadline", "status": "DEADLINE_EXCEEDED"}})


def _too_many():
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})


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


def test_gemini_429_is_retried_after_a_pause():
    """Gemini 429(요청 과다)는 잠깐 쉬고 한 번 더. 10-29 측정에서 재시도 없이 502 로 끝났다."""
    assert rate_limited(_too_many()) and retryable(_too_many())
    assert not rate_limited(_deadline()) and retryable(_deadline())
    r = _rec([_too_many(), 0], hedge_s=5, rate_backoff_s=(0.1, 0.1))
    t = time.perf_counter()
    c = r.call(b"jpeg", "p", {"type": "object"})
    assert c.data == {"tiles": []}
    assert r.client.models.calls == 2
    assert 0.1 <= time.perf_counter() - t < 1


def test_gemini_429_twice_raises_the_429():
    r = _rec([_too_many(), _too_many()], hedge_s=5)
    with pytest.raises(errors.ClientError) as e:
        r.call(b"jpeg", "p", {"type": "object"})
    assert e.value.code == 429 and r.client.models.calls == 2


def test_call_budget_stops_waiting_for_late_requests():
    """두 요청이 다 늦으면 예산(budget_s)에서 TimeoutError — 전엔 마지막 헤지를 기한 없이 기다렸다."""
    r = _rec([1.0, 1.0], hedge_s=0.05, budget_s=0.3)
    t = time.perf_counter()
    with pytest.raises(TimeoutError, match="budget"):
        r.call(b"jpeg", "p", {"type": "object"})
    assert 0.3 <= time.perf_counter() - t < 0.9
    assert r.client.models.calls == 2


def test_pools_are_sized_for_demand():
    # 동시 16 × 자리 읽기 3 × 헤지 2 = 96 (헤지 풀), 16 × 3 = 48 (자리 읽기 풀). 전엔 48·16 고정.
    assert pool_sizes(16, 3, 2) == (96, 48)
    assert rec._hedge_pool._max_workers >= 96 and rec._fanout_pool._max_workers >= 48


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
