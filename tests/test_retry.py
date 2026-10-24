import httpx
import pytest
from google.genai import errors

from app.recognizer import GeminiRecognizer


class _Resp:
    text = '{"tiles": []}'
    usage_metadata = None
    candidates = []


class _Models:
    def __init__(self, fails):
        self.fails = list(fails)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(config.http_options.timeout)
        if self.fails:
            raise self.fails.pop(0)
        return _Resp()


def _rec(fails, attempts=2):
    r = GeminiRecognizer.__new__(GeminiRecognizer)
    from google.genai import types

    r._types = types
    r.model = "fake"
    r.thinking = "default"
    r.locate_thinking = "default"
    r.call_timeout_s = 20.0
    r.locate_timeout_s = 40.0
    r.attempts = attempts
    r.client = type("C", (), {"models": _Models(fails)})()
    return r


def test_timeout_then_success_retries_once():
    r = _rec([httpx.ReadTimeout("slow")])
    c = r.call(b"jpeg", "p", {"type": "object"})
    assert c.data == {"tiles": []}
    assert r.client.models.calls == [20000, 20000]


def test_server_error_retries_then_raises():
    err = errors.ServerError(504, {"error": {"code": 504, "message": "Deadline", "status": "DEADLINE_EXCEEDED"}})
    r = _rec([err, err])
    with pytest.raises(errors.ServerError):
        r.call(b"jpeg", "p", {"type": "object"})
    assert len(r.client.models.calls) == 2


def test_client_error_is_not_retried():
    err = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    r = _rec([err])
    with pytest.raises(errors.ClientError):
        r.call(b"jpeg", "p", {"type": "object"})
    assert len(r.client.models.calls) == 1


class _LocResp(_Resp):
    text = '{"hands": [], "rivers": [], "melds": [], "dead_wall": null, "center": null, "notes": ""}'


def test_locate_uses_longer_timeout():
    r = _rec([])
    r.client.models.generate_content = lambda model, contents, config: (
        r.client.models.calls.append(config.http_options.timeout) or _LocResp())
    r.locate(b"jpeg")
    assert r.client.models.calls == [40000]
