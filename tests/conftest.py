import pytest

from app import main as server


@pytest.fixture(autouse=True)
def _test_defaults(monkeypatch):
    """테스트는 대부분 키 없이 띄우고(기동 거부는 test_startup_refuses_without_key 가 따로 본다), 요청 제한기는
    테스트마다 새로(전역 토큰 버킷이 테스트를 가로질러 소모되지 않게)."""
    monkeypatch.setattr(server, "ALLOW_NO_AUTH", True)
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(server.RATE_PER_MIN, server.RATE_BURST,
                                                                server.RATE_PER_HOUR, server.RATE_PER_DAY))
