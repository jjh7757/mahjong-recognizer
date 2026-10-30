import io

from fastapi.testclient import TestClient
from PIL import Image

from app import main as server
from app.recognizer import GeminiCall, RecognizeResult, prepare_image


def _jpeg(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 110, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _upload(name="a.jpg", data=None):
    return {"image": (name, data if data is not None else _jpeg(), "image/jpeg")}


class FakeRecognizer:
    model = "fake"

    def recognize(self, jpeg: bytes) -> RecognizeResult:
        return RecognizeResult(
            [{"tile": "1m", "conf": 0.9}, {"tile": "0p", "conf": 0.7}], "", 12, 10, 5, "fake",
            melds=[{"type": "pon", "tiles": ["7z", "7z", "7z"], "conf": 0.8}],
            win_tile={"tile": "7s", "conf": 0.6},
        )

    def locate(self, jpeg: bytes) -> GeminiCall:
        return GeminiCall({
            "hands": [{"side": "bottom", "box": [800, 100, 950, 900], "face_up": True, "conf": 0.9}],
            "rivers": [{"side": "bottom", "box": [600, 300, 780, 700], "conf": 0.8}],
            "dead_wall": None,
            "center": {"box": [200, 200, 800, 800]},
            "face_up_sides": ["bottom"],
            "notes": "",
        }, 30, 10, 5, "fake")

    def read_table(self, jpeg: bytes) -> GeminiCall:
        return GeminiCall({
            "dora_indicators": [{"tile": "3p", "conf": 0.9}],
            "ura_dora_indicators": [{"tile": "6m", "conf": 0.8}],
            "riichi": {"bottom": True, "right": False, "top": False, "left": False},
            "riichi_conf": {"bottom": 0.9, "right": 0.5, "top": 0.5, "left": 0.5},
            "last_discards": {"bottom": "7s", "right": None, "top": None, "left": "1z"},
            "last_discards_conf": {"bottom": 0.8, "right": 0, "top": 0, "left": 0.7},
            "notes": "",
        }, 40, 10, 5, "fake")


def _client(monkeypatch, key=""):
    monkeypatch.setattr(server, "API_KEY", key)
    monkeypatch.setattr(server, "recognizer", FakeRecognizer())
    return TestClient(server.app)


def test_health(monkeypatch):
    monkeypatch.setattr(server, "recognizer", None)
    with TestClient(server.app) as client:
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "/v1/locate" in body["endpoints"]
        assert "/v2/handonly" in body["endpoints"]
        assert body["git_sha"] == "unknown"


def test_503_without_recognizer(monkeypatch):
    monkeypatch.setattr(server, "recognizer", None)
    monkeypatch.setattr(server, "API_KEY", "")
    with TestClient(server.app) as client:
        for path in ("/v1/recognize", "/v1/locate", "/v1/table"):
            assert client.post(path, files=_upload()).status_code == 503


def test_api_key_required(monkeypatch):
    with _client(monkeypatch, key="secret") as client:
        assert client.post("/v1/recognize", files=_upload()).status_code == 401
        assert client.post("/v1/locate", files=_upload()).status_code == 401
        r = client.post("/v1/recognize", files=_upload(), headers={"X-Api-Key": "secret"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2 and body["tiles"][1]["tile"] == "0p"


def test_recognize_returns_melds_and_win_tile(monkeypatch):
    with _client(monkeypatch) as client:
        body = client.post("/v1/recognize", files=_upload()).json()
        assert body["melds"] == [{"type": "pon", "tiles": ["7z", "7z", "7z"], "conf": 0.8}]
        assert body["win_tile"] == {"tile": "7s", "conf": 0.6}
        assert body["model"] == "fake" and body["latency_ms"] == 12


def test_locate(monkeypatch):
    with _client(monkeypatch) as client:
        body = client.post("/v1/locate", files=_upload()).json()
        assert body["hands"][0]["side"] == "bottom" and body["hands"][0]["face_up"] is True
        assert body["face_up_sides"] == ["bottom"]
        assert body["center"]["box"] == [200, 200, 800, 800]
        assert body["dead_wall"] is None
        assert body["latency_ms"] == 30


def test_table(monkeypatch):
    with _client(monkeypatch) as client:
        body = client.post("/v1/table", files=_upload()).json()
        assert body["dora_indicators"] == [{"tile": "3p", "conf": 0.9}]
        assert body["ura_dora_indicators"] == [{"tile": "6m", "conf": 0.8}]
        assert body["riichi"] == {"bottom": True, "right": False, "top": False, "left": False}
        assert body["last_discards"]["left"] == "1z" and body["last_discards"]["top"] is None


def test_bad_image(monkeypatch):
    with _client(monkeypatch) as client:
        for path in ("/v1/recognize", "/v1/locate", "/v1/table"):
            assert client.post(path, files=_upload(data=b"not an image")).status_code == 400


def test_prepare_image_resizes():
    out = prepare_image(_jpeg((4000, 2000)))
    img = Image.open(io.BytesIO(out))
    assert max(img.size) == 1568


def test_prepare_image_draft_keeps_exif_rotation():
    """JPEG 는 draft 로 줄여 디코딩해도 EXIF 회전(orientation 6 = 90°)이 그대로 적용된다."""
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (4000, 2000), (40, 110, 60)).save(buf, format="JPEG", exif=exif)
    out = Image.open(io.BytesIO(prepare_image(buf.getvalue())))
    assert out.size == (784, 1568)


def test_prepare_image_rejects_too_many_pixels(monkeypatch):
    import pytest

    from app import recognizer as rec
    monkeypatch.setattr(rec, "MAX_PIXELS", 1000)
    with pytest.raises(ValueError, match="too many pixels"):
        prepare_image(_jpeg((64, 64)))
    with _client(monkeypatch) as client:
        r = client.post("/v1/recognize", files=_upload())
        assert r.status_code == 400 and "too many pixels" in r.json()["detail"]


def test_health_offers_source(monkeypatch):
    monkeypatch.setattr(server, "recognizer", None)
    with TestClient(server.app) as client:
        body = client.get("/").json()
        assert body["source"].startswith("https://github.com/") and body["license"] == "AGPL-3.0-only"


def test_startup_refuses_without_api_key(monkeypatch):
    import pytest

    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "ALLOW_NO_AUTH", False)
    with pytest.raises(RuntimeError, match="RECOGNIZER_API_KEY"):
        with TestClient(server.app):
            pass
    monkeypatch.setattr(server, "ALLOW_NO_AUTH", True)
    with TestClient(server.app) as client:  # 로컬 실험용으로 열어 두면 뜬다
        assert client.get("/").status_code == 200


def test_non_ascii_api_key_is_401_not_500(monkeypatch):
    # str 끼리의 compare_digest 는 비ASCII 에 TypeError(500)였다. 헤더는 latin-1 로 와도 bytes 비교로 401.
    with _client(monkeypatch, key="secret") as client:
        r = client.post("/v1/recognize", files=_upload(), headers={b"x-api-key": "열쇠".encode("utf-8")})
        assert r.status_code == 401
    assert server._check_key(None) is None if not server.API_KEY else True


def test_rate_limiter_is_a_token_bucket_with_hour_and_day_caps():
    now = [1000.0]
    lim = server.RateLimiter(per_min=60, burst=3, per_hour=5, per_day=6, clock=lambda: now[0])
    assert [lim.take() for _ in range(3)] == [0, 0, 0]          # 버스트 3 을 한 번에
    assert lim.take() == 1.0                                     # 비었다: 1초(분당 60 = 초당 1) 뒤
    now[0] += 1.0
    assert lim.take() == 0 and lim.take() == 1.0                 # 1초마다 하나씩 찬다
    now[0] += 30
    assert lim.take() == 0                                       # 5번째(시간 상한 5)
    assert lim.take() == 3600 - 31                               # 시간 상한: 창이 끝날 때까지
    now[0] += 3600
    assert lim.take() == 0                                       # 새 시간 창, 하루 6번째
    assert lim.take() == 86400 - 3631                            # 일 상한
    now[0] += 86400
    assert lim.take() == 0


def test_429_has_retry_after_and_last3_costs_one(monkeypatch):
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(per_min=60, burst=2))
    with _client(monkeypatch) as client:
        server.recognizer.observe_last_multi = lambda jpegs: GeminiCall(
            {"rivers": [{"last_tile": None, "tile_count": 0} for _ in jpegs], "notes": "", "method": "whole"}, 5, 1, 1,
            "fake")
        files = [("images", (f"{i}.jpg", _jpeg(), "image/jpeg")) for i in range(3)]
        assert client.post("/v2/last3", files=files).status_code == 200   # 사진 3장 = 요청 1
        assert client.post("/v1/recognize", files=_upload()).status_code == 200
        r = client.post("/v1/recognize", files=_upload())
        assert r.status_code == 429 and r.headers["Retry-After"] == "1"
        assert "retry after" in r.json()["detail"]


def test_upstream_failures_map_to_429_503_502(monkeypatch):
    from google.genai import errors

    def failing(exc):
        def recognize(jpeg):
            raise exc
        return recognize

    with _client(monkeypatch) as client:
        server.recognizer.recognize = failing(errors.ClientError(429, {"error": {"code": 429, "message": "quota",
                                                                              "status": "RESOURCE_EXHAUSTED"}}))
        r = client.post("/v1/recognize", files=_upload())
        assert r.status_code == 429 and r.headers["Retry-After"] == str(server.UPSTREAM_429_RETRY_S)
        assert "gemini 429" in r.json()["detail"]
        server.recognizer.recognize = failing(errors.ServerError(504, {"error": {"code": 504, "message": "x",
                                                                              "status": "DEADLINE_EXCEEDED"}}))
        r = client.post("/v1/recognize", files=_upload())
        assert r.status_code == 503 and r.headers["Retry-After"] == str(server.UPSTREAM_DOWN_RETRY_S)
        server.recognizer.recognize = failing(TimeoutError("gemini call budget 60s exceeded"))
        assert client.post("/v1/recognize", files=_upload()).status_code == 503
        server.recognizer.recognize = failing(RuntimeError("empty response"))
        assert client.post("/v1/recognize", files=_upload()).status_code == 502


def test_413_before_reading_body_and_by_file_size(monkeypatch):
    calls = []
    with _client(monkeypatch) as client:
        server.recognizer.recognize = lambda jpeg: calls.append(1) or FakeRecognizer().recognize(jpeg)
        monkeypatch.setattr(server, "MAX_UPLOAD", 1000)
        # Content-Length 가 한도 + 여유를 넘으면 본문을 읽기 전에(middleware) 413
        r = client.post("/v1/recognize", files=_upload(data=b"x" * 200_000))
        assert r.status_code == 413 and "body too large" in r.json()["detail"]
        # 겉포장 여유 안이지만 파일이 한도를 넘으면 파일 크기로 413
        r = client.post("/v1/recognize", files=_upload(data=b"x" * 2000))
        assert r.status_code == 413 and "image too large" in r.json()["detail"]
        assert client.post("/v1/recognize", files=_upload()).status_code == 200  # 작은 JPEG(~800B)는 통과
        assert calls == [1]
        # /v2/last3 는 사진 LAST_MULTI_MAX 장까지라 Content-Length 한도가 그만큼 넉넉하다
        monkeypatch.setattr(server, "MAX_UPLOAD", 100_000)
        files = [("images", (f"{i}.jpg", b"x" * 90_000, "image/jpeg")) for i in range(3)]
        assert client.post("/v2/last3", files=files).status_code == 400  # 선검사·크기 통과, 디코딩 실패
        assert client.post("/v1/recognize", files=_upload(data=b"x" * 270_000)).status_code == 413


def test_calls_run_on_a_pool_sized_for_max_concurrent(monkeypatch):
    """asyncio 기본 실행기(2 vCPU 에서 6스레드)가 아니라 MAX_CONCURRENT + 4 짜리 전용 풀에서 돈다: 10개가 동시에 돌아야 통과."""
    import threading

    assert server._executor._max_workers == server.MAX_CONCURRENT + 4
    n = 10
    barrier = threading.Barrier(n, timeout=5)

    def recognize(jpeg):
        barrier.wait()  # 10개가 동시에 들어와야 풀린다; 스레드가 모자라면 5초 뒤 BrokenBarrierError → 502
        return FakeRecognizer().recognize(jpeg)

    with _client(monkeypatch) as client:
        server.recognizer.recognize = recognize
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(client.post("/v1/recognize", files=_upload()).status_code))
                   for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert codes == [200] * n


def test_responses_carry_prep_ms(monkeypatch):
    with _client(monkeypatch) as client:
        assert client.post("/v1/recognize", files=_upload()).json()["prep_ms"] == 0
        assert client.post("/v1/locate", files=_upload()).json()["prep_ms"] == 0
        server.recognizer.observe = lambda j, kind, **kw: GeminiCall({"notes": ""}, 120, 1, 1, "fake", prep_ms=45)
        body = client.post("/v2/river", files=_upload()).json()
        assert body["latency_ms"] == 120 and body["prep_ms"] == 45
