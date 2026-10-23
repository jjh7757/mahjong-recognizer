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
        assert "/v1/locate" in r.json()["endpoints"]


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
