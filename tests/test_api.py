import io

from fastapi.testclient import TestClient
from PIL import Image

from app import main as server
from app.recognizer import RecognizeResult, prepare_image


def _jpeg(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 110, 60)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeRecognizer:
    model = "fake"

    def recognize(self, jpeg: bytes) -> RecognizeResult:
        return RecognizeResult([{"tile": "1m", "conf": 0.9}, {"tile": "0p", "conf": 0.7}], "", 12, 10, 5, "fake")


def test_health(monkeypatch):
    monkeypatch.setattr(server, "recognizer", None)
    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200


def test_503_without_recognizer(monkeypatch):
    monkeypatch.setattr(server, "recognizer", None)
    monkeypatch.setattr(server, "API_KEY", "")
    with TestClient(server.app) as client:
        r = client.post("/v1/recognize", files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 503


def test_api_key_required(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "secret")
    monkeypatch.setattr(server, "recognizer", FakeRecognizer())
    with TestClient(server.app) as client:
        r = client.post("/v1/recognize", files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 401
        r = client.post("/v1/recognize", files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                        headers={"X-Api-Key": "secret"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2 and body["tiles"][1]["tile"] == "0p"


def test_bad_image(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "recognizer", FakeRecognizer())
    with TestClient(server.app) as client:
        r = client.post("/v1/recognize", files={"image": ("a.jpg", b"not an image", "image/jpeg")})
        assert r.status_code == 400


def test_prepare_image_resizes():
    out = prepare_image(_jpeg((4000, 2000)))
    img = Image.open(io.BytesIO(out))
    assert max(img.size) == 1568
