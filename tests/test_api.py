import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as server
from app.detector import Detection, arrange


def test_health_without_model(monkeypatch):
    monkeypatch.setattr(server, "detector", None)
    with TestClient(server.app) as client:
        r = client.get("/")
        assert r.status_code == 200


def test_recognize_503_without_model(monkeypatch):
    monkeypatch.setattr(server, "detector", None)
    monkeypatch.setattr(server, "API_KEY", "")
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="JPEG")
    with TestClient(server.app) as client:
        r = client.post("/v1/recognize", files={"image": ("a.jpg", buf.getvalue(), "image/jpeg")})
        assert r.status_code == 503


def test_api_key_required(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "secret")
    with TestClient(server.app) as client:
        r = client.post("/v1/recognize", files={"image": ("a.jpg", b"x", "image/jpeg")})
        assert r.status_code == 401
        r = client.post("/v1/recognize", files={"image": ("a.jpg", b"x", "image/jpeg")},
                        headers={"X-Api-Key": "secret"})
        assert r.status_code in (400, 503)


def _d(tile, x, y, w=40, h=56, conf=0.9):
    return Detection(tile, conf, x, y, x + w, y + h)


def test_arrange_rows_and_groups():
    dets = [
        _d("3m", 100, 200), _d("1m", 0, 200), _d("2m", 50, 200),      # 손패 왼쪽부터
        _d("1z", 300, 205), _d("1z", 350, 205), _d("1z", 400, 205),  # 간격 띄운 퐁
        _d("5p", 0, 100),                                             # 위쪽 줄: 도라 표시패
    ]
    out = arrange(dets)
    assert [t["tile"] for t in out] == ["5p", "1m", "2m", "3m", "1z", "1z", "1z"]
    assert out[0]["row"] == 0 and out[1]["row"] == 1
    assert out[1]["group"] == out[3]["group"]
    assert out[3]["group"] != out[4]["group"]


@pytest.mark.skipif(not (server.MODEL_DIR / "model.onnx").exists(), reason="모델 없음")
def test_recognize_with_model(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    img = Image.new("RGB", (640, 480), (40, 110, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    with TestClient(server.app) as client:
        r = client.post("/v1/recognize", files={"image": ("a.jpg", buf.getvalue(), "image/jpeg")})
        assert r.status_code == 200
        assert "tiles" in r.json()
