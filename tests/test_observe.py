from fastapi.testclient import TestClient

from app import main as server
from app.normalize import normalize_locate
from app.observe import (
    normalize_center_v2,
    normalize_corner_v2,
    normalize_dora_v2,
    normalize_hand_v2,
    normalize_last_v2,
    normalize_meld_v2,
    normalize_river_v2,
    normalize_sticks_v2,
)
from app.recognizer import GeminiCall
from tests.test_api import FakeRecognizer, _upload


def test_hand_v2_keeps_raw_melds_and_separate_tile():
    out = normalize_hand_v2({
        "tiles": [{"code": "3s", "confidence": 0.9}, {"code": "xx", "confidence": 1}],
        "separate_tile": {"code": "2s", "confidence": 0.8},
        "melds": [
            {"tiles": ["6z", "6z", "6z"], "sideways": [1, 9], "face_down": [], "confidence": 0.9},
            {"tiles": ["1m", "1m"], "sideways": [], "face_down": [], "confidence": 0.9},  # 2장: 제외
            {"tiles": ["4p", "4p", "4p", "4p"], "sideways": [], "face_down": [0, 3], "confidence": 0.9},
        ],
        "notes": "",
    })
    assert out["tiles"] == [{"tile": "3s", "conf": 0.9}]
    assert out["separate_tile"] == {"tile": "2s", "conf": 0.8}
    assert [m["tiles"] for m in out["melds"]] == [["6z", "6z", "6z"], ["4p", "4p", "4p", "4p"]]
    assert out["melds"][0]["sideways"] == [1]          # 범위 밖 9 는 버림
    assert out["melds"][1]["face_down"] == [0, 3]
    assert normalize_hand_v2({"tiles": []})["separate_tile"] is None


def test_river_v2_order_and_sideways():
    out = normalize_river_v2({
        "tiles": [{"code": "1z", "sideways": False, "confidence": 0.9},
                  {"code": "8s", "sideways": True, "confidence": 0.8},
                  {"code": "bad", "sideways": True}],
        "riichi_stick": True,
    })
    assert [(t["tile"], t["sideways"]) for t in out["tiles"]] == [("1z", False), ("8s", True)]
    assert out["riichi_stick"] is True


def test_center_v2_all_sides():
    out = normalize_center_v2({
        "indicators": [{"code": "9s", "confidence": 0.9}, {"code": "5s", "confidence": 0.9}],
        "riichi_sticks": [{"side": "right", "present": True, "confidence": 0.9}, {"side": "nope", "present": True}],
    })
    assert [t["tile"] for t in out["indicators"]] == ["9s", "5s"]
    assert out["riichi_sticks"] == {"bottom": False, "right": True, "top": False, "left": False}


class FakeObserver(FakeRecognizer):
    def observe(self, jpeg: bytes, kind: str) -> GeminiCall:
        data = {"river": {"tiles": [{"tile": "2s", "conf": 0.9, "sideways": False}], "riichi_stick": False, "notes": ""}}
        return GeminiCall(data.get(kind, {"notes": ""}), 20, 7, 3, "fake")


def test_v2_endpoints_return_usage(monkeypatch):
    monkeypatch.setattr(server, "recognizer", FakeObserver())
    monkeypatch.setattr(server, "API_KEY", "")
    client = TestClient(server.app)
    r = client.post("/v2/river", files=_upload())
    assert r.status_code == 200
    j = r.json()
    assert j["tiles"][0]["tile"] == "2s"
    assert j["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_meld_v2_and_locate_melds():
    assert normalize_meld_v2({"tiles": ["6z", "6z", "6z"], "sideways": [2], "face_down": [], "confidence": 0.9})["tiles"] == ["6z"] * 3
    assert normalize_meld_v2({"tiles": ["6z", "6z"], "sideways": [], "face_down": []})["tiles"] == []
    loc = normalize_locate({
        "hands": [], "rivers": [], "dead_wall": None, "center": None, "notes": "",
        "melds": [{"side": "top", "box": [20, 300, 60, 380], "confidence": 0.9},
                  {"side": "top", "box": [20, 400, 60, 480], "confidence": 0.8},
                  {"side": "x", "box": [0, 0, 1, 1]}],
    })
    assert [m["box"] for m in loc["melds"]] == [[20, 300, 60, 380], [20, 400, 60, 480]]


def test_corner_v2_melds_and_loose():
    out = normalize_corner_v2({
        "melds": [{"tiles": ["6m", "7m", "8m"], "sideways": [0], "face_down": [], "confidence": 0.9},
                  {"tiles": ["1z"], "sideways": [], "face_down": []}],  # 1장: 버림
        "loose": [{"code": "2p", "confidence": 0.8}, {"code": "?"}],
        "notes": "",
    })
    assert [m["tiles"] for m in out["melds"]] == [["6m", "7m", "8m"]]
    assert "notes" not in out["melds"][0]
    assert out["loose"] == [{"tile": "2p", "conf": 0.8}]


def test_last_v2_tile_or_none():
    assert normalize_last_v2({"last_tile": {"code": "7m", "confidence": 0.9}, "tile_count": 14})["last_tile"] == {
        "tile": "7m", "conf": 0.9}
    out = normalize_last_v2({"last_tile": None, "tile_count": "x"})
    assert out["last_tile"] is None and out["tile_count"] == 0


def test_sticks_v2_has_no_indicators():
    out = normalize_sticks_v2({"riichi_sticks": [{"side": "top", "present": True, "confidence": 0.8}]})
    assert out["riichi_sticks"] == {"bottom": False, "right": False, "top": True, "left": False}
    assert "indicators" not in out


def test_new_v2_routes(monkeypatch):
    monkeypatch.setattr(server, "recognizer", FakeObserver())
    monkeypatch.setattr(server, "API_KEY", "")
    client = TestClient(server.app)
    for path in ("/v2/corner", "/v2/last", "/v2/sticks", "/v2/dora"):
        assert client.post(path, files=_upload()).status_code == 200


def test_dora_v2_indicators():
    out = normalize_dora_v2({"indicators": [{"code": "4p", "confidence": 0.9}, {"code": "7m", "confidence": 0.8}, {}]})
    assert [t["tile"] for t in out["indicators"]] == ["4p", "7m"]
