"""넓은 구역 틀(app/seat_wide.py)과 /v2/seatwide: 합성 그림 + 가짜 검출기."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as server
from app import observe as ob
from app import seat_wide as sw
from app.recognizer import GeminiCall, GeminiRecognizer


def _canvas(tiles, size=(2000, 1000), blank=()):
    """초록 펠트 위 흰 패 면 + 무늬. [blank] 는 무늬 없는 패(백)."""
    rgb = np.zeros((size[1], size[0], 3), np.uint8)
    rgb[..., 1] = 120
    rgb[..., 2] = 60
    for b in tiles:
        x, y, w, h = b
        rgb[y:y + h, x:x + w] = 245
        if b not in blank:
            rgb[y + h // 3:y + 2 * h // 3, x + w // 3:x + 2 * w // 3] = 20
    return rgb


def _row(n, x0=300, y=800, w=60, h=80, gap=4):
    return [(x0 + i * (w + gap), y, w, h) for i in range(n)]


class _Det:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def detect(self, part):
        # 창 위치를 알 수 없으니 한 창짜리 그림만 쓴다(창 크기 안의 그림).
        self.calls.append(part.shape)
        return list(self.boxes)


def _locate(boxes, rgb, monkeypatch):
    monkeypatch.setattr(sw, "WIN", 4000)  # 한 창으로
    return sw.locate(rgb, _Det(boxes))


def test_row_and_lone_single(monkeypatch):
    row = _row(13)
    lone = (1300, 650, 60, 80)  # 줄 위쪽 펠트의 낱장
    rgb = _canvas(row + [lone])
    fr = _locate(row + [lone], rgb, monkeypatch)
    assert fr.row == row
    assert fr.singles == [lone] and fr.lone == [True]


def test_touching_pair_is_not_lone(monkeypatch):
    row = _row(13)
    pair = [(1300, 650, 60, 80), (1364, 650, 60, 80)]
    fr = _locate(row + pair, _canvas(row + pair), monkeypatch)
    assert sorted(fr.singles) == pair and fr.lone == [False, False]


def test_single_next_to_undetected_tile_is_not_lone(monkeypatch):
    """표시패 둘 중 하나만 검출돼도 옆이 흰 면이라 외톨이가 아니다(133826)."""
    row = _row(13)
    pair = [(1300, 650, 60, 80), (1364, 650, 60, 80)]
    fr = _locate(row + pair[:1], _canvas(row + pair), monkeypatch)
    assert fr.singles == pair[:1] and fr.lone == [False]


def test_missed_tile_in_row_is_filled(monkeypatch):
    row = _row(13)
    fr = _locate(row[:5] + row[6:], _canvas(row), monkeypatch)
    assert len(fr.row) == 13
    assert abs(fr.row[5][0] - row[5][0]) <= 2


def test_river_above_is_dropped_and_left_corner_meld_ignored(monkeypatch):
    row = _row(13)
    river = _row(6, x0=800, y=100)
    left_meld = _row(3, x0=20, y=910, w=60, h=80)
    boxes = row + river + left_meld
    fr = _locate(boxes, _canvas(boxes), monkeypatch)
    assert fr.row == row and fr.singles == []
    assert sorted(fr.others) == sorted(left_meld)


def test_corner_zone_is_at_felt_right_edge():
    rgb = _canvas(_row(13))
    rgb[:, 1800:] = 245  # 펠트 오른쪽 끝 = 1800(흰 테두리)
    x0, y0, x1, y1 = sw.corner_zone(rgb, rx1=1100, ry=800, tw=60, th=80)
    assert x0 == int(1799 - 4.5 * 60) and y0 == 760 and y1 == 1000 and x1 > 1800


def _jpeg(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _recognizer(boxes, answers):
    r = object.__new__(GeminiRecognizer)
    r.seat_reads, r.model = 3, "fake"
    r.seat_locator = type("L", (), {"_detector": lambda self: _Det(boxes)})()
    seen = []

    def fake_call(jpeg, prompt, schema, media_resolution=None, extra_images=()):
        seen.append(prompt)
        if prompt == ob.CORNER_WIDE_PROMPT_V2:
            return GeminiCall({"melds": [{"tiles": ["5z"] * 3, "sideways": [], "face_down": [], "confidence": 0.9}], "loose": []}, 10, 50, 5, "fake")
        return GeminiCall({"tiles": [{"code": t, "confidence": 0.9} for t in next(answers).split()]}, 10, 100, 20, "fake")

    r.call = fake_call
    return r, seen


def test_observe_seat_wide_reads_pieces_and_corner(monkeypatch):
    monkeypatch.setattr(sw, "WIN", 4000)
    row = _row(10)
    lone = (1300, 650, 60, 80)
    rgb = _canvas(row + [lone])
    a = " ".join(["1m"] * 10 + ["7s"])
    r, seen = _recognizer(row + [lone], iter([a] * 3))
    c = r.observe_seat_wide(_jpeg(rgb))
    d = c.data
    assert d["method"] == "wide"
    assert [t["tile"] for t in d["row"]] == ["1m"] * 10
    assert d["singles"] == [{"tile": "7s", "conf": 0.9, "lone": True}]
    assert d["melds"][0]["tiles"] == ["5z"] * 3
    assert d["agreement"]["unanimous"] is True
    assert len(seen) == 6 and c.input_tokens == 450


def test_observe_seat_wide_majority_per_cell(monkeypatch):
    monkeypatch.setattr(sw, "WIN", 4000)
    row = _row(10)
    rgb = _canvas(row)
    base = ["1m"] * 10
    other = ["2m"] + ["1m"] * 9
    r, _ = _recognizer(row, iter([" ".join(base), " ".join(other), " ".join(base)]))
    d = r.observe_seat_wide(_jpeg(rgb)).data
    assert d["row"][0] == {"tile": "1m", "conf": 0.0}  # 엇갈린 칸은 확신도 0
    assert d["agreement"]["unanimous"] is False


def test_observe_seat_wide_frame_not_found():
    r, seen = _recognizer([], iter([]))
    d = r.observe_seat_wide(_jpeg(_canvas([]))).data
    assert d["method"] == "none" and d["row"] == [] and seen == []


def test_seatwide_endpoint(monkeypatch):
    class Fake:
        model = "fake"

        def observe_seat_wide(self, jpeg):
            self.size = Image.open(io.BytesIO(jpeg)).size
            return GeminiCall({"row": [], "singles": [], "melds": [], "method": "none"}, 20, 7, 3, "fake")

    fake = Fake()
    monkeypatch.setattr(server, "recognizer", fake)
    monkeypatch.setattr(server, "API_KEY", "")
    buf = io.BytesIO()
    Image.effect_noise((4000, 1500), 30).convert("RGB").save(buf, "JPEG", quality=60)
    r = TestClient(server.app).post("/v2/seatwide", files={"image": ("half.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert fake.size == (4000, 1500)  # 줄이지 않고 받는다
