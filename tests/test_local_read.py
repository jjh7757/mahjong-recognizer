import io
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import local_read as lr
from app import main as server
from app.recognizer import GeminiCall


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _canvas(boxes: list, blank: tuple = ()) -> np.ndarray:
    """초록 펠트에 흰 패. [blank] 가 아닌 패는 가운데에 무늬."""
    im = Image.new("RGB", (2000, 800), (30, 120, 60))
    d = ImageDraw.Draw(im)
    for b in boxes:
        x, y, w, h = b
        d.rectangle((x, y, x + w, y + h), fill=(240, 240, 235))
        if b not in blank:
            d.rectangle((x + 15, y + 20, x + w - 15, y + h - 20), fill=(20, 20, 20))
    return np.asarray(im)


def _row(n: int, x0: int = 100, y: int = 400) -> list:
    return [(x0 + 66 * i, y, 60, 80) for i in range(n)]


def test_split_frame_drops_neighbours_and_splits_two_rows():
    hand = _row(7)
    melds = _row(3, 1500, 400) + _row(3, 1500, 490)
    neighbours = _row(3, 600, 10)
    boxes = hand + melds + neighbours
    rgb = _canvas(boxes)
    fr = lr.split_frame(rgb, boxes, 1200)
    assert fr.hand == hand
    assert sorted(map(tuple, fr.melds)) == sorted([tuple(_row(3, 1500, 400)), tuple(_row(3, 1500, 490))])
    assert set(neighbours) <= set(fr.loose)
    assert lr.frame_ok(fr, rgb)


def test_blank_tiles_pass_in_hand_row_and_only_whole_meld():
    hand = _row(10)
    meld = _row(3, 1500, 400)
    rgb = _canvas(hand + meld, blank=(hand[0], hand[1], *meld))
    assert lr.frame_ok(lr.split_frame(rgb, hand + meld, 1200), rgb), "손패 白 2장 + 白 퐁"
    rgb = _canvas(hand + meld, blank=(meld[0],))
    assert not lr.frame_ok(lr.split_frame(rgb, hand + meld, 1200), rgb), "치·퐁에 白 한 장만은 없다"


def test_stray_is_lone_tile_right_of_hand_only():
    hand = _row(13)
    lone = (1200, 480, 60, 80)
    far_river = (1200, 100, 60, 80)
    pile = _row(3, 1500, 500)
    boxes = hand + [lone, far_river] + pile
    rgb = _canvas(boxes)
    fr = lr.split_frame(rgb, boxes, 1100)
    assert lr.strays(fr, boxes, rgb) == [lone]


def test_read_seat_with_real_models_finds_no_frame_on_empty_felt():
    r = lr.LocalReader()
    o = r.read_seat(_jpeg(_canvas([])), 0.7)
    assert o["framed"] is False and o["hand"] == [] and o["ms"]["total"] >= 0
    assert r.read_rivers([_jpeg(_canvas([]))])["rivers"] == [None]


def _wall_canvas(faces: list, backs: list) -> np.ndarray:
    im = Image.fromarray(_canvas(faces))
    d = ImageDraw.Draw(im)
    for x, y, w, h in backs:
        d.rectangle((x, y, x + w, y + h), fill=(190, 185, 240))  # 연보라 뒷면
    return np.asarray(im)


def test_wall_score_prefers_tile_between_backs():
    wall = (500, 100, 60, 80)
    river = (800, 400, 60, 80)
    end = (1000, 100, 60, 80)
    backs = [(438, 100, 60, 80), (562, 100, 60, 80), (938, 100, 60, 80)]
    rgb = _wall_canvas([wall, river, end], backs)
    assert lr.wall_score(rgb, wall)[0] == 2
    assert lr.wall_score(rgb, end)[0] == 1
    assert lr.wall_score(rgb, river) == (0, 0.0)


def test_wall_score_ignores_felt_skin_and_white():
    b = (500, 100, 60, 80)
    im = Image.fromarray(_canvas([b, (562, 100, 60, 80)]))  # 오른쪽 이웃은 앞면 패(흰 면)
    ImageDraw.Draw(im).rectangle((438, 100, 498, 180), fill=(225, 175, 145))  # 왼쪽은 피부색
    assert lr.wall_score(np.asarray(im), b)[0] == 0


class _Wall(lr.LocalReader):
    def __init__(self, faces, conf):
        super().__init__()
        self.faces, self.conf = faces, conf

    def detect(self, rgb, low=True):
        return self.faces

    def read_any(self, rgb, b):
        return "5s", self.conf


def test_read_wall_answers_only_flanked_and_confident_tile():
    wall = (500, 100, 60, 80)
    jpeg = _jpeg(_wall_canvas([wall], [(438, 100, 60, 80), (562, 100, 60, 80)]))
    assert _Wall([wall], 0.9).read_wall(jpeg)["indicators"] == [{"tile": "5s", "conf": 0.9}]
    low = _Wall([wall], 0.5).read_wall(jpeg)
    assert low["indicators"] == [] and low["box"] == list(wall)
    assert _Wall([(800, 400, 60, 80)], 0.9).read_wall(jpeg)["indicators"] == []
    assert _Wall([], 0.9).read_wall(jpeg)["indicators"] == []


def test_localdora_route_uses_reader(monkeypatch):
    client = _client(monkeypatch, _Wall([], 0.9))
    r = client.post("/v2/localdora", files=_files())
    assert r.status_code == 200 and r.json()["indicators"] == []


class _Fake:
    model = "fake"

    def observe(self, jpeg, kind, **kw):
        for _ in range(200):  # 그림자가 끝나기를 기다린다(실제 Gemini 는 수 초)
            if server._local_pending == 0:
                break
            time.sleep(0.01)
        return GeminiCall({"notes": ""}, 10, 1, 1, "fake")


def _client(monkeypatch, reader):
    monkeypatch.setattr(server, "LOCAL_SHADOW", True)
    monkeypatch.setattr(server, "_local_reader", reader)
    monkeypatch.setattr(server, "_local_pending", 0)
    monkeypatch.setattr(server, "GEMINI_API_KEY", "")
    client = TestClient(server.app)
    monkeypatch.setattr(server, "recognizer", _Fake())
    return client


def _files():
    return {"image": ("a.jpg", _jpeg(_canvas([])), "image/jpeg")}


def test_seat_response_carries_shadow_answer(monkeypatch):
    seen = []

    class R:
        def read_seat(self, jpeg, frac):
            seen.append(frac)
            return {"framed": True, "hand": ["1m"]}

    r = _client(monkeypatch, R()).post("/v2/seat?hand_frac=0.5", files=_files()).json()
    assert r["local"] == {"framed": True, "hand": ["1m"]} and seen == [0.5]


def test_shadow_failure_does_not_touch_the_answer(monkeypatch):
    class R:
        def read_seat(self, jpeg, frac):
            raise ValueError("boom")

    res = _client(monkeypatch, R()).post("/v2/seat", files=_files())
    assert res.status_code == 200 and res.json()["local"] == {"error": "ValueError"}


def test_shadow_is_skipped_when_queue_is_full(monkeypatch):
    class R:
        def read_seat(self, jpeg, frac):
            raise AssertionError("부르면 안 된다")

    client = _client(monkeypatch, R())
    monkeypatch.setattr(server, "_local_pending", server.LOCAL_QUEUE)
    monkeypatch.setattr(_Fake, "observe", lambda self, jpeg, kind, **kw: GeminiCall({"notes": ""}, 10, 1, 1, "fake"))
    assert "local" not in client.post("/v2/seat", files=_files()).json()


def test_other_kinds_have_no_shadow(monkeypatch):
    class R:
        def read_seat(self, jpeg, frac):
            raise AssertionError("부르면 안 된다")

    assert "local" not in _client(monkeypatch, R()).post("/v2/sticks", files=_files()).json()
