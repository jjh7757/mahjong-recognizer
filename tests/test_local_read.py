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


def test_river_frame_skips_cut_hand_row_and_whole_hand_row():
    river = _row(6, 300, 100) + _row(2, 300, 186)
    cut = [(x, 760, w, 40) for x, _, w, _ in _row(10, 100)]
    fr = lr.river_frame(_canvas(river + cut), river + cut)
    assert fr.count == 8 and fr.last == river[-1], "크롭 아래 끝에 잘린 손패가 아니라 기기 쪽 강"
    whole = _row(9, 100, 600)
    fr = lr.river_frame(_canvas(river[:5] + whole), river[:5] + whole)
    assert fr.count == 5 and fr.last == river[4], "한 줄에 7장 넘으면 강이 아니라 손패 줄"


def test_river_frame_keeps_riichi_tile_but_not_neighbours():
    first = _row(5, 300, 200)
    riichi = (630, 212, 80, 56)  # 첫 줄 끝의 눕힌 패
    second = _row(3, 300, 286)
    neighbour_up = (640, 110, 60, 80)  # 첫 줄 바로 위에 붙은 옆 자리 세운 패
    neighbour_side = (710, 120, 80, 56)  # 그 옆의 옆 자리 눕힌 패
    boxes = first + second + [riichi, neighbour_up, neighbour_side]
    fr = lr.river_frame(_canvas(boxes), boxes)
    seq = [b for r in fr.rows for b in r]
    assert riichi in seq and neighbour_side not in seq and neighbour_up not in seq
    assert [b for b, s in zip(seq, fr.sideways) if s] == [riichi]
    assert fr.last == second[-1]
    assert lr.river_frame(_canvas([]), []) is None


def test_river_frame_split_tile_is_not_riichi():
    row = _row(5, 300, 200)
    top, bottom = (630, 200, 70, 52), (630, 252, 70, 28)  # 세운 패 하나가 위아래 두 상자로 쪼개짐
    boxes = row + [top, bottom]
    fr = lr.river_frame(_canvas(boxes, blank=(top, bottom)), boxes)
    assert not any(fr.sideways) and top in fr.rows[0] and bottom not in fr.rows[0]


class _Seat(lr.LocalReader):
    """검출은 주어진 상자, 분류는 고정 답."""

    def __init__(self, faces, tile="1m", conf=0.95):
        super().__init__()
        self.faces, self.tile, self.conf = faces, tile, conf

    def detect(self, rgb, low=True):
        return self.faces

    def probs(self, ims):
        p = np.full((len(ims), len(lr.CLASSES)), (1 - self.conf) / (len(lr.CLASSES) - 1))
        p[:, lr.CLASSES.index(self.tile)] = self.conf
        return p


def test_read_seat_reports_classifier_confidence():
    hand = _row(13)
    o = _Seat(hand, conf=0.62).read_seat(_jpeg(_canvas(hand)), 0.7)
    assert o["framed"] is True and o["filled"] == 0
    assert o["hand"] == ["1m"] * 13 and o["hand_conf"] == [0.62] * 13


def test_read_seat_fills_blank_gap_as_low_confidence_haku():
    hand = _row(13)
    gap = hand[5]
    seen = [b for b in hand if b != gap]
    jpeg = _jpeg(_canvas(hand, blank=(gap,)))
    o = _Seat(seen).read_seat(jpeg, 0.7)
    assert o["framed"] is True and o["filled"] == 1
    assert o["hand"][5] == "5z" and o["hand_conf"][5] == lr.FILLED_CONF
    assert o["hand"].count("1m") == 12 and o["hand_conf"][0] == 0.95, "다른 패는 분류기 답 그대로"
    assert _Seat(seen).read_seat_app(jpeg, 0.7)["tiles"][5] == {"tile": "5z", "conf": lr.FILLED_CONF}


def test_read_seat_does_not_fill_patterned_gap():
    hand = _row(13)
    seen = [b for b in hand if b != hand[5]]
    o = _Seat(seen).read_seat(_jpeg(_canvas(hand)), 0.7)
    assert o["framed"] is False and o["filled"] == 0 and o["hand"] == [], "무늬 있는 패를 놓친 칸은 白 이 아니다"


class _Wall(lr.LocalReader):
    def __init__(self, faces, conf):
        super().__init__()
        self.faces, self.conf = faces, conf

    def detect(self, rgb, low=True):
        return self.faces

    def read_any(self, rgb, b):
        return "5s", self.conf

    read_small = read_any

    def read_turned(self, rgb, b, tw, th):
        return "5s", self.conf


def test_read_wall_answers_only_flanked_and_confident_tile():
    wall = (500, 100, 60, 80)
    jpeg = _jpeg(_wall_canvas([wall], [(438, 100, 60, 80), (562, 100, 60, 80)]))
    assert _Wall([wall], 0.9).read_wall(jpeg)["indicators"] == [{"tile": "5s", "conf": 0.9}]
    low = _Wall([wall], 0.5).read_wall(jpeg)
    assert low["indicators"] == [] and low["box"] == list(wall)
    assert _Wall([(800, 400, 60, 80)], 0.9).read_wall(jpeg)["indicators"] == []
    assert _Wall([], 0.9).read_wall(jpeg)["indicators"] == []


def test_read_wall_upscales_small_tiles():
    """추적 프레임(1020×764)의 패(약 30px)는 read_small 로, 화료 원본의 패(약 90px)는 read_any 로 읽는다."""
    calls = []

    class _W(_Wall):
        def read_any(self, rgb, b):
            calls.append("any")
            return "5s", 0.9

        def read_small(self, rgb, b):
            calls.append("small")
            return "5s", 0.9

    small = (500, 100, 60, 80)  # 긴 변 80 < SMALL_TILE
    _W([small], 0.9).read_wall(_jpeg(_wall_canvas([small], [(438, 100, 60, 80), (562, 100, 60, 80)])))
    big = (500, 100, 90, 120)  # 화료 원본의 패 크기
    _W([big], 0.9).read_wall(_jpeg(_wall_canvas([big], [(408, 100, 90, 120), (592, 100, 90, 120)])))
    assert calls == ["small", "any"]


def test_read_wall_reads_pairs_turned():
    """탁자 위 짝은 비스듬할 수 있어 read_turned(15° 단위)로 읽는다."""
    calls = []

    class _W(_Wall):
        def read_turned(self, rgb, b, tw, th):
            calls.append(("turned", b))
            return "5s", 0.9

    pair = [(800, 400, 60, 80), (862, 400, 60, 80)]
    _W(pair, 0.9).read_wall(_jpeg(_canvas(pair)))
    assert [c[0] for c in calls] == ["turned", "turned"]


def test_read_small_resizes_crop_to_small_tile(monkeypatch):
    sizes = []
    r = lr.LocalReader()
    monkeypatch.setattr(r, "probs", lambda ims: (sizes.extend(i.size for i in ims), np.eye(len(lr.CLASSES))[:len(ims)])[1])
    r.read_small(np.zeros((400, 600, 3), np.uint8), (100, 100, 24, 32))
    assert sizes and all(max(s) == lr.SMALL_TILE for s in sizes), sizes


def test_localdora_route_uses_reader(monkeypatch):
    client = _client(monkeypatch, _Wall([], 0.9))
    r = client.post("/v2/localdora", files=_files())
    assert r.status_code == 200 and r.json()["indicators"] == []


def test_read_seat_app_splits_separate_tile_and_melds(monkeypatch):
    r = lr.LocalReader()
    base = {"framed": True, "faces": 14, "filled": 0, "ms": {}, "rss_mb": 1}
    monkeypatch.setattr(r, "read_seat", lambda j, f: {**base, "hand": ["1m"] * 10 + ["5s"], "hand_conf": [0.9] * 10 + [0.4], "melds": ["6z 6z 6z"], "meld_conf": [0.6]})
    o = r.read_seat_app(b"", 0.7)
    assert o["face_up"] is True and len(o["tiles"]) == 10 and o["separate_tile"] == {"tile": "5s", "conf": 0.4}
    assert o["tiles"][0] == {"tile": "1m", "conf": 0.9}, "분류기 확신도를 그대로 보낸다"
    assert o["melds"][0]["tiles"] == ["6z", "6z", "6z"] and o["melds"][0]["conf"] == 0.6
    monkeypatch.setattr(r, "read_seat", lambda j, f: {**base, "hand": ["1m"] * 13, "hand_conf": [0.9] * 13, "melds": [], "meld_conf": []})
    assert r.read_seat_app(b"", 0.7)["separate_tile"] is None


def test_read_seat_app_without_frame_has_no_tiles(monkeypatch):
    r = lr.LocalReader()
    base = {"framed": False, "hand": [], "hand_conf": [], "melds": [], "meld_conf": [], "filled": 0, "ms": {}, "rss_mb": 1}
    monkeypatch.setattr(r, "read_seat", lambda j, f: {**base, "faces": 20})
    o = r.read_seat_app(b"", 0.7)
    assert o["face_up"] is True and o["tiles"] == []
    monkeypatch.setattr(r, "read_seat", lambda j, f: {**base, "faces": 0})
    assert r.read_seat_app(b"", 0.7)["face_up"] is False


def test_read_wall_reports_lone_pairs():
    pair = [(800, 400, 60, 80), (862, 400, 60, 80)]
    row = [(100 + 62 * i, 600, 60, 80) for i in range(3)]
    o = _Wall(pair + row, 0.9).read_wall(_jpeg(_canvas(pair + row)))
    assert len(o["pairs"]) == 1 and [x["tile"] for x in o["pairs"][0]] == ["5s", "5s"]
    assert sorted(x["box"] for x in o["pairs"][0]) == sorted(list(b) for b in pair), "짝의 상자(픽셀)도 돌려준다"
    assert o["size"] == [2000, 800]


def test_local_routes_answer_in_app_shape(monkeypatch):
    client = _client(monkeypatch, lr.LocalReader())
    seat = client.post("/v2/localseat", files=_files(), params={"hand_frac": 0.7}).json()
    assert seat["face_up"] is False and seat["tiles"] == []
    rivers = client.post("/v2/locallast3", files=[("images", ("a.jpg", _jpeg(_canvas([])), "image/jpeg"))]).json()
    assert rivers["method"] == "frame" and rivers["rivers"] == [{"last_tile": None, "tile_count": 0, "sideways": []}]


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
