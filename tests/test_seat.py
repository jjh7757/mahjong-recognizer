"""자리 틀(app/seat.py): 규칙·조각 그림·조각 → 자리 판독 변환, 그리고 recognizer 의 틀 → 조각 읽기 → 통째 읽기 복귀."""
import io

import numpy as np
from PIL import Image

from app import observe as ob
from app import seat
from app.recognizer import GeminiCall, GeminiRecognizer


def _row(n, x0=100, y=300, w=60, h=80, gap=4):
    return [(x0 + i * (w + gap), y, w, h) for i in range(n)]


def _canvas(boxes, size=(2000, 600)):
    """초록 펠트 위에 상자마다 흰 패 면 + 가운데 검은 무늬."""
    rgb = np.zeros((size[1], size[0], 3), np.uint8)
    rgb[..., 1] = 120
    for x, y, w, h in boxes:
        rgb[y:y + h, x:x + w] = 245
        rgb[y + h // 3:y + 2 * h // 3, x + w // 3:x + 2 * w // 3] = 20
    return rgb


def test_structure_hand_separate_meld_and_loose():
    hand = _row(10)
    sep = [(hand[-1][0] + 64 + 50, 300, 60, 80)]  # 손패 줄 오른쪽 패 두 개 안
    meld = _row(3, x0=1500)
    meld[1] = (meld[1][0], 310, 80, 60)  # 가운데 패를 눕힘
    loose = [(1000, 100, 60, 80)]  # 띠 위쪽 따로 놓인 패
    rgb = _canvas(hand + sep + meld + loose)
    fr = seat.structure(rgb, hand + sep + meld + loose)
    assert [b[0] for b in fr.hand] == [b[0] for b in hand]
    assert fr.separate == sep[0]
    assert fr.melds == [sorted(meld)] and fr.loose == loose
    assert fr.count == 14 and seat.valid(fr) and seat.safe_ok(fr, rgb)
    kinds = [k for k, _ in fr.pieces()]
    assert kinds == ["hand"] * 10 + ["sep"] + ["meld0"] * 3 + ["loose"]
    assert seat.sideways(meld[1]) and not seat.sideways(meld[0])


def test_valid_needs_13_or_14_and_strict_rejects_uneven_row():
    hand = _row(12)
    fr = seat.structure(_canvas(hand), hand)
    assert not seat.valid(fr)  # 12장
    hand = _row(13)
    hand[5] = (hand[5][0], 300, 60, 110)  # 한 상자만 높이가 다름(+37%)
    fr = seat.structure(_canvas(hand), hand)
    assert seat.valid(fr) and not seat.strict_ok(fr)


def test_safe_ok_rejects_blank_piece():
    hand = _row(13)
    rgb = _canvas(hand)
    x, y, w, h = hand[3]
    rgb[y:y + h, x:x + w] = 245  # 무늬 없는 상자(테이블 테두리처럼)
    fr = seat.structure(rgb, hand)
    assert seat.strict_ok(fr) and not seat.safe_ok(fr, rgb)


def test_sheet_lays_pieces_in_numbered_cells():
    hand = _row(13)
    rgb = _canvas(hand)
    fr = seat.structure(rgb, hand)
    im = Image.open(io.BytesIO(seat.sheet(rgb, fr, cell_h=100, per_row=7)))
    assert im.height == (100 + 40) * 2  # 13칸 = 7 + 6, 두 줄
    assert im.width < 7 * 120


def test_to_seat_maps_pieces_and_rejects_back():
    hand = _row(10)
    meld = _row(3, x0=1500)
    meld[0] = (1500, 310, 80, 60)
    fr = seat.structure(_canvas(hand + meld), hand + meld)
    tiles = [{"tile": "1m", "conf": 0.9}] * 10 + [{"tile": "6z", "conf": 0.8}] * 3
    out = seat.to_seat(fr, tiles)
    assert len(out["tiles"]) == 10 and out["separate_tile"] is None and out["face_up"]
    assert out["melds"] == [{"tiles": ["6z", "6z", "6z"], "sideways": [0], "face_down": [], "conf": 0.8}]
    assert seat.to_seat(fr, tiles[:-1] + [None]) is None  # 울기 조각이 뒷면
    assert seat.to_seat(fr, tiles[:-1]) is None  # 장수가 다름


def test_normalize_pieces_pads_and_drops_back():
    out = ob.normalize_pieces({"tiles": [{"code": "5p", "confidence": 0.7}, {"code": "back", "confidence": 1}]}, 3)
    assert out == [{"tile": "5p", "conf": 0.7}, None, None]
    assert ob.piece_schema(4)["properties"]["tiles"]["minItems"] == 4
    assert "1~4" in ob.piece_prompt(4) and "연보라" not in ob.piece_prompt(4)


class _Locator:
    def __init__(self, found=True):
        self.found = found

    def locate(self, jpeg):
        hand = _row(13)
        rgb = _canvas(hand)
        return (seat.structure(rgb, hand) if self.found else None), rgb


def _recognizer(locator, answers):
    r = object.__new__(GeminiRecognizer)
    r.seat_reads, r.model, r.seat_locator = 3, "fake", locator
    seen = []

    def fake_call(jpeg, prompt, schema, media_resolution=None, extra_images=()):
        seen.append((prompt, media_resolution, len(extra_images)))
        a = next(answers)
        if "tiles" in schema["properties"] and "face_up" not in schema["properties"]:
            return GeminiCall({"tiles": [{"code": t, "confidence": 0.9} for t in a.split()]}, 10, 100, 20, "fake")
        return GeminiCall({"face_up": True, "tiles": [{"code": t, "confidence": 0.9} for t in a.split()],
                           "separate_tile": None, "melds": [], "loose": []}, 10, 200, 20, "fake")

    r.call = fake_call
    return r, seen


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (900, 300), "white").save(buf, format="JPEG")
    return buf.getvalue()


def test_seat_frame_reads_pieces_three_times_and_agrees():
    hand13 = " ".join(["1m"] * 13)
    r, seen = _recognizer(_Locator(), iter([hand13] * 3))
    c = r.observe(_jpeg(), "seat", ultra_high=True)
    assert c.data["method"] == "frame" and c.data["frame"]["hand"] == 13
    assert [t["tile"] for t in c.data["tiles"]] == ["1m"] * 13
    assert c.data["agreement"] == {"reads": 3, "agree": 3, "unanimous": True, "alternatives": [], "requested": 3}
    assert c.input_tokens == 300
    # 조각 그림은 ULTRA_HIGH, 뒤에 참고표 한 장
    assert seen == [(ob.piece_prompt(13), "MEDIA_RESOLUTION_ULTRA_HIGH", 1)] * 3


def test_seat_frame_not_found_falls_back_to_whole():
    hand = "1s 2s 3s"
    r, seen = _recognizer(_Locator(found=False), iter([hand] * 3))
    c = r.observe(_jpeg(), "seat", ultra_high=True)
    assert c.data["method"] == "whole" and "frame" not in c.data
    assert [p for p, _, _ in seen] == [ob.SEAT_PROMPT_V2] * 3


def test_seat_frame_back_piece_falls_back_to_whole():
    pieces = " ".join(["1m"] * 12 + ["back"])
    r, seen = _recognizer(_Locator(), iter([pieces] * 3 + ["1s 2s 3s"] * 3))
    c = r.observe(_jpeg(), "seat", ultra_high=True)
    assert c.data["method"] == "whole" and [t["tile"] for t in c.data["tiles"]] == ["1s", "2s", "3s"]
    assert len(seen) == 6


def test_handonly_ignores_seat_locator_when_frame_off():
    r, seen = _recognizer(_Locator(), iter(["1s 2s 3s"] * 3))
    r.handonly_frame = False
    c = r.observe(_jpeg(), "handonly", ultra_high=True)
    assert c.data["method"] == "whole" and [p for p, _, _ in seen] == [ob.HANDONLY_PROMPT_V2] * 3


def test_valid_adds_tracked_melds():
    hand = _row(10)
    fr = seat.structure(_canvas(hand), hand)
    assert not seat.valid(fr)  # 울기를 잘라 낸 사진: 손패만 10장
    assert seat.valid(fr, extra_melds=1)  # 이미 읽은 울기 하나(3장)를 더하면 13
    assert not seat.valid(fr, extra_melds=2)


class _HandLocator:
    """울기를 잘라 낸 손패 칸(10장). 받은 울기 수로 실제 장수 검사를 한다."""

    def __init__(self):
        self.extra = []

    def locate(self, jpeg, extra_melds=0):
        self.extra.append(extra_melds)
        hand = _row(10)
        rgb = _canvas(hand)
        fr = seat.structure(rgb, hand)
        return (fr if seat.valid(fr, extra_melds) else None), rgb


def test_handonly_frame_uses_tracked_meld_count():
    loc = _HandLocator()
    r, seen = _recognizer(loc, iter([" ".join(["2p"] * 10)] * 3))
    r.handonly_frame = True
    c = r.observe(_jpeg(), "handonly", ultra_high=True, melds=1)
    assert c.data["method"] == "frame" and c.data["frame"]["hand"] == 10
    assert [t["tile"] for t in c.data["tiles"]] == ["2p"] * 10
    assert c.data["melds"] == []  # 울기는 앱이 이미 안다
    assert loc.extra == [1]
    assert seen == [(ob.piece_prompt(10), "MEDIA_RESOLUTION_ULTRA_HIGH", 1)] * 3


def test_handonly_frame_without_meld_count_falls_back_to_whole():
    loc = _HandLocator()
    r, seen = _recognizer(loc, iter(["1s 2s 3s"] * 3))
    r.handonly_frame = True
    c = r.observe(_jpeg(), "handonly", ultra_high=True)  # 울기 수를 안 보낸 예전 앱: 10장은 장수 검사에 걸림
    assert c.data["method"] == "whole" and [p for p, _, _ in seen] == [ob.HANDONLY_PROMPT_V2] * 3
