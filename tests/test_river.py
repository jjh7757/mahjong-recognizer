from pathlib import Path

import numpy as np
import pytest

from app import observe as ob
from app import river
from app.recognizer import GeminiCall, GeminiRecognizer


def _row(y, n, x0=100, w=60, h=80, gap=4):
    return [(x0 + i * (w + gap), y, w, h) for i in range(n)]


def test_rows_keep_sagging_tile_in_its_row():
    # 셋째 줄 첫 패가 패 높이 1/3 쯤 아래로 처져도 같은 줄(마지막 패는 오른쪽 끝 6통 자리)
    boxes = _row(100, 6) + _row(184, 6) + [(100, 295, 60, 80)] + _row(268, 4, x0=164)
    rows = river.to_rows(boxes, 80)
    assert [len(r) for r in rows] == [6, 6, 5]
    assert rows[-1][-1] == (164 + 3 * 64, 268, 60, 80)


def test_own_rows_drops_neighbours_and_needs_full_upper_rows():
    neighbour = _row(0, 2, x0=0)  # 위쪽 이웃 강 2장(6장이 아니라 멈춘다)
    side = [(40, 184, 60, 80)]  # 강 왼쪽 바깥 패는 띠 밖
    rows = river.to_rows(neighbour + _row(100, 6, x0=200) + side + _row(184, 6, x0=200) + _row(268, 5, x0=200), 80)
    own = river.own_rows(rows, 60, 80)
    assert [len(r) for r in own] == [6, 6, 5]


def test_dedupe_keeps_one_box_per_tile():
    assert river.dedupe([(0, 0, 60, 80), (2, 3, 55, 75), (100, 0, 60, 80)]) == [(0, 0, 60, 80), (100, 0, 60, 80)]


def test_nms_and_decode_map_back_to_crop_pixels():
    # 1280×640 크롭 → 640 입력: 축소 0.5, 위아래 160 패딩. 같은 패에 겹친 두 예측 + 낮은 확신도 하나
    out = np.zeros((41, 3), np.float32)
    out[:4, 0] = [320, 320, 30, 40]
    out[:4, 1] = [321, 321, 30, 40]
    out[:4, 2] = [100, 200, 30, 40]
    out[4 + 5, 0], out[4 + 5, 1], out[4 + 7, 2] = 0.9, 0.8, 0.01
    boxes = river.decode(out, 0.5, 0, 160, 1280, 640, conf=0.04, iou=0.7)
    assert boxes == [(610, 280, 60, 80)]


class _Locator:
    def __init__(self, ok=True):
        self.ok = ok

    def locate(self, jpeg):
        rgb = np.full((300, 400, 3), 255, np.uint8)
        if not self.ok:
            return None, rgb
        return river.RiverFrame([_row(10, 6, x0=10, w=50, h=60), _row(80, 3, x0=10, w=50, h=60)], 12,
                                [False] * 9), rgb


def _recognizer(locator):
    r = object.__new__(GeminiRecognizer)
    r.river_locator = locator
    r.model = "fake"
    r.calls = []

    def call(jpeg, prompt, schema, extra_images=(), **kw):
        r.calls.append((prompt, schema, len(extra_images)))
        if "패 한 장을" in prompt:
            data = {f"photo{i + 1}": {"tile": {"code": c, "confidence": 0.9}} for i, c in enumerate(["6s", "3s", "7z"])}
        else:
            data = {f"photo{i + 1}": {"last_tile": {"code": "8s", "confidence": 0.5}, "tile_count": 9} for i in range(3)}
        return GeminiCall(data, 10, 1, 1, "fake")

    r.call = call
    return r


def test_last_multi_reads_cropped_last_tiles_when_frames_found():
    r = _recognizer(_Locator(ok=True))
    c = r.observe_last_multi([b"a", b"b", b"c"])
    assert c.data["method"] == "frame"
    assert [x["last_tile"]["tile"] for x in c.data["rivers"]] == ["6s", "3s", "7z"]
    assert [x["tile_count"] for x in c.data["rivers"]] == [9, 9, 9]
    assert c.data["rivers"][0]["rows"] == [6, 3]
    prompt, schema, extra = r.calls[0]
    assert prompt == ob.last_tile_prompt(3) and extra == 3  # 조각 2장 + 참고표


def test_last_multi_falls_back_to_whole_read_when_any_frame_missing():
    r = _recognizer(_Locator(ok=False))
    c = r.observe_last_multi([b"a", b"b", b"c"])
    assert c.data["method"] == "whole"
    assert r.calls[0][0] == ob.last_multi_prompt(3)


def test_last_tile_prompt_is_the_measured_text():
    p = ob.last_tile_prompt(3)
    assert p.startswith("사진 3장은 각각 리치마작 패 한 장을 위에서 찍은 것입니다")
    assert "- photo1, photo2, photo3: 첫째부터 순서대로" in p and p.endswith("참고표는 답에 넣지 않습니다.")


SAMPLE = Path("C:/Users/jjh7757/dev/mahjong_calc/experiments/field_20261024/grid5_scenes/C/k26_1/last_right.jpg")


@pytest.mark.skipif(not SAMPLE.exists() or not river.MODEL.exists(), reason="실제 강 크롭·모델이 있는 개발 PC 에서만")
def test_real_crop_picks_the_sagging_row_last_tile():
    # C 장면 오른쪽 강: 셋째 줄 첫 8삭이 아래로 처져 있다. 마지막 패는 그 줄 오른쪽 끝(6통) — 처진 8삭이 아니어야 한다.
    f, rgb = river.RiverLocator().locate(SAMPLE.read_bytes())
    assert f is not None
    x, y, w, h = f.last
    assert x > 650 and y < 600  # 6통(오른쪽, 위) — 8삭은 x≈355, y≈575~715
