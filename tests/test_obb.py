"""자리 판독 회전 상자 검출기(app/obb.py)와 모서리 白 퐁 2장 규칙(local_read.corner_pair)."""
from __future__ import annotations

import threading

import numpy as np

from app import local_read as lr
from app import obb
from app import seat as st


def _out(rows: list[tuple]) -> np.ndarray:
    """(cx, cy, w, h, 앞면, 뒷면, 각도 라디안) 행들 → ONNX 출력 모양 7×N."""
    return np.array(rows, np.float32).T


def test_decode_keeps_class_and_wraps_rotated_box():
    out = _out([(100, 100, 40, 60, 0.9, 0.0, np.radians(30)), (300, 100, 40, 60, 0.1, 0.8, 0.0), (102, 101, 40, 60, 0.6, 0.0, np.radians(30))])
    faces = obb.decode(out, 1.0, 0, 0, 1000, 1000)
    assert len(faces) == 1, "겹친 상자는 NMS 로 하나, 뒷면 점수만 높은 상자는 앞면에서 빠진다"
    (x, y, w, h), rect = faces[0]
    assert w > 40 and h > 60, "기운 패는 감싸는 축 상자가 더 크다"
    assert abs(rect[2] - 30) < 1e-3
    backs = obb.decode(out, 1.0, 0, 0, 1000, 1000, cls=1)
    assert [b for b, _ in backs] == [(280, 70, 40, 60)]


def test_slivers_drops_thin_piece_on_crop_edge_only():
    rc = lambda cx, w: ((cx, 50.0), (w, 60.0), 0.0)
    res = [((0, 20, 10, 60), rc(5, 10)), ((100, 20, 40, 60), rc(120, 40)), ((150, 20, 40, 60), rc(170, 40)), ((300, 20, 12, 60), rc(306, 12))]
    kept = [b for b, _ in obb.slivers(res, 1000, 200)]
    assert (0, 20, 10, 60) not in kept, "가장자리에 닿은 가는 조각은 버림"
    assert (300, 20, 12, 60) in kept, "가운데의 가는 상자는 그대로(조각인지 모름)"


def test_kans_pairs_backs_at_both_ends_of_two_faces():
    rc = lambda cx: ((float(cx), 50.0), (40.0, 60.0), 0.0)
    faces = [((180, 20, 40, 60), rc(200)), ((222, 20, 40, 60), rc(242))]
    backs = [((138, 20, 40, 60), rc(158)), ((264, 20, 40, 60), rc(284)), ((600, 20, 40, 60), rc(620))]
    got = obb.kans(faces, backs)
    assert [(b[0], f) for b, f in got] == [((138, 20, 40, 60), (180, 20, 40, 60)), ((264, 20, 40, 60), (222, 20, 40, 60))]
    assert obb.kans(faces, backs[2:]) == [], "양끝에 뒷면이 없으면 안깡 아님"


def test_crop_box_only_inside_session_and_reads_face_for_kan_back():
    rgb = np.zeros((200, 400, 3), np.uint8)
    rgb[20:80, 180:220] = 200
    face = (180, 20, 40, 60)
    back = (138, 20, 40, 60)
    assert obb.crop_box(rgb, face, 0.12) is None, "기록 밖이면 부르는 쪽이 축 상자로"
    with obb.session():
        boxes = obb.record(rgb, [(face, ((200.0, 50.0), (40.0, 60.0), 0.0))], [((back, ((158.0, 50.0), (40.0, 60.0), 0.0)), face)])
        assert boxes == [face, back] and obb.is_kan_back(back) and not obb.is_kan_back(face)
        a, b = np.asarray(obb.crop_box(rgb, face, 0.12)), np.asarray(obb.crop_box(rgb, back, 0.12))
        assert a.shape == b.shape and (a == b).all(), "안깡 뒷면은 옆 앞면을 대신 읽는다"
        assert obb.crop_box(np.ones_like(rgb), face, 0.12) is None, "다른 그림이면 기록을 안 쓴다"
    assert obb.crop_box(rgb, face, 0.12) is None and not obb.is_kan_back(back), "요청이 끝나면 기록이 사라진다"


def test_sessions_do_not_leak_between_threads():
    rgb = np.zeros((100, 100, 3), np.uint8)
    seen = []
    ready, done = threading.Event(), threading.Event()

    def other():
        ready.wait()
        seen.append(obb.crop_box(rgb, (10, 10, 20, 30), 0.12))
        done.set()

    t = threading.Thread(target=other)
    t.start()
    with obb.session():
        obb.record(rgb, [((10, 10, 20, 30), ((20.0, 25.0), (20.0, 30.0), 0.0))], [])
        ready.set()
        done.wait()
    t.join()
    assert seen == [None]


def _frame(hand_right: int = 500) -> st.SeatFrame:
    hand = [(hand_right - 40 * (13 - i), 100, 38, 52) for i in range(10)]
    return st.SeatFrame(hand, None, [], [], 38.0, 52.0, len(hand), split=True)


def test_corner_pair_turns_two_blank_boxes_into_pon():
    rgb = np.full((300, 900, 3), 230, np.uint8)  # 무늬 없는 흰 바탕
    fr = _frame()
    two = [(600, 110, 38, 52), (640, 110, 38, 52)]
    got = lr.corner_pair(rgb, fr, fr.hand + two)
    assert got == [*two, (680, 110, 38, 52)], "같은 간격으로 세 번째 상자"
    assert lr.corner_pair(rgb, fr, fr.hand + two[:1]) is None, "한 장이면 퐁으로 보지 않는다"
    far = [(600, 110, 38, 52), (800, 110, 38, 52)]
    assert lr.corner_pair(rgb, fr, fr.hand + far) is None, "떨어진 두 장은 묶지 않는다"


def test_white_inside_ignores_felt_corners_of_tilted_tile():
    white = np.zeros((300, 300), bool)
    rect = ((150.0, 150.0), (100.0, 120.0), 36.0)
    import cv2

    m = np.zeros(white.shape, np.uint8)
    cv2.fillPoly(m, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 1)
    white[m > 0] = True  # 기운 패 면만 흰색
    p = cv2.boxPoints(rect)
    x0, y0, x1, y1 = int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())
    assert white[y0:y1, x0:x1].mean() < 0.65, "감싸는 축 상자는 귀퉁이 펠트 때문에 낮다"
    assert obb.white_inside(white, rect) > 0.95


def test_keep_seat_keeps_tilted_tile_by_rotated_box():
    import cv2

    rgb = np.zeros((300, 300, 3), np.uint8)
    rgb[...] = (30, 110, 60)  # 펠트
    rect = ((150.0, 150.0), (56.0, 90.0), 45.0)  # 패 면 5040 / 감싸는 축 상자 약 10370 → 0.49
    cv2.fillPoly(rgb, [np.round(cv2.boxPoints(rect)).astype(np.int32)], (235, 235, 235))
    p = cv2.boxPoints(rect)
    box = (int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max() - p[:, 0].min()), int(p[:, 1].max() - p[:, 1].min()))
    with obb.session():
        obb.record(rgb, [(box, rect)], [])
        assert lr.ROT_WHITE, "기본 켬(표 1-73)"
        assert lr.LocalReader.keep_seat(rgb, [box]) == [box]
    from app import river as rv

    assert rv.keep_faces([box], rv.white_mask(rgb)) == [], "축 상자로 재면 버려지는 기운 패"
