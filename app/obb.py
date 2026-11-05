"""자리 판독용 회전 상자(OBB) 검출기: 출력 해석·크롭 끝 조각 버리기·안깡 묶기·세워 자르기(2026-11-05).

검출기 = YOLO26n-obb 8회차(mahjong_calc experiments/tile_classifier, 앞면·뒷면 2종, 라벨 전수 조사 + 재색칠 증강).
ONNX 출력 1×(4+종류+1)×N = cx, cy, w, h, 앞면 점수, 뒷면 점수, 각도(라디안, 시계 방향 +). 서버 규칙은 축 상자를 가정하므로
회전 상자를 감싸는 축 상자로 넘기고, 분류기 크롭만 회전 상자대로 세워 자른다([upright]).
측정: mahjong_calc metrics 표 1-68~1-70(벤치 어댑터 server_bench/obb_detect.py 와 같은 계산).
"""
from __future__ import annotations

import contextlib
import contextvars

import cv2
import numpy as np
from PIL import Image

CONF, IOU = 0.25, 0.7


def fingerprint(rgb: np.ndarray) -> tuple:
    """같은 그림인지(크롭 함수는 그림만 받으므로 그 그림을 검출한 기록을 이것으로 찾는다)."""
    return rgb.shape, hash(np.ascontiguousarray(rgb[::53, ::47]).tobytes())


def decode(out: np.ndarray, r: float, left: int, top: int, W: int, H: int, cls: int = 0) -> list[tuple]:
    """[(축 상자 (x, y, w, h), 회전 상자 ((cx, cy), (w, h), 각도°))] — 종류 cls(0 앞면, 1 뒷면) 점수로, 감싸는 축 상자로 NMS."""
    keep = out[4 + cls] >= CONF
    cx, cy = ((out[0] - left) / r)[keep], ((out[1] - top) / r)[keep]
    w, h, c, ang = (out[2] / r)[keep], (out[3] / r)[keep], out[4 + cls][keep], out[-1][keep]
    rects, boxes = [], []
    for i in range(len(c)):
        rect = ((float(cx[i]), float(cy[i])), (float(w[i]), float(h[i])), float(np.degrees(ang[i])))
        p = cv2.boxPoints(rect)
        x0, y0 = max(0.0, float(p[:, 0].min())), max(0.0, float(p[:, 1].min()))
        x1, y1 = min(float(W), float(p[:, 0].max())), min(float(H), float(p[:, 1].max()))
        rects.append(rect)
        boxes.append((x0, y0, x1, y1))
    order, kept = list(np.argsort(-c)), []
    while order:
        i = order.pop(0)
        kept.append(i)
        rest = []
        for j in order:
            a, b = boxes[i], boxes[j]
            ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            inter = max(0.0, ix) * max(0.0, iy)
            if inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9) <= IOU:
                rest.append(j)
        order = rest
    res = []
    for i in kept:
        x0, y0, x1, y1 = boxes[i]
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            res.append(((int(x0), int(y0), int(x1 - x0), int(y1 - y0)), rects[i]))
    return res


def slivers(res: list, W: int, H: int) -> list:
    """그림 가장자리(2px 안)에 닿아 있고 짧은 변이 다른 패 짧은 변 중앙값의 절반 미만 = 크롭에 잘린 패 조각 → 버린다(표 1-68 123140)."""
    if len(res) < 3:
        return res
    tw = float(np.median([min(rc[1]) for _, rc in res]))
    edge = lambda b: b[0] <= 2 or b[1] <= 2 or b[0] + b[2] >= W - 2 or b[1] + b[3] >= H - 2
    return [(b, rc) for b, rc in res if not (edge(b) and min(rc[1]) < 0.5 * tw)]


def kans(faces: list, backs: list) -> list[tuple]:
    """안깡 모양: 이웃한 앞면 두 장 A·B(중심 거리 0.6~1.35 × 패 짧은 변) 의 줄 방향 양끝에 뒷면이 한 장씩(A − d, B + d 에서 0.45 × 짧은 변 안).
    [(뒷면 (상자, 회전 상자), 대신 읽을 앞면 상자)] 둘씩. 안깡은 네 장이 같은 패라 뒷면 자리는 옆 앞면을 대신 읽는다."""
    if len(faces) < 2 or not backs:
        return []
    tw = float(np.median([min(rc[1]) for _, rc in faces]))
    cen = lambda rc: np.array(rc[0], float)
    out, used = [], set()
    for i, (a, ra) in enumerate(faces):
        for j, (b, rb) in enumerate(faces):
            if j <= i:
                continue
            d = cen(rb) - cen(ra)
            if not 0.6 * tw <= np.linalg.norm(d) <= 1.35 * tw:
                continue
            ca = min(backs, key=lambda x: np.linalg.norm(cen(x[1]) - (cen(ra) - d)))
            cb = min(backs, key=lambda x: np.linalg.norm(cen(x[1]) - (cen(rb) + d)))
            if ca is cb or np.linalg.norm(cen(ca[1]) - (cen(ra) - d)) > 0.45 * tw or np.linalg.norm(cen(cb[1]) - (cen(rb) + d)) > 0.45 * tw:
                continue
            if {ca[0], cb[0]} & used:
                continue
            used |= {ca[0], cb[0]}
            out += [(ca, a), (cb, b)]
    return out


def upright(rgb: np.ndarray, rect, pad: float) -> Image.Image:
    """회전 상자를 가장 작은 회전으로 반듯하게 세워 자른다(긴 변 방향은 그대로 — 눕힌 패는 눕힌 채로, 방향 판단은 축 상자 모양으로 하므로).
    여유는 축 상자 크롭과 같은 비율."""
    (cx, cy), (w, h), a = rect
    a = ((a + 45) % 90) - 45
    if round(((rect[2] - a) / 90)) % 2:
        w, h = h, w
    W2, H2 = w * (1 + 2 * pad), h * (1 + 2 * pad)
    R = int(np.ceil(np.hypot(W2, H2) / 2)) + 2
    H, W = rgb.shape[:2]
    x0, y0 = int(cx) - R, int(cy) - R
    part = np.full((2 * R, 2 * R, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + 2 * R), min(H, y0 + 2 * R)
    part[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    M = cv2.getRotationMatrix2D((cx - x0, cy - y0), a, 1.0)
    rot = cv2.warpAffine(part, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    c = (cx - x0, cy - y0)
    return Image.fromarray(rot[max(0, int(c[1] - H2 / 2)):int(c[1] + H2 / 2), max(0, int(c[0] - W2 / 2)):int(c[0] + W2 / 2)])


class SeatDet:
    """한 요청 동안의 회전 상자 기록: {(그림 지문, 축 상자): 회전 상자}, 안깡 뒷면 → 대신 읽을 앞면, 끼운 뒷면 상자."""

    def __init__(self):
        self.rects: dict = {}
        self.kan_read: dict = {}
        self.kan_backs: set = set()


_CUR: contextvars.ContextVar[SeatDet | None] = contextvars.ContextVar("seat_det", default=None)


@contextlib.contextmanager
def session():
    """자리 판독 한 번. 안에서 검출한 그림의 크롭은 [crop_box] 로 세워 자른다. 요청마다 따로(스레드·요청 사이 섞이지 않게)."""
    tok = _CUR.set(SeatDet())
    try:
        yield _CUR.get()
    finally:
        _CUR.reset(tok)


def current() -> SeatDet | None:
    return _CUR.get()


def record(rgb: np.ndarray, res: list, kan_pairs: list) -> list:
    """검출 결과를 지금 기록에 남기고 축 상자 목록을 돌려준다(안깡 뒷면 상자 포함)."""
    d = _CUR.get()
    boxes = [b for b, _ in res]
    fp = fingerprint(rgb)
    for (bb, rc), face in kan_pairs:
        boxes.append(bb)
        if d is not None:
            d.rects[(fp, bb)] = rc
            d.kan_read[(fp, bb)] = face
            d.kan_backs.add(bb)
    if d is not None:
        d.rects.update({(fp, b): rc for b, rc in res})
    return boxes


def crop_box(rgb: np.ndarray, b, pad: float) -> Image.Image | None:
    """기록에 있는 상자면 세워 자른 크롭(안깡 뒷면이면 옆 앞면), 없으면 None(부르는 쪽이 축 상자로 자른다)."""
    d = _CUR.get()
    if d is None:
        return None
    fp = fingerprint(rgb)
    key = tuple(int(v) for v in b)
    key = d.kan_read.get((fp, key), key)
    rect = d.rects.get((fp, key))
    if rect is None:
        return None if key == tuple(int(v) for v in b) else _axis(rgb, key, pad)
    return upright(rgb, rect, pad)


def _axis(rgb: np.ndarray, b, pad: float) -> Image.Image:
    x, y, w, h = b
    H, W = rgb.shape[:2]
    px, py = int(w * pad), int(h * pad)
    return Image.fromarray(rgb[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)])


def is_kan_back(b) -> bool:
    d = _CUR.get()
    return d is not None and tuple(int(v) for v in b) in d.kan_backs
