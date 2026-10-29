"""자리 판독 틀 잡기: YOLO11 패 검출(위치만) → 손패 줄·떨어진 한 장·울기 묶음 → 조각을 번호 칸에 붙인 그림 한 장.

강 끝(app/river.py)과 같은 역할 분담 — 검출기 = 어디, Gemini = 무엇. 패 종류는 여기서 정하지 않는다.
실험·근거: mahjong_calc/experiments/speed/seat_yolo.py·seat_det_variants.py·overfit_check.py (NOTES 23·24).

입력은 격자 5구역의 한 자리 크롭(손패 + 오른쪽 모서리, 그 사람 방향으로 세운 것). 자리 크롭은 가로로 길어(2~3.3:1)
640 하나로 줄이면 패가 15~20px 라, (1) 통째 한 번으로 손패 줄 띠를 찾고 (2) 띠를 가로 4토막으로 잘라 세로로 쌓아(모자이크)
한 번 더 돌린다(추론 2번, 운영 서버 약 1초). 장수(손패 + 떨어진 한 장 + 울기×3)가 13·14 가 아니거나 모양 검사에 걸리면
None → 통째 읽기(지금 방식)로. 흔든 사진에서 "검사 통과했는데 틀린 틀" 0 이 되게 둔 규칙(토막 몫·무늬 없는 상자 버림·
손패 줄 모양·울기 모양)은 사진에 맞춘 값이 아닌 패 모양의 일반 규칙이다.
"""
from __future__ import annotations

import io
import statistics
import threading
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import river as rv

Box = rv.Box

MOSAIC_K = 4  # 손패 띠 토막 수(k 2~6 모두 같은 결과, 가운데 값)
BAND_HALF = 1.5  # 손패 띠 높이 = 가장 많은 상자가 모인 높이 ± 패 1.5개(민감한 값 — 평가 사진으로 정함)
BAND_PAD = 1.5  # 이웃 토막과 겹침 = 패 1.5개
GAP = 0.35  # 손패 줄 안 이웃 틈 < 패 0.35개
SEP_DIST = 2.5  # 떨어진 한 장: 손패 줄 오른쪽 패 2.5개 안
INK_MIN = 0.03  # 상자 안 무늬 비율 최소(정답 조각 최소 5.5%·중앙 17%, 테이블 테두리 0%). 무늬 없는 白 은 통째 읽기로 간다


@dataclass
class SeatFrame:
    hand: list[Box]  # 손패 줄, 왼쪽부터
    separate: Box | None  # 떨어진 한 장
    melds: list[list[Box]]  # 울기 묶음(왼쪽부터), 묶음 안도 왼쪽부터
    loose: list[Box] = field(default_factory=list)  # 손패·울기가 아닌 앞면 패(도라 표시패 후보 등)
    tw: float = 0.0
    th: float = 0.0
    detected: int = 0
    split: bool = False  # 손패 칸·모서리 칸으로 나눠 잡은 틀(frame_split)

    @property
    def count(self) -> int:
        return len(self.hand) + (1 if self.separate else 0) + 3 * len(self.melds)

    def pieces(self) -> list[tuple[str, Box]]:
        """읽을 순서: 손패 → 떨어진 한 장 → 울기 묶음(왼쪽부터) → 따로 놓인 패."""
        out = [("hand", x) for x in self.hand]
        if self.separate:
            out.append(("sep", self.separate))
        for i, m in enumerate(self.melds):
            out += [(f"meld{i}", x) for x in m]
        out += [("loose", x) for x in self.loose]
        return out

    def summary(self) -> dict:
        return {"hand": len(self.hand), "separate": self.separate is not None, "melds": [len(m) for m in self.melds],
                "loose": len(self.loose), "detected": self.detected, **({"split": True} if self.split else {})}


# ---------------------------------------------------------------- 검출(띠 + 모자이크)

def _band(det: rv.TileDetector, rgb: np.ndarray, white: np.ndarray) -> tuple[int, int, float, list[Box]] | None:
    """통째 한 번(640 letterbox) → 패 면 상자가 가장 많이 모인 높이 ± BAND_HALF 패 = 손패 줄 띠. 못 찾으면 None.
    돌려주는 faces 는 통째 검출의 패 면 전부(띠 밖 따로 놓인 패를 찾는 데 쓴다)."""
    faces = rv.dedupe(rv.keep_faces(det.detect(rgb), white))
    if len(faces) < 3:
        return None
    th = statistics.median(f[3] for f in faces)
    cys = sorted(f[1] + f[3] / 2 for f in faces)
    best = max(cys, key=lambda c: sum(abs(c - d) < th for d in cys))
    return int(best - BAND_HALF * th), int(best + BAND_HALF * th), th, faces


def mosaic(det: rv.TileDetector, rgb: np.ndarray, white: np.ndarray, k: int = MOSAIC_K) \
        -> tuple[list[Box], tuple[int, int], list[Box]] | None:
    """손패 띠를 가로 k 토막(이웃과 BAND_PAD 패 겹침)으로 잘라 세로로 쌓아 한 번에 추론 → (띠 안 상자, 띠 (y0, y1), 통째 faces).
    토막마다 자기 몫(겹친 구간의 가운데까지)에 가운데가 있는 상자만 — 겹친 곳의 같은 패를 두 번 잡지 않는다."""
    got = _band(det, rgb, white)
    if got is None:
        return None
    y0, y1, th, faces = got
    H, W = rgb.shape[:2]
    y0, y1 = max(0, int(y0 - 0.5 * th)), min(H, int(y1 + 0.5 * th))
    h = y1 - y0
    if h < 4 or W < 4:
        return None
    ov = int(BAND_PAD * th * 0.75)  # 패 너비 ≈ 높이 × 0.75
    seg = (W + ov * (k - 1)) // k + 1
    xs = [max(0, min(i * (seg - ov), W - seg)) for i in range(k)]
    canvas = np.full((h * k, seg, 3), 114, np.uint8)
    for i, x in enumerate(xs):
        part = rgb[y0:y1, x:x + seg]
        canvas[i * h:i * h + part.shape[0], :part.shape[1]] = part
    boxes = []
    for bx in det.detect(canvas):
        i = min(k - 1, (bx[1] + bx[3] // 2) // h)
        yy = bx[1] - i * h
        if yy < -0.2 * bx[3] or yy + bx[3] > h + 0.2 * bx[3]:
            continue  # 토막 경계를 넘은 상자
        cx = xs[i] + bx[0] + bx[2] / 2
        lo = 0 if i == 0 else (xs[i] + xs[i - 1] + seg) / 2
        hi = W if i == k - 1 else (xs[i + 1] + xs[i] + seg) / 2
        if not lo <= cx < hi:
            continue
        boxes.append((bx[0] + xs[i], max(0, yy) + y0, bx[2], min(bx[3], h - max(0, yy))))
    return boxes, (y0, y1), faces


# ---------------------------------------------------------------- 틀(규칙만)

def cluster(boxes: list[Box], gx: float, gy: float) -> list[list[Box]]:
    """틈이 (gx, gy) 미만인 이웃끼리 묶기."""
    left = list(boxes)
    out = []
    while left:
        g = [left.pop()]
        i = 0
        while i < len(g):
            a = g[i]
            for x in left[:]:
                dx = max(0, max(a[0], x[0]) - min(a[0] + a[2], x[0] + x[2]))
                dy = max(0, max(a[1], x[1]) - min(a[1] + a[3], x[1] + x[3]))
                if dx < gx and dy < gy:
                    g.append(x)
                    left.remove(x)
            i += 1
        out.append(g)
    return out


def structure(rgb: np.ndarray, boxes: list[Box], white: np.ndarray | None = None) -> SeatFrame | None:
    """패 면 상자 → 손패 줄(가장 많은 묶음) + 떨어진 한 장 + 오른쪽 묶음(울기 3~4장) + 나머지(loose).
    사진은 그 사람 쪽에서 본 방향(손패가 아래, 울기는 손패 줄 오른쪽 모서리 쪽)."""
    white = rv.white_mask(rgb) if white is None else white
    faces = rv.dedupe(rv.keep_faces(boxes, white))
    if not faces:
        return None
    upright = [x for x in faces if x[3] > x[2]] or faces
    tw = statistics.median(x[2] for x in upright)
    th = statistics.median(x[3] for x in upright)
    groups = cluster(faces, tw * GAP, th * 0.5)
    hand_g = max(groups, key=lambda g: (len(g), max(x[1] + x[3] for x in g)))
    hand = sorted(hand_g, key=lambda x: x[0])
    rest = [g for g in groups if g is not hand_g]
    hx1 = max(x[0] + x[2] for x in hand)
    hcy = statistics.mean(x[1] + x[3] / 2 for x in hand)
    sep = None
    near = [g for g in rest if len(g) == 1 and 0 <= g[0][0] - hx1 < SEP_DIST * tw
            and abs(g[0][1] + g[0][3] / 2 - hcy) < th * 0.6]
    if near:
        sep = min(near, key=lambda g: g[0][0])[0]
        rest = [g for g in rest if g[0] is not sep]
    melds, loose = [], []
    for g in rest:
        g = sorted(g, key=lambda x: x[0])
        if 3 <= len(g) <= 4 and min(x[0] for x in g) > hx1 - tw:
            melds.append(g)
        else:
            loose.extend(g)
    melds.sort(key=lambda m: m[0][0])
    return SeatFrame(hand, sep, melds, loose, tw, th, len(boxes))


def valid(fr: SeatFrame | None, extra_melds: int = 0) -> bool:
    """장수 검사(정답 없이): 손패 + 떨어진 한 장 + 울기(묶음마다 3장으로 셈)가 13 또는 14.
    [extra_melds] 는 사진 밖에서 이미 읽은 울기 묶음 수(손패 전용 — 울기를 잘라 낸 사진)."""
    return fr is not None and fr.count + 3 * extra_melds in (13, 14)


def strict_ok(fr: SeatFrame) -> bool:
    """손패 줄 모양(사진과 무관한 일반 규칙): (1) 크기가 고르고(높이·너비 중앙값 ±25%) (2) 한 줄로 늘어서고(가운데 y 가
    직선에서 패 높이 0.25 안) (3) 이웃 간격이 고르다(패 너비 0.6~1.4배)."""
    hand = fr.hand
    if len(hand) < 4:
        return False
    hs = [x[3] for x in hand]
    ws = [x[2] for x in hand]
    mh, mw = statistics.median(hs), statistics.median(ws)
    if any(abs(h - mh) > 0.25 * mh for h in hs) or any(abs(w - mw) > 0.25 * mw for w in ws):
        return False
    xs = np.array([x[0] + x[2] / 2 for x in hand], float)
    ys = np.array([x[1] + x[3] / 2 for x in hand], float)
    a, c = np.polyfit(xs, ys, 1)
    if np.max(np.abs(ys - (a * xs + c))) > 0.25 * mh:
        return False
    steps = np.diff(xs)
    return bool(np.all(steps > 0.6 * mw) and np.all(steps < 1.4 * mw))


def ink(rgb: np.ndarray, x: Box) -> float:
    """상자 안쪽(가장자리 15% 빼고)에서 무늬(글자·동그라미·막대) 픽셀 비율: 어둡거나(V<120) 색이 진함(S>80)."""
    bx, by, bw, bh = x
    mx, my = int(bw * 0.15), int(bh * 0.15)
    c = rgb[by + my:by + bh - my, bx + mx:bx + bw - mx]
    if c.size == 0:
        return 0.0
    hsv = np.asarray(Image.fromarray(c).convert("HSV")).astype(int)
    return float(((hsv[..., 2] < 120) | (hsv[..., 1] > 80)).mean())


def safe_ok(fr: SeatFrame, rgb: np.ndarray) -> bool:
    """strict_ok + 울기 모양(상자끼리 너비 30% 넘게 겹치지 않음, 높이가 손패 패 높이의 0.5~1.3배) + 읽을 조각마다 무늬."""
    if not strict_ok(fr):
        return False
    th = statistics.median(x[3] for x in fr.hand)
    for m in fr.melds:
        for a, c in zip(m, m[1:]):
            if a[0] + a[2] - c[0] > 0.3 * min(a[2], c[2]):
                return False
        if any(not 0.5 * th <= x[3] <= 1.3 * th for x in m):
            return False
    return all(ink(rgb, x) >= INK_MIN for kind, x in fr.pieces() if kind != "loose")


# ---------------------------------------------------------------- 탁자 안만(펠트 끝에서 자르기)
# 사용자 제안(2026-10-28): 자리 크롭에서 탁자 끝(펠트 경계) 밖 — 흰 테두리·자리 표시판·바닥 — 을 지우고 검출한다.
# 흰 테두리는 패 면과 색이 같아, 탁자 끝에 놓인 울기 상자가 테두리까지 번지거나(D 8만) 테두리 조각을 패로 잡았다.

TABLE_SCALE = 320  # 펠트 찾기는 긴 변 320 으로 줄여서
TABLE_WIN = 0.08  # 끝에 놓인 패에 가려진 펠트 끝은 좌우·위아래 폭의 8% 안 이웃 끝으로 메운다(울기 3~4장 너비)
# 탁자 끝에서 폭의 6%(패 약 1.7장)까지는 둔다. 울기는 끝에 걸치거나 모서리 흰 테두리 위에 놓인다(B 發 퐁, D 8만) —
# 1.5% 로 바짝 자르면 B·C·D 틀 6/6 → 5/6·울기 합성 7/14 → 4/14, 3%·4.5% 도 합성 4·5/14, 6% 에서 6/6·8/14(틀린 틀 0).
TABLE_PAD = 0.06
TABLE_FILL = 114  # 지운 곳 색(모자이크 바탕과 같은 회색)


def _majority(m: np.ndarray, r: int) -> np.ndarray:
    """(2r+1)² 창에서 절반 넘게 참이면 참 — 펠트 결의 잡티를 지운다."""
    k = 2 * r + 1
    c = np.pad(m.astype(np.int32), r, mode="edge").cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s * 2 > k * k


def _running_max(a: np.ndarray, k: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(np.pad(a, k, constant_values=-1), 2 * k + 1).max(1)


def table_region(rgb: np.ndarray) -> np.ndarray | None:
    """자리 크롭(그 사람 방향, 탁자 끝이 아래·모서리가 오른쪽)에서 탁자 안 마스크. 펠트를 못 찾으면 None(지우지 않음).
    위쪽 1/4 의 펠트에서 펠트 안으로 번진 덩어리 → 칸마다 가장 아래 펠트 = 아래 끝, 줄마다 가장 오른쪽 펠트 = 오른쪽 끝 →
    끝에 놓인 패가 가린 곳은 이웃 끝의 최댓값으로 메움 → 두 끝 안쪽(+ 여유)."""
    H, W = rgb.shape[:2]
    w = TABLE_SCALE
    h = max(1, round(H * w / W))
    small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
    felt = _majority(felt_mask(small), 1)
    reg = felt.copy()
    reg[h // 4:] = False
    if not reg.any() or felt.mean() < 0.05:
        return None
    while True:  # 펠트 안으로 번지기(가장 큰 덩어리, 바닥의 푸른 물건은 안 닿음)
        g = reg.copy()
        g[1:] |= reg[:-1]
        g[:-1] |= reg[1:]
        g[:, 1:] |= reg[:, :-1]
        g[:, :-1] |= reg[:, 1:]
        g &= felt
        if np.array_equal(g, reg):
            break
        reg = g
    k = max(1, int(TABLE_WIN * w))
    bottom = _running_max(np.where(reg.any(0), h - 1 - np.argmax(reg[::-1], 0), -1), k)
    right = _running_max(np.where(reg.any(1), w - 1 - np.argmax(reg[:, ::-1], 1), -1), k)
    pad = TABLE_PAD * w
    inside = (np.arange(h)[:, None] <= bottom[None, :] + pad) & (np.arange(w)[None, :] <= right[:, None] + pad)
    return np.asarray(Image.fromarray(inside).resize((W, H), Image.NEAREST))


def table_only(rgb: np.ndarray) -> np.ndarray:
    """탁자 밖을 회색으로 지운 사본(못 찾으면 그대로)."""
    region = table_region(rgb)
    if region is None:
        return rgb
    out = rgb.copy()
    out[~region] = TABLE_FILL
    return out


def frame(rgb: np.ndarray, det: rv.TileDetector, extra_melds: int = 0, hand_frac: float | None = None,
          table: bool = False) -> SeatFrame | None:
    """자리 크롭 → 틀. 장수·모양 검사를 못 넘으면 None(통째 읽기로).
    [hand_frac] (격자 손패 칸이 자리 크롭에서 차지하는 폭 비율)이 있으면, 한 장 규칙이 못 잡은 자리를 손패 칸·모서리 칸으로
    나눠 한 번 더 잡아 본다(frame_split). 한 장 규칙이 잡은 자리는 그대로 — 나눠 잡기는 더하기만 한다.
    [table] 이면 탁자 끝 밖을 지운 사진으로 검출한다(table_only). 조각 그림은 원본에서 자른다(좌표는 같다)."""
    if table:
        rgb = table_only(rgb)
    white = rv.white_mask(rgb)
    got = mosaic(det, rgb, white)
    if got is None:
        return None
    boxes, _, _ = got
    boxes = [x for x in boxes if ink(rgb, x) >= INK_MIN]  # 무늬 없는 상자(테이블 테두리 등)는 버림
    fr = structure(rgb, boxes, white)
    if not valid(fr, extra_melds) or not safe_ok(fr, rgb):
        fr = frame_split(rgb, det, hand_frac, boxes, extra_melds) if hand_frac else None
        if fr is None:
            return None
    # 따로 놓인 패(도라 표시패 후보)는 손패 띠 안에서 온전히 보이는 것만: 손패 패와 크기가 비슷하고 사진 가장자리에 안 걸림.
    # 띠 밖(통째 검출)은 이웃 강·잘린 패가 섞여 넣지 않는다 — 가운데 칸 도라 판독이 따로 있다.
    H, W = rgb.shape[:2]
    m = int(0.02 * W)
    fr.loose = [x for x in fr.loose
                if ink(rgb, x) >= INK_MIN and 0.7 * fr.th <= x[3] <= 1.3 * fr.th and x[2] <= 1.6 * fr.tw
                and x[0] >= m and x[0] + x[2] <= W - m and x[1] >= m and x[1] + x[3] <= H - m]
    return fr


# ---------------------------------------------------------------- 손패 칸·모서리 칸 나눠 잡기
# 실험: mahjong_calc/experiments/speed/seat_zones.py(NOTES·metrics 표 2-3). 한 장 규칙은 먼 자리(B)에서 손패 띠를 울기 쪽과
# 섞어 잡아 장수가 안 맞았다. 격자 선으로 손패 칸만 따로 모자이크하면 패가 커져 놓치던 패를 찾고, 울기는 모서리 칸에서 찾는다.

SPLIT_MARGIN = 0.03  # 경계 양쪽으로 자리 폭의 3% 겹쳐 검출(선을 넘은 패)
SPLIT_MELD = {6: (3, 3), 8: (4, 4)}  # 붙어 놓인 울기 둘
RUN_GAP = 0.45  # 손패 줄 안 이웃 틈 < 패 0.45개
TABLE_REACH = 1.3  # 상자 가운데에서 패 1.3개 안에 펠트가 없으면 테이블 밖(울기는 흰 테두리 위에도 놓인다)
CORNER_K = 2  # 모서리만 잘라 검출할 때 토막 수


def fit_line(faces: list[Box]) -> tuple[float, float] | None:
    """가운데점 RANSAC(모든 쌍) → 가장 많은 상자가 패 높이 0.35 안에 드는 직선(기울기 0.3 이하), 최소제곱으로 다듬음."""
    if len(faces) < 3:
        return None
    th = statistics.median(x[3] for x in faces)
    tw = statistics.median(x[2] for x in faces)
    c = [(x[0] + x[2] / 2, x[1] + x[3] / 2) for x in faces]
    best: list[Box] | None = None
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            (x1, y1), (x2, y2) = c[i], c[j]
            if abs(x2 - x1) < tw:
                continue
            m = (y2 - y1) / (x2 - x1)
            if abs(m) > 0.3:
                continue
            k = y1 - m * x1
            inl = [f for f, (cx, cy) in zip(faces, c) if abs(cy - (m * cx + k)) < 0.35 * th]
            if best is None or len(inl) > len(best):
                best = inl
    if best is None or len(best) < 3:
        return None
    m, k = np.polyfit([f[0] + f[2] / 2 for f in best], [f[1] + f[3] / 2 for f in best], 1)
    return float(m), float(k)


def felt_mask(rgb: np.ndarray) -> np.ndarray:
    """펠트(초록~파랑, 채도 있는 면) — 탁자 안쪽(앱 rectify.dart findFeltQuad 와 같은 생각)."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h = hsv[..., 0] * 360 // 255
    return (h >= 70) & (h <= 250) & (hsv[..., 1] > 60) & (hsv[..., 2] > 40)


def on_table(faces: list[Box], rgb: np.ndarray, reach: float) -> list[Box]:
    m = felt_mask(rgb)
    H, W = m.shape
    r = int(reach)
    out = []
    for f in faces:
        cx, cy = int(f[0] + f[2] / 2), int(f[1] + f[3] / 2)
        if m[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)].any():
            out.append(f)
    return out


def hand_cell_boxes(det: rv.TileDetector, rgb: np.ndarray, split: int) -> list[Box]:
    """손패 칸(경계 + 여유까지)만 모자이크한 상자."""
    hand_rgb = np.ascontiguousarray(rgb[:, :split + int(rgb.shape[1] * SPLIT_MARGIN)])
    got = mosaic(det, hand_rgb, rv.white_mask(hand_rgb))
    return [x for x in (got[0] if got else []) if ink(hand_rgb, x) >= INK_MIN]


def corner_cell_boxes(det: rv.TileDetector, rgb: np.ndarray, split: int) -> list[Box]:
    """모서리 칸만 잘라 모자이크한 상자(원래 좌표)."""
    x0 = max(0, split - int(rgb.shape[1] * SPLIT_MARGIN))
    corner = np.ascontiguousarray(rgb[:, x0:])
    got = mosaic(det, corner, rv.white_mask(corner), k=CORNER_K)
    out = []
    for bx in (got[0] if got else []):
        nb = (bx[0] + x0, bx[1], bx[2], bx[3])
        if ink(rgb, nb) >= INK_MIN:
            out.append(nb)
    return out


def structure_split(rgb: np.ndarray, faces: list[Box], split: int) -> SeatFrame | None:
    """손패 = 손패 칸 상자 중 한 직선 위(세운 패)에 틈 없이 이어진 가장 긴 줄, 그 뒤 패 2.5개 안 한 장 = 떨어진 패.
    울기 = 나머지 중 손패 줄 시작보다 오른쪽 것을 틈으로 묶은 3·4장(6 → 3+3, 8 → 4+4). 나머지는 loose."""
    if not faces:
        return None
    W = rgb.shape[1]
    up = [x for x in faces if x[3] > x[2]] or faces
    tw = statistics.median(x[2] for x in up)
    th = statistics.median(x[3] for x in up)
    faces = on_table(faces, rgb, TABLE_REACH * th)
    hand_side = [f for f in faces if f[0] + f[2] / 2 < split + SPLIT_MARGIN * W]
    line = fit_line(hand_side)
    if line is None:
        return None
    a, c = line
    on = sorted([f for f in hand_side if abs(f[1] + f[3] / 2 - (a * (f[0] + f[2] / 2) + c)) < 0.5 * th and f[3] > f[2]],
                key=lambda x: x[0])
    if not on:
        return None
    runs = [[on[0]]]
    for f in on[1:]:
        if f[0] - (runs[-1][-1][0] + runs[-1][-1][2]) > RUN_GAP * tw:
            runs.append([f])
        else:
            runs[-1].append(f)
    hand = max(runs, key=len)
    after = runs[runs.index(hand) + 1:]
    hx1 = hand[-1][0] + hand[-1][2]
    sep = after[0][0] if after and len(after[0]) == 1 and after[0][0][0] - hx1 < SEP_DIST * tw else None
    used = set(hand) | ({sep} if sep else set())
    rest = [f for f in faces if f not in used and f[0] > hand[0][0]]
    loose = [f for f in faces if f not in used and f not in rest]
    melds = []
    for g in cluster(rest, tw * GAP, th * 0.5):
        g = sorted(g, key=lambda x: x[0])
        if len(g) in (3, 4):
            melds.append(g)
        elif len(g) in SPLIT_MELD:
            k = SPLIT_MELD[len(g)][0]
            melds += [g[:k], g[k:]]
        else:
            loose.extend(g)
    melds.sort(key=lambda m: m[0][0])
    return SeatFrame(hand, sep, melds, loose, tw, th, len(faces), split=True)


def frame_split(rgb: np.ndarray, det: rv.TileDetector, hand_frac: float, full: list[Box],
                extra_melds: int = 0) -> SeatFrame | None:
    """손패 칸·모서리 칸으로 나눠 틀 잡기 — 검사(장수·모양·무늬)를 넘는 첫 것. 손패 칸은 따로 모자이크(패가 커져 놓치던 패를
    찾음), 모서리 칸은 자리 전체 모자이크 상자([full] 중 경계 오른쪽)를 먼저 쓰고, 안 되면 모서리만 잘라 모자이크한다
    (따로 자르면 상자가 커져 서로 겹치기 쉬워 두 번째로)."""
    split = int(rgb.shape[1] * hand_frac)
    hand = hand_cell_boxes(det, rgb, split)
    white = rv.white_mask(rgb)
    for corner in (lambda: [x for x in full if x[0] + x[2] / 2 >= split], lambda: corner_cell_boxes(det, rgb, split)):
        fr = structure_split(rgb, rv.dedupe(rv.keep_faces(hand + corner(), white)), split)
        if valid(fr, extra_melds) and safe_ok(fr, rgb):
            return fr
    return None


def sideways(x: Box) -> bool:
    return x[2] > 1.15 * x[3]


# ---------------------------------------------------------------- 조각 그림

def sheet(rgb: np.ndarray, fr: SeatFrame, cell_h: int = 200, per_row: int = 7) -> bytes:
    """조각을 번호(1~n) 칸에 순서대로 붙인 그림 한 장(JPEG). 조각 13장을 따로 보내면 입력 토큰 17.9k, 한 장이면 3.7k."""
    cells = []
    for _, x in fr.pieces():
        im = Image.open(io.BytesIO(rv.crop(rgb, x))).convert("RGB")
        im = im.resize((max(1, round(im.width * cell_h / im.height)), cell_h), Image.LANCZOS)
        cells.append(im)
    cw = max(c.width for c in cells) + 12
    lab = 40
    rows = (len(cells) + per_row - 1) // per_row
    out = Image.new("RGB", (cw * min(per_row, len(cells)), (cell_h + lab) * rows), "white")
    d = ImageDraw.Draw(out)
    f = ImageFont.load_default(size=30)
    for i, c in enumerate(cells):
        x, y = (i % per_row) * cw, (i // per_row) * (cell_h + lab)
        d.text((x + cw // 2 - 10, y + 2), str(i + 1), fill=(200, 0, 0), font=f)
        out.paste(c, (x + (cw - c.width) // 2, y + lab))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def to_seat(fr: SeatFrame, tiles: list[dict | None]) -> dict | None:
    """조각 읽기(순서대로 {tile, conf}, 뒷면은 None) → normalize_seat_v2 와 같은 모양. 손패·떨어진 한 장·울기 조각이
    뒷면이거나 비면 None(틀이 틀렸을 수 있으니 통째 읽기로)."""
    ps = fr.pieces()
    if len(tiles) != len(ps):
        return None
    hand, sep, melds, loose = [], None, [[] for _ in fr.melds], []
    for (kind, x), t in zip(ps, tiles):
        if kind == "loose":
            if t:
                loose.append(t)
            continue
        if t is None:
            return None
        if kind == "hand":
            hand.append(t)
        elif kind == "sep":
            sep = t
        else:
            melds[int(kind[4:])].append(t)
    return {
        "face_up": True,
        "tiles": hand,
        "separate_tile": sep,
        "melds": [{"tiles": [t["tile"] for t in ts], "sideways": [i for i, x in enumerate(m) if sideways(x)],
                   "face_down": [], "conf": min(t["conf"] for t in ts)} for m, ts in zip(fr.melds, melds)],
        "loose": loose,
        "notes": "",
    }


class SeatLocator:
    """자리 크롭 JPEG → (SeatFrame 또는 None, rgb). 검출기는 강 틀(RiverLocator)과 같이 쓴다(모델 하나, 잠금 하나)."""

    def __init__(self, shared: rv.RiverLocator | None = None, conf: float = 0.04, table: bool = False):
        self._shared = shared
        self.conf = conf
        self.table = table  # 탁자 끝 밖을 지우고 검출(frame 의 table)
        self._det: rv.TileDetector | None = None
        self._init = threading.Lock()

    def _detector(self) -> rv.TileDetector:
        if self._shared is not None:
            return self._shared._detector()
        with self._init:
            if self._det is None:
                self._det = rv.TileDetector(conf=self.conf)
            return self._det

    def locate(self, jpeg: bytes, extra_melds: int = 0, hand_frac: float | None = None) \
            -> tuple[SeatFrame | None, np.ndarray]:
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        return frame(rgb, self._detector(), extra_melds, hand_frac, self.table), rgb
