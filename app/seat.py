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
                "loose": len(self.loose), "detected": self.detected}


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


def frame(rgb: np.ndarray, det: rv.TileDetector, extra_melds: int = 0) -> SeatFrame | None:
    """자리 크롭 → 틀. 장수·모양 검사를 못 넘으면 None(통째 읽기로)."""
    white = rv.white_mask(rgb)
    got = mosaic(det, rgb, white)
    if got is None:
        return None
    boxes, _, _ = got
    boxes = [x for x in boxes if ink(rgb, x) >= INK_MIN]  # 무늬 없는 상자(테이블 테두리 등)는 버림
    fr = structure(rgb, boxes, white)
    if not valid(fr, extra_melds) or not safe_ok(fr, rgb):
        return None
    # 따로 놓인 패(도라 표시패 후보)는 손패 띠 안에서 온전히 보이는 것만: 손패 패와 크기가 비슷하고 사진 가장자리에 안 걸림.
    # 띠 밖(통째 검출)은 이웃 강·잘린 패가 섞여 넣지 않는다 — 가운데 칸 도라 판독이 따로 있다.
    H, W = rgb.shape[:2]
    m = int(0.02 * W)
    fr.loose = [x for x in fr.loose
                if ink(rgb, x) >= INK_MIN and 0.7 * fr.th <= x[3] <= 1.3 * fr.th and x[2] <= 1.6 * fr.tw
                and x[0] >= m and x[0] + x[2] <= W - m and x[1] >= m and x[1] + x[3] <= H - m]
    return fr


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

    def __init__(self, shared: rv.RiverLocator | None = None, conf: float = 0.04):
        self._shared = shared
        self.conf = conf
        self._det: rv.TileDetector | None = None
        self._init = threading.Lock()

    def _detector(self) -> rv.TileDetector:
        if self._shared is not None:
            return self._shared._detector()
        with self._init:
            if self._det is None:
                self._det = rv.TileDetector(conf=self.conf)
            return self._det

    def locate(self, jpeg: bytes, extra_melds: int = 0) -> tuple[SeatFrame | None, np.ndarray]:
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        return frame(rgb, self._detector(), extra_melds), rgb
