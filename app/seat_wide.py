"""넓은 구역 틀 잡기: 화료자 쪽 테이블 절반에서 YOLO11 로 패 위치만 찾는다(패 종류는 조각으로 Gemini).

자리 크롭(/v2/seat)은 떼어 둔 화료패·탁자에 내놓은 표시패·모서리 백 3장을 놓친다. 그렇다고 절반을 통째로 읽으면 패 종류를 크게
틀린다(B·C·D 12/12 → 8/12). 그래서 앱이 자리 판독으로 답을 못 정한 장면에서만 이 경로를 부른다.
실험·근거: mahjong_calc/experiments/speed/wide_pieces.py·wide_lone.py, docs/metrics.md 표 1-5·1-6(개발 세트로 정한 값).

- 손패 줄 = 사진 아래쪽에서 가장 많은 패가 놓인 직선. 놓친 패는 간격으로 채운다(흰 패 면 70% 이상인 칸만).
- 낱장 = 줄보다 패 2.5개 위까지의 나머지. 위쪽은 강이고, 왼쪽 아래 묶음은 옆 사람 울기.
- 외톨이 = 다른 낱장과 안 붙고 좌우가 펠트(왕패 위 표시패는 옆이 패산). 탁자 위 표시패는 리치의 도라 + 뒷도라라 2장 이상이므로(사용자)
  외톨이는 표시패가 아니라 떼어 둔 패다.
- 모서리 칸 = 펠트 오른쪽 끝에서 패 4.5개 안쪽. 흰 패는 YOLO·무늬 검사가 못 잡아 칸을 통째로 읽는다.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field

import numpy as np

from . import river as rv
from . import seat as st

Box = rv.Box

WIN, OVER = 1280, 320  # 창 크기·겹침. 640 으로 줄여도 패가 약 44px
ROW_GAP = 2.0  # 손패를 벌려 놓은 틈은 패 2개까지 같은 줄
RIVER_ABOVE = 2.5  # 줄보다 패 2.5개 넘게 위는 강
FILL_WHITE = 0.7
LONE_CLUSTER = 0.45
CORNER_TILES = 4.5


@dataclass
class WideFrame:
    row: list[Box]
    singles: list[Box]
    lone: list[bool]  # singles 와 같은 순서
    corner: tuple[int, int, int, int]  # x0, y0, x1, y1
    tw: float = 0.0
    th: float = 0.0
    detected: int = 0
    others: list[Box] = field(default_factory=list)  # 왼쪽 아래 옆 사람 묶음(읽지 않음)

    def summary(self) -> dict:
        return {"row": len(self.row), "singles": len(self.singles), "lone": sum(self.lone), "detected": self.detected}


def detect_all(rgb: np.ndarray, det: rv.TileDetector) -> list[Box]:
    """겹치는 창마다 검출 → 창 경계에 잘린 상자는 옆 창에 맡기고 패 면만."""
    H, W = rgb.shape[:2]
    xs = sorted({*range(0, max(1, W - WIN), WIN - OVER), max(0, W - WIN)})
    ys = sorted({*range(0, max(1, H - WIN), WIN - OVER), max(0, H - WIN)})
    boxes = []
    for x0, y0 in itertools.product(xs, ys):
        part = np.ascontiguousarray(rgb[y0:y0 + WIN, x0:x0 + WIN])
        ph, pw = part.shape[:2]
        for bx in det.detect(part):
            if (bx[0] <= 1 and x0 > 0) or (bx[1] <= 1 and y0 > 0) \
                    or (bx[0] + bx[2] >= pw - 1 and x0 + WIN < W) or (bx[1] + bx[3] >= ph - 1 and y0 + WIN < H):
                continue
            boxes.append((bx[0] + x0, bx[1] + y0, bx[2], bx[3]))
    return rv.dedupe(rv.keep_faces(boxes, rv.white_mask(rgb)))


def _white_at(rgb: np.ndarray, x: Box) -> float:
    if x[0] < 0 or x[1] < 0 or x[0] + x[2] > rgb.shape[1] or x[1] + x[3] > rgb.shape[0]:
        return 0.0
    c = rgb[x[1]:x[1] + x[3], x[0]:x[0] + x[2]]
    return float(rv.white_mask(c).mean()) if c.size else 0.0


def fill_row(row: list[Box], rgb: np.ndarray, tw: float) -> list[Box]:
    """YOLO 가 놓친 줄 안 패를 간격으로 채운다. 줄 끝 바깥도 한 칸씩."""
    if len(row) < 2:
        return row
    pitch = min(statistics.median(b[0] - a[0] for a, b in zip(row, row[1:])), tw * 1.25)
    out = [row[0]]
    for nx in row[1:]:
        prev = out[-1]
        k = round((nx[0] - prev[0]) / pitch)
        for i in range(1, k):
            box = (round(prev[0] + (nx[0] - prev[0]) * i / k), round(prev[1] + (nx[1] - prev[1]) * i / k), prev[2], prev[3])
            if _white_at(rgb, box) >= FILL_WHITE:
                out.append(box)
        out.append(nx)
    for step in (-1, 1):
        for _ in range(14):  # 흰 테두리에 닿아도 끝없이 늘지 않게
            e = out[0] if step < 0 else out[-1]
            box = (round(e[0] + step * pitch), e[1], e[2], e[3])
            if _white_at(rgb, box) < FILL_WHITE:
                break
            out.insert(0, box) if step < 0 else out.append(box)
    return out


def corner_zone(rgb: np.ndarray, rx1: float, ry: float, tw: float, th: float, slope: float = 0.0) -> tuple[int, int, int, int]:
    """오른쪽 아래 모서리 칸. 기운 사진(오른쪽이 올라감)이면 칸도 올린다."""
    H, W = rgb.shape[:2]
    y0 = int(max(0, ry - 0.5 * th))
    cols = np.where(st.felt_mask(rgb[y0:H]).mean(0) > 0.2)[0]
    right = int(cols.max()) if cols.size else W
    x0 = int(max(rx1 + 0.3 * tw, right - CORNER_TILES * tw))
    y0 = int(max(0, y0 + min(0.0, slope) * (right - rx1)))
    return min(x0, W - 1), min(y0, H - 1), min(W, int(right + 2 * tw)), H


def lone_flags(rgb: np.ndarray, singles: list[Box], tw: float, th: float) -> list[bool]:
    groups = st.cluster(singles, tw * LONE_CLUSTER, th * 0.5)
    alone = {g[0] for g in groups if len(g) == 1}
    felt = st.felt_mask(rgb)
    w = max(4, int(tw * 0.4))
    out = []
    for x in singles:
        ok = x in alone
        if ok:
            left = felt[x[1]:x[1] + x[3], max(0, x[0] - w):x[0]]
            right = felt[x[1]:x[1] + x[3], x[0] + x[2]:x[0] + x[2] + w]
            ok = left.size > 0 and right.size > 0 and left.mean() > 0.5 and right.mean() > 0.5
        out.append(bool(ok))
    return out


def locate(rgb: np.ndarray, det: rv.TileDetector) -> WideFrame | None:
    """테이블 절반(그 사람 방향으로 세운 것, 손패가 아래) → 틀. 손패 줄을 못 찾으면 None."""
    faces = detect_all(rgb, det)
    if len(faces) < 3:
        return None
    H = rgb.shape[0]
    long = statistics.median(max(x[2], x[3]) for x in faces)
    faces = [x for x in faces if 0.6 * long <= max(x[2], x[3]) <= 1.4 * long]
    faces = st.on_table(faces, rgb, 1.3 * long)
    up = [x for x in faces if x[3] > x[2]]
    th = statistics.median(x[3] for x in up) if up else long
    tw = statistics.median(x[2] for x in up) if up else long * 0.75
    low = [f for f in faces if f[1] + f[3] / 2 > 0.35 * H]
    line = st.fit_line(low)
    if line is None:
        return None
    a, c = line
    on = sorted((f for f in low if abs(f[1] + f[3] / 2 - (a * (f[0] + f[2] / 2) + c)) < 0.6 * th), key=lambda x: x[0])
    if not on:
        return None
    runs = [[on[0]]]
    for f in on[1:]:
        if f[0] - (runs[-1][-1][0] + runs[-1][-1][2]) > ROW_GAP * tw:
            runs.append([f])
        else:
            runs[-1].append(f)
    row = fill_row(max(runs, key=len), rgb, tw)
    rx0, rx1 = row[0][0], row[-1][0] + row[-1][2]
    ry = statistics.mean(f[1] for f in row)
    corner = corner_zone(rgb, rx1, ry, tw, th, a)
    used = set(row)
    near = [f for f in faces if f not in used and f[1] + f[3] > ry - RIVER_ABOVE * th and st.ink(rgb, f) >= st.INK_MIN]
    singles, others = [], []
    for g in st.cluster(near, tw * LONE_CLUSTER, th * 0.5):
        cx = statistics.mean(x[0] + x[2] / 2 for x in g)
        cy = statistics.mean(x[1] + x[3] / 2 for x in g)
        if corner[0] <= cx and corner[1] <= cy:
            continue  # 모서리 칸은 통째로 읽는다
        if len(g) >= 3 and max(x[0] + x[2] for x in g) < rx0:
            others.extend(g)  # 왼쪽 아래 모서리: 옆 사람 울기
        else:
            singles.extend(g)
    return WideFrame(row, singles, lone_flags(rgb, singles, tw, th), corner, tw, th, len(faces), others)


def grown(x: Box, rgb: np.ndarray) -> Box:
    """돌아간 낱장은 상자가 패 일부만 잡기도 해 넉넉히 자른다."""
    H, W = rgb.shape[:2]
    px, py = int(x[2] * 0.2), int(x[3] * 0.2)
    x0, y0 = max(0, x[0] - px), max(0, x[1] - py)
    return x0, y0, min(W, x[0] + x[2] + px) - x0, min(H, x[1] + x[3] + py) - y0
