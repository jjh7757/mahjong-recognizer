"""넓은 구역 틀 잡기: 화료자 쪽 테이블 절반에서 YOLO11 로 패 위치만 찾는다(패 종류는 조각으로 Gemini).

자리 크롭(/v2/seat)은 떼어 둔 화료패·탁자에 내놓은 표시패·모서리 백 3장을 놓친다. 그렇다고 절반을 통째로 읽으면 패 종류를 크게
틀린다(B·C·D 12/12 → 8/12). 그래서 앱이 자리 판독으로 답을 못 정한 장면에서만 이 경로를 부른다.
실험·근거: mahjong_calc/experiments/speed/wide_pieces.py·wide_lone.py, docs/metrics.md 표 1-5·1-6(개발 세트로 정한 값).

- 손패 줄 = 사진 아래쪽에서 가장 많은 패가 놓인 직선. 놓친 패는 간격으로 채운다(흰 패 면 70% 이상인 칸만).
- 낱장 = 줄보다 패 2.5개 위까지의 나머지. 위쪽은 강이고, 왼쪽 아래 묶음은 옆 사람 울기.
- 외톨이 = 다른 낱장과 안 붙고 좌우가 펠트(왕패 위 표시패는 옆이 패산). 탁자 위 표시패는 리치의 도라 + 뒷도라라 2장 이상이므로(사용자)
  외톨이는 표시패가 아니라 떼어 둔 패다.
- 모서리 칸 = 펠트 오른쪽 끝에서 패 4.5개 안쪽. 흰 패는 YOLO·무늬 검사가 못 잡아 칸을 통째로 읽는다.
- 표시패 짝(pair) = 줄보다 위(낱장 높이든 강 높이든)에서 둘레가 펠트이고 다른 패와 한 패 넘게 떨어져 둘만 붙어 놓인 패. 리치 화료자가
  탁자에 내놓은 도라·뒷도라 표시패(매장 습관, 2026-11-01: 143104·172424 강 옆, 142535·171025 줄 가까이, 겹쳐 놓기도). 강 줄·왕패 위
  표시패(옆이 패)·줄 아래 낱장(떨어뜨린 손패)은 아니다. 앱이 리치 화료의 표시패로 쓴다.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import river as rv
from . import seat as st

Box = rv.Box

WIN, OVER = 1280, 320  # 창 크기·겹침. 640 으로 줄여도 패가 약 44px
ROW_GAP = 2.0  # 손패를 벌려 놓은 틈은 패 2개까지 같은 줄
RIVER_ABOVE = 2.5  # 줄보다 패 2.5개 넘게 위는 강
FILL_WHITE = 0.7
LONE_CLUSTER = 0.45
CORNER_TILES = 4.5
BLOB_SCALE = 4  # 덩어리 찾기는 1/4 로 줄여서
BLOB_AREA = (0.6, 1.9)  # 패 한 장 넓이 배수(두께 포함, 돌아가도 넓이는 같다)
BLOB_LONG = 1.9  # 상자 긴 변 < 패 높이 × 1.9(돌아간 패의 대각선까지)
BLOB_WHITE = 0.35  # 상자 안 흰 면 비율(돌아간 패는 상자 모서리가 펠트라 낮다)


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
    pair: list[bool] = field(default_factory=list)  # singles 와 같은 순서, 탁자에 내놓은 표시패 짝

    def summary(self) -> dict:
        return {"row": len(self.row), "singles": len(self.singles), "lone": sum(self.lone), "pair": sum(self.pair), "detected": self.detected}


def detect_all(rgb: np.ndarray, det: rv.TileDetector) -> list[Box]:
    """겹치는 창마다 검출 → 창 경계에 잘린 상자는 옆 창에 맡기고 패 면만."""
    H, W = rgb.shape[:2]
    xs = sorted({*range(0, max(1, W - WIN), WIN - OVER), max(0, W - WIN)})
    ys = sorted({*range(0, max(1, H - WIN), WIN - OVER), max(0, H - WIN)})
    boxes = []
    for x0, y0 in itertools.product(xs, ys):
        part = np.ascontiguousarray(rgb[y0:y0 + WIN, x0:x0 + WIN])
        ph, pw = part.shape[:2]
        for bx in det.detect(part, low=True):  # 자리·강 틀 잡기가 진행 중이면 창마다 양보
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


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4방향 연결 덩어리 → (x0, y0, x1, y1, 넓이). opencv 없이(서버 의존성에 없다), 줄인 그림이라 파이썬 반복으로 충분."""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack = [(int(y0), int(x0))]
        ys, xs, area = [int(y0), int(y0)], [int(x0), int(x0)], 0
        while stack:
            y, x = stack.pop()
            area += 1
            ys = [min(ys[0], y), max(ys[1], y)]
            xs = [min(xs[0], x), max(xs[1], x)]
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((xs[0], ys[0], xs[1] + 1, ys[1] + 1, area))
    return out


def tile_blobs(rgb: np.ndarray, known: list[Box], tw: float, th: float, min_y: float) -> list[Box]:
    """펠트 위의 패 한 장 크기 덩어리(YOLO 가 놓치는 돌아간 낱장, 181209 9s·183627 4m). 이미 찾은 상자와 겹치는 것은 뺀다."""
    S = BLOB_SCALE
    H, W = rgb.shape[:2]
    small = np.asarray(Image.fromarray(rgb).resize((max(1, W // S), max(1, H // S)), Image.BILINEAR))
    white = rv.white_mask(small)
    full = tw * th / (S * S)
    out = []
    for x0, y0, x1, y1, area in _components(~st.felt_mask(small)):
        if not BLOB_AREA[0] * full <= area <= BLOB_AREA[1] * full or max(x1 - x0, y1 - y0) * S > BLOB_LONG * th:
            continue
        if white[y0:y1, x0:x1].mean() < BLOB_WHITE:  # 손·기기·패산 뒷면
            continue
        b = (x0 * S, y0 * S, (x1 - x0) * S, (y1 - y0) * S)
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        if cy < min_y or any(k[0] <= cx <= k[0] + k[2] and k[1] <= cy <= k[1] + k[3] for k in known):
            continue
        out.append(b)
    return out


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


def table_pairs(rgb: np.ndarray, far: list[Box], faces: list[Box], corner: tuple[int, int, int, int], tw: float, th: float) -> list[Box]:
    """줄보다 위에서 둘만 붙어 놓인 패 — 탁자에 내놓은 표시패 짝. 다른 패(검출된 면)가 한 패 안에 있으면 강·울기, 둘레가 펠트가
    아니면 왕패 위 표시패라 뺀다."""
    felt = st.felt_mask(rgb)
    H, W = felt.shape
    out: list[Box] = []
    for g in st.cluster(far, tw * LONE_CLUSTER, th * 0.5):
        if len(g) != 2:
            continue
        x0, y0 = min(x[0] for x in g), min(x[1] for x in g)
        x1, y1 = max(x[0] + x[2] for x in g), max(x[1] + x[3] for x in g)
        if corner[0] <= (x0 + x1) / 2 and corner[1] <= (y0 + y1) / 2:
            continue
        if any(f not in g and max(0, max(x0, f[0]) - min(x1, f[0] + f[2])) < tw and max(0, max(y0, f[1]) - min(y1, f[1] + f[3])) < th
               for f in faces):
            continue
        w, h = max(4, int(tw * 0.4)), max(4, int(th * 0.4))
        bands = [felt[y0:y1, max(0, x0 - w):x0], felt[y0:y1, x1:min(W, x1 + w)], felt[max(0, y0 - h):y0, x0:x1], felt[y1:min(H, y1 + h), x0:x1]]
        if all(b.size > 0 and b.mean() > 0.5 for b in bands):
            out.extend(sorted(g))
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
    near += [b for b in tile_blobs(rgb, row + faces, tw, th, ry - RIVER_ABOVE * th) if st.ink(rgb, b) >= st.INK_MIN]
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
    far = [f for f in faces if f not in used and f[1] + f[3] <= ry - RIVER_ABOVE * th and st.ink(rgb, f) >= st.INK_MIN]
    above = [x for x in singles if x[1] + x[3] / 2 < ry - 0.5 * th]  # 줄보다 위의 낱장(줄 아래는 떨어뜨린 손패)
    pair_set = set(table_pairs(rgb, far + above, faces, corner, tw, th))
    extra = [x for x in pair_set if x not in singles]
    lone = lone_flags(rgb, singles, tw, th) + [False] * len(extra)
    all_singles = singles + extra
    return WideFrame(row, all_singles, lone, corner, tw, th, len(faces), others=others, pair=[x in pair_set for x in all_singles])


def grown(x: Box, rgb: np.ndarray) -> Box:
    """돌아간 낱장은 상자가 패 일부만 잡기도 해 넉넉히 자른다."""
    H, W = rgb.shape[:2]
    px, py = int(x[2] * 0.2), int(x[3] * 0.2)
    x0, y0 = max(0, x[0] - px), max(0, x[1] - py)
    return x0, y0, min(W, x[0] + x[2] + px) - x0, min(H, x[1] + x[3] + py) - y0
