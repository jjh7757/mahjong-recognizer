"""자체 모델 판독: 자리 크롭·강 크롭·사진 전체를 Gemini 없이 읽는다(/v2/localseat·/v2/locallast3·/v2/localdora, 그림자).

검출 = YOLO26n "앞면 패" 1종(mahjong_calc/experiments/tile_classifier, 입력 긴 변 1280, NMS 필요), 분류 = MobileNetV3 Small 37종(128×128).
자리: 통째 검출 1번 → seat.structure_split 복사본(이웃 패를 울기에서 뺌, 위아래 울기를 줄로 나눔) → 모양·무늬 검사(옅은 무늬·白 허용, 직선 0.30패)
→ 분류, 13장이면 손패 줄 오른쪽 홀로 놓인 패를 떼어 둔 패로. 틀을 못 잡고 장수가 모자라면 白 을 코드로 채운다(빈칸·모서리 퐁).
강: 검출 → river.frame → 마지막 패를 위아래 두 방향으로 읽어 확신 높은 쪽. 확신도는 분류기 확률, 채운 白 은 [FILLED_CONF](앱이 자동 반영하지 않는다).
측정: mahjong_calc metrics 표 1-23·1-26·1-28·1-29·1-33(같은 매장 사진, t4g.small 자리 0.69 / 0.91 / 1.02초).
"""
from __future__ import annotations

import io
import logging
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import obb
from . import river as rv
from . import seat as st
from . import seat_wide as sw
from .shared import Shared, digest

log = logging.getLogger("local_read")

# 환경 변수는 측정용: 그 사진을 학습하지 않은 나눔 모델로 재기 위해.
DET_MODEL = Path(os.environ.get("LOCAL_DET_MODEL") or Path(__file__).parent / "assets" / "tile-yolo26n.onnx")
# 자리 판독(/v2/localseat·/v2/localroles) 검출기: 회전 상자 8회차(app/obb.py). 파일이 없으면 위 DET_MODEL 로 읽는다.
# 강·표시패(/v2/locallast3·/v2/localdora)는 DET_MODEL(2회차) 그대로: 8회차로 바꾸면 헛 리치·강 끝·표시패 짝이 나빠졌다
# (mahjong_calc metrics 표 1-80, 강 틀·눕힌 패·짝 규칙이 2회차 상자에 맞춰져 있음).
# LOCAL_OTHER_DET: 강·표시패 검출도 자리 판독 검출기(SEAT_MODEL)로(시험, 기본 끔). "obb" = 회전 상자를 감싸는 축 상자만,
# "obb-up" = 거기에 분류기 크롭을 회전 상자대로 세워 자르기(자리 판독과 같음). 앞면만 쓰고 뒷면·조각 버리기·안깡은 안 씀.
OTHER_DET = os.environ.get("LOCAL_OTHER_DET", "")
SEAT_MODEL = Path(os.environ.get("LOCAL_SEAT_MODEL") or Path(__file__).parent / "assets" / "tile-obb-r8.onnx")
ROT_WHITE = os.environ.get("LOCAL_ROT_WHITE", "1") == "1"  # 자리 판독 흰 비율을 회전 상자 안에서(metrics 표 1-73, 2026-11-05 켬)
# 7-2a(2026-11-06, mahjong_calc metrics 표 1-99·1-100): v2 = 합성 장면 크롭 사전학습 + 같은 미세 조정(씨드 10). 옛 tile-cls-mnv3s.onnx 는 env 로 되돌릴 때 쓰려고 남김.
# v3(2026-11-08, mahjong_calc metrics 표 1-116·1-118): v2 설정 + 180° 뒤집기 증강 + 학습 라벨 고침(뒤집힌 三萬·一萬 이 2m 이던 4장, 잘린 적5통 2장 뺌).
# v2 는 뒤집힌 만수를 2m 으로 확신해 read_any(강 끝·떼어 둔 패)가 바로 놓인 3만을 2만으로 골랐다(나눔 3m 19/37 → v3 0/37). v2·옛 파일은 env 로 되돌릴 때 쓰려고 남김.
CLS_MODEL = Path(os.environ.get("LOCAL_CLS_MODEL") or Path(__file__).parent / "assets" / "tile-cls-mnv3s-v3.onnx")
# 두 번째 분류기(불확실성 신호 S3, 개선 계획 5단계): 같은 학습 묶음·다른 씨드. 답에는 안 쓴다. 파일이 없으면 S3 를 안 낸다.
ALT_CLS_MODEL = Path(os.environ.get("LOCAL_ALT_CLS_MODEL") or Path(__file__).parent / "assets" / "tile-cls-alt-v3.onnx")  # v3 = 같은 설정 씨드 11
CLASSES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["0m", "0p", "0s"] + [f"{n}z" for n in range(1, 8)]
IMG = 1280
CONF, IOU = 0.25, 0.7
CLS_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
PAD = 0.12  # 분류 크롭 여유(상자 크기 비율)

BLANK = 0.01  # 이보다 무늬가 적으면 白 후보(정답 있는 白 상자는 0.0)
LOW_INK = 0.015  # 읽을 패의 무늬 기준(운영 틀 0.03 은 옅은 세운 패 0.020~0.024 를 막았다)
LINE_TOL = 0.30  # 손패 줄 직선 허용(패 높이 비율, 운영 틀 0.25)
ABOVE = 1.5  # 손패 줄보다 패 몇 개 위(가운데 쪽)까지 울기 후보로 보나
STRAY_UP, STRAY_DOWN = 1.0, 3.5  # 떼어 둔 패: 손패 줄 기준 패 몇 개 위·아래까지
ANGLES = range(0, 360, 15)

FACE_UP_MIN = 8  # 틀을 못 잡아도 앞면 패가 이만큼 보이면 화료자 자리로 친다
WALL_SIDE = 0.5  # 패산 표시패: 앞면 패 바로 옆 칸이 이 비율 이상 패 뒷면이면 그쪽에 패산이 붙어 있다
LOCAL_SIDEWAYS_HEIGHT = (0.6, 0.9)  # YOLO26 상자는 눕힌 패 높이가 운영 기준(YOLO11, ~0.8)보다 크게 잡힌다(metrics 표 3-2)
SMALL_TILE = CLS_SIZE * 3 // 4  # 이보다 작은 상자(추적 프레임의 패 약 30px)는 키워서 읽는다(metrics 표 1-34·1-53)
WALL_CONF = 0.7  # 이보다 확신이 낮으면 답을 내지 않는다(측정에서 맞은 답 0.81 이상, 돌아가 겹친 패 0.33·0.58, metrics 표 1-35)

# 白 채우기(metrics 표 1-33). 뒷면이 색 있는 패 세트에서는 무늬 없는 흰 앞면 = 白 이라 위치만 찾으면 종류가 정해진다.
FILLED_CONF = 0.5  # 앱 uncertainBelow(0.7)보다 낮게: 노란 테두리·자동 반영 막힘
WHITE_MIN = sw.FILL_WHITE
RISE = 0.5  # 탁자 끝이 패 높이의 이만큼 넘게 올라오면 패가 놓인 곳
CORNER_W = (2.4, 3.9)  # 패 3장 폭(패 폭 배수)
CORNER_H = (0.6, 1.5)  # 올라온 높이(패 높이 배수)
CORNER_REACH = 1.0  # 오른쪽 탁자 끝에서 패 폭 1개 안에서 끝나야 모서리
WINDOW = 6  # 오른쪽 끝에서 패 몇 개 폭을 보나
SEAM_JUMP, SEAM_MIN = 0.12, 0.6  # 패 나누기: 위쪽 끝 꺾임 최소(패 폭 배수), 첫 패 최소 폭(패 폭 배수)
TILE_W_MAX, TILE_H = 1.3, 1.05  # 모서리 패 상자: 폭 상한(패 폭 배수), 높이(패 높이 배수)
EDGE = 0.1  # 모서리 무늬 검사에서 뺄 경계 폭(패 폭 배수)
CORNER_INK = 0.03  # 모서리 칸 무늬 비율 상한. 패 옆면 줄이 조금 들어와 낱장 기준 0.01 보다 넉넉히


HAND_EDGE = 0.01  # 강 크롭 아래 끝에서 이 비율 안까지 닿는 상자는 잘린 손패다
RIVER_GAP = 1.2  # 강 묶기 틈(패 폭 배수). 검출기가 놓친 패(白 등) 한 장은 건너 잇는다
SIDE_ROW = 0.5  # 눕힌 패가 줄 높이에서 벗어나도 되는 폭(패 높이 배수, rv.to_rows 와 같은 기준). 기기 옆 옆자리 패는 줄보다 위에 있다
RIVER_ROW_MIN = 4  # 아래 줄이 있는 줄의 세운 패 최소 수: 6장에서 눕힌 패 1장·검출 누락 1장을 뺀 값


def _split_piece(b, faces: list, th: float) -> bool:
    """세운 패 하나가 위아래 두 상자로 쪼개진 조각인가: 바로 위나 아래에 붙은 상자와 합쳐 패 한 장 높이. rv.sideways 는 겹친 상자만 본다."""
    for o in faces:
        if o == b or min(b[0] + b[2], o[0] + o[2]) - max(b[0], o[0]) <= 0.5 * min(b[2], o[2]):
            continue
        if abs(max(o[1] - (b[1] + b[3]), b[1] - (o[1] + o[3]))) < 0.1 * th and b[3] + o[3] <= 1.2 * th:
            return True
    return False


def river_frame(rgb: np.ndarray, boxes: list, height: tuple = LOCAL_SIDEWAYS_HEIGHT) -> rv.RiverFrame | None:
    """그 자리 쪽으로 세운 강 크롭 → 그 사람 강. 운영 rv.frame 은 크롭 아래쪽 패부터 묶어 공개한 손패 줄이나 옆 자리 강을 강으로 잡았다.
    크롭 아래 끝에 걸린 상자(잘린 손패)를 빼고, 세운 패끼리 묶어 가장 큰 묶음을 강으로 본다(옆 자리 강은 눕혀 보여 안 묶인다).
    눕힌 패는 그 강의 줄에 나란히 붙은 것만 더한다(리치 선언패)."""
    H, W = rgb.shape[:2]
    faces = [b for b in rv.dedupe(rv.keep_faces(boxes, rv.white_mask(rgb))) if b[1] + b[3] < (1 - HAND_EDGE) * H]
    stand = [b for b in faces if b[2] <= rv.SIDEWAYS_RATIO * b[3]]
    if not stand:
        return None
    tw, th = statistics.median(b[2] for b in stand), statistics.median(b[3] for b in stand)
    # 한 줄에 7장 넘는 묶음은 손패 줄이다(강은 앞 줄을 6장 채워야 다음 줄로 가므로 첫 줄은 6장까지)
    groups = [g for g in st.cluster(stand, RIVER_GAP * tw, 0.5 * th) if len(g) <= 6 or len(rv.to_rows(g, th)) > 1]
    if not groups:
        return None
    group = max(groups, key=lambda g: (len(g), -min(abs(b[0] + b[2] / 2 - W / 2) + b[1] for b in g)))  # 같으면 기기(크롭 위 가운데)에 가까운 쪽
    rows = rv.to_rows(group, th)
    while len(rows) > 1 and len(rows[0]) < RIVER_ROW_MIN:  # 첫 줄이 짧은데 아래 줄이 있으면 옆 자리 패가 붙은 것
        rows.pop(0)
    for b in faces:
        if b in stand or b[3] < height[0] * th:  # 눕힌 패보다 낮으면 쪼개진 조각이다
            continue
        cy = b[1] + b[3] / 2
        for r in rows:
            near = min(max(b[0] - (x[0] + x[2]), x[0] - (b[0] + b[2])) for x in r) < RIVER_GAP * tw
            # 강은 앞 줄을 6장 채워야 다음 줄로 간다. 이미 6장인 줄 옆의 눕힌 패는 옆 자리 패다
            if len(r) < 6 and near and abs(cy - statistics.mean(x[1] + x[3] / 2 for x in r)) < SIDE_ROW * th:
                r.append(b)
                r.sort(key=lambda x: x[0])
                break
    own = [b for r in rows for b in r]
    return rv.RiverFrame(rows, len(boxes), [rv.sideways(b, th, own, height) and not _split_piece(b, faces, th) for b in own])


LAST_ZONE_BAND = 0.6  # 앱 lastZoneBand
ROT_CW = {"bottom": 0, "right": 90, "top": 180, "left": 270}


def _round_half_up(v: float) -> int:
    return int(np.floor(v + 0.5)) if v >= 0 else -int(np.floor(-v + 0.5))  # Dart round(0.5 은 0 에서 먼 쪽)


def last_zone(g: dict, side: str, band: float = LAST_ZONE_BAND) -> tuple[int, int, int, int]:
    """앱 GridZones.lastZone 과 같은 강 끝 상자 (ymin, xmin, ymax, xmax) 1000 분율: 가운데 칸 한가운데부터 그 자리 쪽 끝까지, 옆으로 band."""
    x1, x2, y1, y2 = g["x1"], g["x2"], g["y1"], g["y2"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hw, hh = _round_half_up((x2 - x1) * band / 2), _round_half_up((y2 - y1) * band / 2)
    return {"bottom": (cy, cx - hw, y2, cx + hw), "top": (y1, cx - hw, cy, cx + hw),
            "left": (cy - hh, x1, cy + hh, cx), "right": (cy - hh, cx, cy + hh, x2)}[side]


def last_zone_crop(im: Image.Image, g: dict, side: str) -> Image.Image:
    """[last_zone] 을 잘라 그 자리 방향으로 세운 것(roles.crop_seat 과 같은 돌리기)."""
    W, H = im.size
    ymin, xmin, ymax, xmax = last_zone(g, side)
    out = im.crop((round(xmin * W / 1000), round(ymin * H / 1000), round(xmax * W / 1000), round(ymax * H / 1000)))
    return out.rotate(-ROT_CW[side], expand=True) if ROT_CW[side] else out


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


# 사진 전체 디코딩을 같은 바이트면 한 번만(/v2/localroles·/v2/localdora 가 같은 화료 사진을 받는다, app/shared.py).
# 12MP 배열은 36MB 라 하나만, 화료 한 번의 요청들이 다 들어올 만큼만 들고 있는다. 배열은 읽기 전용(전에도 np.asarray(PIL) 이라 같음).
_decoded = Shared(size=1, ttl=10.0)


def _decode(jpeg: bytes) -> np.ndarray:
    a = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    a.flags.writeable = False
    return a


def decode_rgb(jpeg: bytes) -> np.ndarray:
    return _decoded.get(digest(jpeg), lambda: _decode(jpeg))


def rss_mb() -> int | None:
    """이 프로세스의 최대 메모리(MB, 리눅스). 그림자 판독이 컨테이너 한도에 얼마나 다가가는지 응답으로 본다."""
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


# ---------------------------------------------------------------- 틀 규칙

def _line(hand: list) -> tuple[float, float]:
    a, c = np.polyfit([x[0] + x[2] / 2 for x in hand], [x[1] + x[3] / 2 for x in hand], 1)
    return float(a), float(c)


def split_frame(rgb: np.ndarray, faces: list, split: int) -> st.SeatFrame | None:
    """seat.structure_split 복사본 + 이웃 패 빼기 + 위아래 울기 나누기."""
    if not faces:
        return None
    W = rgb.shape[1]
    up = [x for x in faces if x[3] > x[2]] or faces
    tw = statistics.median(x[2] for x in up)
    th = statistics.median(x[3] for x in up)
    faces = st.on_table(faces, rgb, st.TABLE_REACH * th)
    hand_side = [f for f in faces if f[0] + f[2] / 2 < split + st.SPLIT_MARGIN * W]
    line = st.fit_line(hand_side)
    if line is None:
        return None
    a, c = line
    on = sorted([f for f in hand_side if abs(f[1] + f[3] / 2 - (a * (f[0] + f[2] / 2) + c)) < 0.5 * th and f[3] > f[2]], key=lambda x: x[0])
    if not on:
        return None
    runs = [[on[0]]]
    for f in on[1:]:
        if f[0] - (runs[-1][-1][0] + runs[-1][-1][2]) > st.RUN_GAP * tw:
            runs.append([f])
        else:
            runs[-1].append(f)
    hand = max(runs, key=len)
    after = runs[runs.index(hand) + 1:]
    hx1 = hand[-1][0] + hand[-1][2]
    sep = after[0][0] if after and len(after[0]) == 1 and after[0][0][0] - hx1 < st.SEP_DIST * tw else None
    used = set(hand) | ({sep} if sep else set())
    rest = [f for f in faces if f not in used and f[0] > hand[0][0]]
    loose = [f for f in faces if f not in used and f not in rest]
    far = [f for f in rest if f[1] + f[3] / 2 < a * (f[0] + f[2] / 2) + c - ABOVE * th]  # 크롭 위쪽 이웃 패
    loose += far
    rest = [f for f in rest if f not in far]
    melds = []
    for g in st.cluster(rest, tw * st.GAP, th * 0.5):
        # 모양으로만(임시 나눔 없음): 이 경로의 덩어리엔 울기 아닌 패도 섞여 임시 나눔이 틀을 바꾼다(robust ev_p1 124654 손패 22장).
        # 못 나눈 한 줄 깡 + 퐁은 틀이 안 맞아 hybrid 가 model 경로(역할 meld 만, meld_parts)로 넘어간다.
        parts = st.split_meld_group(g, th)
        if parts:
            melds += parts
        else:
            loose.extend(g)
    melds.sort(key=lambda m: m[0][0])
    return st.SeatFrame(hand, sep, melds, loose, tw, th, len(faces), split=True)


def resplit_melds(fr: st.SeatFrame, read) -> tuple[list[list], list[list[tuple]], list[str]]:
    """틀의 울기를 붙은 덩어리로 다시 모아(틀 잡을 때와 같은 틈) 패를 모두 읽은 뒤 [st.choose_melds] 로 다시 나눈다.
    덩어리마다 틀이 준 울기 수는 그대로(장수 검사 유지). 반환 (울기 상자들, 울기 읽기들, 울기별 표시 ok|disagree|alt|fallback), 왼쪽부터."""
    owner = {tuple(b): i for i, m in enumerate(fr.melds) for b in m}
    boxes = list(owner)
    out = []
    for g in st.cluster(boxes, fr.tw * st.GAP, fr.th * 0.5):
        g = sorted(g, key=lambda b: b[0])
        reads = [read(b) for b in g]
        n = len({owner[b] for b in g})
        parts, used, flag = st.choose_melds(g, reads, fr.tw, fr.th, n)
        out += [([g[i] for i in p], [used[i] for i in p], flag) for p in parts]
    out.sort(key=lambda m: m[0][0][0])
    return [m[0] for m in out], [m[1] for m in out], [m[2] for m in out]


def meld_conf(ts: list[tuple], flag: str) -> float:
    """울기 확신 = 패 확신 최솟값, 가드레일에 걸리면([st.choose_melds] 표시가 ok 아님) [st.MELD_FLAG_CONF] 이하."""
    c = min(t[1] for t in ts)
    return c if flag == "ok" else min(c, st.MELD_FLAG_CONF)


def frame_ok(fr: st.SeatFrame | None, rgb: np.ndarray) -> bool:
    """장수 13·14 + 손패 줄 모양 + 울기 모양 + 무늬(白 은 손패 줄 4장까지, 울기는 전부 무늬 없을 때만)."""
    if not st.valid(fr):
        return False
    hand = fr.hand
    if len(hand) < 4:
        return False
    hs, ws = [x[3] for x in hand], [x[2] for x in hand]
    mh, mw = statistics.median(hs), statistics.median(ws)
    if any(abs(h - mh) > 0.25 * mh for h in hs) or any(abs(w - mw) > 0.25 * mw for w in ws):
        return False
    xs = np.array([x[0] + x[2] / 2 for x in hand], float)
    ys = np.array([x[1] + x[3] / 2 for x in hand], float)
    a, c = np.polyfit(xs, ys, 1)
    if np.max(np.abs(ys - (a * xs + c))) > LINE_TOL * mh:
        return False
    steps = np.diff(xs)
    if not (np.all(steps > 0.6 * mw) and np.all(steps < 1.4 * mw)):
        return False
    for m in fr.melds:
        for p, q in zip(m, m[1:]):
            if p[0] + p[2] - q[0] > 0.3 * min(p[2], q[2]):
                return False
        if any(not 0.5 * mh <= x[3] <= 1.3 * mh for x in m):
            return False
    blank = lambda x: st.ink(rgb, x) < BLANK
    if sum(blank(x) for x in hand) > 4:
        return False
    for x in hand + ([fr.separate] if fr.separate else []):
        if st.ink(rgb, x) < LOW_INK and not blank(x):
            return False
    for m in fr.melds:
        if any(blank(x) for x in m) and not all(blank(x) for x in m):
            return False
        if any(st.ink(rgb, x) < LOW_INK and not blank(x) for x in m):
            return False
    return True


def strays(fr: st.SeatFrame, faces: list, rgb: np.ndarray) -> list:
    """손패 줄 오른쪽, 탁자 위에 홀로 놓인 앞면 패(떼어 둔 쯔모패 후보)."""
    a, c = _line(fr.hand)
    used = set(fr.hand) | {x for m in fr.melds for x in m}
    hx1 = fr.hand[-1][0] + fr.hand[-1][2]
    out = []
    table = st.on_table(faces, rgb, st.TABLE_REACH * fr.th)
    for g in st.cluster([f for f in table if f not in used], fr.tw * st.GAP, fr.th * 0.5):
        if len(g) != 1:
            continue
        f = g[0]
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        if cx > hx1 and -STRAY_UP * fr.th < cy - (a * cx + c) < STRAY_DOWN * fr.th:
            out.append(f)
    return out


# ---------------------------------------------------------------- 白 채우기

def blank_at(rgb: np.ndarray, b) -> bool:
    return sw._white_at(rgb, b) >= WHITE_MIN and st.ink(rgb, b) < BLANK


def overlaps(b, faces: list) -> bool:
    return any(min(b[0] + b[2], f[0] + f[2]) - max(b[0], f[0]) > 0.3 * b[2] and min(b[1] + b[3], f[1] + f[3]) - max(b[1], f[1]) > 0.3 * b[3] for f in faces)


def row_blanks(rgb: np.ndarray, faces: list, fr: st.SeatFrame, split: int) -> list:
    """손패 줄의 틈·양 끝에서 무늬 없는 흰 칸. 줄에 든 상자(앞뒤 덩어리 모두)로 간격을 잰다."""
    W = rgb.shape[1]
    line = st.fit_line([f for f in faces if f[0] + f[2] / 2 < split + st.SPLIT_MARGIN * W])
    if line is None:
        return []
    a, c = line
    on = sorted([f for f in faces if abs(f[1] + f[3] / 2 - (a * (f[0] + f[2] / 2) + c)) < 0.5 * fr.th and f[3] > f[2]], key=lambda x: x[0])
    if len(on) < 2:
        return []
    steps = [q[0] - p[0] for p, q in zip(on, on[1:]) if q[0] - p[0] < 1.4 * fr.tw]
    if not steps:
        return []
    pitch = statistics.median(steps)
    out = []
    for p, q in zip(on, on[1:]):
        k = round((q[0] - p[0]) / pitch)
        for i in range(1, k):
            out.append((round(p[0] + (q[0] - p[0]) * i / k), round(p[1] + (q[1] - p[1]) * i / k), p[2], p[3]))
    for e, step in ((on[0], -1), (on[-1], 1)):
        for i in range(1, 5):
            out.append((round(e[0] + step * i * pitch), round(e[1] + step * i * pitch * a), e[2], e[3]))
    # 끝쪽 후보는 안쪽부터 이어져야 한다(떨어진 흰 곳을 건너뛰어 잡지 않게)
    keep = [b for b in out if not overlaps(b, faces) and blank_at(rgb, b)]
    lo, hi = on[0][0], on[-1][0]
    inner = [b for b in keep if lo < b[0] < hi]
    left = [b for b in keep if b[0] <= lo and round((lo - b[0]) / pitch) <= 1 + sum(1 for x in keep if b[0] < x[0] <= lo)]
    right = [b for b in keep if b[0] >= hi and round((b[0] - hi) / pitch) <= 1 + sum(1 for x in keep if hi <= x[0] < b[0])]
    return inner + left + right


def first_seam(tops: np.ndarray, tw: float) -> int | None:
    """왼쪽 첫 패의 오른쪽 끝: 위쪽 경계의 큰 꺾임 두 곳 중 왼쪽. 나란히 놓여 안 꺾이면 None."""
    span = 24  # 패 모서리가 둥글어 꺾임이 약 30픽셀에 걸치기도 한다(171245)
    if len(tops) <= span:
        return None
    jump = np.abs(tops[span:].astype(int) - tops[:-span].astype(int))
    cuts: list[int] = []
    for i in np.argsort(-jump):
        c = int(i) + span // 2
        if jump[i] < SEAM_JUMP * tw or len(cuts) == 2:
            break
        if c >= SEAM_MIN * tw and len(tops) - c >= SEAM_MIN * tw and all(abs(c - k) >= SEAM_MIN * tw for k in cuts):
            cuts.append(c)
    return min(cuts) if cuts else None


def corner_blank(rgb: np.ndarray, fr: st.SeatFrame, faces: list) -> list | None:
    """오른쪽 모서리 白 퐁. 펠트 계단으로 찾고([corner_steps]), 못 찾으면 무늬 없는 상자 2장 이상을 퐁 3장으로([corner_pair])."""
    return corner_steps(rgb, fr, faces) or corner_pair(rgb, fr, faces)


def corner_pair(rgb: np.ndarray, fr: st.SeatFrame, faces: list) -> list | None:
    """검출기가 모서리 白 을 일부 잡아 계단 찾기가 포기한 경우(겹침): 손패 줄 오른쪽의 손패·울기에 안 쓰인 무늬 없는 상자가
    이웃해(중심 거리 ≤ 1.6 × 패 긴 변) 2장 이상이면 白 퐁. 2장이면 같은 간격으로 한 칸 더 간 자리에 세 번째 상자.
    퐁은 3장이라는 규칙에서 나온 값(mahjong_calc metrics 표 1-70, 2026-11-05 사용자 "2개 이상 잡히면 3개로")."""
    if not fr.hand:
        return None
    hand_right = max(b[0] + b[2] for b in fr.hand)
    used = {tuple(b) for b in fr.hand} | {tuple(b) for m in fr.melds for b in m} | ({tuple(fr.separate)} if fr.separate else set())
    cand = [b for b in faces if tuple(b) not in used and b[0] > hand_right and st.ink(rgb, tuple(b)) < BLANK]
    if len(cand) < 2:
        return None
    big = 1.6 * max(fr.tw, fr.th)
    cen = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[3] / 2], float)
    groups: list[list] = []
    for b in sorted(cand, key=lambda b: b[0]):
        for g in groups:
            if any(np.linalg.norm(cen(b) - cen(x)) <= big for x in g):
                g.append(b)
                break
        else:
            groups.append([b])
    g = max(groups, key=len)
    if len(g) < 2:
        return None
    g = sorted(g, key=lambda b: b[0])[:3]
    if len(g) == 2:
        a, b = g
        c = cen(b) + (cen(b) - cen(a))
        g.append((int(round(c[0] - b[2] / 2)), int(round(c[1] - b[3] / 2)), b[2], b[3]))
    return [tuple(int(v) for v in x) for x in g]


def corner_steps(rgb: np.ndarray, fr: st.SeatFrame, faces: list) -> list | None:
    """오른쪽 모서리 白 퐁: 펠트 아래쪽 경계가 패 3장 폭만큼 올라온 곳. 세 상자(왼쪽부터) 또는 None."""
    felt = st.felt_mask(rgb)
    y_ref = int(max(b[1] + b[3] for b in fr.hand) + fr.th)  # 손패 줄 아래 펠트에서 아래로 내려가며 끝을 찾는다(탁자 밖 초록 옷·그늘을 안 잡게)
    if y_ref >= felt.shape[0]:
        return None
    # 오른쪽 탁자 끝 = 패 폭 2개 넘게 이어진 펠트 중 가장 오른쪽(테두리 너머 작은 조각을 안 잡게). 탁자 끝은 모서리 패에 가릴 수 있어
    # 손패 바로 아래(y_top)부터 y_ref 까지 여러 줄 중 가장 오른쪽 끝(한 줄만 보면 白 위쪽 끝에 닿아 한 장 좁게 읽었다, metrics 표 1-70 171245)
    y_top = int(max(b[1] + b[3] for b in fr.hand) + 0.15 * fr.th)
    ends = []
    for yy in range(y_top, y_ref + 1, 3):
        cols = np.nonzero(felt[yy])[0]
        if not cols.size:
            continue
        runs = np.split(cols, np.nonzero(np.diff(cols) > 0.5 * fr.tw)[0] + 1)
        long_runs = [c for c in runs if c[-1] - c[0] >= 2 * fr.tw]
        if long_runs:
            ends.append(int(long_runs[-1][-1]))
    if not ends:
        return None
    right = max(ends)
    x0 = max(0, int(right - WINDOW * fr.tw))

    def edge(x: int) -> int:
        if not felt[y_top, x]:
            return -1  # 손·팔 등으로 가려진 줄
        stop = np.nonzero(~felt[y_top:, x])[0]
        return y_top + int(stop[0]) if stop.size else felt.shape[0]

    prof = np.array([edge(x) for x in range(x0, right + 1)])
    # 탁자 끝은 손패 줄과 나란하다. 그만큼 빼고 봐야 기운 사진의 비스듬한 끝을 패로 잘못 보지 않는다
    tilt = _line(fr.hand)[0] * np.arange(len(prof))
    level = prof - tilt
    seen = level[prof >= 0]
    if not seen.size:
        return None
    base = float(np.percentile(seen, 90))  # 모서리는 패가 절반 넘게 가려 높은 쪽으로 잡는다
    up = (prof >= 0) & (level < base - RISE * fr.th)
    idx = np.nonzero(up)[0]
    if not idx.size:
        return None
    end = int(idx.max())
    start = end
    while start > 0 and (up[start - 1] or (start >= 2 and up[start - 2])):
        start -= 1
    x1, x2 = x0 + start, x0 + end + 1
    if right - x2 > CORNER_REACH * fr.tw or not CORNER_W[0] * fr.tw <= x2 - x1 <= CORNER_W[1] * fr.tw:
        return None
    tops = prof[start:end + 1]
    if not CORNER_H[0] * fr.th <= base - float(np.median(level[start:end + 1])) <= CORNER_H[1] * fr.th:
        return None
    y2 = int(base + tilt[end])
    # 계단이라 칸마다 위쪽 끝이 달라 상자 대신 경계 안쪽 픽셀만 본다(펠트·패 옆면 줄 제외)
    hsv = np.asarray(Image.fromarray(rgb[:y2, x1:x2]).convert("HSV")).astype(int)
    trim = int(EDGE * fr.tw)
    inside = np.zeros(hsv.shape[:2], bool)
    for i, t in enumerate(tops):
        if trim <= i < len(tops) - trim and t + trim < y2 - trim:
            inside[t + trim:y2 - trim, i] = True
    if inside.sum() == 0:
        return None
    ink = ((hsv[..., 2] < 120) | (hsv[..., 1] > 80))[inside].mean()
    if ink >= CORNER_INK:
        return None
    # 패마다 상자. 오른쪽·아래는 흰 테두리 위에 걸쳐 경계가 안 보이므로, 폭은 첫 꺾임(없으면 손패 패 폭), 높이는 손패 패 높이로 정한다
    seam = first_seam(tops, fr.tw)
    pw = float(np.clip(seam if seam else fr.tw, fr.tw, TILE_W_MAX * fr.tw))
    boxes = []
    for i in range(3):
        col = tops[int(i * pw):int((i + 1) * pw)]
        top = int(np.median(col)) if col.size else boxes[-1][1]
        boxes.append((round(x1 + i * pw), top, round(pw), round(TILE_H * fr.th)))
    return None if any(overlaps(b, faces) for b in boxes) else boxes


def fill(rgb: np.ndarray, faces: list, split: int) -> tuple[st.SeatFrame | None, list]:
    """틀이 실패한 사진에 白 을 채워 다시 잡는다. (통과한 틀, 채운 상자) 또는 (None, [])."""
    fr = split_frame(rgb, faces, split)
    if fr is None or fr.count >= 13:
        return None, []
    added = row_blanks(rgb, faces, fr, split)
    if fr.count + len(added) > 14:
        return None, []
    if added:
        fr = split_frame(rgb, faces + added, split)
        if fr is None:
            return None, []
    if fr.count + 3 in (13, 14):
        boxes = corner_blank(rgb, fr, faces + added)
        if boxes:
            fr.melds = [*fr.melds, boxes]
            added = added + boxes
    return (fr, added) if added and frame_ok(fr, rgb) else (None, [])


def cut(rgb: np.ndarray, b) -> Image.Image:
    """분류기 크롭. 자리 판독 중 회전 상자로 검출한 상자면 세워 자른다(obb.crop_box), 아니면 축 상자 + 여유."""
    got = obb.crop_box(rgb, b, PAD)
    if got is not None:
        return got
    x, y, w, h = b
    H, W = rgb.shape[:2]
    px, py = int(w * PAD), int(h * PAD)
    return Image.fromarray(rgb[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)])


# ---------------------------------------------------------------- 모델

def _back_mask(rgb: np.ndarray) -> np.ndarray:
    """패 뒷면 후보: 펠트(초록)·피부·흰 면·어두운 곳이 아닌 색 있는 면. PIL HSV 0~255."""
    h, s, v = np.moveaxis(np.asarray(Image.fromarray(rgb).convert("HSV")), 2, 0)
    return (s >= 25) & (v >= 150) & (h > 30) & ((h < 55) | (h > 130))


def wall_score(rgb: np.ndarray, b) -> tuple[int, float]:
    """앞면 패 상자의 앞뒤(한 축)에 패 뒷면이 붙은 정도: (붙은 쪽 수, 점수). 패산에 뒤집어 둔 표시패면 높다."""
    x, y, w, h = (int(v) for v in b)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x - w - 2), max(0, y - h - 2), min(W, x + 2 * w + 2), min(H, y + 2 * h + 2)
    if x1 <= x0 or y1 <= y0:
        return 0, 0.0
    m = _back_mask(rgb[y0:y1, x0:x1])

    def frac(a: int, c: int, p: int, q: int) -> float:
        a, c, p, q = max(a - x0, 0), max(c - y0, 0), min(p - x0, x1 - x0), min(q - y0, y1 - y0)
        return float(m[c:q, a:p].mean()) if p > a and q > c else 0.0

    across = [frac(x - w - 2, y, x - 2, y + h), frac(x + w + 2, y, x + 2 * w + 2, y + h)]
    along = [frac(x, y - h - 2, x + w, y - 2), frac(x, y + h + 2, x + w, y + 2 * h + 2)]
    pair = across if sum(across) >= sum(along) else along
    n = sum(p >= WALL_SIDE for p in pair)
    return n, n + sum(pair) / 2


def top2(q: np.ndarray) -> tuple[str, float, str, float]:
    """확률 벡터 → (1순위, 확신도, 2순위, 그 확신도). 1순위는 예전 `CLASSES[argmax]`·`round(max, 3)` 그대로."""
    i = int(q.argmax())
    rest = q.copy()
    rest[i] = -1.0
    j = int(rest.argmax())
    return CLASSES[i], round(float(q[i]), 3), CLASSES[j], round(float(q[j]), 3)


def tile_json(r: tuple) -> dict:
    """(종류, 확신도[, 2순위, 그 확신도]) → 응답의 패 한 장. 2순위가 없는 것(白 채우기)은 예전 모양 그대로 alt·conf2 를 안 넣는다."""
    return {"tile": r[0], "conf": r[1], **({"alt": r[2], "conf2": r[3]} if len(r) > 3 and r[2] is not None else {})}


def _letterbox(rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """ultralytics 예측과 같게: 긴 변 1280, 짧은 변은 32 배수로 가운데 패딩."""
    H, W = rgb.shape[:2]
    r = min(IMG / H, IMG / W)
    nw, nh = round(W * r), round(H * r)
    cw, ch = -(-nw // 32) * 32, -(-nh // 32) * 32
    left, top = round((cw - nw) / 2 - 0.1), round((ch - nh) / 2 - 0.1)
    canvas = np.full((ch, cw, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    return canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255, r, left, top


class LocalReader:
    """처음 쓸 때 모델을 올린다. [gate] 를 주면 검출·분류를 낮은 우선순위로(자리·강 틀 잡기가 진행 중이면 기다린다)."""

    def __init__(self, gate=None, threads: int = 2, det_model: Path = DET_MODEL, cls_model: Path = CLS_MODEL, seat_model: Path | None = SEAT_MODEL,
                 alt_cls_model: Path | None = ALT_CLS_MODEL):
        self.gate, self.threads, self.det_model, self.cls_model = gate, threads, det_model, cls_model
        self.alt_cls_model = alt_cls_model if alt_cls_model is not None and Path(alt_cls_model).exists() else None
        self.seat_model = seat_model if seat_model is not None and Path(seat_model).exists() else None
        self._lock = threading.Lock()
        self._det = self._cls = self._seat = None
        # 사진 전체 자리 판독 검출(회전 상자 앞면·뒷면)을 같은 바이트면 한 번만(개선 계획 8단계 P1·8b): localroles 의 맥락 검출,
        # WALL_FRAME 기준점 잡기(localroles·localdora·locallast3)가 나눠 쓴다. 판독기마다 따로(테스트의 가짜 검출기끼리 안 섞이게).
        self._full = Shared(size=1, ttl=10.0)

    def _sessions(self):
        with self._lock:
            if self._det is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.threads
                opts.enable_cpu_mem_arena = False  # 쓰지 않을 때 메모리를 쥐고 있지 않게(작은 서버)
                self._det = ort.InferenceSession(str(self.det_model), sess_options=opts, providers=["CPUExecutionProvider"])
                self._cls = ort.InferenceSession(str(self.cls_model), sess_options=opts, providers=["CPUExecutionProvider"])
                if self.seat_model is not None:
                    self._seat = ort.InferenceSession(str(self.seat_model), sess_options=opts, providers=["CPUExecutionProvider"])
            return self._det, self._cls

    def _run(self, fn):
        return self.gate.run(fn, low=True) if self.gate is not None else fn()

    def detect(self, rgb: np.ndarray, low: bool = True) -> list:
        """ultralytics 예측과 같게: 긴 변 1280, 짧은 변은 32 배수로 가운데 패딩. 출력 1×5×N(cx, cy, w, h, conf)."""
        det, _ = self._sessions()
        H, W = rgb.shape[:2]
        r = min(IMG / H, IMG / W)
        nw, nh = round(W * r), round(H * r)
        cw, ch = -(-nw // 32) * 32, -(-nh // 32) * 32
        left, top = round((cw - nw) / 2 - 0.1), round((ch - nh) / 2 - 0.1)
        canvas = np.full((ch, cw, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self._run(lambda: det.run(None, {det.get_inputs()[0].name: x})[0][0])
        keep = out[4] >= CONF
        cx, cy, w, h, c = out[0][keep], out[1][keep], out[2][keep], out[3][keep], out[4][keep]
        x0, y0, x1, y1 = (cx - w / 2 - left) / r, (cy - h / 2 - top) / r, (cx + w / 2 - left) / r, (cy + h / 2 - top) / r
        order, boxes = list(np.argsort(-c)), []
        while order:
            i = order.pop(0)
            boxes.append((max(0.0, float(x0[i])), max(0.0, float(y0[i])), min(float(W), float(x1[i])), min(float(H), float(y1[i]))))
            rest = []
            for j in order:
                ix, iy = min(x1[i], x1[j]) - max(x0[i], x0[j]), min(y1[i], y1[j]) - max(y0[i], y0[j])
                inter = max(0.0, ix) * max(0.0, iy)
                if inter / ((x1[i] - x0[i]) * (y1[i] - y0[i]) + (x1[j] - x0[j]) * (y1[j] - y0[j]) - inter + 1e-9) <= IOU:
                    rest.append(j)
            order = rest
        return [(int(a), int(b), int(p - a), int(q - b)) for a, b, p, q in boxes if p - a >= 2 and q - b >= 2]

    def detect_seat(self, rgb: np.ndarray) -> list:
        """자리 판독 검출: 회전 상자 모델로, 크롭 끝에 잘린 조각을 버리고, 안깡(앞면 2 + 양끝 뒷면)의 뒷면 상자를 끼운다.
        회전 상자·안깡 기록은 obb.session() 안에서만 남는다(분류기 크롭이 세워 자를 때 쓴다). 모델이 없으면 [detect]."""
        self._sessions()
        if self._seat is None:
            return self.detect(rgb)
        res, backs, pairs, scores = self._seat_raw(rgb)
        return obb.record(rgb, res, pairs, backs, scores)

    def _seat_raw(self, rgb: np.ndarray) -> tuple[list, list, list, dict]:
        """자리 판독 검출기 출력 해석: (앞면 [(축 상자, 회전 상자)], 뒷면 [(축 상자, 회전 상자)], 안깡 짝, 앞면 점수). 기록은 안 남김."""
        x, r, left, top = _letterbox(rgb)
        sess = self._seat
        out = self._run(lambda: sess.run(None, {sess.get_inputs()[0].name: x})[0][0])
        H, W = rgb.shape[:2]
        scores: dict = {}
        res = obb.slivers(obb.decode(out, r, left, top, W, H, scores=scores), W, H)
        backs = obb.decode(out, r, left, top, W, H, cls=1) if out.shape[0] >= 7 else []
        pairs = obb.kans(res, backs) if backs else []
        return res, backs, pairs, scores

    def full_raw(self, jpeg: bytes, rgb: np.ndarray) -> tuple[list, list, list, dict]:
        """사진 전체(jpeg 를 decode_rgb 한 rgb)의 자리 판독 검출 — 같은 바이트면 한 번만(8단계 P1). 기록은 안 남김([detect_full] 이 남김).
        자리 판독 검출기가 없으면 [detect] 의 축 상자를 (상자, None) 으로(뒷면·점수 없음)."""
        self._sessions()
        if self._seat is None:
            return self._full.get(digest(jpeg), lambda: ([(b, None) for b in self.detect(rgb)], [], [], {}))
        return self._full.get(digest(jpeg), lambda: self._seat_raw(rgb))

    def detect_full(self, jpeg: bytes, rgb: np.ndarray) -> list:
        """[detect_seat] 과 같은 답(사진 전체), 검출은 [full_raw] 로 나눠 쓴다. 회전 상자·뒷면 기록은 지금 obb.session 에 남긴다."""
        res, backs, pairs, scores = self.full_raw(jpeg, rgb)
        if self._seat is None:
            return [b for b, _ in res]
        return obb.record(rgb, res, pairs, backs, scores)

    def detect_other(self, rgb: np.ndarray) -> list:
        """강·표시패 검출. [OTHER_DET] 가 켜져 있고 자리 판독 검출기가 있으면 그 모델의 앞면(감싸는 축 상자), 아니면 [detect].
        "obb-up" 이면 회전 상자를 지금 기록(obb.session 안일 때)에 남겨 분류기 크롭이 세워 자른다."""
        self._sessions()
        if not OTHER_DET or self._seat is None:
            return self.detect(rgb)
        x, r, left, top = _letterbox(rgb)
        sess = self._seat
        out = self._run(lambda: sess.run(None, {sess.get_inputs()[0].name: x})[0][0])
        H, W = rgb.shape[:2]
        res = obb.decode(out, r, left, top, W, H)
        return obb.record(rgb, res, []) if OTHER_DET == "obb-up" else [b for b, _ in res]

    @staticmethod
    def keep_seat(rgb: np.ndarray, boxes: list) -> list:
        """앞면 거르기(흰 비율) + 겹침 정리. 안깡 뒷면 상자는 흰 비율 거르기를 건너뛴다.
        [ROT_WHITE] 면 회전 상자를 아는 상자는 흰 비율을 회전 상자 안에서 잰다(기운 떼어 둔 패, metrics 표 1-73)."""
        backs = [b for b in boxes if obb.is_kan_back(b)]
        white = rv.white_mask(rgb)
        faces = []
        for b in boxes:
            if b in backs:
                continue
            rect = obb.rect_of(rgb, b) if ROT_WHITE else None
            if rect is None:
                faces += rv.keep_faces([b], white)
            elif obb.white_inside(white, rect) >= 0.5:
                faces.append(b)
        return rv.dedupe(faces + backs)

    def _tensor(self, im: Image.Image) -> np.ndarray:
        im = im.copy()
        im.thumbnail((CLS_SIZE, CLS_SIZE))
        out = Image.new("RGB", (CLS_SIZE, CLS_SIZE), (114, 114, 114))
        out.paste(im, ((CLS_SIZE - im.width) // 2, (CLS_SIZE - im.height) // 2))
        return (np.asarray(out, np.float32).transpose(2, 0, 1) / 255 - MEAN) / STD

    def probs(self, ims: list[Image.Image]) -> np.ndarray:
        _, cls = self._sessions()
        x = np.stack([self._tensor(i) for i in ims])
        y = self._run(lambda: cls.run(None, {cls.get_inputs()[0].name: x})[0])
        e = np.exp(y - y.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def read(self, ims: list[Image.Image]) -> list[tuple[str, float]]:
        """자리 크롭처럼 이미 세운 패. (종류, 확신도)."""
        return [r[:2] for r in self.read2(ims)]

    def read2(self, ims: list[Image.Image]) -> list[tuple[str, float, str, float]]:
        """[read] + 분류기 2순위(종류, 확신도). 답(1순위)은 read 와 같다(개선 계획 4단계 V3 후보용, 2026-11-05)."""
        return [top2(q) for q in self.probs(ims)] if ims else []

    def read_any(self, rgb: np.ndarray, b) -> tuple[str, float]:
        """방향을 모르는 패: 상자 모양에 맞는 방향 중 확신이 가장 높은 값(매장 강 패는 뒤집혀 놓이기도 한다)."""
        return self.read_any2(rgb, b)[:2]

    def read_any2(self, rgb: np.ndarray, b) -> tuple[str, float, str, float]:
        """[read_any] + 고른 방향에서의 2순위."""
        im = cut(rgb, b)
        ts = (0, 2) if b[3] > 1.15 * b[2] else (1, 3) if b[2] > 1.15 * b[3] else (0, 1, 2, 3)
        p = self.probs([im.rotate(90 * k, expand=True) for k in ts])
        i = int(p.max(1).argmax())
        return top2(p[i])

    def read_small(self, rgb: np.ndarray, b) -> tuple[str, float]:
        """추적 프레임(1020×764)처럼 패가 작을 때: 크롭을 긴 변 [SMALL_TILE] 로 키워 읽는다. _tensor 는 줄이기만 해서
        작은 패가 회색 바탕에 그대로 들어가 0/8 이었다(metrics 표 1-34, 키우면 6/8). 자리·강 크롭은 패가 커서 안 거친다."""
        im = cut(rgb, b)
        s = SMALL_TILE / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
        ts = (0, 2) if b[3] > 1.15 * b[2] else (1, 3) if b[2] > 1.15 * b[3] else (0, 1, 2, 3)
        p = self.probs([im.rotate(90 * k, expand=True) for k in ts])
        i = int(p.max(1).argmax())
        return CLASSES[int(p[i].argmax())], round(float(p[i].max()), 3)

    def read_meld_tile(self, rgb: np.ndarray, b) -> tuple[str, float]:
        """울기: 세운 패는 그대로, 눕힌 패는 두 방향 중 확신 높은 쪽."""
        return self.read_meld_tile2(rgb, b)[:2]

    def read_meld_tile2(self, rgb: np.ndarray, b) -> tuple[str, float, str, float]:
        """[read_meld_tile] + 2순위."""
        im = cut(rgb, b)
        if b[2] <= 1.15 * b[3]:
            return self.read2([im])[0]
        p = self.probs([im.rotate(90 * k, expand=True) for k in (1, 3)])
        i = int(p.max(1).argmax())
        return top2(p[i])

    def read_turned(self, rgb: np.ndarray, b, tw: float, th: float) -> tuple[str, float]:
        """돌아간 낱장: 각도마다 패 크기로 잘라 가장 확신 높은 값."""
        return self.read_turned2(rgb, b, tw, th)[:2]

    def read_turned2(self, rgb: np.ndarray, b, tw: float, th: float) -> tuple[str, float, str, float]:
        """[read_turned] + 2순위(각도별 최댓값 벡터에서)."""
        cx, cy, r = b[0] + b[2] / 2, b[1] + b[3] / 2, int(max(b[2], b[3], th))
        H, W = rgb.shape[:2]
        part = Image.fromarray(rgb).crop((max(0, int(cx - r)), max(0, int(cy - r)), min(W, int(cx + r)), min(H, int(cy + r))))
        m = part.width / 2, part.height / 2
        box = (int(m[0] - tw / 2), int(m[1] - th / 2), int(m[0] + tw / 2), int(m[1] + th / 2))
        p = self.probs([part.rotate(a).crop(box) for a in ANGLES]).max(0)
        return top2(p)

    # ------------------------------------------------------------ 판독

    def read_wall(self, jpeg: bytes, frame=None, grid: dict | None = None, grid_src: str | None = None) -> dict:
        with obb.session():
            return self._read_wall(jpeg, frame, grid, grid_src)

    def _read_wall(self, jpeg: bytes, frame=None, grid: dict | None = None, grid_src: str | None = None) -> dict:
        """테이블 전체 사진 → 패산에 뒤집어 둔 표시패 한 장. 옆에 패 뒷면이 가장 많이 붙은 앞면 패를 고른다.
        못 찾거나 확신이 낮으면 indicators 가 빈다(白 은 검출기가 자주 놓친다). 리치 화료의 표시패 짝은 다루지 않는다.
        측정: mahjong_calc metrics 표 1-34·1-35(리치 아닌 화료 9/9, 같은 매장 사진).
        WALL_FRAME(개선 계획 8b): 기준점(frame = 앱이 준 것, 없으면 사진에서 — 자리 판독 검출을 localroles 와 나눠 씀)을 잡아 후보·짝을
        구역표 표시패 탐색 칸(dora_search) ∪ 왕패 칸(dead_wall, 네 변)에 드는 것만 남기고 짝마다 바람개비 칸(zone)을 단다. 손패/표시패 구분은
        지금처럼 검출·짝 규칙 몫(크롭으로 가르지 않음). rot 이면 그 둘레 정사각형을 θ 만큼 세워 자른 그림에서 검출·패산 옆 검사·읽기를 하고
        상자는 원본 좌표로 돌려준다. 못 잡으면 frame_error 를 넣고 지금처럼.
        rot2(8b-3b 후보 3): 세워 자르지 않고 지금 경로와 같은 사진 전체 검출([detect_other])에서, 패산 표시패 후보만 dora_search ∪ dead_wall 로
        거르고 탁자 위 짝은 지금처럼 사진 전체 앞면에서 묶는다(짝마다 zone) — 크롭 끝에 잘린 납작한 조각의 헛 짝(표 1-105 165614)과 13u 밖
        진짜 짝(B·C·D k15·k26, 가운데에서 약 14u)을 둘 다 피한다. 자리 판독 검출(full_raw)을 나눠 쓰면 검출기가 바뀌어 짝 후보가 달라지므로
        (개발 단위 확인에서 132239·134815 에 5z 낀 새 짝) 쓰지 않는다.
        hybrid(8b-3c): 격자 출처 grid_src(없으면 WALL_FRAME_GRID_SRC, 그것도 없으면 saved — 8b-4a)가 saved|manual|felt
        면 지금 경로, none|default 면 rot2(table_locate.route)."""
        from . import table_locate as tl

        t0 = time.perf_counter()
        rgb = decode_rgb(jpeg)
        t_dec = time.perf_counter()
        wf_on, zc = tl.mode() is not None, None
        wf, tz, fmeta = tl.route(self, jpeg, rgb, frame, grid, grid_src)
        if wf:
            if tz is not None and wf == "rot":  # rot2 는 세워 자르지 않음(아래 whole)
                s = tl.dora_reach(tz)
                zc = tl.warp_zone(rgb, tz, "bottom", (-s, s), (-s, s))
                if zc is None:
                    tz, fmeta["frame_error"] = None, "표시패 구역이 사진 밖"
        img = zc.rgb if zc is not None else rgb
        to_orig = (lambda b: zc.box_to_orig(b)[0]) if zc is not None else (lambda b: b)
        t_frame = time.perf_counter()
        whole = wf == "rot2" and tz is not None  # 사진 전체(img = rgb), 후보만 구역으로
        faces = self.detect_other(img)
        cands = faces
        if tz is not None:
            n_all = len(faces)
            cands = [b for b in faces if tl.in_dora_zone(tz, to_orig(b))]
            if not whole:
                faces = cands
            fmeta["kept"] = [len(cands), n_all]
        t_det = time.perf_counter()
        scored = sorted(((*wall_score(img, b), b) for b in cands), key=lambda c: -c[1])
        top = scored[0] if scored and scored[0][0] else None
        t_rule = time.perf_counter()
        out = {"indicators": [], "box": None, "sides": 0, "conf": 0.0, "faces": len(cands), "pairs": []}
        if top:
            tile, conf = self.read_small(img, top[2]) if max(top[2][2], top[2][3]) < SMALL_TILE else self.read_any(img, top[2])
            out.update(box=[round(v) for v in to_orig(top[2])], sides=top[0], conf=conf)
            if conf >= WALL_CONF:
                out["indicators"] = [{"tile": tile, "conf": conf}]
        tw = statistics.median(min(f[2], f[3]) for f in faces) if faces else 1
        th = statistics.median(max(f[2], f[3]) for f in faces) if faces else 1
        for group in st.cluster(faces, tw * 0.5, tw * 0.5):  # 리치 화료의 탁자 위 표시패 짝 후보: 딱 2장 붙은 묶음
            if len(group) == 2:
                # 비스듬히 내놓은 짝은 0·90° 로는 틀리게 읽힌다(170105: 3p·6z 0.58·0.33 → 돌려 읽으면 5p·5s 0.98). 15° 단위로 돌려 읽는다.
                # 상자(픽셀)도 같이 돌려줘 앱이 격자 구역(자리 띠·강·리치봉)으로 짝의 자리를 가린다(mahjong_calc metrics 표 1-54).
                read = [self.read_turned(img, b, tw, th) for b in group]
                pair = [{"tile": t, "conf": c, "box": [round(v) for v in to_orig(b)]} for (t, c), b in zip(read, group)]
                if tz is not None:  # 기준점이 있으면 짝마다 바람개비 칸(앱이 격자 대신 쓸 수 있게)
                    for p in pair:
                        x, y, w, h = p["box"]
                        p["zone"] = tz.zone_of(x + w / 2, y + h / 2)
                out["pairs"].append(pair)
        out["size"] = [rgb.shape[1], rgb.shape[0]]
        out["ms"] = {"decode": round((t_dec - t0) * 1000), "detect": round((t_det - t_frame) * 1000), "rules": round((t_rule - t_det) * 1000), "read": _ms(t_rule), "total": _ms(t0)}
        if wf_on:
            out["ms"]["frame"] = round((t_frame - t_dec) * 1000)
            fmeta.pop("frame_ms", None)
            out.update(fmeta)
            out["cut"] = "rot" if zc is not None else "whole" if whole else "axis" if tz is not None else "grid"
        out["rss_mb"] = rss_mb()
        return out

    def read_seat(self, jpeg: bytes, hand_frac: float | None, fill_blank: bool = True) -> dict:
        """자리 크롭(그 자리 방향으로 세운 것) → framed, hand(떼어 둔 패는 끝에)·hand_conf, melds·meld_conf(묶음에서 가장 낮은 값),
        block(홀로 놓인 패가 둘 이상), filled(코드로 채운 白 장수), 시간."""
        with obb.session():
            return self._read_seat(jpeg, hand_frac, fill_blank)

    def _read_seat(self, jpeg: bytes, hand_frac: float | None, fill_blank: bool) -> dict:
        t0 = time.perf_counter()
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        boxes = self.detect_seat(rgb)
        t_det = time.perf_counter()
        faces = self.keep_seat(rgb, boxes)
        split = int(rgb.shape[1] * (hand_frac if hand_frac else 0.7))
        fr = split_frame(rgb, faces, split)
        framed = frame_ok(fr, rgb)
        added: list = []
        if not framed and fill_blank:
            got, added = fill(rgb, faces, split)
            if got is not None:
                fr, framed = got, True
        t_rule = time.perf_counter()
        out = {"framed": framed, "hand": [], "hand_conf": [], "melds": [], "meld_conf": [], "block": False, "faces": len(faces), "filled": len(added)}
        if framed:
            filled = set(added)

            def blank_or(b, read):
                """白 을 채운 사진은 무늬 없는 상자도 白 으로(분류기 답을 안 쓴다). 틀을 그냥 잡은 사진은 분류기 답 그대로."""
                return ("5z", FILLED_CONF, None, None) if filled and (b in filled or st.ink(rgb, b) < BLANK) else read(b)

            hand = [blank_or(b, lambda x: self.read2([cut(rgb, x)])[0]) for b in fr.hand + ([fr.separate] if fr.separate else [])]
            fr.melds, melds, flags = resplit_melds(fr, lambda b: blank_or(b, lambda x: self.read_meld_tile2(rgb, x)))
            if fr.count == 13 and not fr.separate and not added:
                got = strays(fr, faces, rgb)
                if len(got) == 1:
                    hand.append(self.read_turned2(rgb, got[0], fr.tw, fr.th))
                elif got:
                    out["block"] = True
            out["hand"], out["hand_conf"] = [t[0] for t in hand], [t[1] for t in hand]
            out["melds"], out["meld_conf"] = [" ".join(t[0] for t in m) for m in melds], [meld_conf(m, f) for m, f in zip(melds, flags)]
            out["meld_split"] = flags
            # 분류기 2순위(답은 위 그대로, 개선 계획 4단계 V3 후보용)
            out["hand_alt"], out["hand_conf2"] = [t[2] for t in hand], [t[3] for t in hand]
            out["meld_alts"], out["meld_conf2s"] = [[t[2] for t in m] for m in melds], [[t[3] for t in m] for m in melds]
        out["ms"] = {"detect": round((t_det - t0) * 1000), "rules": round((t_rule - t_det) * 1000), "read": _ms(t_rule), "total": _ms(t0)}
        out["rss_mb"] = rss_mb()
        return out

    def read_seat_app(self, jpeg: bytes, hand_frac: float | None) -> dict:
        """read_seat 을 앱의 /v2/seat 응답 모양으로. 틀을 못 잡으면 패 없이 답한다(Gemini 로 넘기지 않음)."""
        r = self.read_seat(jpeg, hand_frac)
        count = len(r["hand"]) + 3 * len(r["melds"])
        separate = r["hand"][-1] if r["hand"] and count % 3 == 2 else None
        hand = r["hand"][:-1] if separate else r["hand"]
        none = [None] * len(r["hand"])
        tiles = [tile_json(x) for x in zip(r["hand"], r["hand_conf"], r.get("hand_alt", none), r.get("hand_conf2", none))]
        return {"face_up": r["framed"] or r["faces"] >= FACE_UP_MIN, "tiles": tiles[:len(hand)], "separate_tile": tiles[-1] if separate else None,
                "melds": [{"tiles": m.split(), "sideways": [], "face_down": [], "conf": c, "alts": a, "conf2s": c2}
                          for m, c, a, c2 in zip(r["melds"], r["meld_conf"], r.get("meld_alts", [None] * len(r["melds"])),
                                                 r.get("meld_conf2s", [None] * len(r["melds"])))], "loose": [],
                "framed": r["framed"], "filled": r["filled"], "ms": r["ms"], "rss_mb": r["rss_mb"]}

    def read_rivers_app(self, jpegs: list[bytes]) -> dict:
        """강 크롭들 → 앱의 /v2/last3 응답 모양(마지막 패·장수·눕힌 패 순번). 못 잡은 자리는 패 없음."""
        with obb.session():
            return self._read_rivers_app(jpegs)

    def _read_rivers_app(self, jpegs: list[bytes]) -> dict:
        t0 = time.perf_counter()
        rivers = []
        for j in jpegs:
            rgb = np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))
            fr = river_frame(rgb, self.detect_other(rgb))
            if fr is None:
                rivers.append({"last_tile": None, "tile_count": 0, "sideways": []})
                continue
            tile, conf = self.read_any(rgb, fr.last)
            rivers.append({"last_tile": {"tile": tile, "conf": conf}, "tile_count": fr.count, "sideways": [i for i, s in enumerate(fr.sideways) if s]})
        return {"rivers": rivers, "method": "frame", "ms": {"total": _ms(t0)}, "rss_mb": rss_mb()}

    def read_rivers_full(self, jpeg: bytes, sides: list[str], grid: dict | None = None, frame=None, grid_src: str | None = None) -> dict:
        """사진 전체 + 자리들 → [read_rivers_app] 과 같은 모양(개선 계획 8b, 앱이 강 끝 크롭을 안 보내는 길). 강 끝 크롭은
        WALL_FRAME=rot 이면 구역표 강 칸(river)을 세워 자르고, axis 면 기준점에서 만든 격자로, 꺼져 있거나 기준점을 못 잡으면 [grid] 로
        앱 GridZones.lastZone 과 같게 자른다. 격자도 없으면 강이 다 빈다(frame_error)."""
        with obb.session():
            return self._read_rivers_full(jpeg, sides, grid, frame, grid_src)

    def _read_rivers_full(self, jpeg: bytes, sides: list[str], grid: dict | None, frame, grid_src: str | None = None) -> dict:
        from . import table_locate as tl

        t0 = time.perf_counter()
        rgb = decode_rgb(jpeg)
        wf_on = tl.mode() is not None
        wf, tz, fmeta = tl.route(self, jpeg, rgb, frame, grid, grid_src)  # hybrid: 격자를 가졌거나 출처 모름이면 지금 경로, 명시적 none 이면 rot2
        fmeta.pop("frame_ms", None)
        t_frame = time.perf_counter()
        crops: list = []
        if tz is not None and wf in tl.ROT:
            for s in sides:
                zc = tl.warp_rect(rgb, tz, "river", s)
                crops.append(None if zc is None else zc.rgb)
        else:
            g = tl.grid_from_frame(tz) if tz is not None else None
            if tz is not None and g is None:
                fmeta["frame_error"], tz = "격자 선이 사진 밖", None
            g = g or grid
            im = Image.fromarray(rgb)
            crops = [None if g is None else np.asarray(last_zone_crop(im, g, s)) for s in sides]
        rivers = []
        for c in crops:
            fr = river_frame(c, self.detect_other(c)) if c is not None and c.size else None
            if fr is None:
                rivers.append({"last_tile": None, "tile_count": 0, "sideways": []})
                continue
            tile, conf = self.read_any(c, fr.last)
            rivers.append({"last_tile": {"tile": tile, "conf": conf}, "tile_count": fr.count, "sideways": [i for i, s in enumerate(fr.sideways) if s]})
        out = {"rivers": rivers, "method": "frame", "sides": list(sides), "ms": {"total": _ms(t0)}, "rss_mb": rss_mb()}
        if wf_on:
            out["ms"]["frame"] = round((t_frame - t0) * 1000)
            out.update(fmeta)
            out["cut"] = "rot" if tz is not None and wf in tl.ROT else "axis" if tz is not None else "grid"
        return out

    def read_rivers(self, jpegs: list[bytes]) -> dict:
        """강 크롭들(각 자리 쪽으로 세운 것, 보낸 순서) → rivers[{tile, conf, count}|None], 시간."""
        t0 = time.perf_counter()
        rivers = []
        for j in jpegs:
            rgb = np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))
            fr = rv.frame(rgb, self.detect(rgb))
            if fr is None:
                rivers.append(None)
                continue
            tile, conf = self.read_any(rgb, fr.last)
            rivers.append({"tile": tile, "conf": conf, "count": fr.count})
        return {"rivers": rivers, "ms": {"total": _ms(t0)}, "rss_mb": rss_mb()}
