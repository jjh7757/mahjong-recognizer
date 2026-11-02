"""자체 모델 판독(그림자): 자리 크롭·강 크롭을 Gemini 없이 읽어 비교용으로 남긴다. 사용자에게 보이는 답에는 쓰지 않는다.

검출 = YOLO26n "앞면 패" 1종(mahjong_calc/experiments/tile_classifier, 입력 긴 변 1280, NMS 필요), 분류 = MobileNetV3 Small 37종(128×128).
자리: 통째 검출 1번 → seat.structure_split 복사본(이웃 패를 울기에서 뺌, 위아래 울기를 줄로 나눔) → 모양·무늬 검사(옅은 무늬·白 허용, 직선 0.30패)
→ 분류, 13장이면 손패 줄 오른쪽 홀로 놓인 패를 떼어 둔 패로. 강: 검출 → river.frame → 마지막 패를 위아래 두 방향으로 읽어 확신 높은 쪽.
측정: mahjong_calc metrics 표 1-23·1-26·1-28·1-29(같은 매장 사진, t4g.small 자리 0.69 / 0.91 / 1.02초).
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

from . import river as rv
from . import seat as st

log = logging.getLogger("local_read")

# 환경 변수는 측정용: 그 사진을 학습하지 않은 나눔 모델로 재기 위해.
DET_MODEL = Path(os.environ.get("LOCAL_DET_MODEL") or Path(__file__).parent / "assets" / "tile-yolo26n.onnx")
CLS_MODEL = Path(os.environ.get("LOCAL_CLS_MODEL") or Path(__file__).parent / "assets" / "tile-cls-mnv3s.onnx")
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
WALL_CONF = 0.7  # 이보다 확신이 낮으면 답을 내지 않는다(측정에서 맞은 답 0.81 이상, 돌아가 겹친 패 0.33·0.58, metrics 표 1-35)


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


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
        g = sorted(g, key=lambda x: x[0])
        if len(g) in (3, 4):
            melds.append(g)
        elif len(g) in st.SPLIT_MELD:
            k = st.SPLIT_MELD[len(g)][0]
            by_y = sorted(g, key=lambda x: x[1] + x[3] / 2)
            if by_y[k][1] + by_y[k][3] / 2 - (by_y[k - 1][1] + by_y[k - 1][3] / 2) > 0.4 * th:  # 위아래 두 줄
                melds += [sorted(by_y[:k], key=lambda x: x[0]), sorted(by_y[k:], key=lambda x: x[0])]
            else:
                melds += [g[:k], g[k:]]
        else:
            loose.extend(g)
    melds.sort(key=lambda m: m[0][0])
    return st.SeatFrame(hand, sep, melds, loose, tw, th, len(faces), split=True)


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


def cut(rgb: np.ndarray, b) -> Image.Image:
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


class LocalReader:
    """처음 쓸 때 모델을 올린다. [gate] 를 주면 검출·분류를 낮은 우선순위로(자리·강 틀 잡기가 진행 중이면 기다린다)."""

    def __init__(self, gate=None, threads: int = 2, det_model: Path = DET_MODEL, cls_model: Path = CLS_MODEL):
        self.gate, self.threads, self.det_model, self.cls_model = gate, threads, det_model, cls_model
        self._lock = threading.Lock()
        self._det = self._cls = None

    def _sessions(self):
        with self._lock:
            if self._det is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.threads
                opts.enable_cpu_mem_arena = False  # 쓰지 않을 때 메모리를 쥐고 있지 않게(작은 서버)
                self._det = ort.InferenceSession(str(self.det_model), sess_options=opts, providers=["CPUExecutionProvider"])
                self._cls = ort.InferenceSession(str(self.cls_model), sess_options=opts, providers=["CPUExecutionProvider"])
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

    def read(self, ims: list[Image.Image]) -> list[str]:
        """자리 크롭처럼 이미 세운 패."""
        return [CLASSES[int(k)] for k in self.probs(ims).argmax(1)] if ims else []

    def read_any(self, rgb: np.ndarray, b) -> tuple[str, float]:
        """방향을 모르는 패: 상자 모양에 맞는 방향 중 확신이 가장 높은 값(매장 강 패는 뒤집혀 놓이기도 한다)."""
        im = cut(rgb, b)
        ts = (0, 2) if b[3] > 1.15 * b[2] else (1, 3) if b[2] > 1.15 * b[3] else (0, 1, 2, 3)
        p = self.probs([im.rotate(90 * k, expand=True) for k in ts])
        i = int(p.max(1).argmax())
        return CLASSES[int(p[i].argmax())], round(float(p[i].max()), 3)

    def read_meld_tile(self, rgb: np.ndarray, b) -> str:
        """울기: 세운 패는 그대로, 눕힌 패는 두 방향 중 확신 높은 쪽."""
        im = cut(rgb, b)
        if b[2] <= 1.15 * b[3]:
            return self.read([im])[0]
        p = self.probs([im.rotate(90 * k, expand=True) for k in (1, 3)])
        return CLASSES[int(p[int(p.max(1).argmax())].argmax())]

    def read_turned(self, rgb: np.ndarray, b, tw: float, th: float) -> str:
        """돌아간 낱장: 각도마다 패 크기로 잘라 가장 확신 높은 값."""
        cx, cy, r = b[0] + b[2] / 2, b[1] + b[3] / 2, int(max(b[2], b[3], th))
        H, W = rgb.shape[:2]
        part = Image.fromarray(rgb).crop((max(0, int(cx - r)), max(0, int(cy - r)), min(W, int(cx + r)), min(H, int(cy + r))))
        m = part.width / 2, part.height / 2
        box = (int(m[0] - tw / 2), int(m[1] - th / 2), int(m[0] + tw / 2), int(m[1] + th / 2))
        p = self.probs([part.rotate(a).crop(box) for a in ANGLES])
        return CLASSES[int(p.max(0).argmax())]

    # ------------------------------------------------------------ 판독

    def read_wall(self, jpeg: bytes) -> dict:
        """테이블 전체 사진 → 패산에 뒤집어 둔 표시패 한 장. 옆에 패 뒷면이 가장 많이 붙은 앞면 패를 고른다.
        못 찾거나 확신이 낮으면 indicators 가 빈다(白 은 검출기가 자주 놓친다). 리치 화료의 표시패 짝은 다루지 않는다.
        측정: mahjong_calc metrics 표 1-34·1-35(리치 아닌 화료 9/9, 같은 매장 사진)."""
        t0 = time.perf_counter()
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        t_dec = time.perf_counter()
        faces = self.detect(rgb)
        t_det = time.perf_counter()
        scored = sorted(((*wall_score(rgb, b), b) for b in faces), key=lambda c: -c[1])
        top = scored[0] if scored and scored[0][0] else None
        t_rule = time.perf_counter()
        out = {"indicators": [], "box": None, "sides": 0, "conf": 0.0, "faces": len(faces), "pairs": []}
        if top:
            tile, conf = self.read_any(rgb, top[2])
            out.update(box=list(top[2]), sides=top[0], conf=conf)
            if conf >= WALL_CONF:
                out["indicators"] = [{"tile": tile, "conf": conf}]
        tw = statistics.median(min(f[2], f[3]) for f in faces) if faces else 1
        for group in st.cluster(faces, tw * 0.5, tw * 0.5):  # 리치 화료의 탁자 위 표시패 짝 후보: 딱 2장 붙은 묶음
            if len(group) == 2:
                read = [self.read_any(rgb, b) for b in group]
                out["pairs"].append([{"tile": t, "conf": c} for t, c in read])
        out["ms"] = {"decode": round((t_dec - t0) * 1000), "detect": round((t_det - t_dec) * 1000), "rules": round((t_rule - t_det) * 1000), "read": _ms(t_rule), "total": _ms(t0)}
        out["rss_mb"] = rss_mb()
        return out

    def read_seat(self, jpeg: bytes, hand_frac: float | None) -> dict:
        """자리 크롭(그 자리 방향으로 세운 것) → framed, hand(떼어 둔 패는 끝에), melds, block(홀로 놓인 패가 둘 이상), 시간."""
        t0 = time.perf_counter()
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        boxes = self.detect(rgb)
        t_det = time.perf_counter()
        faces = rv.dedupe(rv.keep_faces(boxes, rv.white_mask(rgb)))
        split = int(rgb.shape[1] * (hand_frac if hand_frac else 0.7))
        fr = split_frame(rgb, faces, split)
        framed = frame_ok(fr, rgb)
        t_rule = time.perf_counter()
        out = {"framed": framed, "hand": [], "melds": [], "block": False, "faces": len(faces)}
        if framed:
            out["hand"] = self.read([cut(rgb, b) for b in fr.hand] + ([cut(rgb, fr.separate)] if fr.separate else []))
            out["melds"] = [" ".join(self.read_meld_tile(rgb, b) for b in m) for m in fr.melds]
            if fr.count == 13 and not fr.separate:
                got = strays(fr, faces, rgb)
                if len(got) == 1:
                    out["hand"].append(self.read_turned(rgb, got[0], fr.tw, fr.th))
                elif got:
                    out["block"] = True
        out["ms"] = {"detect": round((t_det - t0) * 1000), "rules": round((t_rule - t_det) * 1000), "read": _ms(t_rule), "total": _ms(t0)}
        out["rss_mb"] = rss_mb()
        return out

    def read_seat_app(self, jpeg: bytes, hand_frac: float | None) -> dict:
        """read_seat 을 앱의 /v2/seat 응답 모양으로. 틀을 못 잡으면 패 없이 답한다(Gemini 로 넘기지 않음)."""
        r = self.read_seat(jpeg, hand_frac)
        count = len(r["hand"]) + 3 * len(r["melds"])
        separate = r["hand"][-1] if r["hand"] and count % 3 == 2 else None
        hand = r["hand"][:-1] if separate else r["hand"]
        tile = lambda t: {"tile": t, "conf": 0.9}  # noqa: E731
        return {"face_up": r["framed"] or r["faces"] >= FACE_UP_MIN, "tiles": [tile(t) for t in hand], "separate_tile": tile(separate) if separate else None,
                "melds": [{"tiles": m.split(), "sideways": [], "face_down": [], "conf": 0.9} for m in r["melds"]], "loose": [], "framed": r["framed"], "ms": r["ms"], "rss_mb": r["rss_mb"]}

    def read_rivers_app(self, jpegs: list[bytes]) -> dict:
        """강 크롭들 → 앱의 /v2/last3 응답 모양(마지막 패·장수·눕힌 패 순번). 못 잡은 자리는 패 없음."""
        t0 = time.perf_counter()
        rivers = []
        for j in jpegs:
            rgb = np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))
            fr = rv.frame(rgb, self.detect(rgb), LOCAL_SIDEWAYS_HEIGHT)
            if fr is None:
                rivers.append({"last_tile": None, "tile_count": 0, "sideways": []})
                continue
            tile, conf = self.read_any(rgb, fr.last)
            rivers.append({"last_tile": {"tile": tile, "conf": conf}, "tile_count": fr.count, "sideways": [i for i, s in enumerate(fr.sideways) if s]})
        return {"rivers": rivers, "method": "frame", "ms": {"total": _ms(t0)}, "rss_mb": rss_mb()}

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
