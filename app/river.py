"""강(버림패) 틀 잡기: YOLO11 패 검출(위치만) → 그 사람 강의 줄·칸 → 마지막 패 상자.

패 종류는 여기서 정하지 않는다(Gemini 가 잘라 낸 패를 읽는다). 검출 모델은 nikmomo/Mahjong-YOLO 의 YOLO11s
(MIT, 학습 데이터 Kaggle shinz114514 MIT, Ultralytics AGPL-3.0 — THIRD_PARTY_NOTICES.md). ultralytics 코드 없이 onnxruntime 으로 돌린다.

입력 크롭은 가운데 칸을 그 사람 쪽으로 세운 것(그 사람이 사진 아래). 손패 사진으로 학습한 모델이라 강 패를 낮은 확신도로 찾아
기준을 0.04 로 둔다(B·C·D 실측: 0.25 → 구조 2/18, 0.04 → 15/18, 마지막 패 위치 18/18 — mahjong_calc/experiments/speed/river_yolo.py).
"""
from __future__ import annotations

import io
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = Path(__file__).parent / "assets" / "mahjong-yolo11s.onnx"
INPUT = 640

Box = tuple[int, int, int, int]  # x, y, w, h (원본 크롭 픽셀)


@dataclass
class RiverFrame:
    rows: list[list[Box]]  # 그 사람 강, 위(테이블 가운데 쪽) → 아래, 줄 안은 왼쪽 → 오른쪽
    detected: int = 0  # 거르기 전 검출 수
    sideways: list[bool] = field(default_factory=list)  # 버린 순서대로 눕힌 패인지

    @property
    def last(self) -> Box:
        return self.rows[-1][-1]

    @property
    def count(self) -> int:
        return sum(len(r) for r in self.rows)


# ---------------------------------------------------------------- 검출(onnxruntime)

class TileDetector:
    """YOLO11 detect ONNX(입력 1×3×640×640, 출력 1×(4+37)×8400, NMS 없음)."""

    def __init__(self, path: Path = MODEL, conf: float = 0.04, iou: float = 0.7):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.conf, self.iou = conf, iou
        self._lock = threading.Lock()

    def detect(self, rgb: np.ndarray) -> list[Box]:
        H, W = rgb.shape[:2]
        x, scale, px, py = letterbox(rgb)
        with self._lock:  # 작은 서버라 동시에 돌리지 않는다(메모리·CPU)
            out = self.session.run(None, {self.input: x})[0][0]  # (41, 8400)
        return decode(out, scale, px, py, W, H, self.conf, self.iou)


def letterbox(rgb: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """비율 유지 축소 + 가운데 회색(114) 패딩. ultralytics 예측과 같은 방식."""
    H, W = rgb.shape[:2]
    s = min(INPUT / H, INPUT / W)
    nw, nh = round(W * s), round(H * s)
    im = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)
    px, py = (INPUT - nw) / 2, (INPUT - nh) / 2
    top, left = round(py - 0.1), round(px - 0.1)
    canvas[top:top + nh, left:left + nw] = im
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
    return x, s, left, top


def decode(out: np.ndarray, scale: float, px: float, py: float, W: int, H: int, conf: float, iou: float) -> list[Box]:
    boxes, scores = out[:4].T, out[4:].T  # (8400, 4) cx cy w h, (8400, 37)
    cls = scores.argmax(1)
    best = scores[np.arange(len(cls)), cls]
    keep = best >= conf
    boxes, best, cls = boxes[keep], best[keep], cls[keep]
    xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
    # 종류별 NMS(ultralytics 기본과 같게): 종류마다 좌표를 크게 띄워 한 번에
    picked = nms(xyxy + cls[:, None] * 4096.0, best, iou)
    res = []
    for i in picked:
        x0, y0, x1, y1 = xyxy[i]
        x0, x1 = (x0 - px) / scale, (x1 - px) / scale
        y0, y1 = (y0 - py) / scale, (y1 - py) / scale
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            res.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return res


def nms(xyxy: np.ndarray, score: np.ndarray, iou: float, max_det: int = 300) -> list[int]:
    order = score.argsort()[::-1]
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(xyxy[i, 0], xyxy[order[1:], 0])
        yy0 = np.maximum(xyxy[i, 1], xyxy[order[1:], 1])
        xx1 = np.minimum(xyxy[i, 2], xyxy[order[1:], 2])
        yy1 = np.minimum(xyxy[i, 3], xyxy[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        o = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][o <= iou]
    return keep


# ---------------------------------------------------------------- 틀(정답·모델 없이 규칙만)

def white_mask(rgb: np.ndarray) -> np.ndarray:
    """패 면 = 채도 낮고 밝음(HSV S<50, V>140)."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    return (hsv[..., 1] < 50) & (hsv[..., 2] > 140)


def keep_faces(boxes: list[Box], white: np.ndarray, min_white: float = 0.5) -> list[Box]:
    """빈 펠트·기계 위 가짜 상자 거르기: 패 면이면 상자 안 절반 이상이 흰색."""
    return [b for b in boxes if white[b[1]:b[1] + b[3], b[0]:b[0] + b[2]].mean() >= min_white]


def dedupe(boxes: list[Box]) -> list[Box]:
    """같은 패에 겹친 상자(작은 쪽의 70% 이상이 겹침)는 큰 상자 하나만."""
    keep: list[Box] = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        dup = False
        for k in keep:
            ix = max(0, min(b[0] + b[2], k[0] + k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[1] + b[3], k[1] + k[3]) - max(b[1], k[1]))
            if ix * iy > 0.7 * min(b[2] * b[3], k[2] * k[3]):
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def own_group(boxes: list[Box], W: int, H: int) -> tuple[list[Box], float, float] | None:
    """사진 아래쪽 가운데에 가장 가까운 상자에서 시작해 이웃(틈 < 패 반 개)으로 넓힌 묶음 + 패 너비·높이."""
    if not boxes:
        return None
    upright = [b for b in boxes if b[3] > b[2]] or boxes
    tw = statistics.median(b[2] for b in upright)
    th = statistics.median(b[3] for b in upright)
    cand = [b for b in boxes if b[1] + b[3] / 2 > H * 0.25]
    if not cand:
        return None
    start = min(cand, key=lambda b: abs(b[0] + b[2] / 2 - W / 2) + abs(b[1] + b[3] - H * 0.75))
    group, todo = {start}, [start]
    while todo:
        a = todo.pop()
        for b in boxes:
            if b in group:
                continue
            dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
            dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
            if dx < tw * 0.5 and dy < th * 0.5:
                group.add(b)
                todo.append(b)
    return sorted(group, key=lambda b: (b[1], b[0])), tw, th


def to_rows(boxes: list[Box], th: float) -> list[list[Box]]:
    """가운데 y 가 줄 평균과 패 높이 반 안이면 같은 줄(처진 패도 같은 줄로)."""
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        cy = b[1] + b[3] / 2
        if rows and abs(cy - statistics.mean(r[1] + r[3] / 2 for r in rows[-1])) < th * 0.5:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [sorted(r, key=lambda b: b[0]) for r in rows]


def own_rows(rows: list[list[Box]], tw: float, th: float) -> list[list[Box]]:
    """맨 아래 줄부터 위로: 강 왼쪽 끝 L 기준 [L - 0.3패, L + 6.5패] 안의 패만. 강은 앞 줄을 6장 채워야 다음 줄로 가므로
    마지막 줄이 아닌 줄은 5~6장(검출 누락 하나 허용)이어야 하고, 줄 사이가 패 높이 반 넘게 벌어지면 멈춘다."""
    if not rows:
        return []
    acc = [rows[-1]]
    left = min(b[0] for b in rows[-1])
    for row in reversed(rows[:-1]):
        keep = [b for b in row if left - 0.3 * tw <= b[0] + b[2] / 2 <= left + 6.5 * tw]
        if not 5 <= len(keep) <= 6:
            break
        if min(b[1] for b in acc[-1]) - max(b[1] + b[3] for b in keep) > 0.5 * th:
            break
        acc.append(sorted(keep, key=lambda b: b[0]))
        left = min(left, min(b[0] for b in keep))
    return list(reversed(acc))


def frame(rgb: np.ndarray, boxes: list[Box]) -> RiverFrame | None:
    H, W = rgb.shape[:2]
    faces = dedupe(keep_faces(boxes, white_mask(rgb)))
    got = own_group(faces, W, H)
    if got is None:
        return None
    group, tw, th = got
    rows = own_rows(to_rows(group, th), tw, th)
    if not rows:
        return None
    return RiverFrame(rows, len(boxes), [b[2] > 1.15 * b[3] for r in rows for b in r])


def crop(rgb: np.ndarray, box: Box, pad: float = 0.12) -> bytes:
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    H, W = rgb.shape[:2]
    c = rgb[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
    buf = io.BytesIO()
    Image.fromarray(c).save(buf, "JPEG", quality=92)
    return buf.getvalue()


class RiverLocator:
    """크롭 JPEG → RiverFrame(못 잡으면 None). 모델은 처음 쓸 때 읽는다."""

    def __init__(self, conf: float = 0.04):
        self.conf = conf
        self._det: TileDetector | None = None
        self._init = threading.Lock()

    def _detector(self) -> TileDetector:
        with self._init:
            if self._det is None:
                self._det = TileDetector(conf=self.conf)
            return self._det

    def locate(self, jpeg: bytes) -> tuple[RiverFrame | None, np.ndarray]:
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        return frame(rgb, self._detector().detect(rgb)), rgb
