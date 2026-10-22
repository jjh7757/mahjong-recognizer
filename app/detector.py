"""ONNX Runtime 로 YOLO 모델을 돌려 마작 패를 검출한다.

ultralytics 가 내보낸 ONNX(출력 (1, 4+클래스수, N), xywh 픽셀 좌표) 를 전제로 한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Detection:
    tile: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


class Detector:
    def __init__(self, model_path: Path, classes_path: Path, conf: float = 0.5, iou: float = 0.5):
        import onnxruntime as ort

        self.classes: list[str] = json.loads(classes_path.read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.conf = conf
        self.iou = iou
        self.version = model_path.stem

    # ---------- 전처리 ----------

    def _letterbox(self, img: Image.Image) -> tuple[np.ndarray, float, int, int]:
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(resized, (px, py))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None]  # 1x3xHxW
        return np.ascontiguousarray(arr), scale, px, py

    # ---------- 추론 ----------

    def detect(self, img: Image.Image) -> tuple[list[Detection], float]:
        arr, scale, px, py = self._letterbox(img)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: arr})[0]
        infer_s = time.perf_counter() - t0

        pred = out[0]  # (4+nc, N)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # (N, 4+nc)
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores.max(axis=1)
        keep = conf >= self.conf
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]

        # xywh(letterbox 픽셀) → xyxy(원본 픽셀)
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - px) / scale
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - py) / scale
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - px) / scale
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - py) / scale

        idx = nms(xyxy, conf, self.iou)
        dets = [
            Detection(self.classes[int(cls[i])], float(conf[i]), *map(float, xyxy[i]))
            for i in idx
        ]
        return dets, infer_s


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """클래스 구분 없는 NMS. 패는 겹치지 않으므로 클래스 무관하게 억제하는 편이 안전하다."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def arrange(dets: list[Detection]) -> list[dict]:
    """검출 결과를 줄(row)·묶음(group) 번호와 함께 왼쪽→오른쪽 순으로 정렬한다.

    - row: y 중심이 패 높이의 60% 이상 떨어지면 다른 줄.
    - group: 같은 줄에서 옆 패와의 간격이 패 폭의 60% 이상 벌어지면 다른 묶음
      (손패와 부른 묶음을 띄워 놓으면 자동으로 나뉜다).
    """
    if not dets:
        return []
    med_h = float(np.median([d.h for d in dets]))
    med_w = float(np.median([d.w for d in dets]))
    rows: list[list[Detection]] = []
    for d in sorted(dets, key=lambda d: d.cy):
        if rows and abs(d.cy - np.mean([r.cy for r in rows[-1]])) < med_h * 0.6:
            rows[-1].append(d)
        else:
            rows.append([d])

    result: list[dict] = []
    group = 0
    for ri, row in enumerate(rows):
        row.sort(key=lambda d: d.cx)
        prev: Detection | None = None
        for d in row:
            if prev is not None and d.x1 - prev.x2 > med_w * 0.6:
                group += 1
            result.append({
                "tile": d.tile,
                "conf": round(d.conf, 3),
                "box": [round(d.x1), round(d.y1), round(d.w), round(d.h)],
                "row": ri,
                "group": group,
            })
            prev = d
        group += 1
    return result
