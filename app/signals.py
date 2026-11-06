"""판독 불확실성 신호(개선 계획 5단계, 2026-11-05): `/v2/localroles` 응답에 더하는 부가 정보. 답(tiles·melds·separate_tile·block)은 안 바꾼다.

패마다(손패 줄·떼어 둔 패·울기 패, 白 채우기 패는 None):
  S1 margin  보정 확률(온도 [TEMP])의 1·2순위 차, pconf = 보정 1순위 확률
  S2 det     자리 크롭 검출기(회전 상자 8회차) 앞면 점수
  S3 cls2    두 번째 분류기(같은 학습 묶음·다른 씨드, assets/tile-cls-alt-v2.onnx)가 같은 크롭을 읽은 답
  S5 jitter  같은 크롭을 ×0.9 축소 + JPEG q75 로 다시 저장해 운영 분류기로 다시 읽은 답(검출은 다시 안 한다)
  S6 red     분류가 5m·5p·5s·0m·0p·0s 인 크롭의 붉은 잉크 비율(빨강 픽셀 / 잉크 픽셀, 패 가운데만)
자리 요약(signals): margin_min·det_min(손패 줄 + 떼어 둔 패)·cls2_diff·jitter_diff(답과 다른 패 수)·role_diff(S4 역할 모델
3씨드 투표가 갈린 상자 수, 손패·울기·떼어 둔 패 후보인 것만)·red(패별 붉은 비율 목록). 문턱은 앱(mahjong_calc lib/table/read_signals.dart)이 갖는다
— 서버는 값만 낸다. LOCAL_SIGNALS=0 이면 안 낸다(시간 비교용).
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import local_read as lr
from . import obb

ON = os.environ.get("LOCAL_SIGNALS", "1") == "1"
TOP = int(os.environ.get("LOCAL_SIGNALS_TOP", "0"))  # 측정용: 패마다 보정 전 확률 상위 몇 개를 같이 낸다(온도 다시 맞추기)
# 온도 0.5: 분류기가 라벨 스무딩(0.05) 탓에 확신을 낮게 낸다(보정 전 0.5~0.6 인 패도 96~98% 맞음). 정한 곳은 fold 회귀 21변형의 서버 경로 패
# 5,261장(그 사진을 안 배운 모델, 오답 50) 1순위 맞음 NLL 최소(mahjong_calc experiments/speed/select_thresholds.py temp, ECE 0.067 → 0.003,
# results/temperature_tiles.json). 나눔 크롭 홀드아웃(prep_alt_cls.py)은 오답 0 이라 T 를 못 정했다. argmax 는 안 바뀐다.
TEMP = float(os.environ.get("LOCAL_CLS_TEMP", "0.5"))
ALT_CLS_MODEL = Path(os.environ.get("LOCAL_ALT_CLS_MODEL") or Path(__file__).parent / "assets" / "tile-cls-alt-v2.onnx")
FIVES = {"5m", "5p", "5s", "0m", "0p", "0s"}
JITTER_SCALE, JITTER_Q = 0.9, 75
RED_TRIM = 0.16  # 크롭 여유(PAD 0.12) + 패 가장자리·옆면을 빼고 가운데만
RED_MIN_INK = 30  # 잉크 픽셀이 이보다 적으면 비율을 안 낸다


class AltCls:
    """두 번째 분류기 세션(경로별 하나). 없으면 S3 를 안 낸다."""

    _cache: dict = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, path: Path | None, threads: int = 2):
        if path is None or not Path(path).exists():
            return None
        with cls._lock:
            key = str(path)
            if key not in cls._cache:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = threads
                opts.enable_cpu_mem_arena = False
                cls._cache[key] = ort.InferenceSession(key, sess_options=opts, providers=["CPUExecutionProvider"])
            return cls._cache[key]


def calibrate(p: np.ndarray, T: float = TEMP) -> np.ndarray:
    """softmax 확률 → softmax(logit / T). log p 는 logit 과 상수 차이라 같은 결과."""
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def jitter(im: Image.Image) -> Image.Image:
    """×0.9 축소 + JPEG q75 재저장(입력 흔들기, 검출은 그대로)."""
    small = im.resize((max(1, round(im.width * JITTER_SCALE)), max(1, round(im.height * JITTER_SCALE))), Image.BILINEAR)
    buf = io.BytesIO()
    small.convert("RGB").save(buf, "JPEG", quality=JITTER_Q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def red_ratio(im: Image.Image) -> float | None:
    """패 가운데의 붉은 잉크 비율. 잉크 = 어둡거나(V<110) 색이 진함(S>80, V>60), 빨강 = 진한 색 중 색상 0~14·235~255(PIL HSV 0~255).
    적5 는 무늬 전체가 빨강, 보통 5m 은 萬 만, 5p·5s 는 가운데 하나만 빨강이라 이 비율로 갈린다. 문턱은 앱이 갖는다."""
    a = np.asarray(im.convert("HSV")).astype(int)
    H, W = a.shape[:2]
    my, mx = int(H * RED_TRIM), int(W * RED_TRIM)
    a = a[my:H - my, mx:W - mx]
    if a.size == 0:
        return None
    h, s, v = a[..., 0], a[..., 1], a[..., 2]
    vivid = (s > 80) & (v > 60)
    ink = (v < 110) | vivid
    red = vivid & ((h <= 14) | (h >= 235))
    n = int(ink.sum())
    return round(float(red.sum()) / n, 3) if n >= RED_MIN_INK else None


def _orient(reader: lr.LocalReader, rgb: np.ndarray, b, kind: str, tw: float = 0, th: float = 0) -> list[Image.Image]:
    """답을 낸 판독 함수와 같은 방향 후보들(첫 판독이 고른 방향 = 운영 분류기 최댓값)."""
    if kind == "turned":
        cx, cy, r = b[0] + b[2] / 2, b[1] + b[3] / 2, int(max(b[2], b[3], th))
        H, W = rgb.shape[:2]
        part = Image.fromarray(rgb).crop((max(0, int(cx - r)), max(0, int(cy - r)), min(W, int(cx + r)), min(H, int(cy + r))))
        m = part.width / 2, part.height / 2
        box = (int(m[0] - tw / 2), int(m[1] - th / 2), int(m[0] + tw / 2), int(m[1] + th / 2))
        return [part.rotate(a).crop(box) for a in lr.ANGLES]
    im = lr.cut(rgb, b)
    if kind == "hand" or (kind == "meld" and b[2] <= 1.15 * b[3]):
        return [im]
    ks = (1, 3) if kind == "meld" else (0, 2) if b[3] > 1.15 * b[2] else (1, 3) if b[2] > 1.15 * b[3] else (0, 1, 2, 3)
    return [im.rotate(90 * k, expand=True) for k in ks]


def collect(reader: lr.LocalReader, items: list, role_seeds: list | None) -> dict:
    """items = [(그룹, 그림, 상자, kind, 답 패, tw, th)] — 그룹 'hand'|'sep'|'meld<i>', kind 'hand'|'meld'|'turned'|'any', 답 패 None = 白 채우기.
    role_seeds = 씨드별 역할 [[역할]] (손패·울기·떼어 둔 패 후보 상자만 센다) 또는 None. 돌려주는 것: 자리 요약 + 패별 값."""
    t0 = time.perf_counter()
    alt = AltCls.get(getattr(reader, "alt_cls_model", None), reader.threads)
    per: list = [None] * len(items)
    cands, owner = [], []
    for k, (_, rgb, b, kind, tile, tw, th) in enumerate(items):
        if tile is None:
            continue
        ims = _orient(reader, rgb, b, kind, tw, th)
        cands += ims
        owner += [k] * len(ims)
    if cands:
        p = reader.probs(cands)
        best: dict = {}
        for i, k in enumerate(owner):
            if k not in best or p[i].max() > p[best[k]].max():
                best[k] = i
        keys = sorted(best)
        chosen = [cands[best[k]] for k in keys]
        pm = calibrate(np.stack([p[best[k]] for k in keys]))
        pj = reader.probs([jitter(im) for im in chosen])
        pa = None
        if alt is not None:
            x = np.stack([reader._tensor(im) for im in chosen])
            ya = reader._run(lambda: alt.run(None, {alt.get_inputs()[0].name: x})[0])
            pa = ya
        for n, k in enumerate(keys):
            grp, rgb, b, kind, tile, _, _ = items[k]
            q = np.sort(pm[n])[::-1]
            mine = lr.CLASSES[int(pm[n].argmax())]
            d = {"pconf": round(float(q[0]), 3), "margin": round(float(q[0] - q[1]), 3), "det": obb.score_of(rgb, b),
                 "jitter": lr.CLASSES[int(pj[n].argmax())], "same": mine == tile}
            if pa is not None:
                d["cls2"] = lr.CLASSES[int(pa[n].argmax())]
            if tile in FIVES:
                d["red"] = red_ratio(chosen[n])
            if TOP:
                raw = p[best[k]]
                d["top"] = [[lr.CLASSES[int(i)], round(float(raw[i]), 5)] for i in np.argsort(-raw)[:TOP]]
            per[k] = d
    rows = [(items[k][0], items[k][4], d) for k, d in enumerate(per)]
    read = [(g, t, d) for g, t, d in rows if d is not None]
    hand_dets = [d["det"] for g, _, d in read if g in ("hand", "sep") and d["det"] is not None]
    out = {
        "margin_min": min((d["margin"] for _, _, d in read), default=None),
        "pconf_min": min((d["pconf"] for _, _, d in read), default=None),
        "det_min": min(hand_dets, default=None),
        "det_min_all": min((d["det"] for _, _, d in read if d["det"] is not None), default=None),
        "cls2_diff": sum(d.get("cls2", t) != t for _, t, d in read) if alt is not None else None,
        "jitter_diff": sum(d["jitter"] != t for _, t, d in read),
        "red": [{"tile": t, "ratio": d["red"]} for _, t, d in read if "red" in d],
        "temp": TEMP,
        "n": len(read),
        "filled": len(rows) - len(read),
    }
    if role_seeds is not None:
        mine = {"hand", "meld", "loose"}
        out["role_diff"] = sum(1 for votes in zip(*role_seeds) if len(set(votes)) > 1 and set(votes) & mine)
    out["tiles"] = [d for g, _, d in rows if g == "hand"]
    out["separate"] = next((d for g, _, d in rows if g == "sep"), None)
    out["melds"] = []
    for g, _, d in rows:
        if g.startswith("meld"):
            i = int(g[4:])
            while len(out["melds"]) <= i:
                out["melds"].append([])
            out["melds"][i].append(d)
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out
