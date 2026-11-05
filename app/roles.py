"""사진 전체 + 격자 + 화료자 자리 → 상자 배치만으로 역할(손패·울기·강·패산 표시패·짝·떼어 둔 패) → 자리 판독(`/v2/localroles`, 2026-11-04).

왜: 자리 크롭만 보는 규칙(split_frame)은 다른 자리 패·갈라진 손패 줄에서 틀린다. 탁자 전체 상자를 보는 Transformer(mahjong_calc
experiments/tile_classifier/role_set.py, 합성 배치 사전학습 + 매장 25장 미세 조정, metrics 표 1-58·1-59)가 상자마다 역할을 정하면 그 자리 것만 남는다.
흐름: 사진 전체 검출(맥락) + 자리 크롭(격자 40 키움, 그 자리 방향으로 세움) 검출(고해상) → 크롭 상자를 원본 좌표로 → 펠트 격자 기준 벡터 → 역할 →
      hybrid: 모델이 이 자리 것으로 고른 상자만 split_frame(규칙)에 / model: 역할로 바로 줄·울기 묶기 → 패 읽기(read_seat 과 같은 분류기·白 채우기).
입력 벡터(role_set.scene_tokens 과 같아야 한다): 펠트 격자(x1·x2·y1·y2, 1000 분율) 기준 0~1 좌표·크기×10, 화료자 자리가 아래로 오게 돌림,
  뒷면 플래그, 눕힘, 리치봉 상자 중심과의 차, 1.
  모델이 11칸(2026-11-05 회전 상자 특징, ONNX 입력 모양에서 읽음)이면 크기·눕힘은 회전 상자(패에 맞는 변)로 재고 sin·cos(2×기울기)를 더하며,
  검출기의 뒷면 상자(obb.backs_of)도 뒷면 플래그 1 로 넣는다(역할 결과에는 안 나감). 9칸(예전 모델)이면 축 상자·뒷면 없음 그대로.
"""
from __future__ import annotations

import io
import os
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from . import local_read as lr
from . import obb
from . import seat as st

# 역할 모델: 11칸(회전 상자 크기·기울기 + 뒷면 토큰, 라벨 수정 뒤 재학습, mahjong_calc metrics 표 1-78·1-79, 2026-11-05 사용자 "11칸 서버 코드 변경").
# 예전 9칸 tile-roles.onnx(124654·125839 손패 끝 3장을 울기로 배운 라벨)는 되돌리기용으로 남김 — LOCAL_ROLE_MODEL 로 끼우면 tokens 가 입력 모양으로 9칸을 만든다.
ROLE_MODEL = Path(os.environ.get("LOCAL_ROLE_MODEL") or Path(__file__).parent / "assets" / "tile-roles-rot.onnx")
ROLES = ["hand", "meld", "river", "wall", "pair", "loose"]
SIDES = ("bottom", "right", "top", "left")
ROT_CW = {"bottom": 0, "right": 90, "top": 180, "left": 270}
GROW = 40  # 앱 seatReadGrow
LOOSE_REACH = 3.0  # 크롭 밖 떼어 둔 화료패: 손패 패 높이의 몇 배 안
FEAT = 9  # 예전 특징(축 상자, 뒷면 없음)
FEAT_ROT = 11  # 회전 상자 특징(+ sin·cos 2×기울기, 뒷면 상자 넣음)


def seat_box(g: dict, side: str, grow: int = GROW) -> tuple[int, int, int, int]:
    """(ymin, xmin, ymax, xmax) 1000 분율. 앱 GridZones.seatForRead 와 같다."""
    x1, x2, y1, y2 = g["x1"], g["x2"], g["y1"], g["y2"]
    return {"bottom": (y2 - grow, x1, 1000, 1000), "right": (0, x2 - grow, y2, 1000), "top": (0, 0, y1 + grow, x2), "left": (y1, 0, 1000, x1 + grow)}[side]


def hand_frac(g: dict, side: str) -> float:
    x1, x2, y1, y2 = g["x1"], g["x2"], g["y1"], g["y2"]
    return {"bottom": (x2 - x1) / (1000 - x1), "right": (y2 - y1) / y2, "top": (x2 - x1) / x2, "left": (y2 - y1) / (1000 - y1)}[side]


def turn(u: float, v: float, side: str) -> tuple[float, float, bool]:
    if side == "bottom":
        return u, v, False
    if side == "right":
        return 1 - v, u, True
    if side == "top":
        return 1 - u, 1 - v, False
    return v, 1 - u, True


def tokens(boxes: list, W: int, H: int, g: dict, side: str, feat: int = FEAT, pts: list | None = None, backs: list | None = None) -> np.ndarray:
    """상자 벡터 (N, feat). pts = 상자마다 회전 상자 꼭짓점(원본 좌표) 또는 None, backs = [(축 상자, 꼭짓점)] 뒷면(feat 11 일 때만 뒤에 붙음)."""
    x1, x2, y1, y2 = (g[k] / 1000 for k in ("x1", "x2", "y1", "y2"))
    fw, fh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    s = g.get("sticks")
    su, sv = ((s[1] + s[3]) / 2000, (s[0] + s[2]) / 2000) if s else ((x1 + x2) / 2, (y1 + y2) / 2)
    su, sv, _ = turn((su - x1) / fw, (sv - y1) / fh, side)
    rot = feat == FEAT_ROT
    items = [(b, (pts[i] if rot and pts else None), 0.0) for i, b in enumerate(boxes)]
    if rot and backs:
        items += [(b, p, 1.0) for b, p in backs]
    rows = []
    for (x, y, w, h), p, back in items:
        u, v, swap = turn(((x + w / 2) / W - x1) / fw, ((y + h / 2) / H - y1) / fh, side)
        pw, ph, tilt = obb.tile_geom((x, y, w, h), p if rot else None)
        bw, bh = pw / W / fw, ph / H / fh
        if swap:
            bw, bh = bh, bw
        row = [u, v, bw * 10, bh * 10, back, float(bw > bh * 1.15), u - su, v - sv, 1.0]
        if rot:
            t = np.radians(2 * tilt)
            row += [float(np.sin(t)), float(np.cos(t))]
        rows.append(row)
    return np.array(rows, np.float32).reshape(-1, feat)


def unrotate(b, rot: int, cw: int, ch: int) -> tuple:
    """자리 방향으로 세운 크롭(rotate(-rot, expand))의 상자 → 돌리기 전 크롭 좌표."""
    x, y, w, h = b
    pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    if rot == 90:
        back = [(py, ch - px) for px, py in pts]
    elif rot == 180:
        back = [(cw - px, ch - py) for px, py in pts]
    elif rot == 270:
        back = [(cw - py, px) for px, py in pts]
    else:
        back = pts
    xs, ys = [p[0] for p in back], [p[1] for p in back]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def unrotate_pts(pts, rot: int, cw: int, ch: int, x0: int = 0, y0: int = 0) -> np.ndarray:
    """세운 크롭 좌표의 꼭짓점들 → 원본 사진 좌표(돌리기 전 크롭 + 크롭 원점)."""
    out = []
    for px, py in np.asarray(pts, float):
        if rot == 90:
            q = (py, ch - px)
        elif rot == 180:
            q = (cw - px, ch - py)
        elif rot == 270:
            q = (cw - py, px)
        else:
            q = (px, py)
        out.append((q[0] + x0, q[1] + y0))
    return np.array(out, float)


def _pts_of(rgb: np.ndarray, b) -> np.ndarray | None:
    rc = obb.rect_of(rgb, b)
    return None if rc is None else obb.corners(rc)


def crop_seat(im: Image.Image, g: dict, side: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """앱과 같은 자리 크롭(세운 것)과 돌리기 전 크롭의 (x0, y0, cw, ch)."""
    W, H = im.size
    ymin, xmin, ymax, xmax = seat_box(g, side)
    x0, y0, x1, y1 = round(xmin * W / 1000), round(ymin * H / 1000), round(xmax * W / 1000), round(ymax * H / 1000)
    out = im.crop((x0, y0, x1, y1))
    if ROT_CW[side]:
        out = out.rotate(-ROT_CW[side], expand=True)
    return out, (x0, y0, x1 - x0, y1 - y0)


def iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


class RoleNet:
    def __init__(self, path: Path = ROLE_MODEL, threads: int = 2):
        self.path, self.threads = path, threads
        self._lock = threading.Lock()
        self._sess = None

    def _session(self):
        with self._lock:
            if self._sess is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.threads
                opts.enable_cpu_mem_arena = False
                self._sess = ort.InferenceSession(str(self.path), sess_options=opts, providers=["CPUExecutionProvider"])
            return self._sess

    @property
    def feat(self) -> int:
        """모델 입력 칸 수(ONNX 입력 모양 [1, n, feat]) — 9 = 예전, 11 = 회전 상자 특징."""
        shape = self._session().get_inputs()[0].shape
        return int(shape[-1]) if isinstance(shape[-1], int) else FEAT

    def predict(self, x: np.ndarray) -> tuple[list[str], list[float]]:
        if len(x) == 0:
            return [], []
        p = self._session().run(None, {"x": x[None]})[0][0]
        return [ROLES[int(i)] for i in p.argmax(1)], [round(float(v), 3) for v in p.max(1)]


_role_net: RoleNet | None = None


def role_net() -> RoleNet:
    global _role_net
    if _role_net is None:
        _role_net = RoleNet()
    return _role_net


def frame_from_roles(rgb: np.ndarray, faces: list, roles: list[str], split: int, mode: str) -> tuple[st.SeatFrame | None, list]:
    """(틀, 채운 白 상자). hybrid: 이 자리 것(손패·울기·떼어 둔 패)만 split_frame 에, 실패하면 model 방식. model: 역할로 줄·울기 묶기."""
    mine = [f for f, r in zip(faces, roles) if r in ("hand", "meld", "loose")]
    fr = None
    if mode == "hybrid" and len(mine) >= 2:
        fr = lr.split_frame(rgb, mine, split)
        if fr is not None and not lr.frame_ok(fr, rgb):
            fr = None
    if fr is None:
        hand = sorted([f for f, r in zip(faces, roles) if r == "hand"], key=lambda b: b[0])
        if len(hand) < 2:
            return None, []
        up = [f for f in faces if f[3] > f[2]] or faces
        tw, th = statistics.median(f[2] for f in up), statistics.median(f[3] for f in up)
        a, c = lr._line(hand)
        loose = [f for f, r in zip(faces, roles) if r == "loose"]
        hx1 = hand[-1][0] + hand[-1][2]
        near = [f for f in loose if abs(f[1] + f[3] / 2 - (a * (f[0] + f[2] / 2) + c)) < th and 0 < f[0] - hx1 < st.SEP_DIST * tw]
        sep = near[0] if len(near) == 1 else None
        melds = []
        for grp in st.cluster([f for f, r in zip(faces, roles) if r == "meld"], tw * st.GAP, th * 0.5):
            grp = sorted(grp, key=lambda b: b[0])
            if len(grp) in (3, 4):
                melds.append(grp)
            elif len(grp) in st.SPLIT_MELD:
                k = st.SPLIT_MELD[len(grp)][0]
                by_y = sorted(grp, key=lambda b: b[1] + b[3] / 2)
                if by_y[k][1] + by_y[k][3] / 2 - (by_y[k - 1][1] + by_y[k - 1][3] / 2) > 0.4 * th:
                    melds += [sorted(by_y[:k], key=lambda b: b[0]), sorted(by_y[k:], key=lambda b: b[0])]
                else:
                    melds += [grp[:k], grp[k:]]
        melds.sort(key=lambda m: m[0][0])
        fr = st.SeatFrame(hand, sep, melds, [f for f in loose if f is not sep], tw, th, len(faces), split=True)
    added: list = []
    if fr.count < 13:  # 서버 fill 과 같은 白 채우기: 손패 줄 틈·양 끝, 그래도 3장 모자라면 모서리 白 울기
        added = lr.row_blanks(rgb, faces, fr, split)
        if fr.count + len(added) in (13, 14):
            fr.hand = sorted(fr.hand + [tuple(b) for b in added], key=lambda b: b[0])
        elif fr.count + len(added) + 3 in (13, 14):
            if added:
                fr.hand = sorted(fr.hand + [tuple(b) for b in added], key=lambda b: b[0])
            boxes = lr.corner_blank(rgb, fr, faces + added)
            if boxes:
                fr.melds = [*fr.melds, boxes]
                added = added + boxes
            else:
                fr.hand = [b for b in fr.hand if tuple(b) not in set(map(tuple, added))]
                added = []
        else:
            added = []
    return fr, added


def read_roles(reader: lr.LocalReader, jpeg: bytes, side: str, grid: dict, mode: str = "hybrid", net: RoleNet | None = None) -> dict:
    """사진 전체 → 화료자 자리의 손패·울기(/v2/seat 모양) + 상자별 역할(roles: 원본 좌표 box, role, conf, in_seat)."""
    with obb.session():
        out = _read_roles(reader, jpeg, side, grid, mode, net)
        _loose_outside(reader, jpeg, out)
        return out


def _loose_outside(reader: lr.LocalReader, jpeg: bytes, out: dict) -> None:
    """자리 크롭 밖의 떼어 둔 화료패(2026-11-05, metrics 표 1-70 143104): 크롭 안에서 떼어 둔 패를 못 찾았고 손패 + 울기 × 3 = 13 이면,
    크롭 밖 상자 중 역할이 loose 이고 화료자 손패에서(가장 가까운 손패 중심까지) 패 높이 [LOOSE_REACH] 배 안인 것이 딱 하나면
    그것을 원본 사진에서 읽어 떼어 둔 화료패로."""
    if not out.get("framed") or out.get("separate_tile") or len(out["tiles"]) + 3 * len(out["melds"]) != 13:
        return
    hand = [x["box"] for x in out["roles"] if x["in_seat"] and x["role"] == "hand"]
    if not hand:
        return
    th = float(np.median([max(b[2], b[3]) for b in hand]))
    cen = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[3] / 2], float)
    near = [x for x in out["roles"] if not x["in_seat"] and x["role"] == "loose"
            and min(np.linalg.norm(cen(x["box"]) - cen(h)) for h in hand) <= LOOSE_REACH * th]
    if len(near) != 1:
        return
    full = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    tile, conf = reader.read_any(full, tuple(near[0]["box"]))
    out["separate_tile"] = {"tile": tile, "conf": conf, "box": near[0]["box"], "from": "loose_outside"}


def _read_roles(reader: lr.LocalReader, jpeg: bytes, side: str, grid: dict, mode: str, net: RoleNet | None) -> dict:
    t0 = time.perf_counter()
    net = net or role_net()
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    W, H = im.size
    full = np.asarray(im)
    t_dec = time.perf_counter()
    ctx = reader.detect_seat(full)  # 맥락용(작게 보이는 다른 자리 패)
    crop_im, (x0, y0, cw, ch) = crop_seat(im, grid, side)
    rgb = np.asarray(crop_im)
    faces = reader.keep_seat(rgb, reader.detect_seat(rgb))
    t_det = time.perf_counter()
    rot = ROT_CW[side]
    faces_full, pts = [], []
    for b in faces:
        x, y, w, h = unrotate(b, rot, cw, ch)
        faces_full.append((x + x0, y + y0, w, h))
        p = _pts_of(rgb, b)
        pts.append(None if p is None else unrotate_pts(p, rot, cw, ch, x0, y0))
    # 크롭 상자 우선, 크롭과 겹치지 않는 전체 사진 상자를 맥락으로 더한다
    extra = [b for b in ctx if all(iou(b, f) < 0.3 for f in faces_full)]
    allb = faces_full + extra
    pts += [_pts_of(full, b) for b in extra]
    backs: list = []
    if net.feat == FEAT_ROT:  # 뒷면 상자(패산·다른 자리 손패·안깡 양끝): 크롭 것 우선, 겹치지 않는 전체 사진 것
        for b, rc in obb.backs_of(rgb):
            x, y, w, h = unrotate(b, rot, cw, ch)
            backs.append(((x + x0, y + y0, w, h), unrotate_pts(obb.corners(rc), rot, cw, ch, x0, y0)))
        backs += [(b, obb.corners(rc)) for b, rc in obb.backs_of(full) if all(iou(b, f) < 0.3 for f, _ in backs) and all(iou(b, f) < 0.3 for f in allb)]
    roles, confs = net.predict(tokens(allb, W, H, grid, side, net.feat, pts, backs))
    roles, confs = roles[: len(allb)], confs[: len(allb)]  # 뒷면 토큰의 역할은 버린다
    seat_roles = roles[: len(faces)]
    t_role = time.perf_counter()
    split = int(rgb.shape[1] * hand_frac(grid, side))
    fr, added = frame_from_roles(rgb, faces, seat_roles, split, mode) if faces else (None, [])
    out = {"face_up": fr is not None or len(faces) >= lr.FACE_UP_MIN, "tiles": [], "separate_tile": None, "melds": [], "loose": [], "framed": fr is not None,
           "filled": len(added), "mode": mode, "roles": [{"box": [round(v) for v in b], "role": r, "conf": c, "in_seat": i < len(faces)} for i, (b, r, c) in enumerate(zip(allb, roles, confs))],
           "backs": len(backs), "feat": net.feat}
    if fr is not None:
        filled = set(map(tuple, added))

        def blank_or(b, read):
            return ("5z", lr.FILLED_CONF) if filled and (tuple(b) in filled or st.ink(rgb, b) < lr.BLANK) else read(b)

        hand = [blank_or(b, lambda x: reader.read([lr.cut(rgb, x)])[0]) for b in fr.hand]
        sep = blank_or(fr.separate, lambda x: reader.read([lr.cut(rgb, x)])[0]) if fr.separate else None
        melds = [[blank_or(b, lambda x: reader.read_meld_tile(rgb, x)) for b in m] for m in fr.melds]
        block = False
        if fr.count == 13 and not fr.separate and not added:
            got = lr.strays(fr, faces, rgb)
            if len(got) == 1:
                sep = reader.read_turned(rgb, got[0], fr.tw, fr.th)
            elif got:
                block = True
        out["tiles"] = [{"tile": t, "conf": c} for t, c in hand]
        out["separate_tile"] = {"tile": sep[0], "conf": sep[1]} if sep else None
        out["melds"] = [{"tiles": [t for t, _ in m], "sideways": [], "face_down": [], "conf": min(c for _, c in m)} for m in melds]
        out["block"] = block
    out["counts"] = dict(Counter(seat_roles))
    out["ms"] = {"decode": round((t_dec - t0) * 1000), "detect": round((t_det - t_dec) * 1000), "roles": round((t_role - t_det) * 1000), "read": lr._ms(t_role), "total": lr._ms(t0)}
    out["rss_mb"] = lr.rss_mb()
    return out
