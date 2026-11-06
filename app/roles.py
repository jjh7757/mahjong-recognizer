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
from . import signals as sig
from . import table_frame as tf
from . import table_locate as tl

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


def tokens_frame(boxes: list, tz: tf.TableZones, side: str, feat: int = FEAT, pts: list | None = None, backs: list | None = None) -> np.ndarray:
    """[tokens] 와 같은 칸을 탁자 좌표(기준점, u)로(WALL_FRAME=rot, 개선 계획 8b). 가운데 칸 [−R, R]² 가 지금 격자의 x1~x2·y1~y2 자리,
    리치봉 상자 가운데 = 기기 가운데(0, 0). 상자는 네 점(회전 상자, 없으면 축 상자 모서리)을 탁자 좌표로 돌린 뒤 같은 식으로 잰다 —
    크기·기울기는 탁자 축 기준이라 사진이 θ 만큼 돌아가도 학습 때(거의 축 정렬)와 같은 값이 된다."""
    R = tz.R
    rot = feat == FEAT_ROT
    su, sv, _ = turn(0.5, 0.5, side)
    items = [(b, (pts[i] if pts else None), 0.0) for i, b in enumerate(boxes)]
    if rot and backs:
        items += [(b, p, 1.0) for b, p in backs]
    rows = []
    for (x, y, w, h), p, back in items:
        corners = p if p is not None else [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        q = tl.frame_pts(tz, corners)
        cx, cy = q.mean(0)
        if rot:
            pw, ph, tilt = obb.tile_geom(None, q)
        else:
            pw, ph, tilt = float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1])), 0.0
        u, v, swap = turn((cx + R) / (2 * R), (cy + R) / (2 * R), side)
        bw, bh = pw / (2 * R), ph / (2 * R)
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
        return self.predict_full(x)[:2]

    def predict_full(self, x: np.ndarray) -> tuple[list[str], list[float], list[list[str]] | None]:
        """[predict] + 씨드별 역할(ONNX 둘째 출력 seeds[S, 1, N, 6] 이 있을 때, 불확실성 신호 S4). 답(평균 확률 argmax)은 그대로."""
        if len(x) == 0:
            return [], [], None
        outs = self._session().run(None, {"x": x[None]})
        p = outs[0][0]
        seeds = [[ROLES[int(i)] for i in s[0].argmax(1)] for s in outs[1]] if len(outs) > 1 else None
        return [ROLES[int(i)] for i in p.argmax(1)], [round(float(v), 3) for v in p.max(1)], seeds


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


def read_roles(reader: lr.LocalReader, jpeg: bytes, side: str, grid: dict, mode: str = "hybrid", net: RoleNet | None = None,
               frame: tf.TableFrame | None = None) -> dict:
    """사진 전체 → 화료자 자리의 손패·울기(/v2/seat 모양) + 상자별 역할(roles: 원본 좌표 box, role, conf, in_seat).
    WALL_FRAME(axis|rot|rot2, 개선 계획 8b·8b-3b)이 켜져 있으면 기준점(frame = 앱이 준 것, 없으면 사진에서 잡음)으로 자른다 — 응답에 wall_frame·frame
    (또는 frame_error, 그때는 [grid] 로 지금처럼). 꺼져 있으면 frame 을 무시하고 지금과 같다."""
    with obb.session():
        out = _read_roles(reader, jpeg, side, grid, mode, net, frame)
        ok = out.pop("_inside_ok", None)
        _loose_outside(reader, jpeg, out, inside=ok if out.get("wall_frame") == "rot2" and out.get("cut") == "rot" else None)
        items, seeds = out.pop("_sig_items", None), out.pop("_role_seeds", None)
        if sig.ON and items is not None:  # 불확실성 신호(개선 계획 5단계): 답은 그대로, 값만 더한다
            if (s := out.get("separate_tile")) and s.get("from") in ("loose_outside", "loose_inside"):
                items.append(("sep", lr.decode_rgb(jpeg), tuple(s["box"]), "any", s["tile"], 0, 0))
            out["signals"] = sig.collect(reader, items, seeds)
            out["ms"]["signals"] = out["signals"]["ms"]
            out["ms"]["total"] += out["signals"]["ms"]
        return out


def _loose_outside(reader: lr.LocalReader, jpeg: bytes, out: dict, inside: set | None = None) -> None:
    """자리 크롭 밖의 떼어 둔 화료패(2026-11-05, metrics 표 1-70 143104): 크롭 안에서 떼어 둔 패를 못 찾았고 손패 + 울기 × 3 = 13 이면,
    크롭 밖 상자 중 역할이 loose 이고 화료자 손패에서(가장 가까운 손패 중심까지) 패 높이 [LOOSE_REACH] 배 안인 것이 딱 하나면
    그것을 원본 사진에서 읽어 떼어 둔 화료패로.
    inside(WALL_FRAME=rot2, 8b-3b 후보 2) = 후보가 될 수 있는 크롭 상자 번호(틀에 안 쓰였고 홀로 놓인 것 — _read_roles 의 _inside_ok) — 주어지면
    크롭 안의 loose 상자 중 그 번호인 것도 같은 조건(손패 가까이, 크롭 밖 것과 합쳐 딱 하나)으로 후보에 넣는다(넓어진 세운 크롭 안으로 들어온
    떼어 둔 쯔모패, 표 1-105 143104)."""
    if not out.get("framed") or out.get("separate_tile") or len(out["tiles"]) + 3 * len(out["melds"]) != 13:
        return
    hand = [x["box"] for x in out["roles"] if x["in_seat"] and x["role"] == "hand"]
    if not hand:
        return
    th = float(np.median([max(b[2], b[3]) for b in hand]))
    cen = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[3] / 2], float)
    near = [x for i, x in enumerate(out["roles"]) if (not x["in_seat"] or (inside is not None and i in inside)) and x["role"] == "loose"
            and min(np.linalg.norm(cen(x["box"]) - cen(h)) for h in hand) <= LOOSE_REACH * th]
    if len(near) != 1:
        return
    full = lr.decode_rgb(jpeg)  # _read_roles 가 디코딩한 것을 다시 쓴다
    out["separate_tile"] = {**lr.tile_json(reader.read_any2(full, tuple(near[0]["box"]))), "box": near[0]["box"],
                            "from": "loose_inside" if near[0]["in_seat"] else "loose_outside"}


def seat_grow_u(tz: tf.TableZones, side: str) -> float:
    """[GROW](앱 seatReadGrow, 1000 분율)를 u 로: 아래·위 자리는 사진 높이, 오른쪽·왼쪽은 폭의 GROW/1000 — 지금 크롭과 같은 픽셀 폭."""
    return GROW / 1000 * (tz.H if side in ("bottom", "top") else tz.W) / tz.frame.u


def _read_roles(reader: lr.LocalReader, jpeg: bytes, side: str, grid: dict, mode: str, net: RoleNet | None,
                frame: tf.TableFrame | None = None) -> dict:
    t0 = time.perf_counter()
    net = net or role_net()
    full = lr.decode_rgb(jpeg)  # 같은 사진을 받은 /v2/localdora 와 디코딩을 나눈다(app/shared.py)
    im = Image.fromarray(full)
    W, H = im.size
    t_dec = time.perf_counter()
    ctx = reader.detect_full(jpeg, full)  # 맥락용(작게 보이는 다른 자리 패) — 같은 사진이면 localdora·기준점 잡기와 한 번만(8단계 P1)
    # 탁자 기준점(WALL_FRAME, 개선 계획 8b): axis = 기준점에서 만든 같은 뜻의 격자로 지금 경로, rot = 구역표 회전 사각형을 세워 자름
    wf, tz, fmeta = tl.mode(), None, {}
    if wf:
        tz, fmeta = tl.frame_for(reader, jpeg, full, frame)
        if tz is not None and wf == "axis":
            g = tl.grid_from_frame(tz)
            if g is None:
                tz, fmeta["frame_error"] = None, "격자 선이 사진 밖"
            else:
                grid, fmeta["grid"] = g, g
    zc = None
    if wf in tl.ROT and tz is not None:  # rot2(8b-3b 후보 1): 읽기 크롭은 이웃 자리와 3u 겹친 seat_read, 구역(채점·역할)은 seat 그대로
        zc = tl.warp_rect(full, tz, "seat_read" if wf == "rot2" else "seat", side, grow_r0=seat_grow_u(tz, side))
        if zc is None:
            tz, fmeta["frame_error"] = None, "자리 구역이 사진 밖"
    if zc is not None:
        rgb = zc.rgb
    else:
        crop_im, (x0, y0, cw, ch) = crop_seat(im, grid, side)
        rgb = np.asarray(crop_im)
    faces = reader.keep_seat(rgb, reader.detect_seat(rgb))
    t_det = time.perf_counter()
    rot = ROT_CW[side]
    faces_full, pts = [], []
    for b in faces:
        p = _pts_of(rgb, b)
        if zc is not None:
            bb, q = zc.box_to_orig(b, p)
            faces_full.append(bb)
            pts.append(None if p is None else q)
            continue
        x, y, w, h = unrotate(b, rot, cw, ch)
        faces_full.append((x + x0, y + y0, w, h))
        pts.append(None if p is None else unrotate_pts(p, rot, cw, ch, x0, y0))
    # 크롭 상자 우선, 크롭과 겹치지 않는 전체 사진 상자를 맥락으로 더한다
    extra = [b for b in ctx if all(iou(b, f) < 0.3 for f in faces_full)]
    allb = faces_full + extra
    pts += [_pts_of(full, b) for b in extra]
    backs: list = []
    if net.feat == FEAT_ROT:  # 뒷면 상자(패산·다른 자리 손패·안깡 양끝): 크롭 것 우선, 겹치지 않는 전체 사진 것
        for b, rc in obb.backs_of(rgb):
            if zc is not None:
                backs.append(zc.box_to_orig(b, obb.corners(rc)))
                continue
            x, y, w, h = unrotate(b, rot, cw, ch)
            backs.append(((x + x0, y + y0, w, h), unrotate_pts(obb.corners(rc), rot, cw, ch, x0, y0)))
        backs += [(b, obb.corners(rc)) for b, rc in obb.backs_of(full) if all(iou(b, f) < 0.3 for f, _ in backs) and all(iou(b, f) < 0.3 for f in allb)]
    full_pred = getattr(net, "predict_full", None)
    x_tok = tokens_frame(allb, tz, side, net.feat, pts, backs) if zc is not None else tokens(allb, W, H, grid, side, net.feat, pts, backs)
    roles, confs, seeds = full_pred(x_tok) if full_pred else (*net.predict(x_tok), None)
    roles, confs = roles[: len(allb)], confs[: len(allb)]  # 뒷면 토큰의 역할은 버린다
    seeds = [s[: len(allb)] for s in seeds] if seeds else None
    seat_roles = roles[: len(faces)]
    t_role = time.perf_counter()
    # 손패 칸 폭: 지금 격자는 선 사이 변(x1~x2), 탁자 좌표는 같은 뜻의 a = −R ~ R(자리 크롭은 a = −R 부터)
    split = int(rgb.shape[1] * (min(1.0, zc.frac_a(tz.R)) if zc is not None else hand_frac(grid, side)))
    fr, added = frame_from_roles(rgb, faces, seat_roles, split, mode) if faces else (None, [])
    out = {"face_up": fr is not None or len(faces) >= lr.FACE_UP_MIN, "tiles": [], "separate_tile": None, "melds": [], "loose": [], "framed": fr is not None,
           "filled": len(added), "mode": mode, "roles": [{"box": [round(v) for v in b], "role": r, "conf": c, "in_seat": i < len(faces)} for i, (b, r, c) in enumerate(zip(allb, roles, confs))],
           "backs": len(backs), "feat": net.feat}
    if fr is not None:
        filled = set(map(tuple, added))

        def blank_or(b, read):
            return ("5z", lr.FILLED_CONF, None, None) if filled and (tuple(b) in filled or st.ink(rgb, b) < lr.BLANK) else read(b)

        # 패마다 (종류, 확신도, 분류기 2순위, 그 확신도) — 2순위는 답을 안 바꾸는 부가 정보(개선 계획 4단계 V3 후보용, 2026-11-05)
        hand = [blank_or(b, lambda x: reader.read2([lr.cut(rgb, x)])[0]) for b in fr.hand]
        sep = blank_or(fr.separate, lambda x: reader.read2([lr.cut(rgb, x)])[0]) if fr.separate else None
        melds = [[blank_or(b, lambda x: reader.read_meld_tile2(rgb, x)) for b in m] for m in fr.melds]
        block, got = False, []
        if fr.count == 13 and not fr.separate and not added:
            got = lr.strays(fr, faces, rgb)
            if len(got) == 1:
                sep = reader.read_turned2(rgb, got[0], fr.tw, fr.th)
            elif got:
                block = True
        out["tiles"] = [lr.tile_json(t) for t in hand]
        out["separate_tile"] = lr.tile_json(sep) if sep else None
        out["melds"] = [{"tiles": [t[0] for t in m], "sideways": [], "face_down": [], "conf": min(t[1] for t in m),
                         "alts": [t[2] for t in m], "conf2s": [t[3] for t in m]} for m in melds]
        out["block"] = block
        # 신호용(read_roles 가 꺼내 signals.collect 에): 답과 같은 크롭·방향. 白 채우기 패는 답 None
        is_fill = lambda b, t: t[2] is None and t[1] == lr.FILLED_CONF
        out["_sig_items"] = ([("hand", rgb, b, "hand", None if is_fill(b, t) else t[0], 0, 0) for b, t in zip(fr.hand, hand)]
                             + ([("sep", rgb, fr.separate, "hand", None if is_fill(fr.separate, sep) else sep[0], 0, 0)] if fr.separate and sep else [])
                             + ([("sep", rgb, got[0], "turned", sep[0], fr.tw, fr.th)] if not fr.separate and sep and len(got) == 1 else [])
                             + [(f"meld{i}", rgb, b, "meld", None if is_fill(b, t) else t[0], 0, 0) for i, m in enumerate(fr.melds) for b, t in zip(m, melds[i])])
        out["_role_seeds"] = seeds
        # rot2 크롭 안 떼어 둔 패 후보(8b-3b 후보 2): 틀(손패·떼어 둔 패·울기)에 안 쓴 크롭 상자 중 홀로 놓인 것 — 표시패 짝과 같은 간격(tw×0.5,
        # read_wall)으로 묶어 혼자인 것만(손패 뒤 탁자 위 표시패 짝 한 장이 손패 3배 거리 안에 걸리는 것을 막음, 개발 B·C·D k15·k26)
        used = {tuple(b) for b in [*fr.hand, *([fr.separate] if fr.separate else []), *(x for m in fr.melds for x in m)]}
        alone = {tuple(g[0]) for g in st.cluster([tuple(b) for b in faces], fr.tw * 0.5, fr.tw * 0.5) if len(g) == 1}
        out["_inside_ok"] = {i for i, b in enumerate(faces) if tuple(b) in alone and tuple(b) not in used}
    out["counts"] = dict(Counter(seat_roles))
    out["ms"] = {"decode": round((t_dec - t0) * 1000), "detect": round((t_det - t_dec) * 1000), "roles": round((t_role - t_det) * 1000), "read": lr._ms(t_role), "total": lr._ms(t0)}
    if wf:  # 기준점(축·회전 어느 쪽으로 잘랐는지는 wall_frame, 못 잡아 격자로 떨어졌으면 frame_error)
        out["ms"]["frame"] = fmeta.pop("frame_ms", 0)
        out.update(fmeta)
        out["cut"] = "rot" if zc is not None else "axis" if tz is not None else "grid"
    out["rss_mb"] = lr.rss_mb()
    return out
