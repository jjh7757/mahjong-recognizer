"""탁자 기준점 잡기 + 구역표로 자르기(개선 계획 8b "할 일 3", 서버 실험 옵션 `WALL_FRAME`, 2026-11-06).

기준점 찾기는 mahjong_calc `experiments/zone_points/wall_frame.py`(8b-1, metrics 표 1-101)의 `orient`·`find_device`·`lying`·`find_wall`·
`estimate` 를 그대로 옮긴 것이다(상수·로직 같음, 색 안 씀). 입력은 사진 전체를 자리 판독 검출기(회전 상자 8회차)로 한 번 검출한 앞면·뒷면
회전 상자 — 새로 검출하지 않고 `LocalReader.detect_full` 이 localroles·localdora 와 나눠 쓰는 것(app/shared.py, 8단계 P1).
  화료 사진 = 가운데 기기(앞면 패 무리 안 빈 정사각형)   배패 때 = 패산이 올라온 첫 프레임(눕힌 뒷면 사각형)   앱이 준 것 = 요청 frame=
결과는 `table_frame.TableFrame`(가운데·θ·u·D·사진 크기·source). 구역은 `table_frame.TableZones` + 구역표 JSON(앱과 같은 파일).

자르기 두 가지(`WALL_FRAME`):
  axis = 기준점에서 지금 격자와 같은 뜻의 축 평행 x1·x2·y1·y2(가운데 칸 외접 상자, 1000 분율)와 리치봉 상자를 만들어 기존 경로에 그대로 —
         역할 모델·분류기 입력 분포 무변경([grid_from_frame]).
  rot  = 구역표의 회전 사각형을 그 자리 사람이 아래에 오게 세워 자른다([warp_zone], 축 평행 외접 상자 아님). 상자는 [ZoneCrop] 으로 원본 좌표로.
  rot2 = rot 의 크롭 정의 손실 세 가지를 고친 후보(8b-3b, mahjong_calc metrics 표 1-105 '다음 후보'): ① 자리 읽기 크롭은 구역표 `seat_read`
         (seat 를 이웃 자리 쪽으로 3u 겹침, 채점·역할 구역은 seat 그대로) ② 크롭 안 `loose` 역할도 떼어 둔 화료패 후보(roles._loose_near)
         ③ 표시패는 세워 자르지 않고 사진 전체 검출(full_raw 공유) — 패산 표시패 후보만 dora_search ∪ dead_wall 로 거르고 짝은 지금처럼 전체에서.
  hybrid = 요청마다 고른다(8b-3c, mahjong_calc metrics 표 1-108): 앱이 이 탁자 격자를 가졌으면(grid_src=saved|manual|felt) 지금 경로
         (격자 그대로, WALL_FRAME 없음과 같은 답 = wall_frame "saved", 기준점은 잡아 응답에만), 기본값 격자·자동 맞추기 실패(none|default)
         이거나 격자 질의가 없으면 rot2("rot2"). 기준점을 못 잡으면 지금 경로("none"). 문턱·색 없음. 측정용 강제값 WALL_FRAME_GRID_SRC.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from . import table_frame as tf

MODES = ("axis", "rot", "rot2", "hybrid")
ROT = ("rot", "rot2")  # 세워 자르는 모드(강 끝·자리 크롭)


def mode() -> str | None:
    """WALL_FRAME 환경 변수: axis | rot | rot2 | hybrid(1 은 axis 로 친다), 없거나 그 밖이면 None(기존 격자 경로)."""
    v = os.environ.get("WALL_FRAME", "").strip().lower()
    return "axis" if v == "1" else v if v in MODES else None


_TABLE: tf.ZoneTable | None = None


def table() -> tf.ZoneTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = tf.ZoneTable.load()
    return _TABLE


# ---------------------------------------------------------------- 기준점 찾기(wall_frame.py 그대로)

ROW_BAND = (0.2, 1.4)  # 강 첫 줄 패 가운데가 기기 변에서 떨어진 거리(u)


def rects_array(res: list) -> np.ndarray:
    """[(축 상자, 회전 상자 ((cx, cy), (w, h), 각도°))] 또는 [축 상자] → (N, 5) cx, cy, w, h, 각도°. 축 상자만이면 각도 0."""
    rows = []
    for r in res:
        if len(r) == 2 and r[1] is not None:
            (cx, cy), (w, h), a = r[1]
            rows.append((cx, cy, w, h, a))
        else:
            x, y, w, h = r[0] if len(r) == 2 else r
            rows.append((x + w / 2, y + h / 2, w, h, 0.0))
    return np.array(rows, float).reshape(-1, 5)


def rot_frame(theta: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([math.cos(theta), math.sin(theta)]), np.array([-math.sin(theta), math.cos(theta)])


def to_frame(pts: np.ndarray, c: np.ndarray, theta: float, u: float) -> np.ndarray:
    ex, ey = rot_frame(theta)
    d = np.asarray(pts, float) - c
    return np.stack([d @ ex, d @ ey], 1) / u


def from_frame(q: np.ndarray, c: np.ndarray, theta: float, u: float) -> np.ndarray:
    ex, ey = rot_frame(theta)
    q = np.asarray(q, float) * u
    return c + q[:, :1] * ex + q[:, 1:2] * ey


def orient(rects: np.ndarray) -> float:
    """회전 상자 각도(90° 주기)의 평균, 라디안 (-45°, 45°]."""
    a = np.radians(rects[:, 4]) * 4
    w = rects[:, 2] * rects[:, 3]
    return float(math.atan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum()) / 4)


def side_coords(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """바람개비 자리(0 아래·1 오른쪽·2 위·3 왼쪽)와 그 자리 사람 기준 (a = 그 사람 오른쪽, r = 바깥)."""
    x, y = q[:, 0], q[:, 1]
    side = np.where(np.abs(y) >= np.abs(x), np.where(y >= 0, 0, 2), np.where(x >= 0, 1, 3))
    a = np.choose(side, [x, -y, -x, y])
    r = np.choose(side, [y, x, -y, -x])
    return side, a, r


def find_device(faces: np.ndarray, theta: float, u: float) -> dict | None:
    """앞면 패 무리 안 빈 정사각형: 네 변에 강 첫 줄이 붙고 안에는 패가 없는 곳."""
    if len(faces) < 6:
        return None
    c0 = np.median(faces[:, :2], 0)
    q = to_frame(faces[:, :2], c0, theta, u)
    best = None
    grid = np.arange(-6, 6.01, 0.25)
    DX = q[None, :, 0] - grid[:, None]  # (cx, 패)
    DY = q[None, :, 1] - grid[:, None]  # (cy, 패)
    for D in np.arange(2.0, 4.51, 0.25):
        dx, dy = DX[:, None, :], DY[None, :, :]  # (cx, cy, 패)
        rho = np.maximum(np.abs(dx), np.abs(dy))
        n_in = (rho < D - 0.15).sum(-1)
        s = np.zeros_like(n_in)
        sides = np.zeros_like(n_in)
        for t, a in ((dy, dx), (dx, -dy), (-dy, -dx), (-dx, dy)):
            k = ((t >= D + ROW_BAND[0]) & (t <= D + ROW_BAND[1]) & (np.abs(a) <= D + 1)).sum(-1)
            s += np.minimum(k, 6)
            sides += k > 0
        sc = np.where(n_in > 2, -99, s + sides - 3 * n_in)
        i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        if best is None or sc[i, j] > best[0]:
            best = (int(sc[i, j]), float(grid[i]), float(grid[j]), float(D), int(sides[i, j]))
    score, cx, cy, D, sides = best
    if sides < 2 or score < 8:
        return None
    # 변마다 첫 줄 안쪽 끝으로 가운데·반폭을 다시 잡는다.
    dx, dy = q[:, 0] - cx, q[:, 1] - cy
    half_r = []  # 패의 바깥쪽 방향 반길이(u): 회전 상자의 두 축 중 r 방향 투영
    ex, ey = rot_frame(theta)
    for f in faces:
        ax = np.array([math.cos(math.radians(f[4])), math.sin(math.radians(f[4]))])
        ay = np.array([-ax[1], ax[0]])
        half_r.append((abs(ax @ ex) * f[2] + abs(ay @ ex) * f[3], abs(ax @ ey) * f[2] + abs(ay @ ey) * f[3]))
    half_r = np.array(half_r) / 2 / u  # [:,0] x 방향 반길이, [:,1] y 방향 반길이
    edge = {}
    for k, (t, a, h) in enumerate(((dy, dx, half_r[:, 1]), (dx, -dy, half_r[:, 0]), (-dy, -dx, half_r[:, 1]), (-dx, dy, half_r[:, 0]))):
        m = (t >= D + ROW_BAND[0] - 0.3) & (t <= D + ROW_BAND[1] + 0.3) & (np.abs(a) <= D + 1)
        if m.sum() >= 1:
            edge[k] = float(np.median((t - h)[m]))
    sx, sy, Ds = cx, cy, []
    if 0 in edge and 2 in edge:
        sy, Ds = cy + (edge[0] - edge[2]) / 2, Ds + [(edge[0] + edge[2]) / 2]
    if 1 in edge and 3 in edge:
        sx, Ds = cx + (edge[1] - edge[3]) / 2, Ds + [(edge[1] + edge[3]) / 2]
    Dn = float(np.mean(Ds)) if Ds else float(np.mean(list(edge.values())))
    c = from_frame(np.array([[sx, sy]]), c0, theta, u)[0]
    return {"c": c, "D": Dn, "score": int(score), "sides": int(sides), "edges": len(edge)}


def lying(backs: np.ndarray, theta: float, c0: np.ndarray, u: float) -> np.ndarray:
    """눕힌 뒷면(패산 더미)만: 긴 축이 가운데 쪽(바깥 방향)을 향하고 긴 변/짧은 변 ≥ 1.25. 세운 손패 뒷면은 긴 축이 줄을 따라 눕는다."""
    if not len(backs):
        return backs
    ex, ey = rot_frame(theta)
    q = to_frame(backs[:, :2], c0, theta, u)
    keep = []
    for b, (x, y) in zip(backs, q):
        a = math.radians(b[4])
        ax = np.array([math.cos(a), math.sin(a)]) if b[2] >= b[3] else np.array([-math.sin(a), math.cos(a)])
        radial = ey if abs(y) >= abs(x) else ex
        keep.append(max(b[2], b[3]) / max(min(b[2], b[3]), 1) >= 1.25 and abs(ax @ radial) > 0.7)
    return backs[np.array(keep, bool)]


def find_wall(backs: np.ndarray, theta: float, u: float, c0: np.ndarray, W_range: tuple[float, float], search: float) -> dict | None:
    """패산 사각형: 뒷면 상자 가운데가 둘레 |ρ − W| ≤ 0.6u 띠에 가장 많이 드는 (가운데, W). 변마다 줄 수도 센다."""
    backs = lying(backs, theta, c0, u)
    if len(backs) < 6:
        return None
    q = to_frame(backs[:, :2], c0, theta, u)
    best = None
    grid = np.arange(-search, search + 0.01, 0.25)
    DX = np.abs(q[None, :, 0] - grid[:, None])[:, None, :]
    DY = np.abs(q[None, :, 1] - grid[:, None])[None, :, :]
    rho = np.maximum(DX, DY)  # (cx, cy, 패)
    for W in np.arange(W_range[0], W_range[1] + 0.01, 0.25):
        cnt = (np.abs(rho - W) <= 0.6).sum(-1)
        i, j = np.unravel_index(int(np.argmax(cnt)), cnt.shape)
        if best is None or cnt[i, j] > best[0]:
            best = (int(cnt[i, j]), float(grid[i]), float(grid[j]), float(W))
    n, cx, cy, W = best
    side, a, r = side_coords(q - [cx, cy])
    on = np.abs(r - W) <= 0.6
    per = [int((on & (side == s)).sum()) for s in range(4)]
    rows = sum(p >= 4 for p in per)
    if n < 8 or rows < 1:
        return None
    # 맞은편 두 변이 다 있으면 그 축의 가운데는 정해진다, 한 변뿐이면 W(가정)에 기댄다.
    fixed = {"y": per[0] >= 4 and per[2] >= 4, "x": per[1] >= 4 and per[3] >= 4}
    c = from_frame(np.array([[cx, cy]]), c0, theta, u)[0]
    return {"c": c, "W": float(W), "n": n, "per_side": per, "rows": rows, "fixed": fixed}


def estimate(faces: np.ndarray, backs: np.ndarray, profile: dict) -> dict:
    """사진 한 장의 기준점(wall_frame.estimate 와 같음, 기종 값은 구역표 프로필에서). u·θ 는 상자에서, 가운데는 기기(앞면 빈 곳) → 없으면 패산."""
    faces = np.asarray(faces, float).reshape(-1, 5)
    backs = np.asarray(backs, float).reshape(-1, 5)
    wall_per_d, u_back = profile["wall_half_per_device"], profile["u_per_back_short"]
    allr = np.concatenate([faces, backs]) if len(faces) + len(backs) else faces
    if len(allr) < 6:
        return {"ok": False, "why": "상자 부족"}
    theta = orient(allr)
    if len(faces) >= 8:
        u, u_src = float(np.median(np.minimum(faces[:, 2], faces[:, 3]))), "faces"
    else:
        u, u_src = float(np.median(np.minimum(backs[:, 2], backs[:, 3]))) * u_back, "backs"
    dev = find_device(faces, theta, u)
    out = {"ok": True, "theta": theta, "u": u, "u_src": u_src, "device": None, "wall": None}
    if dev:
        out["device"] = {"c": dev["c"].tolist(), "D": dev["D"], "score": dev["score"], "sides": dev["sides"], "edges": dev["edges"]}
        W0 = wall_per_d * dev["D"]
        wall = find_wall(backs, theta, u, dev["c"], (W0 - 1.5, W0 + 1.5), 2.0)
    else:
        wall = find_wall(backs, theta, u, np.median(backs[:, :2], 0) if len(backs) else np.zeros(2), (6.0, 15.0), 4.0)
    if wall:
        out["wall"] = {"c": wall["c"].tolist(), "W": wall["W"], "n": wall["n"], "per_side": wall["per_side"], "rows": wall["rows"], "fixed": wall["fixed"]}
    if dev:
        out["c"], out["D"], out["src"] = dev["c"].tolist(), dev["D"], "device"
    elif wall:
        out["c"], out["D"], out["src"] = wall["c"].tolist(), wall["W"] / wall_per_d, "wall"
    else:
        out.update(ok=False, why="기기·패산 못 찾음")
    return out


def locate(faces_res: list, backs_res: list, size: tuple[int, int], profile: dict | None = None) -> tuple[tf.TableFrame | None, dict]:
    """검출 결과(앞면·뒷면 [(축 상자, 회전 상자)]) → (기준점, 잡은 내용 요약). 못 잡으면 (None, {"error": 까닭}).
    가운데가 사진 밖이면 못 잡은 것으로 친다(가운데 칸이 사진 끝에 걸린 장면은 기존 격자 경로가 낫다)."""
    prof = profile or table().profile()
    e = estimate(rects_array(faces_res), rects_array(backs_res), prof)
    if not e.get("ok"):
        return None, {"error": e.get("why", "못 찾음")}
    W, H = size
    cx, cy = e["c"]
    if not (0 <= cx < W and 0 <= cy < H):
        return None, {"error": "가운데가 사진 밖"}
    fr = tf.TableFrame(cx=float(cx), cy=float(cy), theta=float(e["theta"]), u=float(e["u"]), D=float(e["D"]),
                       size=(float(W), float(H)), source="device" if e["src"] == "device" else "wall")
    info = {"source": fr.source, "u_src": e["u_src"]}
    if e["device"]:
        info["device"] = {k: e["device"][k] for k in ("score", "sides", "edges")}
    if e["wall"]:
        info["wall"] = {"n": e["wall"]["n"], "rows": e["wall"]["rows"], "W": round(e["wall"]["W"], 2)}
    return fr, info


def frame_for(reader, jpeg: bytes, rgb: np.ndarray, given: tf.TableFrame | None = None) -> tuple[tf.TableZones | None, dict]:
    """요청 하나의 기준점: given(앱이 준 frame=)이 있으면 그대로, 없으면 사진 전체 검출(reader.full_raw — 같은 사진이면 나눠 씀)에서 잡는다.
    (구역, 응답에 더할 값 {wall_frame, frame | frame_error, frame_info, ms}) — 못 잡으면 구역 None(부르는 쪽이 기존 격자 경로로)."""
    import time

    t0 = time.perf_counter()
    H, W = rgb.shape[:2]
    meta: dict = {"wall_frame": mode()}
    if given is not None:
        fr, info = given, {"source": given.source, "given": True}
    else:
        res, backs, _, _ = reader.full_raw(jpeg, rgb)
        fr, info = locate(res, backs, (W, H))
    tz = None
    if fr is None:
        meta["frame_error"] = info["error"]
    else:
        tz = tf.TableZones(table(), fr, W, H)
        meta["frame"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tz.frame.to_json().items()}
        meta["frame_info"] = info
    meta["frame_ms"] = round((time.perf_counter() - t0) * 1000)
    return tz, meta


GRID_HAVE = ("saved", "manual", "felt")  # 앱이 이 탁자 격자를 가짐(저장·손으로 맞춤·펠트 자동 맞추기 성공) → 지금 경로
GRID_NONE = ("none", "default")  # 기본값 격자·자동 맞추기 실패 → rot2
GRID_SRCS = GRID_HAVE + GRID_NONE


def grid_src_env() -> str | None:
    """WALL_FRAME_GRID_SRC 환경 변수(측정용 강제값 — 앱이 아직 grid_src 를 안 보냄, 8b-3c). 요청에 grid_src 가 오면 그쪽이 먼저."""
    v = os.environ.get("WALL_FRAME_GRID_SRC", "").strip().lower()
    return v if v in GRID_SRCS else None


def grid_check(tz: tf.TableZones, grid: dict) -> dict:
    """앱 격자와 기준점의 차이(진단만, 경로를 정하지 않음 — 8b-3c 측정 전 확인에서 저장 격자 가운데 칸 중심은 기기 중심에서 거치마다 1~3u
    떨어져 있었고 사진 한 장 θ 는 0.3~4.2° 흔들렸다, mahjong_calc metrics 표 1-108): 격자 가운데 칸 중심 ↔ 기준점 가운데 거리(u, 기준점의 u)·
    격자 중심의 기준점 좌표(dx·dy, u)·기준점 θ(90° 주기, 0° 와의 차, 도)."""
    px = (grid["x1"] + grid["x2"]) / 2 / 1000 * tz.W
    py = (grid["y1"] + grid["y2"]) / 2 / 1000 * tz.H
    qx, qy = tz.to_frame(px, py)
    return {"dist_u": round(math.hypot(qx, qy), 3), "dx_u": round(qx, 3), "dy_u": round(qy, 3),
            "theta_deg": round(tf.theta_diff_deg(tz.frame.theta, 0.0), 3)}


def route(reader, jpeg: bytes, rgb: np.ndarray, given: tf.TableFrame | None = None, grid: dict | None = None,
          grid_src: str | None = None) -> tuple[str | None, tf.TableZones | None, dict]:
    """요청 하나가 실제로 탈 경로 (모드, 구역, 응답에 더할 값). 모드 None = 지금 격자 경로(구역도 None).
    WALL_FRAME 이 axis|rot|rot2 면 그 모드 + [frame_for] 그대로.
    hybrid(8b-3c, 문턱 없음)는 **앱이 격자를 가졌는지**로 고른다 — 격자 출처 src = 요청 grid_src, 없으면 WALL_FRAME_GRID_SRC, 그것도 없으면
    격자 질의가 있으면 saved·없으면 none 으로 친다. src 가 saved|manual|felt 면 지금 경로(wall_frame "saved", 기준점은 잡아 응답에만),
    none|default 면 rot2(wall_frame "rot2"). 기준점을 못 잡으면 어느 쪽이든 지금 경로(rot2 로 가려던 것은 wall_frame "none").
    응답 wall_frame_src = [출처, 어디서(query|env|grid|no_grid)]·wall_frame_check(격자가 있으면 [grid_check], 진단만)."""
    m = mode()
    if m is None:
        return None, None, {}
    tz, meta = frame_for(reader, jpeg, rgb, given)
    if m != "hybrid":
        return m, tz, meta
    env = grid_src_env()
    if grid_src:
        src, why = grid_src, "query"
    elif env:
        src, why = env, "env"
    else:
        src, why = ("saved", "grid") if grid else ("none", "no_grid")
    meta.update(wall_frame_mode="hybrid", wall_frame_src=[src, why])
    if tz is not None and grid:
        meta["wall_frame_check"] = grid_check(tz, grid)
    if src in GRID_HAVE:
        meta["wall_frame"] = "saved"
        return None, None, meta
    if tz is None:
        meta["wall_frame"] = "none"
        return None, None, meta
    meta["wall_frame"] = "rot2"
    return "rot2", tz, meta


def parse_frame(s: str | dict | None) -> tf.TableFrame | None:
    """요청 frame=(TableFrame.to_json 의 JSON 문자열) → 기준점. 없으면 None, 틀리면 ValueError."""
    if s is None or s == "":
        return None
    import json

    j = json.loads(s) if isinstance(s, str) else s
    try:
        return tf.TableFrame.from_json(j)
    except (KeyError, TypeError) as e:
        raise ValueError(f"frame: {e}") from e


# ---------------------------------------------------------------- axis: 지금 격자와 같은 뜻의 선

def grid_from_frame(tz: tf.TableZones) -> dict | None:
    """기준점 → 지금 격자(1000 분율 x1·x2·y1·y2 = 가운데 칸 외접 상자, sticks = 리치봉 상자 외접 [ymin, xmin, ymax, xmax]).
    선이 사진 안(1~999)에 안 들면 None(기존 격자 경로로)."""
    c = tz.center().norm(tz.W, tz.H)
    if c is None:
        return None
    y1, x1, y2, x2 = c
    if not (1 <= x1 < x2 <= 999 and 1 <= y1 < y2 <= 999):
        return None
    g = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
    s = tz.sticks().norm(tz.W, tz.H)
    if s is not None:
        g["sticks"] = s
    return g


# ---------------------------------------------------------------- rot: 회전 사각형으로 세워 자르기

class ZoneCrop:
    """회전 사각형을 세워 자른 그림 + 좌표 변환. rgb = 그 자리 사람이 아래에 선 크롭((a0, r0) 이 왼쪽 위, 픽셀 크기 그대로),
    M = 원본 → 크롭 2×3, Minv = 크롭 → 원본. a·r = 실제로 자른 끝값(u, 사진 밖은 잘라 냄)."""

    def __init__(self, rgb: np.ndarray, M: np.ndarray, a: tuple[float, float], r: tuple[float, float]):
        self.rgb, self.M, self.Minv = rgb, M, cv2.invertAffineTransform(M)
        self.a, self.r = a, r

    def to_orig(self, pts) -> np.ndarray:
        p = np.asarray(pts, float).reshape(-1, 2)
        return p @ self.Minv[:, :2].T + self.Minv[:, 2]

    def to_crop(self, pts) -> np.ndarray:
        p = np.asarray(pts, float).reshape(-1, 2)
        return p @ self.M[:, :2].T + self.M[:, 2]

    def box_to_orig(self, b, pts=None) -> tuple[tuple[float, float, float, float], np.ndarray]:
        """크롭 상자(축 상자 x, y, w, h · 있으면 회전 상자 네 점) → (원본 축 상자(외접), 원본 네 점)."""
        if pts is None:
            x, y, w, h = b
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        q = self.to_orig(pts)
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        return (float(x0), float(y0), float(x1 - x0), float(y1 - y0)), q

    def frac_a(self, a: float) -> float:
        """자리 기준 a(u) → 크롭 가로 비율(0 왼쪽 ~ 1 오른쪽)."""
        return (a - self.a[0]) / max(self.a[1] - self.a[0], 1e-9)


def photo_bounds(tz: tf.TableZones, side: str) -> tuple[float, float, float, float]:
    """사진 네 모서리의 그 자리 기준 (a, r) 범위 (a_min, a_max, r_min, r_max)."""
    q = [tz.local(side, x, y) for x, y in ((0, 0), (tz.W, 0), (tz.W, tz.H), (0, tz.H))]
    return min(p[0] for p in q), max(p[0] for p in q), min(p[1] for p in q), max(p[1] for p in q)


def warp_zone(rgb: np.ndarray, tz: tf.TableZones, side: str, a: tuple[float, float], r: tuple[float, float], clip: bool = True) -> ZoneCrop | None:
    """그 자리 기준 사각형 a0~a1 × r0~r1(u)을 세워 자른다: (a0, r0) → 크롭 (0, 0), a 가 오른쪽, r 이 아래(그 사람 쪽). 픽셀 크기는 원본 그대로.
    clip = 사진이 안 닿는 쪽 끝을 사진 모서리 범위로 줄인다(회색으로 채운 빈 곳을 검출기에 덜 넣게). 비면 None."""
    a0, a1 = a
    r0, r1 = r
    if clip:
        pa0, pa1, pr0, pr1 = photo_bounds(tz, side)
        a0, a1, r0, r1 = max(a0, pa0), min(a1, pa1), max(r0, pr0), min(r1, pr1)
    if a1 - a0 <= 0.5 or r1 - r0 <= 0.5:
        return None
    u = tz.frame.u
    w, h = int(round((a1 - a0) * u)), int(round((r1 - r0) * u))
    src = np.float32([tz.from_frame(*tz.side_to_frame(side, aa, rr)) for aa, rr in ((a0, r0), (a1, r0), (a0, r1))])
    dst = np.float32([(0, 0), (w, 0), (0, h)])
    M = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(np.ascontiguousarray(rgb), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return ZoneCrop(out, M, (a0, a1), (r0, r1))


def warp_rect(rgb: np.ndarray, tz: tf.TableZones, name: str, side: str = "bottom", grow_r0: float = 0.0, clip: bool = True,
              except_=()) -> ZoneCrop | None:
    """구역표 구역 이름 → 세운 크롭. grow_r0 = 가운데 쪽 끝을 이만큼(u) 더 넓힘(자리 판독 seatReadGrow)."""
    if tz.table.rules["zones"][name]["kind"] == "center":
        side = "bottom"
    a0, a1, r0, r1 = tz.bounds(name, side, except_)
    return warp_zone(rgb, tz, side, (a0, a1), (r0 - grow_r0, r1), clip)


def dora_reach(tz: tf.TableZones) -> float:
    """표시패를 볼 정사각형 반폭(u): 왕패 칸 바깥 끝과 표시패 탐색 칸(넓힌 것) 중 큰 쪽."""
    a0, a1, r0, r1 = tz.bounds("dora_search")
    return max(tz.bounds("dead_wall", "bottom")[3], -a0, a1, -r0, r1)


def in_dora_zone(tz: tf.TableZones, box) -> bool:
    """원본 축 상자 가운데가 표시패 탐색 칸 ∪ 왕패 칸(네 변) 안인지(구역표, 검토 ① — 크롭으로 손패를 가르지 않고 후보만 거른다)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return tz.inside("dora_search", "bottom", cx, cy) or any(tz.inside("dead_wall", s, cx, cy) for s in tf.SIDES)


def frame_pts(tz: tf.TableZones, pts) -> np.ndarray:
    """사진 점들 → 탁자 좌표(u, 아래 자리 기준 x·y)."""
    return np.array([tz.to_frame(float(x), float(y)) for x, y in np.asarray(pts, float).reshape(-1, 2)], float)
