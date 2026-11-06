"""탁자 기준점(TableFrame) + 패 단위 구역표 로더 — 서버 판(개선 계획 8b "할 일 3", 2026-11-06).

mahjong_calc `experiments/zone_points/table_frame.py`(8b-2, 커밋 70f57af)를 옮긴 것이다 — **로직 변경 없음**. 바뀐 것은 구역표 경로
(`app/assets/table_frame_zones.json` = 앱 `assets/table_frame/zones.json` 복사본)와, 예제 파일을 다시 만드는 `make_fixture`(실험 결과
파일에 기대므로 앱 저장소에만 둠)를 뺀 것뿐이다. 앱 `lib/table/table_frame.dart` 와 같은 기준점 + 사진 크기에 같은 구역 상자를 낸다
(예제 `tests/fixtures/table_frame.json` = 앱 `test/fixtures/table_frame.json` 복사본, 점 허용 오차 0.5px — tests/test_table_frame.py).
표준 라이브러리(json·math)만 쓴다. 호모그래피 없음(비슷변환).
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

ZONES = Path(__file__).resolve().parent / "assets" / "table_frame_zones.json"
SIDES = ("bottom", "right", "top", "left")
SOURCES = ("device", "wall", "manual")
TOL_PX = 0.5


# ---------------------------------------------------------------- 기준점

@dataclass(frozen=True)
class TableFrame:
    """가운데 기기 가운데 (cx, cy, 사진 픽셀) · 방향 theta(라디안) · u(픽셀, 패 한 장 짧은 변) · D(u, 기기 반폭).
    size = 이 기준점을 잰 사진 크기(가로, 세로) — 다른 크기 사진에 쓸 때 scaled_to 로 옮긴다.
    source = device(화료 시점 기기) / wall(패산이 올라온 첫 프레임) / manual(사용자가 찍음), at = 잰 시각(ISO 문자열, 없으면 None)."""

    cx: float
    cy: float
    theta: float
    u: float
    D: float
    size: tuple[float, float]
    source: str = "device"
    at: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source {self.source!r}")
        if not (self.u > 0 and self.D > 0 and self.size[0] > 0 and self.size[1] > 0):
            raise ValueError("u·D·size 는 양수")

    @staticmethod
    def from_json(j: dict) -> "TableFrame":
        return TableFrame(cx=float(j["cx"]), cy=float(j["cy"]), theta=float(j["theta"]), u=float(j["u"]), D=float(j["D"]),
                          size=(float(j["size"][0]), float(j["size"][1])), source=j.get("source", "device"), at=j.get("at"))

    def to_json(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "theta": self.theta, "u": self.u, "D": self.D, "size": list(self.size),
                "source": self.source, "at": self.at}

    def scaled_to(self, W: float, H: float) -> "TableFrame":
        """같은 카메라·같은 화각의 다른 크기 사진(예: 1020 미리보기 프레임 → 4080 사진). 비슷변환이라 가로 비 하나로 늘린다."""
        k = W / self.size[0]
        return replace(self, cx=self.cx * k, cy=self.cy * k, u=self.u * k, size=(float(W), float(H)))

    @staticmethod
    def from_wall(cx: float, cy: float, theta: float, back_short_px: float, wall_half_px: float, size, profile: dict,
                  at: str | None = None) -> "TableFrame":
        """패산 사각형(배패 때 첫 프레임)에서: u = 눕힌 뒷면 짧은 변 × u_per_back_short, D = (패산 반폭 / u) / wall_half_per_device."""
        u = back_short_px * profile["u_per_back_short"]
        return TableFrame(cx=cx, cy=cy, theta=theta, u=u, D=wall_half_px / u / profile["wall_half_per_device"],
                          size=(float(size[0]), float(size[1])), source="wall", at=at)

    def moved_from(self, other: "TableFrame", rules: dict) -> bool:
        """두 기준점이 거치 이동으로 볼 만큼 다른지(가운데 > move_center_u 또는 방향 > move_theta_deg, 방향은 90° 주기)."""
        o = other.scaled_to(*self.size)
        fc = rules["frame_check"]
        dc = math.hypot(self.cx - o.cx, self.cy - o.cy) / self.u
        return dc > fc["move_center_u"] or theta_diff_deg(self.theta, o.theta) > fc["move_theta_deg"]


def theta_diff_deg(a: float, b: float) -> float:
    """두 방향 차이(도, 90° 주기) 0~45."""
    return abs((math.degrees(a - b) + 45.0) % 90.0 - 45.0)


# ---------------------------------------------------------------- 구역표

@dataclass(frozen=True)
class ZoneTable:
    rules: dict
    profiles: dict
    default_profile: str

    @staticmethod
    def from_json(j: dict) -> "ZoneTable":
        if j.get("version") != 1:
            raise ValueError(f"구역표 version {j.get('version')!r}")
        for name, z in j["rules"]["zones"].items():
            if z["kind"] not in ("side", "center") or len(z["a"]) != 2 or len(z["r"]) != 2:
                raise ValueError(f"구역 {name}")
        return ZoneTable(rules=j["rules"], profiles=j["profiles"], default_profile=j["default_profile"])

    @staticmethod
    def load(path: Path | str = ZONES) -> "ZoneTable":
        return ZoneTable.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def profile(self, name: str | None = None) -> dict:
        return self.profiles[name or self.default_profile]

    @property
    def zone_names(self) -> list[str]:
        return list(self.rules["zones"])


@dataclass(frozen=True)
class RotRect:
    """회전 사각형. pts = 사진 픽셀 네 점((a0,r0) (a1,r0) (a1,r1) (a0,r1) — 그 자리 사람이 아래에 오게 세웠을 때 왼쪽 위부터 시계 방향),
    angle_deg = a 축이 사진에서 향하는 각. 비면(끝값이 엇갈림 — 사진이 그 구역까지 안 닿음) pts = ()."""

    pts: tuple[tuple[float, float], ...]
    angle_deg: float
    a: tuple[float, float]
    r: tuple[float, float]

    @property
    def empty(self) -> bool:
        return not self.pts

    def bbox(self, W: float, H: float) -> tuple[float, float, float, float] | None:
        """축 평행 외접 상자(x0, y0, x1, y1)를 사진 안으로 자른 것. 사진 밖이면 None."""
        if self.empty:
            return None
        xs, ys = [p[0] for p in self.pts], [p[1] for p in self.pts]
        x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
        x1, y1 = min(float(W), max(xs)), min(float(H), max(ys))
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def norm(self, W: float, H: float) -> list[int] | None:
        """앱 NormBox 와 같은 [ymin, xmin, ymax, xmax](0~1000, 반올림은 0.5 를 올림 — Dart round 와 같게)."""
        b = self.bbox(W, H)
        if b is None:
            return None
        x0, y0, x1, y1 = b
        return [_round(y0 / H * 1000), _round(x0 / W * 1000), _round(y1 / H * 1000), _round(x1 / W * 1000)]


def _round(v: float) -> int:
    return int(math.floor(v + 0.5))


class TableZones:
    """기준점 + 사진 크기 → 구역 상자. 기준점이 다른 크기 사진에서 잰 것이면 이 사진 크기로 옮겨 쓴다."""

    def __init__(self, table: ZoneTable, frame: TableFrame, W: float, H: float) -> None:
        self.table = table
        self.frame = frame if (frame.size[0], frame.size[1]) == (float(W), float(H)) else frame.scaled_to(W, H)
        self.W, self.H = float(W), float(H)
        f = self.frame
        self._c, self._s = math.cos(f.theta), math.sin(f.theta)
        self.D = f.D
        ch = table.rules["center_half"]
        self.R = ch.get("c", 0.0) + ch.get("D", 0.0) * self.D
        self.edge = max(max(abs(q) for q in self.to_frame(x, y)) for x, y in ((0, 0), (self.W, 0), (self.W, self.H), (0, self.H)))

    # 좌표 ------------------------------------------------------------
    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.frame.cx, y - self.frame.cy
        return ((dx * self._c + dy * self._s) / self.frame.u, (-dx * self._s + dy * self._c) / self.frame.u)

    def from_frame(self, qx: float, qy: float) -> tuple[float, float]:
        u = self.frame.u
        return (self.frame.cx + u * (qx * self._c - qy * self._s), self.frame.cy + u * (qx * self._s + qy * self._c))

    @staticmethod
    def side_to_frame(side: str, a: float, r: float) -> tuple[float, float]:
        return {"bottom": (a, r), "right": (r, -a), "top": (-a, -r), "left": (-r, a)}[side]

    def local(self, side: str, x: float, y: float) -> tuple[float, float]:
        """사진 픽셀 → 그 자리 기준 (a, r)(u)."""
        qx, qy = self.to_frame(x, y)
        return {"bottom": (qx, qy), "right": (-qy, qx), "top": (-qx, -qy), "left": (qy, -qx)}[side]

    def angle_deg(self, side: str) -> float:
        rot = {"bottom": 0.0, "right": -90.0, "top": 180.0, "left": 90.0}[side]
        return (math.degrees(self.frame.theta) + rot + 180.0) % 360.0 - 180.0

    def value(self, e: dict) -> float:
        if "edge" in e:
            return e["edge"] * self.edge
        return e.get("c", 0.0) + e.get("D", 0.0) * self.D + e.get("R", 0.0) * self.R

    # 구역 ------------------------------------------------------------
    def bounds(self, name: str, side: str = "bottom", except_: tuple[str, ...] | list[str] = ()) -> tuple[float, float, float, float]:
        """구역 끝값 (a0, a1, r0, r1)(u, 그 자리 기준). kind=center 는 side 무시(아래 자리 기준)."""
        z = self.table.rules["zones"][name]
        a0, a1 = (self.value(e) for e in z["a"])
        r0, r1 = (self.value(e) for e in z["r"])
        g = z.get("grow_u")
        if g:  # 표시패 탐색: except 가 아닌 자리 쪽으로 넓힘(아래 자리 기준 a = x 오른쪽, r = y 아래)
            a0 -= 0.0 if "left" in except_ else g
            a1 += 0.0 if "right" in except_ else g
            r0 -= 0.0 if "top" in except_ else g
            r1 += 0.0 if "bottom" in except_ else g
        return a0, a1, r0, r1

    def rect(self, name: str, side: str = "bottom", except_: tuple[str, ...] | list[str] = ()) -> RotRect:
        z = self.table.rules["zones"][name]
        if z["kind"] == "center":
            side = "bottom"
        a0, a1, r0, r1 = self.bounds(name, side, except_)
        if a1 <= a0 or r1 <= r0:
            return RotRect((), self.angle_deg(side), (a0, a1), (r0, r1))
        pts = tuple(self.from_frame(*self.side_to_frame(side, a, r)) for a, r in ((a0, r0), (a1, r0), (a1, r1), (a0, r1)))
        return RotRect(pts, self.angle_deg(side), (a0, a1), (r0, r1))

    def inside(self, name: str, side: str, x: float, y: float, except_=()) -> bool:
        """사진 점이 그 구역 안인지(끝 포함)."""
        if self.table.rules["zones"][name]["kind"] == "center":
            side = "bottom"
        a, r = self.local(side, x, y)
        a0, a1, r0, r1 = self.bounds(name, side, except_)
        return a0 <= a <= a1 and r0 <= r <= r1

    def zone_of(self, x: float, y: float) -> str:
        """바람개비 5칸 중 어디인지: 자리 이름 또는 'center'. 자리 칸은 아래쪽 끝값을 넘어야(a > a0 · r > r0) — 앱 GridZones 와 같은 경계."""
        for side in SIDES:
            a, r = self.local(side, x, y)
            a0, _, r0, _ = self.bounds("seat", side)
            if a > a0 and r > r0:
                return side
        return "center"

    # 앱 GridZones 와 같은 종류의 크롭 -------------------------------------------
    def seat(self, side: str) -> RotRect:
        return self.rect("seat", side)

    def hand(self, side: str) -> RotRect:
        return self.rect("hand", side)

    def corner(self, side: str) -> RotRect:
        return self.rect("corner", side)

    def river(self, side: str) -> RotRect:
        return self.rect("river", side)

    def stick_strip(self, side: str) -> RotRect:
        return self.rect("stick_strip", side)

    def dead_wall(self, side: str) -> RotRect:
        return self.rect("dead_wall", side)

    def center(self) -> RotRect:
        return self.rect("center")

    def device(self) -> RotRect:
        return self.rect("device")

    def sticks(self) -> RotRect:
        return self.rect("sticks")

    def dora_search(self, except_=()) -> RotRect:
        return self.rect("dora_search", except_=except_)

    def all_rects(self, except_=()) -> dict[str, RotRect]:
        """예제 파일용: 'seat.bottom' … 'center' 처럼 이름(.자리) → 상자."""
        out = {}
        for name, z in self.table.rules["zones"].items():
            if z["kind"] == "side":
                for s in SIDES:
                    out[f"{name}.{s}"] = self.rect(name, s)
            else:
                out[name] = self.rect(name, except_=except_)
        return out


# ---------------------------------------------------------------- 예제 파일

def _r(v: float, n: int = 3) -> float:
    return round(float(v), n)


def case_output(table: ZoneTable, frame: TableFrame, W: float, H: float, except_: list[str], points: list) -> dict:
    tz = TableZones(table, frame, W, H)
    rects = {}
    for k, rr in tz.all_rects(except_).items():
        rects[k] = None if rr.empty else {"pts": [[_r(x), _r(y)] for x, y in rr.pts], "angle_deg": _r(rr.angle_deg, 4),
                                          "bbox": [_r(v) for v in b] if (b := rr.bbox(W, H)) else None, "norm": rr.norm(W, H)}
    return {"R": _r(tz.R, 4), "edge": _r(tz.edge, 4), "rects": rects, "zone_of": [tz.zone_of(x, y) for x, y in points]}


def grid_points(W: float, H: float, nx: int = 9, ny: int = 7) -> list[list[float]]:
    return [[_r((i + 0.5) / nx * W, 1), _r((j + 0.5) / ny * H, 1)] for j in range(ny) for i in range(nx)]


def run_test(path: Path | str, zones: Path | str = ZONES) -> int:
    table = ZoneTable.load(zones)
    fx = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = 0
    n = 0
    for c in fx["cases"]:
        f = TableFrame.from_json(c["frame"])
        W, H = c["size"]
        got = case_output(table, f, W, H, c["except"], c["points"])
        exp = c["expect"]
        errs = []
        if abs(got["R"] - exp["R"]) > 1e-3 or abs(got["edge"] - exp["edge"]) > 1e-3:
            errs.append(f"R/edge {got['R']},{got['edge']} != {exp['R']},{exp['edge']}")
        if set(got["rects"]) != set(exp["rects"]):
            errs.append("구역 이름이 다름")
        for k, e in exp["rects"].items():
            g = got["rects"].get(k)
            n += 1
            if (g is None) != (e is None):
                errs.append(f"{k}: 빈 구역 여부")
                continue
            if e is None:
                continue
            if any(abs(a - b) > TOL_PX for p, q in zip(g["pts"], e["pts"]) for a, b in zip(p, q)):
                errs.append(f"{k}: pts {g['pts']} != {e['pts']}")
            if abs((g["angle_deg"] - e["angle_deg"] + 180) % 360 - 180) > 1e-3:
                errs.append(f"{k}: angle")
            if (g["bbox"] is None) != (e["bbox"] is None) or (e["bbox"] and any(abs(a - b) > TOL_PX for a, b in zip(g["bbox"], e["bbox"]))):
                errs.append(f"{k}: bbox {g['bbox']} != {e['bbox']}")
            if (g["norm"] is None) != (e["norm"] is None) or (e["norm"] and any(abs(a - b) > 1 for a, b in zip(g["norm"], e["norm"]))):
                errs.append(f"{k}: norm")
        if got["zone_of"] != exp["zone_of"]:
            errs.append("zone_of")
        if errs:
            bad += 1
            print("✗", c["name"], *errs[:5], sep="\n   ")
    for w in fx["from_wall"]:
        i = w["input"]
        f = TableFrame.from_wall(i["cx"], i["cy"], i["theta"], i["back_short_px"], i["wall_half_px"], i["size"], table.profile())
        if abs(f.u - w["expect"]["u"]) > 1e-3 or abs(f.D - w["expect"]["D"]) > 1e-3 or f.source != w["expect"]["source"]:
            bad += 1
            print("✗ from_wall", w["name"], f.u, f.D)
    for m in fx["moved"]:
        got = TableFrame.from_json(m["a"]).moved_from(TableFrame.from_json(m["b"]), table.rules)
        if got != m["expect"]:
            bad += 1
            print("✗ moved", m["name"], got)
    print(f"cases {len(fx['cases'])} (상자 {n}) · from_wall {len(fx['from_wall'])} · moved {len(fx['moved'])} — {'통과' if not bad else f'틀림 {bad}'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(run_test(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "table_frame.json"))
