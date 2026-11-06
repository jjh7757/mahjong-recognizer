"""WALL_FRAME(탁자 기준점, 개선 계획 8b "할 일 3") 사진 몇 장 단위 확인 — 서버 코드를 프로세스 안에서 불러(HTTP 없음, ONNX 2 스레드)
같은 사진을 WALL_FRAME 없음 / axis / rot 세 가지로 /v2/localroles(read_roles hybrid)·/v2/localdora(read_wall) 처럼 읽는다.

사진·격자·정답은 mahjong_calc 저장소(MAHJONG_CALC_REPO, 기본 본 체크아웃)에서 읽기만 한다: 개발 = experiments/speed/seat_grow.scenes()
(저장 격자), 테스트·색 바꾼 = field_bench.SCENES 장면 격자(펴기 없이) + rectify/recheck_test TRUTH(자리 13장). 받은 사진은 서버처럼
prepare_image(4080). 기준점은 8b-1 결과(experiments/zone_points/results/wall_frame.json, 있으면)와 가운데 차이(u)도 적는다.
시간은 모드마다 디코딩·검출 공유 캐시를 비우고 잰다(한 장 첫 요청과 같게), localroles 다음 localdora 는 같은 사진이라 공유가 걸린다.

  python -X utf8 scripts/wall_frame_check.py [--out 로그.json] [--reps 3] [키 ...]   (--reps: 모드를 번갈아 여러 번, 시간은 중앙값)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2  # noqa: E402

cv2.setNumThreads(2)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP = Path(os.environ.get("MAHJONG_CALC_REPO", "C:/Users/jjh7757/dev/mahjong_calc"))
sys.path.insert(0, str(APP / "experiments" / "speed"))

from app import local_read as lr  # noqa: E402
from app import roles  # noqa: E402
from app import table_locate as tl  # noqa: E402
from app.recognizer import prepare_image  # noqa: E402

FIELD = APP / "experiments" / "field_20261024" / "captures"
RECOLOR = APP / "experiments" / "field_20261024" / "captures_recolor"
DEFAULT = ["k25", "20261031_133826", "20261031_171245", "20261024_122922", "20261024_140918", "recolor/20261024_122922", "recolor/20261024_133011"]
MODES = ["", "axis", "rot"]


def field_consts() -> tuple[dict, dict]:
    src = (APP / "experiments" / "tile_classifier" / "server_bench" / "field_bench.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(src[src.index("STANDARD = dict"):src.index("\n\n\ndef truth")], ns)  # noqa: S102  상수 블록만
    grids = {f"20261024_{ph}": (g, sc) for sc, (photos, g) in ns["SCENES"].items() for ph in photos}
    t = (APP / "experiments" / "rectify" / "recheck_test.py").read_text(encoding="utf-8")
    block = t[t.index("TRUTH = {"):t.index("}\n", t.index("TRUTH = {")) + 1]
    tr = eval(block.split("=", 1)[1])  # noqa: S307  상수 dict 리터럴
    truth = {k: dict(side=v[0], tiles=v[2].split()[:-1] + [x for m in v[3] for x in re.sub(r"[\[\]()]", "", m).split()]) for k, v in tr.items()}
    return grids, truth


def items(keys: list[str]) -> list[dict]:
    import seat_grow as sg

    dev = {s["name"]: s for s in sg.scenes()}
    fgrid, ftruth = field_consts()
    out = []
    for k in keys:
        if k in dev:
            s = dev[k]
            tiles = (s["hand"] or []) + [x for m in (s["melds"] or []) for x in m.split()]
            lab = Path(s["path"]).parent / "label.json"
            dora = json.loads(lab.read_text(encoding="utf-8")).get("dora_indicators") if lab.exists() else None
            out.append(dict(key=k, set="dev", path=Path(s["path"]), side=s["side"], grid=s["grid"], tiles=tiles, dora=dora))
            continue
        base = k.split("/")[-1]
        folder = (RECOLOR if k.startswith("recolor/") else FIELD) / base
        g, sc = fgrid[base]
        t = ftruth.get(sc)
        lab = FIELD / base / "label.json"
        out.append(dict(key=k, set="recolor" if k.startswith("recolor/") else "test", path=folder / "original.jpg", side=t["side"] if t else None,
                        grid=g, tiles=t["tiles"] if t else None,
                        dora=json.loads(lab.read_text(encoding="utf-8")).get("dora_indicators") if lab.exists() else None))
    return out


def hit(got: list[str], want: list[str] | None) -> int | None:
    return None if want is None else sum((Counter(got) & Counter(want)).values())


def answer(o: dict) -> list[str]:
    return [t["tile"] for t in o["tiles"]] + ([o["separate_tile"]["tile"]] if o.get("separate_tile") else []) + [x for m in o["melds"] for x in m["tiles"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "wall_frame_check.json"))
    ap.add_argument("--reps", type=int, default=1, help="시간 잴 때 모드를 번갈아 몇 번(답은 첫 번째, 시간은 중앙값)")
    ap.add_argument("--modes", default=",axis,rot", help="WALL_FRAME 값들(쉼표, 빈 값 = 없음) — 8b-3b 는 ',rot2'")
    a = ap.parse_args()
    global MODES
    MODES = a.modes.split(",")
    ref = {}
    rp = APP / "experiments" / "zone_points" / "results" / "wall_frame.json"
    if rp.exists():
        ref = json.loads(rp.read_text(encoding="utf-8"))["rows"]
    reader = lr.LocalReader(threads=2)
    net = roles.RoleNet(threads=2)
    rows = []
    its = items(a.keys)
    warm = prepare_image(its[0]["path"].read_bytes(), 4080)  # 예열(모델 올리기)은 시간에서 뺀다
    roles.read_roles(reader, warm, its[0]["side"], its[0]["grid"], "hybrid", net)
    reader.read_wall(warm)
    for it in its:
        jpeg = prepare_image(it["path"].read_bytes(), 4080)
        row = {"key": it["key"], "set": it["set"], "side": it["side"], "want": len(it["tiles"]) if it["tiles"] else None, "dora_want": it["dora"]}
        for rep, m in [(k, m) for k in range(a.reps) for m in MODES]:
            os.environ["WALL_FRAME"] = m
            reader._full.clear()
            lr._decoded.clear()
            t0 = time.perf_counter()
            o = roles.read_roles(reader, jpeg, it["side"], it["grid"], "hybrid", net)
            t1 = time.perf_counter()
            d = reader.read_wall(jpeg)
            t2 = time.perf_counter()
            if rep:
                row[m or "none"]["times"].append((round(t1 - t0, 3), round(t2 - t1, 3), o["ms"].get("frame", 0), d["ms"].get("frame", 0)))
                continue
            got = answer(o)
            r = {"hit": hit(got, it["tiles"]), "n": len(got), "framed": o["framed"], "tiles": " ".join(got), "roles_s": round(t1 - t0, 3),
                 "dora_s": round(t2 - t1, 3), "ms": o["ms"], "dora": [x["tile"] for x in d["indicators"]],
                 "pairs": [[p["tile"] for p in pr] for pr in d["pairs"]], "dora_ms": d["ms"],
                 "times": [(round(t1 - t0, 3), round(t2 - t1, 3), o["ms"].get("frame", 0), d["ms"].get("frame", 0))]}
            if m:
                r.update(cut=o.get("cut"), frame=o.get("frame"), frame_error=o.get("frame_error"), frame_info=o.get("frame_info"),
                         grid=o.get("grid"), dora_cut=d.get("cut"), dora_kept=d.get("kept"))
                e = (ref.get(it["key"]) or {}).get("est")
                if e and o.get("frame"):
                    f = o["frame"]
                    r["vs_8b1"] = {"dc_u": round(((f["cx"] - e["c"][0]) ** 2 + (f["cy"] - e["c"][1]) ** 2) ** 0.5 / e["u"], 2),
                                   "dtheta_deg": round(abs(f["theta"] - e["theta"]) * 57.2958, 2), "du": round(f["u"] - e["u"], 2)}
            row[m or "none"] = r
            print(it["key"], m or "none", r["hit"], "/", row["want"], r["n"], "framed" if r["framed"] else "-", f"{r['roles_s']:.2f}s",
                  r.get("cut", ""), r.get("frame_error") or "", r.get("vs_8b1", ""), "dora", r["dora"], r["pairs"], f"{r['dora_s']:.2f}s", flush=True)
        for m in MODES:
            t = row[m or "none"]["times"]
            row[m or "none"]["med"] = {k: sorted(x[i] for x in t)[len(t) // 2] for i, k in enumerate(("roles_s", "dora_s", "roles_frame_ms", "dora_frame_ms"))}
        print("  중앙", {m or "none": row[m or "none"]["med"] for m in MODES}, flush=True)
        rows.append(row)
    os.environ.pop("WALL_FRAME", None)
    Path(a.out).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
