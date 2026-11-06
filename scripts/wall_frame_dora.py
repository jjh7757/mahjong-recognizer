"""WALL_FRAME 표시패 서버 단독 채점(개선 계획 8b 검토 ① "표시패 맞음 수", 2026-11-06) — /v2/localdora(read_wall)를 프로세스 안에서
WALL_FRAME 없음 / axis / rot 로 읽고, 읽은 패(패산 표시패 indicators + 탁자 위 짝 pairs)에 정답 표시패(도라 + 뒷도라)가 몇 장 드는지 센다.
기준점 잡힘(frame / frame_error)도 사진마다 남긴다.

정답(mahjong_calc, 읽기만): 개발 = seat_grow.eval_scenes() 중 10-31 매장 사진(visit_20261031/truth.json `indicators`, 없으면 label.json
dora_indicators + ura_indicators — B·C·D 는 정답 없음), 테스트·색 바꾼 = field_20261024/captures/<사진>/label.json 이 있는 것.
모델은 서버 assets(운영) — 개발 사진은 2회차 검출기가 배운 사진이라 절대값보다 모드 사이 차이를 본다.

  python -X utf8 scripts/wall_frame_dora.py [--sets dev,test,recolor] --out 결과.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from app.recognizer import prepare_image  # noqa: E402

FIELD = APP / "experiments" / "field_20261024" / "captures"
RECOLOR = APP / "experiments" / "field_20261024" / "captures_recolor"
VISIT = APP / "experiments" / "visit_20261031"
MODES = ["", "axis", "rot"]


def label_inds(p: Path) -> list[str] | None:
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    return list(j.get("dora_indicators") or []) + list(j.get("ura_indicators") or [])


def items(sets: list[str]) -> list[dict]:
    out = []
    if "dev" in sets:
        import seat_grow as sg

        truth = json.loads((VISIT / "truth.json").read_text(encoding="utf-8"))["scenes"]
        for s in sg.eval_scenes():
            if s["set"] != "visit":
                continue
            t = truth[s["name"]]["indicators"] if s["name"] in truth and "indicators" in truth[s["name"]] else label_inds(Path(s["path"]).parent / "label.json")
            out.append(dict(key=s["name"], set="dev", path=Path(s["path"]), truth=t))
    for st, base in (("test", FIELD), ("recolor", RECOLOR)):
        if st not in sets:
            continue
        for d in sorted(FIELD.glob("20261024_1*")):
            if d.name == "20261024_131823" or not (base / d.name / "original.jpg").exists():  # EXIF 회전이 다른 사진(field_bench 와 같이 뺌)
                continue
            out.append(dict(key=(f"recolor/{d.name}" if st == "recolor" else d.name), set=st, path=base / d.name / "original.jpg", truth=label_inds(d / "label.json")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="dev,test,recolor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default=",axis,rot", help="WALL_FRAME 값들(쉼표, 빈 값 = 없음) — 8b-3b 는 ',rot2'")
    a = ap.parse_args()
    global MODES
    MODES = a.modes.split(",")
    its = items(a.sets.split(","))
    field_names = {f"20261024_{p}" for p in ("122922", "124007", "125053", "130533", "131210", "132647", "133011", "133620", "134323", "135148",
                                              "135912", "140302", "140750", "140918", "143443")}
    its = [it for it in its if it["set"] == "dev" or it["key"].split("/")[-1] in field_names]
    reader = lr.LocalReader(threads=2)
    reader.read_wall(prepare_image(its[0]["path"].read_bytes(), 4080))  # 예열
    rows = []
    for it in its:
        jpeg = prepare_image(it["path"].read_bytes(), 4080)
        row = {"key": it["key"], "set": it["set"], "truth": it["truth"]}
        for m in MODES:
            os.environ["WALL_FRAME"] = m
            reader._full.clear()
            lr._decoded.clear()
            t0 = time.perf_counter()
            d = reader.read_wall(jpeg)
            got = [x["tile"] for x in d["indicators"]] + [p["tile"] for pr in d["pairs"] for p in pr]
            r = {"got": got, "ind": [x["tile"] for x in d["indicators"]], "pairs": [[p["tile"] for p in pr] for pr in d["pairs"]], "s": round(time.perf_counter() - t0, 3)}
            if it["truth"] is not None:
                r["hit"] = sum((Counter(got) & Counter(it["truth"])).values())
                r["all"] = r["hit"] == len(it["truth"])
            if m:
                r.update(cut=d.get("cut"), frame=d.get("frame"), frame_error=d.get("frame_error"), kept=d.get("kept"),
                         zones=[[p.get("zone") for p in pr] for pr in d["pairs"]])
            row[m or "none"] = r
        rows.append(row)
        print(it["key"], "정답", it["truth"], *[f"{m or 'none'}:{row[m or 'none']['got']}({row[m or 'none'].get('hit')}){row[m or 'none'].get('cut') or ''}" for m in MODES], flush=True)
    os.environ.pop("WALL_FRAME", None)
    summ = {}
    for st in ("dev", "test", "recolor"):
        rs = [r for r in rows if r["set"] == st]
        if not rs:
            continue
        sc = [r for r in rs if r["truth"] is not None]
        summ[st] = {"photos": len(rs), "scored": len(sc), "truth_tiles": sum(len(r["truth"]) for r in sc)}
        for m in MODES:
            k = m or "none"
            summ[st][k] = {"hit": sum(r[k]["hit"] for r in sc), "all": sum(r[k]["all"] for r in sc), "got_tiles": sum(len(r[k]["got"]) for r in rs)}
            if m:
                summ[st][k]["frame_ok"] = sum(1 for r in rs if r[k].get("frame"))
    print(json.dumps(summ, ensure_ascii=False))
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
