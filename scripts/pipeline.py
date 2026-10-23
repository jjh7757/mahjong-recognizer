"""전경 사진 한 장으로 2단계 판독을 끝까지 돌린다 (앱이 할 일을 PC 에서 흉내).

1. 긴 변 1024 로 줄인 사본 → /v1/locate
2. 원본에서 손패 상자(face_up 인 자리, 또는 --side)와 center 상자를 잘라 자리별로 회전
3. 손패 크롭 → /v1/recognize, 중앙 크롭 → /v1/table (병렬)
4. 결과 JSON 과 크롭·상자 그림을 --out 폴더에 저장

python scripts/pipeline.py photo.jpg --out results/run1 [--url http://127.0.0.1:8000] [--key-file ..] [--side bottom]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from try_endpoint import PATHS, post_image  # noqa: E402

# 자리별 회전(PIL rotate 는 반시계가 양수). 글자 윗부분이 가운데를 향하므로:
#   right: 윗부분이 왼쪽 → 시계 90 → rotate(-90)
#   left : 윗부분이 오른쪽 → 반시계 90 → rotate(90)
#   top  : 180
ROTATE = {"bottom": 0, "right": -90, "top": 180, "left": 90}
PAD = {"hand": 0.04, "center": 0.05}  # 상자를 비율만큼 넓힌다. 손패는 넓히면 옆자리 버림패가 들어와 화료패로 오인된다


def shrink(img: Image.Image, max_side: int = 1024) -> bytes:
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def crop(img: Image.Image, box: list[int], pad: float, rotate: int) -> Image.Image:
    W, H = img.size
    y0, x0, y1, x1 = box
    bw, bh = (x1 - x0) * pad, (y1 - y0) * pad
    px0 = max(0, int((x0 - bw) * W / 1000)); px1 = min(W, int((x1 + bw) * W / 1000))
    py0 = max(0, int((y0 - bh) * H / 1000)); py1 = min(H, int((y1 + bh) * H / 1000))
    out = img.crop((px0, py0, px1, py1))
    return out.rotate(rotate, expand=True) if rotate else out


def post_bytes(url: str, data: bytes, name: str, key: str, out_dir: Path) -> dict:
    p = out_dir / name
    p.write_bytes(data)
    return post_image(url, p, key)


def draw_boxes(img: Image.Image, loc: dict, path: Path) -> None:
    im = img.convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    items = [(f"hand {h['side']}{' *' if h['face_up'] else ''}", h["box"], "red") for h in loc["hands"]]
    items += [(f"river {r['side']}", r["box"], "yellow") for r in loc["rivers"]]
    if loc.get("dead_wall"):
        items.append(("dead_wall", loc["dead_wall"]["box"], "cyan"))
    if loc.get("center"):
        items.append(("center", loc["center"]["box"], "white"))
    for name, (y0, x0, y1, x1), c in items:
        r = [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]
        d.rectangle(r, outline=c, width=3); d.text((r[0] + 4, r[1] + 4), name, fill=c)
    im.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--side", choices=list(ROTATE), help="손패를 읽을 자리. 없으면 face_up 인 자리 전부")
    a = ap.parse_args()
    key = a.key_file.read_text(encoding="utf-8").strip() if a.key_file else ""
    a.out.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
    print(f"원본 {img.size}", file=sys.stderr)

    t0 = time.perf_counter()
    loc = post_bytes(a.url + PATHS["locate"], shrink(img), "locate_input.jpg", key, a.out)
    t_loc = time.perf_counter() - t0
    draw_boxes(img, loc, a.out / "locate_boxes.jpg")
    print(f"locate {t_loc:.1f}s hands={[h['side'] for h in loc['hands']]} face_up={loc['face_up_sides']} "
          f"dead_wall={'o' if loc.get('dead_wall') else 'x'} center={'o' if loc.get('center') else 'x'}",
          file=sys.stderr)

    sides = [a.side] if a.side else loc["face_up_sides"]
    hands = {h["side"]: h for h in loc["hands"]}
    jobs: list[tuple[str, str, bytes]] = []
    for s in sides:
        if s not in hands:
            print(f"손패 상자 없음: {s}", file=sys.stderr); continue
        im = crop(img, hands[s]["box"], PAD["hand"], ROTATE[s])
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=92)
        jobs.append((f"hand_{s}", PATHS["hand"], buf.getvalue()))
    if loc.get("center"):
        im = crop(img, loc["center"]["box"], PAD["center"], 0)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=92)
        jobs.append(("center", PATHS["table"], buf.getvalue()))

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {name: ex.submit(post_bytes, a.url + path, data, f"{name}.jpg", key, a.out)
                for name, path, data in jobs}
        results = {name: f.result() for name, f in futs.items()}
    t_2 = time.perf_counter() - t1

    summary = {"image": str(a.image), "locate": loc, "results": results,
               "timing_s": {"locate": round(t_loc, 1), "stage2_parallel": round(t_2, 1),
                            "total": round(t_loc + t_2, 1)}}
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    for name, r in results.items():
        if name.startswith("hand"):
            tiles = " ".join(t["tile"] for t in r["tiles"])
            melds = " | ".join(f"{m['type']}:{' '.join(m['tiles'])}" for m in r["melds"]) or "-"
            win = r["win_tile"]["tile"] if r.get("win_tile") else "-"
            print(f"{name}: tiles[{r['count']}]={tiles}  melds={melds}  win={win}  ({r['latency_ms']}ms)")
        else:
            dora = " ".join(t["tile"] for t in r["dora_indicators"]) or "-"
            riichi = [s for s, v in r["riichi"].items() if v]
            print(f"center: dora={dora}  riichi={riichi}  last={r['last_discards']}  ({r['latency_ms']}ms)")
        if r.get("notes"):
            print(f"  notes: {r['notes']}")
    print(f"timing: locate {t_loc:.1f}s + stage2 {t_2:.1f}s = {t_loc + t_2:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
