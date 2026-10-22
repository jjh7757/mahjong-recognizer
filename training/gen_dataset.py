"""CC0 패 이미지로 YOLO 형식 합성 데이터셋을 만든다.

    python gen_dataset.py <출력 폴더> --train 600 --val 100 [--seed 1]

출력:
    <out>/images/{train,val}/NNNN.jpg
    <out>/labels/{train,val}/NNNN.txt   (class cx cy w h, 0~1 정규화)
    <out>/dataset.yaml
    <out>/classes.json

클래스 38개: 1m~9m 1p~9p 1s~9s 1z~7z 0m 0p 0s back
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

TILE_DIR = Path(__file__).parent / "tiles"

NAMES = {
    **{f"{n}m": f"Man{n}" for n in range(1, 10)},
    **{f"{n}p": f"Pin{n}" for n in range(1, 10)},
    **{f"{n}s": f"Sou{n}" for n in range(1, 10)},
    "1z": "Ton", "2z": "Nan", "3z": "Shaa", "4z": "Pei", "5z": "Haku", "6z": "Hatsu", "7z": "Chun",
    "0m": "Man5-Dora", "0p": "Pin5-Dora", "0s": "Sou5-Dora",
    "back": "Back",
}
CLASSES = list(NAMES)
TILE_CODES = CLASSES[:-1]

_cache: dict[str, Image.Image] = {}


def tile_image(code: str, width: int) -> Image.Image:
    if code not in _cache:
        if code == "back":
            img = Image.open(TILE_DIR / "Back.png").convert("RGBA")
        else:
            front = Image.open(TILE_DIR / "Front.png").convert("RGBA")
            face = Image.open(TILE_DIR / f"{NAMES[code]}.png").convert("RGBA")
            img = Image.alpha_composite(front, face)
        _cache[code] = img
    img = _cache[code]
    h = round(width * img.height / img.width)
    return img.resize((width, h), Image.LANCZOS)


def random_hand(rng: random.Random, n: int) -> list[str]:
    counts: dict[str, int] = {}
    hand: list[str] = []

    def base(code: str) -> str:
        return "5" + code[1] if code[0] == "0" else code

    def add(code: str) -> None:
        if counts.get(base(code), 0) >= 4:
            return
        if code[0] == "0" and code in hand:
            return
        counts[base(code)] = counts.get(base(code), 0) + 1
        hand.append(code)

    plausible = rng.random() < 0.5
    while len(hand) < n:
        if plausible and rng.random() < 0.6 and n - len(hand) >= 3:
            suit = rng.choice("mps")
            start = rng.randint(1, 7)
            seq = [f"{start + i}{suit}" for i in range(3)]
            if rng.random() < 0.3 and "5" + suit in seq:
                seq[seq.index("5" + suit)] = "0" + suit
            for c in seq:
                add(c)
            continue
        add(rng.choice(TILE_CODES))
    return hand[:n]


def felt(w: int, h: int, rng: random.Random) -> Image.Image:
    palette = [
        (rng.randint(20, 50), rng.randint(90, 130), rng.randint(40, 70)),    # 녹색 매트
        (rng.randint(30, 60), rng.randint(50, 90), rng.randint(120, 170)),   # 파란 매트
        (rng.randint(120, 160), rng.randint(90, 120), rng.randint(60, 90)),  # 나무 탁자
        (rng.randint(200, 235), rng.randint(200, 235), rng.randint(200, 235)),  # 밝은 탁자
    ]
    bg = Image.new("RGB", (w, h), rng.choice(palette))
    draw = ImageDraw.Draw(bg)
    for _ in range(3000):
        x, y = rng.randrange(w), rng.randrange(h)
        d = rng.randint(-14, 14)
        px = bg.getpixel((x, y))
        draw.point((x, y), tuple(max(0, min(255, c + d)) for c in px))
    return bg


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.array(matrix, dtype=float)
    b = np.array(pb, dtype=float).reshape(8)
    return np.linalg.solve(a, b)


def apply_h(coeffs, x, y):
    a, b, c, d, e, f, g, h = coeffs
    den = g * x + h * y + 1
    return (a * x + b * y + c) / den, (d * x + e * y + f) / den


def render(rng: random.Random) -> tuple[Image.Image, list[tuple[str, tuple[float, float, float, float]]]]:
    """이미지와 (클래스, xyxy 박스) 목록."""
    n = rng.choice([14, 14, 14, 13, 11, 8, 5, 3, 1, 2])
    hand = random_hand(rng, n)
    if rng.random() < 0.7:
        hand.sort(key=lambda c: ("mpsz".index(c[1]), 5 if c[0] == "0" else int(c[0])))
    # 안깡 흉내: 뒷면 2장 + 같은 패 2장
    if n >= 8 and rng.random() < 0.25:
        t = rng.choice(TILE_CODES)
        hand = hand[: n - 4] + ["back", t, t, "back"]
    gaps = set()
    if n >= 8:
        if rng.random() < 0.5:
            gaps.add(n - 3)
        if rng.random() < 0.4:
            gaps.add(n - 1 if n - 1 not in gaps else n - 6)

    tile_w = rng.randint(50, 110)
    tiles = [tile_image(c, tile_w) for c in hand]
    tile_h = tiles[0].height
    spacing = rng.randint(0, 8)
    two_rows = n >= 10 and rng.random() < 0.3
    per_row = (n + 1) // 2 if two_rows else n

    canvas_w = rng.randint(900, 1400)
    canvas_h = rng.randint(500, 900)
    bg = felt(canvas_w, canvas_h, rng)
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []

    row_w = sum(t.width + spacing for t in tiles[:per_row]) + tile_w // 2 * len(gaps)
    x0 = max(10, (canvas_w - row_w) // 2 + rng.randint(-60, 60))
    y0 = max(10, (canvas_h - tile_h * (2 if two_rows else 1)) // 2 + rng.randint(-60, 60))
    x, y = x0, y0
    for i, (code, t) in enumerate(zip(hand, tiles)):
        if two_rows and i == per_row:
            x, y = x0, y0 + tile_h + rng.randint(8, 30)
        angle = rng.uniform(-6, 6)
        rotated = t.rotate(angle, resample=Image.BICUBIC, expand=True)
        jy = rng.randint(-5, 5)
        alpha = rotated.split()[3]
        shadow = Image.new("RGBA", rotated.size, (0, 0, 0, 0))
        shadow.paste((0, 0, 0, rng.randint(50, 110)), mask=alpha)
        bg.paste(shadow, (x + 5, y + jy + 7), shadow)
        bg.paste(rotated, (x, y + jy), rotated)
        bb = alpha.getbbox() or (0, 0, rotated.width, rotated.height)
        boxes.append((code, (x + bb[0], y + jy + bb[1], x + bb[2], y + jy + bb[3])))
        x += t.width + spacing
        if i + 1 in gaps:
            x += tile_w // 2

    # 도라 표시패를 위쪽에 하나 놓기도 한다
    if rng.random() < 0.3 and y0 > tile_h + 40:
        code = rng.choice(TILE_CODES)
        t = tile_image(code, tile_w)
        dx, dy = rng.randint(x0, max(x0, canvas_w - tile_w - 10)), max(5, y0 - tile_h - rng.randint(20, 40))
        bg.paste(t, (dx, dy), t)
        boxes.append((code, (dx, dy, dx + t.width, dy + t.height)))

    img = bg
    if rng.random() < 0.7:
        w, h = img.size
        s = rng.uniform(0.02, 0.12)
        dx = int(w * s * rng.uniform(0.5, 1.0))
        dy = int(h * s * rng.uniform(0.0, 0.5))
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(dx, dy), (w - dx, dy), (w, h), (0, h)]
        inv = find_coeffs(dst, src)
        fwd = find_coeffs(src, dst)
        img = img.transform((w, h), Image.PERSPECTIVE, tuple(inv), Image.BICUBIC, fillcolor=img.getpixel((5, h - 5)))
        new_boxes = []
        for code, (x1, y1, x2, y2) in boxes:
            pts = [apply_h(fwd, px, py) for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            new_boxes.append((code, (min(xs), min(ys), max(xs), max(ys))))
        boxes = new_boxes

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.15))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.3:
        arr = np.asarray(img).astype(np.int16)
        arr += np.random.default_rng(rng.randrange(1 << 30)).integers(-10, 10, arr.shape, dtype=np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    r, g, b = img.split()
    r = r.point(lambda v: min(255, int(v * rng.uniform(0.92, 1.1))))
    b = b.point(lambda v: min(255, int(v * rng.uniform(0.9, 1.08))))
    return Image.merge("RGB", (r, g, b)), boxes


def write_split(out: Path, split: str, count: int, rng: random.Random) -> None:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img, boxes = render(rng)
        w, h = img.size
        name = f"{i:04d}"
        img.save(out / "images" / split / f"{name}.jpg", quality=rng.randint(65, 92))
        lines = []
        for code, (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{CLASSES.index(code)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if (i + 1) % 100 == 0:
            print(f"{split}: {i + 1}/{count}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    write_split(args.out, "train", args.train, rng)
    write_split(args.out, "val", args.val, rng)
    (args.out / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8")
    print("done:", args.out)


if __name__ == "__main__":
    main()
