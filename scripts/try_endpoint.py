"""서버 엔드포인트를 사진 한 장으로 호출해 JSON 을 찍는다.

python scripts/try_endpoint.py --mode hand|locate|table [--url http://127.0.0.1:8000] [--key-file ...] 사진.jpg
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PATHS = {"hand": "/v1/recognize", "locate": "/v1/locate", "table": "/v1/table"}


def post_image(url: str, path: Path, key: str) -> dict:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if key:
        req.add_header("X-Api-Key", key)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--mode", choices=PATHS, default="hand")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key-file", type=Path, help="X-Api-Key 를 담은 파일 (없으면 키 없이)")
    a = ap.parse_args()
    key = a.key_file.read_text(encoding="utf-8").strip() if a.key_file else ""
    t0 = time.perf_counter()
    try:
        out = post_image(a.url + PATHS[a.mode], a.image, key)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"# wall {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
