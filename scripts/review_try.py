"""패보 리뷰 해설을 서버 없이 한 번 돌려 본다: 사실표 JSON → explain(+ GeminiTextGenerator) → 응답 JSON(UTF-8).

python scripts/review_try.py <사실표.json> [--out 응답.json] [--fake]

키는 환경 변수 GEMINI_API_KEY 에서만 읽고 어디에도 찍지 않는다. 모델은 REVIEW_MODEL(없으면 GEMINI_MODEL),
생각 수준은 REVIEW_THINKING(default | low | off, 없으면 low). --fake 는 Gemini 를 부르지 않고 가짜 generate 로 흐름만 돈다
(첫 답의 첫 차례에 일부러 금지어를 넣어 재시도까지 보인다).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import review  # noqa: E402


def fake_generate():
    """사실표만 보고 규칙에 맞는 견본 글을 쓰는 가짜 generate. 첫 호출의 첫 차례에는 금지어를 넣는다."""
    calls = {"n": 0}

    def generate(prompt: str, schema: dict) -> tuple[dict, int, int]:
        calls["n"] += 1
        facts = json.loads(prompt.split("## 사실표(JSON)\n", 1)[1])
        redo = None
        if "## 다시 쓰기" in prompt:
            redo = {int(line.split()[2].rstrip(":")) for line in prompt.splitlines() if line.startswith("- 차례 ")
                    and "앞선 title" in line}
        turns = []
        for i, t in enumerate(facts["turns"]):
            if redo is not None and t["turn"] not in redo:
                continue
            name = review.tile_name(t["discard"])
            text = f"패효율 기준 견본 해설이다. [{name}]을 버린 뒤 유효패는 {t['actual']['total']}장이었다."
            if calls["n"] == 1 and i == 0:
                text = "반드시 " + text
            turns.append({"turn": t["turn"], "title": f"[{name}] 버림", "text": text})
        data = {"turns": turns}
        if "summary" in schema["properties"]:
            eff = facts["game"]["efficiency_pct"]
            data["summary"] = "견본 총평이다." + (f" 패효율 {eff}%였다." if eff is not None else "")
        return data, len(prompt), 50

    return generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("facts", type=Path, help="사실표(요청 본문) JSON 파일")
    ap.add_argument("--out", type=Path, help="응답을 저장할 파일(UTF-8)")
    ap.add_argument("--fake", action="store_true", help="Gemini 대신 가짜 generate(키 없이 흐름만)")
    args = ap.parse_args()

    req = json.loads(args.facts.read_text(encoding="utf-8"))
    try:
        review.validate_request(req)
    except ValueError as e:
        print(f"규격 밖 요청: {e}", file=sys.stderr)
        return 2
    model = os.environ.get("REVIEW_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
    if args.fake:
        generate, model = fake_generate(), "fake"
    else:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("GEMINI_API_KEY 가 없다(--fake 로 흐름만 돌릴 수 있다)", file=sys.stderr)
            return 2
        thinking = os.environ.get("REVIEW_THINKING", review.DEFAULT_THINKING)
        generate = review.GeminiTextGenerator(key, model=model, thinking=thinking)
    out = review.explain(req, generate, model)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
