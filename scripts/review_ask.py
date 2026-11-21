"""패보 복기 에이전트를 서버 없이 한 판 · 한 질문 돌려 본다: 에이전트 사실표 JSON + 질문 → review_agent 그래프 → 답.

python scripts/review_ask.py <에이전트 사실표.json> "<질문>" [--history 앞선문답.json] [--out 응답.json] [--fake]

사실표는 앱의 `dart run tool/paifu_agent_facts.dart <패보.json> [출력.json]` 으로 만든다(시험 견본: tests/fixtures/agent_facts/).
흐름(질문 → 도구 호출 · 결과 → 답 → 검증 결과)을 사람이 읽는 꼴로 stdout 에 적고, 응답 JSON(라우트와 같은 꼴)은 --out 에 쓴다.

--fake 는 Gemini 를 부르지 않는 가짜 LLM 이다(키 · 네트워크 없이 그래프만 돈다): 질문에 차례 번호가 있으면 turn_info →
(숫자 있는 차례면) discard_options 를 부르고, 없으면 game_summary 를 부른 뒤 도구 결과로 답을 쓴다. 첫 답에는 일부러 도구 결과에
없는 숫자를 넣어 검증 실패 → 재시도까지 보인다.
--fake 가 없으면 GEMINI_API_KEY(환경 변수에서만 읽고 찍지 않음)로 실제 호출한다 — 돈이 드니 사용자 허락 뒤에만.
모델은 REVIEW_MODEL(없으면 GEMINI_MODEL), 생각 수준은 REVIEW_THINKING(없으면 low).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import review, review_agent  # noqa: E402

_TOOL_LINE = re.compile(r"^- 도구 (\S+) (\{.*\})$")


def _section(prompt: str, head: str) -> str:
    body = prompt.split(f"\n{head}\n", 1)[1] if f"\n{head}\n" in prompt else ""
    return body.split("\n\n## ", 1)[0].strip()


def _tool_results(prompt: str) -> list[tuple[str, dict, dict]]:
    """지시문의 "지금까지" 에서 (도구, 인자, 결과)를 읽는다."""
    lines = prompt.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = _TOOL_LINE.match(line)
        if m and i + 1 < len(lines) and lines[i + 1].startswith("  결과: "):
            out.append((m.group(1), json.loads(m.group(2)), json.loads(lines[i + 1][len("  결과: "):])))
    return out


def fake_generate():
    """지시문만 보고 움직이는 가짜 LLM(generate). 계산하지 않고 도구 결과의 값만 옮긴다."""

    def generate(prompt: str, schema: dict) -> tuple[dict, int, int]:
        question = _section(prompt, "## 질문")
        done = _tool_results(prompt)
        retried = "- 검사에 걸림" in prompt
        answer_only = schema["properties"]["action"].get("enum") == ["answer"]

        def found(name: str, turn: int | None = None) -> dict | None:
            for tool, args, result in done:
                if tool == name and (turn is None or args.get("turn") == turn):
                    return result
            return None

        def call(name: str, args: dict) -> tuple[dict, int, int]:
            return {"action": "tool", "tool": name, "args": args}, len(prompt) // 4, 20

        turns = review_agent.question_turns(question)
        if turns:
            n = turns[0]
            info = found("turn_info", n)
            if info is None and not answer_only:
                return call("turn_info", {"turn": n})
            if info is not None and "error" not in info and info.get("actual") and found("discard_options", n) is None \
                    and not answer_only:
                return call("discard_options", {"turn": n})
            if info is None or "error" in info:
                text = "질문한 차례는 이 판에서 찾지 못했다. 다른 차례 번호로 물어 주면 도구로 확인한다."
            else:
                text = " ".join(review_agent._turn_sentences(info))
                if info.get("actual"):
                    text += " 패효율만 본 평가라 수비 · 타점 판단은 들어 있지 않다."
        else:
            s = found("game_summary")
            if s is None and not answer_only:
                return call("game_summary", {})
            s = s or {}
            eff = s.get("efficiency_pct")
            text = (f"이 판 패효율은 {eff}%다." if eff is not None else "이 판은 등급을 매긴 버림이 없다.") + \
                f" 내 버림은 {s.get('hero_discards', 0)}번이다. 패효율만 본 평가라 수비 · 타점 판단은 들어 있지 않다."
        if not retried:
            text += " 유효패 차이는 99장이다."  # 일부러 넣은 틀린 숫자 → 검증 실패 → 재시도
        return {"action": "answer", "text": text}, len(prompt) // 4, 60

    return generate


def _short(result: dict, limit: int = 160) -> str:
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("facts", type=Path, help="에이전트 사실표 JSON 파일")
    ap.add_argument("question", help="질문(1~300자)")
    ap.add_argument("--history", type=Path, help='앞선 문답 [{"q": "...", "a": "..."}] JSON 파일(최대 4개)')
    ap.add_argument("--out", type=Path, help="응답을 저장할 파일(UTF-8)")
    ap.add_argument("--fake", action="store_true", help="Gemini 대신 가짜 LLM(키 · 네트워크 없이 흐름만)")
    args = ap.parse_args()

    req = {"facts": json.loads(args.facts.read_text(encoding="utf-8")), "question": args.question}
    if args.history:
        req["history"] = json.loads(args.history.read_text(encoding="utf-8"))
    try:
        review_agent.validate_request(req)
    except ValueError as e:
        print(f"규격 밖 요청: {e}", file=sys.stderr)
        return 2
    if args.fake:
        generate, model = fake_generate(), "fake"
    else:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("GEMINI_API_KEY 가 없다(--fake 로 흐름만 돌릴 수 있다)", file=sys.stderr)
            return 2
        model = os.environ.get("REVIEW_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
        thinking = os.environ.get("REVIEW_THINKING", review.DEFAULT_THINKING)
        generate = review.GeminiTextGenerator(key, model=model, thinking=thinking)

    state, elapsed = review_agent.run(req, generate)
    out = review_agent.to_response(state, elapsed, model)

    lines = [f"질문: {req['question']}"]
    for m in state["messages"]:
        if m["role"] == "tool":
            lines.append(f"  도구 {m['tool']} {json.dumps(m['args'], ensure_ascii=False)} → {_short(m['result'])}")
        elif m["role"] == "answer":
            lines.append(f"  답(LLM): {m['text']}")
        elif m["role"] == "check":
            lines.append("  검증 실패 → 다시: " + " / ".join(m["violations"]))
        elif m["role"] == "note":
            lines.append(f"  알림: {m['text']}")
    lines.append(f"최종 답: {out['answer']}")
    lines.append(f"결과: {out['outcome']} · 재시도 {out['retries']} · 도구 {len(out['tool_calls'])}회 · "
                 f"LLM {out['usage']['llm_calls']}회 · 위반 {out['violations'] or '없음'} · {out['elapsed_s']}초")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    if args.out:
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
