"""패보 복기 에이전트(POST /v2/review/ask) — 도구 · LangGraph 그래프 · 검증 · 대체 · 지시문.

설계: mahjong_calc/docs/paifu_review_agent_design_1121.md. 사용자가 패보 한 판을 보다가 질문하면 LLM 이 도구로 사실을 찾아
한국어로 답한다. **도구는 계산하지 않고 찾기만 한다** — 숫자는 전부 앱(Dart 엔진)이 만든 "에이전트 사실표"에 있다.

그래프(StateGraph):

    START → plan ─(tool)→ act → plan          도구 6회를 쓰면 plan 은 답만 받는다(스키마를 answer 로 좁힘)
              └─(answer)→ verify ─(통과)→ END
                            ├─(실패, 재시도 0회)→ plan(위반을 덧붙여)
                            └─(실패, 재시도 1회)→ fallback → END

한도: 도구 6회 · 재시도 1회 · LLM 호출 8회 · 전체 60초 — 넘으면 fallback(LLM 없이 코드가 만든 답).
LLM 은 해설(app/review.py)과 같은 주입 함수 `generate(prompt, schema) -> (dict, 입력 토큰, 출력 토큰)` — 운영은
GeminiTextGenerator, 시험은 가짜. 검증 규칙(대괄호 패 · 숫자 단위 · 영어 · 금지어)은 review.py 의 것을 다시 쓴다.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import review as rv
from .review import Generate, tile_name

PROMPT_VERSION = "a2"
FACTS_VERSION = 1

# 요청 상한(설계서 5절). 사실표는 예시 11판에서 압축 2.3 ~ 32.6KB.
MAX_REQUEST_BYTES = 200 * 1024
QUESTION_MAX = 300
HISTORY_MAX = 4
HISTORY_TEXT_MAX = 1000
MAX_TURNS = 40
MAX_OTHERS = 64

# 답 모양: 지시문은 3~5문장. 검사는 위쪽만 본다(짧고 정직한 "모른다" 를 막지 않게).
ANSWER_MAX_CHARS = 500
ANSWER_MAX_SENTENCES = 5


@dataclass(frozen=True)
class Limits:
    """설계서 4절 한도."""

    tool_calls: int = 6
    retries: int = 1
    llm_calls: int = 8
    seconds: float = 60.0


# ---------------------------------------------------------------- 이름

STATUSES = ["graded", "forced", "defense", "complete", "unknownHand", "inconsistent", "guessed"]
GRADES = ["best", "good", "inaccuracy", "mistake", "blunder"]
NUMBERLESS = {"unknownHand", "inconsistent", "guessed"}
GRADE_KO = {"best": "최선", "good": "좋음", "inaccuracy": "부정확", "mistake": "실수", "blunder": "큰 실수"}
STATUS_KO = {"graded": "등급 매김", "forced": "리치 중", "defense": "수비", "complete": "화료 형태",
             "unknownHand": "손패 기록 없음", "inconsistent": "손패 기록 어긋남", "guessed": "짐작 손패"}
RESULT_KO = {"ron": "론", "tsumo": "쯔모", "draw": "유국"}
CALL_KO = {"pon": "퐁", "chi": "치", "kan": "깡", "kakan": "가깡", "ankan": "안깡", "call": "울기"}
WINDS = ["동", "남", "서", "북"]
_TILE_SET = set(rv.TILE_CODES)


def _ko_lookup(table: dict[str, str]) -> dict[str, str]:
    """영어 이름 · 한국어 이름(띄어쓰기 무시) → 영어 이름."""
    out = {k.lower(): k for k in table}
    out.update({v.replace(" ", ""): k for k, v in table.items()})
    return out


_GRADE_ARG = _ko_lookup(GRADE_KO)
_STATUS_ARG = _ko_lookup(STATUS_KO)
_STATUS_ARG.update({"리치중": "forced", "손패모름": "unknownHand"})


def _jong(word: str) -> bool:
    """끝 글자에 받침이 있나. 끝의 닫는 괄호(`[3만]` · `(또는 [3삭])`)는 건너뛰고, 한글이 아니면 있다고 본다(숫자 · % 뒤)."""
    word = word.rstrip(")]")
    if not word:
        return False
    c = ord(word[-1])
    if 0xAC00 <= c <= 0xD7A3:
        return (c - 0xAC00) % 28 != 0
    return True


def with_eul(word: str) -> str:
    """을/를."""
    return word + ("을" if _jong(word) else "를")


def with_ida(word: str) -> str:
    """이다/다."""
    return word + ("이다" if _jong(word) else "다")


def with_eun(word: str) -> str:
    """은/는."""
    return word + ("은" if _jong(word) else "는")


def bracket(code: str) -> str:
    return f"[{tile_name(code)}]"


def shanten_text(s: int) -> str:
    return "화료 형태" if s < 0 else "텐파이" if s == 0 else f"{s}샹텐"


def ukeire_word(shanten_after: int) -> str:
    return "대기" if shanten_after <= 0 else "유효패"


# ---------------------------------------------------------------- 요청 검사


def validate_request(req: Any) -> None:
    """`{"facts": 사실표, "question": "...", "history": [{"q", "a"}]}` 검사. 맞지 않으면 ValueError.
    도구가 기대는 칸(차례 번호 · 후보 · 패 코드)을 본다. 모르는 칸은 무시한다(앞으로 더해도 깨지지 않게)."""
    need, tile_ok, int_in, tiles_ok = rv._need, rv._tile_ok, rv._int_in, rv._tiles_ok
    need(isinstance(req, dict), "요청", "JSON 객체가 아님")
    size = len(json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode())
    need(size <= MAX_REQUEST_BYTES, "요청", f"너무 큼({size} > {MAX_REQUEST_BYTES} bytes)")
    q = req.get("question")
    need(isinstance(q, str) and 1 <= len(q.strip()) <= QUESTION_MAX, "question", f"1~{QUESTION_MAX}자 글")
    hist = req.get("history", [])
    need(hist is None or (isinstance(hist, list) and len(hist) <= HISTORY_MAX), "history", f"최대 {HISTORY_MAX}개 목록")
    for i, h in enumerate(hist or []):
        need(isinstance(h, dict) and all(isinstance(h.get(k), str) and len(h[k]) <= HISTORY_TEXT_MAX for k in "qa"),
             f"history[{i}]", f"{{q, a}} 글(각 {HISTORY_TEXT_MAX}자 안)")

    f = req.get("facts")
    need(isinstance(f, dict), "facts", "객체가 아님")
    need(f.get("version") == FACTS_VERSION, "facts.version", f"{FACTS_VERSION} 이어야 함")
    for k in ("game", "turns", "others"):
        need(k in f, "facts", f"{k} 없음")
    g = f["game"]
    need(isinstance(g, dict), "facts.game", "객체가 아님")
    for k in ("round", "honba", "hero_wind", "dora_indicators", "efficiency_pct", "grades", "statuses", "result"):
        need(k in g, "facts.game", f"{k} 없음")
    need(isinstance(g["round"], str) and len(g["round"]) <= 16, "facts.game", "round")
    need(int_in(g["honba"], 0, 99), "facts.game", "honba")
    need(g["hero_wind"] in WINDS, "facts.game", "hero_wind")
    need(tiles_ok(g["dora_indicators"], 10), "facts.game", "dora_indicators")
    need(g["efficiency_pct"] is None or int_in(g["efficiency_pct"], 0, 100), "facts.game", "efficiency_pct")
    for k, keys in (("grades", GRADES), ("statuses", STATUSES)):
        d = g[k]
        need(isinstance(d, dict) and all(x in keys and int_in(v, 0, 999) for x, v in d.items()), "facts.game", k)
    res = g["result"]
    need(res is None or (isinstance(res, dict) and res.get("type") in RESULT_KO), "facts.game", "result")

    turns = f["turns"]
    need(isinstance(turns, list) and len(turns) <= MAX_TURNS, "facts.turns", f"목록(최대 {MAX_TURNS})")
    for i, t in enumerate(turns):
        w = f"facts.turns[{i}]"
        need(isinstance(t, dict), w, "객체가 아님")
        for k in ("turn", "step", "status", "grade", "hand", "drawn", "discard", "tsumogiri", "melds", "riichi_declared",
                  "shanten_prev", "shanten_before", "shanten_loss", "ukeire_loss", "options", "actual", "best",
                  "riichi_winds", "genbutsu", "safe_tiles"):
            need(k in t, w, f"{k} 없음")
        need(t["turn"] == i + 1, w, "turn 은 1 부터 차례대로")
        need(int_in(t["step"], 1, 9999), w, "step")
        need(t["status"] in STATUSES, w, "status")
        need(t["grade"] is None or t["grade"] in GRADES, w, "grade")
        need(t["hand"] is None or tiles_ok(t["hand"], 14), w, "hand")
        need(t["drawn"] is None or tile_ok(t["drawn"]), w, "drawn")
        need(tile_ok(t["discard"]), w, "discard")
        need(t["tsumogiri"] is None or isinstance(t["tsumogiri"], bool), w, "tsumogiri")
        need(int_in(t["melds"], 0, 4), w, "melds")
        need(isinstance(t["riichi_declared"], bool), w, "riichi_declared")
        for k in ("shanten_prev", "shanten_before"):
            need(t[k] is None or int_in(t[k], -1, 8), w, k)
        need(t["shanten_loss"] is None or int_in(t["shanten_loss"], -8, 8), w, "shanten_loss")
        need(t["ukeire_loss"] is None or int_in(t["ukeire_loss"], -136, 136), w, "ukeire_loss")
        opts = t["options"]
        need(isinstance(opts, list) and len(opts) <= 34, w, "options")
        for j, o in enumerate(opts):
            ow = f"{w}.options[{j}]"
            need(isinstance(o, dict) and tile_ok(o.get("discard")), ow, "discard")
            need(int_in(o.get("shanten"), -1, 8) and int_in(o.get("total"), 0, 136) and int_in(o.get("kinds"), 0, 34),
                 ow, "shanten · total · kinds")
            tl = o.get("tiles")
            need(isinstance(tl, dict) and len(tl) <= 34 and all(tile_ok(k) and int_in(v, 0, 4) for k, v in tl.items()),
                 ow, "tiles = {패: 남은 장수}")
        if opts:
            need(t["hand"] is not None and t["actual"] == t["discard"], w, "후보가 있는 차례는 hand 가 있고 actual = discard")
            need(t["actual"] in {o["discard"] for o in opts}, w, "actual 이 options 에 없음")
        else:
            need(t["actual"] is None, w, "후보가 없는 차례의 actual 은 null")
        need(tiles_ok(t["best"]) and set(t["best"]) <= {o["discard"] for o in opts}, w, "best 는 options 의 패")
        need(isinstance(t["riichi_winds"], list) and len(t["riichi_winds"]) <= 3
             and all(x in WINDS for x in t["riichi_winds"]), w, "riichi_winds")
        need(t["genbutsu"] is None or isinstance(t["genbutsu"], bool), w, "genbutsu")
        need(tiles_ok(t["safe_tiles"]), w, "safe_tiles")

    others = f["others"]
    need(isinstance(others, list) and len(others) <= MAX_OTHERS, "facts.others", f"목록(최대 {MAX_OTHERS})")
    for i, o in enumerate(others):
        need(isinstance(o, dict) and int_in(o.get("step"), 1, 9999) and isinstance(o.get("kind"), str)
             and (o.get("wind") is None or o.get("wind") in WINDS), f"facts.others[{i}]", "{step, wind, kind}")


# ---------------------------------------------------------------- 도구(사실표에서 찾기만)

TOOLS: dict[str, dict] = {
    "turn_info": {"args": ["turn"], "desc": "그 차례의 손패 · 뽑은 패 · 버린 패 · 상태 · 등급 · 샹텐 · 유효패(실제와 최선)"},
    "discard_options": {"args": ["turn"],
                        "desc": "그 차례 버림 후보 전부(버릴 패 · 버린 뒤 샹텐 · 유효패 장수 · 종류 수), 실제 · 최선 표시, "
                                "그 차례 상태 · 고를 수 있었나(can_choose) · 알림(note)"},
    "what_if": {"args": ["turn", "tile"],
                "desc": "그 차례에 tile 을 버렸다면: 샹텐 · 유효패 목록(패마다 남은 장수), 실제 버림과의 차이(한쪽에만 있는 유효패), "
                        "그 차례 상태 · 고를 수 있었나(can_choose, 리치 중이면 false — 숫자는 참고) · 리치한 자리가 있으면 그 패가 현물인가"},
    "safe_tiles": {"args": ["turn"], "desc": "그 차례에 리치한 다른 자리 · 버린 패가 현물인가 · 손 안의 현물"},
    "find_turns": {"args": ["grade?", "status?", "min_ukeire_loss?", "tile?", "after_riichi?"],
                   "desc": "조건에 맞는 내 버림 차례를 판 전체에서 한 번에: 줄마다 버린 패 · 상태 · 등급 · 쯔모기리 · 리치한 자리 · "
                           "현물인가. after_riichi 에 자리 바람(동 · 남 · 서 · 북)을 주면 그 자리가 리치한 뒤의 내 버림"
                           "(조건은 모두 그리고, 없으면 전부)"},
    "game_summary": {"args": [], "desc": "판 요약: 국 · 패효율 · 등급 개수 · 상태 개수 · 결과 · 남의 리치 · 울기 · 화료"},
    "rule": {"args": ["name"], "desc": "용어 풀이: 유효패 · 샹텐 · 현물 · 쯔모기리 · 등급 · 패효율"},
}
TOOL_NAMES = list(TOOLS)
PER_TURN_TOOLS = {"turn_info", "discard_options", "what_if", "safe_tiles"}

# rule 도구의 글. 숫자는 등급 문턱(10 · 30%)만 — 지시문과 앱(paifu_review.dart ReviewThresholds)의 임시 값.
RULES: dict[str, str] = {
    "유효패": "버린 뒤의 손에서 샹텐을 하나 줄여 주는 패다. 장수는 그 패가 아직 몇 장 남았나를 더한 것이다"
            "(한 종류 넉 장에서 내 손 · 네 자리 강 · 울기 · 도라 표시패로 보이는 장수를 뺀다). 텐파이에서는 대기라고 부른다.",
    "샹텐": "텐파이까지 패를 몇 번 더 바꿔야 하나를 센 수다. 영은 텐파이, 마이너스 일은 화료 형태다. "
          "버린 뒤 샹텐이 최선 후보보다 크면 샹텐이 뒤로 갔다고 한다.",
    "현물": "리치한 자리가 이미 버린 패, 그리고 리치 뒤 다른 자리가 버렸는데 그 자리가 론하지 않은 패다. 그 자리에는 론 당하지 않는다. "
          "이 도우미는 현물만 보고 스지 · 벽 같은 수비 판단은 하지 않는다.",
    "쯔모기리": "뽑은 패를 손에 넣지 않고 그대로 버리는 것이다. 리치 뒤에는 쯔모기리만 할 수 있다.",
    "등급": "패효율만 본 임시 기준이다. 버린 뒤 샹텐이 최선 후보보다 크면 큰 실수, 샹텐이 같고 유효패가 최선보다 줄지 않았으면 최선, "
          "줄어든 장수가 최선 유효패의 10% 이하면 좋음, 30% 이하면 부정확, 그보다 많으면 실수다. "
          "다른 자리가 리치 중이면 수비, 내가 리치 중이면 리치 중으로 보고 등급을 매기지 않는다.",
    "패효율": "등급을 매긴 차례마다 내 버림의 유효패 장수를 최선 후보의 유효패 장수로 나눠 평균하고 백을 곱한 값이다. "
            "큰 실수 차례는 영으로 친다. 수비 · 리치 중 · 손패 모름 차례는 넣지 않는다.",
}
_RULE_ALIASES = {
    "유효패": "유효패", "대기": "유효패", "ukeire": "유효패", "샹텐": "샹텐", "향청": "샹텐", "shanten": "샹텐",
    "텐파이": "샹텐", "현물": "현물", "genbutsu": "현물", "쯔모기리": "쯔모기리", "tsumogiri": "쯔모기리",
    "등급": "등급", "등급기준": "등급", "grade": "등급", "grades": "등급", "패효율": "패효율", "efficiency": "패효율",
}


class ToolError(Exception):
    pass


def tile_arg(v: Any) -> str | None:
    """패 인자: 코드(`3m` · 적5 `0m`) · 이름(`3만` · `[3만]` · `중`). 모르면 None."""
    if not isinstance(v, str):
        return None
    s = v.strip().strip("[]").strip()
    if len(s) == 2 and s[0] == "0" and s[1] in "mps":
        s = "5" + s[1]
    if s in _TILE_SET:
        return s
    return rv.NAME_TO_CODE.get(s)


def _turn_arg(args: dict) -> int:
    v = args.get("turn")
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if not rv._int(v):
        raise ToolError("turn 인자(몇 번째 버림인가, 정수)가 필요하다")
    return v


def _find_turn(facts: dict, n: int) -> dict:
    turns = facts.get("turns") or []
    for t in turns:
        if t.get("turn") == n:
            return t
    raise ToolError(f"{n}번째 버림은 이 판에 없다(내 버림은 1~{len(turns)}번째)")


def _need_numbers(t: dict) -> None:
    if not t.get("options"):
        raise ToolError(f"{t['turn']}번째 차례는 {STATUS_KO.get(t.get('status'), '손패 모름')} 상태라 후보 숫자가 없다"
                        "(손패를 몰라 다른 패를 버렸을 때를 알 수 없다)")


def _tile_genbutsu(t: dict, code: str) -> bool | None:
    """그 차례에 code 가 리치한 자리의 현물인가. 리치한 자리가 없거나 손패를 모르면 None.
    safe_tiles 는 버리기 전 손(뽑은 패 포함) 안의 현물 — 손에 있는 패면 들어 있나로 가린다."""
    if not t.get("riichi_winds") or t.get("hand") is None:
        return None
    return code in (t.get("safe_tiles") or [])


def _status_view(t: dict, tile: str | None = None) -> dict:
    """what_if · discard_options 에 붙는 차례 상태: status · status_name · can_choose · riichi_winds · (있으면) note.
    숫자는 그대로 두고, 그 숫자를 어떻게 읽어야 하나를 note 로 알린다(f4: 리치 중 차례에 "같았다" 고 답한 것을 막으려고)."""
    s = t["status"]
    winds = " · ".join(t.get("riichi_winds") or [])
    out: dict = {"status": s, "status_name": STATUS_KO.get(s, s), "can_choose": s != "forced",
                 "riichi_winds": t.get("riichi_winds") or []}
    note = None
    if s == "forced":
        note = "리치 중이라 뽑은 패를 그대로 버려야 한다 — 다른 패는 고를 수 없다(숫자는 참고)"
        if winds:
            note += f". 이때 {winds} 자리도 리치 중이었다"
    elif s == "defense":
        note = f"{winds or '다른'} 자리가 리치 중이라 패효율 등급을 매기지 않은 수비 차례다(숫자는 패효율만 본 참고)"
        g = _tile_genbutsu(t, tile) if tile else None
        if g is not None:
            note += f". {with_eun(bracket(tile))} 현물{'이다' if g else '이 아니다'}"
    elif s == "complete":
        note = "버리기 전 손이 이미 화료 형태였다 — 화료하지 않고 버린 차례라 숫자는 참고"
    elif s in NUMBERLESS:
        note = f"{STATUS_KO.get(s, s)} 차례라 숫자를 믿을 수 없다"
    if note:
        out["note"] = note
    return out


def _tiles_view(m: dict) -> list[dict]:
    return [{"tile": k, "left": v} for k, v in (m or {}).items()]


def _live(m: dict) -> set[str]:
    return {k for k, v in (m or {}).items() if v > 0}


def _sorted_codes(codes) -> list[str]:
    return sorted(codes, key=rv._tile_key)


def _option(t: dict, code: str) -> dict | None:
    for o in t.get("options") or []:
        if o.get("discard") == code:
            return o
    return None


def _opt_view(o: dict | None, with_tiles: bool = True) -> dict | None:
    if o is None:
        return None
    out = {"discard": o["discard"], "shanten": o["shanten"], "total": o["total"], "kinds": o["kinds"]}
    if with_tiles:
        out["tiles"] = _tiles_view(o.get("tiles"))
    return out


def _turn_info(facts: dict, args: dict) -> dict:
    t = _find_turn(facts, _turn_arg(args))
    best = t.get("best") or []
    out = {
        "turn": t["turn"], "step": t["step"], "status": t["status"], "status_name": STATUS_KO.get(t["status"]),
        "grade": t["grade"], "grade_name": GRADE_KO.get(t["grade"]) if t["grade"] else None,
        "hand": t["hand"], "drawn": t["drawn"], "discard": t["discard"], "tsumogiri": t["tsumogiri"],
        "melds": t["melds"], "riichi_declared": t["riichi_declared"],
        "shanten_prev": t["shanten_prev"], "shanten_before": t["shanten_before"],
        "shanten_loss": t["shanten_loss"], "ukeire_loss": t["ukeire_loss"],
        "actual": _opt_view(_option(t, t["actual"])) if t.get("actual") else None,
        "best": best, "best_option": _opt_view(_option(t, best[0])) if best else None,
        "riichi_winds": t["riichi_winds"], "genbutsu": t["genbutsu"],
    }
    if not t.get("options"):
        out["note"] = f"{STATUS_KO.get(t['status'], '손패 모름')} 차례라 손패 · 샹텐 · 유효패 숫자가 없다"
    return out


def _discard_options(facts: dict, args: dict) -> dict:
    t = _find_turn(facts, _turn_arg(args))
    _need_numbers(t)
    best = set(t.get("best") or [])
    defense = t["status"] == "defense"

    def row(o: dict) -> dict:
        r = {**_opt_view(o, with_tiles=False), "actual": o["discard"] == t["actual"], "best": o["discard"] in best}
        if defense:  # 수비 차례에만 후보마다 현물인가(리치 중 차례는 고를 수 없으니 넣지 않는다)
            r["genbutsu"] = _tile_genbutsu(t, o["discard"])
        return r

    return {"turn": t["turn"], **_status_view(t), "actual": t["actual"], "best": t.get("best") or [],
            "options": [row(o) for o in t["options"]]}


def _what_if(facts: dict, args: dict) -> dict:
    t = _find_turn(facts, _turn_arg(args))
    _need_numbers(t)
    raw = args.get("tile")
    code = tile_arg(raw)
    if code is None:
        raise ToolError(f"tile 인자(패 코드 3m 또는 이름 3만)가 필요하다: {raw!r}")
    o = _option(t, code)
    if o is None:
        raise ToolError(f"{t['turn']}번째 차례의 손에 없는 패다: {code}")
    a = _option(t, t["actual"])
    same = o["shanten"] == a["shanten"]
    return {
        "turn": t["turn"], "tile": code, **_status_view(t, code), "tile_genbutsu": _tile_genbutsu(t, code),
        "shanten": o["shanten"], "total": o["total"], "kinds": o["kinds"],
        "tiles": _tiles_view(o.get("tiles")),
        "is_actual": code == t["actual"], "is_best": code in (t.get("best") or []),
        "actual": _opt_view(a, with_tiles=False),
        "shanten_diff": o["shanten"] - a["shanten"],
        # 샹텐이 같을 때만 장수 차이를 낸다(샹텐이 다르면 장수 비교는 뜻이 없다 — 해설 규격 A-8 과 같은 까닭).
        "total_diff": o["total"] - a["total"] if same else None,
        "only_this": _sorted_codes(_live(o.get("tiles")) - _live(a.get("tiles"))),
        "only_actual": _sorted_codes(_live(a.get("tiles")) - _live(o.get("tiles"))),
    }


def _safe_tiles(facts: dict, args: dict) -> dict:
    t = _find_turn(facts, _turn_arg(args))
    out = {"turn": t["turn"], "riichi_winds": t["riichi_winds"], "discard": t["discard"], "genbutsu": t["genbutsu"],
           "safe_tiles": t["safe_tiles"], "hand_known": t.get("hand") is not None}
    if not t["riichi_winds"]:
        out["note"] = "이 차례에는 리치한 다른 자리가 없다"
    elif t.get("hand") is None:
        out["note"] = "손패를 몰라 손 안의 현물은 알 수 없다"
    return out


def _turn_line(t: dict) -> str:
    """find_turns 의 한 줄: "6순 [4만] 버림 · 실수 · 유효패 5장 손해"."""
    label = GRADE_KO[t["grade"]] if t.get("grade") else STATUS_KO.get(t["status"], t["status"])
    line = f"{t['turn']}순 {bracket(t['discard'])} 버림 · {label}"
    sl, ul = t.get("shanten_loss"), t.get("ukeire_loss")
    if rv._int(sl) and sl > 0:
        line += f" · 샹텐 {sl} 후퇴"
    elif sl == 0 and rv._int(ul) and ul > 0:
        line += f" · 유효패 {ul}장 손해"
    if t.get("riichi_winds"):
        g = t.get("genbutsu")
        safe = "현물 판단 없음" if g is None else "현물" if g else "현물 아님"
        line += f" · {' · '.join(t['riichi_winds'])} 리치에 {safe}"
    return line


_WIND_EN = {"east": "동", "south": "남", "west": "서", "north": "북", "e": "동", "s": "남", "w": "서", "n": "북"}


def wind_arg(v: Any) -> str | None:
    """자리 바람 인자: `북` · `[북]` · `북 자리` · 패 코드 `4z` · 영어(north). 모르면 None."""
    if not isinstance(v, str):
        return None
    s = v.strip().strip("[]").replace(" ", "").removesuffix("자리").strip("[]")
    if s in WINDS:
        return s
    if s.lower() in _WIND_EN:
        return _WIND_EN[s.lower()]
    code = tile_arg(s)
    if code in ("1z", "2z", "3z", "4z"):
        return WINDS[int(code[0]) - 1]
    return None


def _after_riichi(facts: dict, wind: str) -> tuple[Callable[[dict], bool], dict | None, str | None]:
    """after_riichi 조건 → (차례 거르개, 리치 사건, 알림). 남의 자리는 차례의 riichi_winds 에 그 바람이 있나(리치 뒤 내 버림),
    내 자리는 리치 선언한 버림 다음 차례부터(선언패 자체는 빼고)."""
    turns = facts.get("turns") or []
    if wind == (facts.get("game") or {}).get("hero_wind"):
        declared = next((t["turn"] for t in turns if t.get("riichi_declared")), None)
        if declared is None:
            return (lambda t: False), None, f"{wind} 자리는 내 자리이고, 나는 이 판에서 리치하지 않았다"
        d = next(t for t in turns if t["turn"] == declared)
        event = {"wind": wind, "step": d["step"], "tile": d["discard"], "hero": True, "turn": declared}
        return (lambda t: t["turn"] > declared), event, None
    ev = next((o for o in facts.get("others") or [] if o.get("kind") == "riichi" and o.get("wind") == wind), None)
    event = None if ev is None else {"wind": wind, "step": ev.get("step"), "tile": ev.get("tile"), "hero": False}
    note = None
    if ev is None and not any(wind in (t.get("riichi_winds") or []) for t in turns):
        note = f"{wind} 자리는 이 판에서 리치하지 않았다"
    elif not any(wind in (t.get("riichi_winds") or []) for t in turns):
        note = f"{wind} 자리가 리치한 뒤에는 내 버림이 없다"
    return (lambda t: wind in (t.get("riichi_winds") or [])), event, note


def _find_turns(facts: dict, args: dict) -> dict:
    cond: dict = {}
    if args.get("grade") not in (None, ""):
        g = _GRADE_ARG.get(str(args["grade"]).replace(" ", "").lower())
        if g is None:
            raise ToolError(f"모르는 grade: {args['grade']!r}(쓸 수 있는 값: {', '.join(GRADES)})")
        cond["grade"] = g
    if args.get("status") not in (None, ""):
        s = _STATUS_ARG.get(str(args["status"]).replace(" ", "").lower())
        if s is None:
            raise ToolError(f"모르는 status: {args['status']!r}(쓸 수 있는 값: {', '.join(STATUSES)})")
        cond["status"] = s
    if args.get("min_ukeire_loss") not in (None, ""):
        v = args["min_ukeire_loss"]
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if not rv._int(v):
            raise ToolError("min_ukeire_loss 는 정수")
        cond["min_ukeire_loss"] = v
    if args.get("tile") not in (None, ""):
        code = tile_arg(args["tile"])
        if code is None:
            raise ToolError(f"모르는 tile: {args['tile']!r}")
        cond["tile"] = code
    keep_riichi, event, note = (lambda t: True), None, None
    if args.get("after_riichi") not in (None, ""):
        w = wind_arg(args["after_riichi"])
        if w is None:
            raise ToolError(f"모르는 after_riichi: {args['after_riichi']!r}(자리 바람: {', '.join(WINDS)})")
        cond["after_riichi"] = w
        keep_riichi, event, note = _after_riichi(facts, w)
    # 줄 수 상한은 두지 않는다 — 사실표의 차례가 최대 MAX_TURNS(40)라 판 전체가 한 번에 들어간다(g4: 차례마다 부르다 한도에 닿음).
    rows = []
    for t in facts.get("turns") or []:
        if "grade" in cond and t.get("grade") != cond["grade"]:
            continue
        if "status" in cond and t.get("status") != cond["status"]:
            continue
        if "min_ukeire_loss" in cond and not (rv._int(t.get("ukeire_loss")) and t["ukeire_loss"] >= cond["min_ukeire_loss"]):
            continue
        if "tile" in cond and t.get("discard") != cond["tile"]:
            continue
        if not keep_riichi(t):
            continue
        rows.append({"turn": t["turn"], "discard": t["discard"], "status": t["status"], "grade": t.get("grade"),
                     "tsumogiri": t.get("tsumogiri"), "riichi_winds": t.get("riichi_winds") or [],
                     "genbutsu": t.get("genbutsu"),
                     "shanten_loss": t.get("shanten_loss"), "ukeire_loss": t.get("ukeire_loss"), "line": _turn_line(t)})
    out: dict = {"conditions": cond, "count": len(rows), "turns": rows}
    if "after_riichi" in cond:
        out["riichi"] = event
    if note:
        out["note"] = note
    return out


def _game_summary(facts: dict, args: dict) -> dict:
    g = facts["game"]
    res = g.get("result")
    result = None
    if isinstance(res, dict):
        result = {"type": res.get("type"), "type_name": RESULT_KO.get(res.get("type")), "winner": res.get("winner"),
                  "loser": res.get("loser"), "win_tile": res.get("win_tile"),
                  "hero_won": res.get("winner") is not None and res.get("winner") == g.get("hero_wind")}
    others = []
    for o in facts.get("others") or []:
        name = CALL_KO.get(o.get("call"), "울기") if o.get("kind") == "call" else \
            {"riichi": "리치", "ron": "론", "tsumo": "쯔모"}.get(o.get("kind"), o.get("kind"))
        others.append({**o, "kind_name": name})
    turns = facts.get("turns") or []
    return {
        "round": g.get("round"), "honba": g.get("honba"), "hero_wind": g.get("hero_wind"),
        "dora_indicators": g.get("dora_indicators"), "efficiency_pct": g.get("efficiency_pct"),
        "grades": {GRADE_KO[k]: v for k, v in (g.get("grades") or {}).items() if k in GRADE_KO},
        "statuses": {STATUS_KO[k]: v for k, v in (g.get("statuses") or {}).items() if k in STATUS_KO},
        "hero_discards": len(turns), "result": result, "others": others,
    }


def _rule(facts: dict, args: dict) -> dict:
    name = args.get("name")
    key = _RULE_ALIASES.get(str(name).replace(" ", "").lower()) if isinstance(name, str) else None
    if key is None:
        raise ToolError(f"모르는 용어: {name!r}(쓸 수 있는 이름: {', '.join(RULES)})")
    return {"name": key, "text": RULES[key]}


_TOOL_FNS: dict[str, Callable[[dict, dict], dict]] = {
    "turn_info": _turn_info, "discard_options": _discard_options, "what_if": _what_if, "safe_tiles": _safe_tiles,
    "find_turns": _find_turns, "game_summary": _game_summary, "rule": _rule,
}


def run_tool(facts: dict, name: Any, args: Any) -> dict:
    """도구 하나. 없는 도구 · 없는 차례 · 손에 없는 패 · 숫자 없는 차례 · 모르는 인자는 `{"error": "..."}` (예외 없음)."""
    if name not in _TOOL_FNS:
        return {"error": f"없는 도구: {name!r}(쓸 수 있는 도구: {', '.join(TOOL_NAMES)})"}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"error": "args 는 객체여야 한다"}
    allowed = {a.rstrip("?") for a in TOOLS[name]["args"]}
    # Gemini 스키마는 모든 도구의 인자 칸을 한 객체에 둔다 — 빈 값(null · "")으로 온 남의 칸은 넘어간다.
    extra = [k for k, v in args.items() if k not in allowed and v not in (None, "")]
    if extra:
        return {"error": f"{name} 이 받지 않는 인자: {', '.join(extra)}(받는 인자: {', '.join(sorted(allowed)) or '없음'})"}
    try:
        return _TOOL_FNS[name](facts, {k: v for k, v in args.items() if k in allowed})
    except ToolError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- 검증

_PCT = re.compile(r"(\d+)\s?(?:%|퍼센트)")
_QUESTION_TURN = re.compile(r"(\d+)\s?(?:순(?!위)|번째|차례|턴)")
# 존댓말 끝: ㅂ받침 + 니다(습니다 · 합니다 · 입니다 · 됩니다 …, "아니다" 는 해라체라 통과) · 요 끝.
_POLITE_YO = re.compile(r"(?:해요|에요|예요|어요|아요|세요|네요|까요|지요|죠)(?=[\s.!?,]|$)")
_SHANTEN_KEYS = {"shanten", "shanten_prev", "shanten_before"}


def _polite(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"([가-힣])니다", text):
        c = ord(m.group(1)) - 0xAC00
        if c % 28 == 17:  # 받침 ㅂ
            out.append(m.group(0))
    out += [m.group(0) for m in _POLITE_YO.finditer(text)]
    return out


def evidence_allow(evidence: list[dict]) -> tuple[rv._Allow, set[int]]:
    """이번 답이 근거로 삼을 수 있는 패 · 숫자: evidence(도구 이름 · 인자 · 결과)에 나온 것만.

    단위마다 칸 이름으로 고른다 — 장: total · left · 손패 한 종류 장수 · ukeire_loss(샹텐 손실 0 이고 양수일 때만) ·
    |total_diff| · 종: kinds · 샹텐: shanten · shanten_prev · shanten_before(1 이상) · 순: turn · 위: 후보 목록 자리 ·
    %: efficiency_pct 와 글 속 "N%". 패: 패 코드인 값 전부, 글 속 대괄호 이름."""
    a = rv._Allow()
    pct: set[int] = set()

    def walk(v, key: str | None, parent: dict | None):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k, v)
            if key == "tiles" and all(isinstance(k, str) and k in _TILE_SET for k in v):  # 사실표 꼴 맵
                a.tiles.update(v)
                a.jang.update(x for x in v.values() if rv._int(x))
            return
        if isinstance(v, list):
            if key == "options":
                a.rank.update(range(1, len(v) + 1))
            if key == "hand":
                a.jang.update(Counter(x for x in v if isinstance(x, str)).values())
            for x in v:
                walk(x, key, parent)
            return
        if isinstance(v, str):
            code = tile_arg(v) if len(v) <= 4 else None
            if code is not None and (v in _TILE_SET or key == "tile"):
                a.tiles.add(code)
            for m in rv._BRACKET.finditer(v):
                if m.group(1) in rv.NAME_TO_CODE:
                    a.tiles.add(rv.NAME_TO_CODE[m.group(1)])
            pct.update(int(m.group(1)) for m in _PCT.finditer(v))
            return
        if not rv._int(v):
            return
        if key in ("total", "left"):
            a.jang.add(v)
        elif key == "total_diff":
            a.jang.add(abs(v))
        elif key == "ukeire_loss" and v > 0 and (parent or {}).get("shanten_loss") == 0:
            a.jang.add(v)
        elif key == "kinds":
            a.jong.add(v)
        elif key in _SHANTEN_KEYS and v >= 1:
            a.shanten.add(v)
        elif key == "turn":
            a.jun.add(v)
        elif key == "efficiency_pct":
            pct.add(v)

    for e in evidence:
        walk(e, None, None)
    return a, pct


def question_turns(question: str) -> list[int]:
    return list(dict.fromkeys(int(m.group(1)) for m in _QUESTION_TURN.finditer(question or "")))


def checked_turns(evidence: list[dict]) -> set[int]:
    """차례 도구(turn_info · discard_options · what_if · safe_tiles)로 확인한 차례 번호(오류 결과도 확인한 것으로 친다)."""
    out = set()
    for e in evidence:
        if e.get("tool") in PER_TURN_TOOLS:
            v = (e.get("args") or {}).get("turn")
            if isinstance(v, str) and v.strip().isdigit():
                v = int(v.strip())
            if rv._int(v):
                out.add(v)
    return out


def verify_answer(text: Any, evidence: list[dict], question: str) -> list[str]:
    """답 검증(설계서 4절 verify). 위반마다 사람이 읽는 한 줄, 통과면 빈 목록.

    1 패: 대괄호 `[3만]` 만 패로 보고, evidence 에 나온 패여야 한다. 대괄호 밖 `3만` · `3m` 은 위반.
    2 숫자: `N장 · N종 · N샹텐 · N순(N번째) · N위 · N%` 는 evidence 안의 값(`0샹텐` 은 늘 위반 — 텐파이). `N점` 위반.
    3 영어(대괄호 밖 영문자 2자 이상) · 금지어(반드시 · 무조건 · 절대) · 존댓말(해라체로).
    4 질문이 물은 차례(`N순 · N번째 · N차례`)를 차례 도구로 확인하지 않고 답함.
    5 길이 1~500자 · 문장 5개 이하 · 한글 한 글자 이상."""
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return ["답 없음"]
    allow, pct = evidence_allow(evidence)
    out = rv._check_tiles_numbers(text, allow, "도구로 찾은")
    plain = rv._BRACKET.sub(" ", text)
    for m in _PCT.finditer(plain):
        if int(m.group(1)) not in pct:
            out.append(f"도구로 찾은 사실표에 없는 숫자: {m.group(0).strip()}")
    for w in _polite(plain):
        out.append(f"존댓말(해라체로 써야 함): {w}")
    seen = checked_turns(evidence)
    for n in question_turns(question):
        if n not in seen:
            out.append(f"질문한 {n}번째 차례를 도구로 확인하지 않고 답함")
    if len(text) > ANSWER_MAX_CHARS:
        out.append(f"답 {len(text)}자(1~{ANSWER_MAX_CHARS}자)")
    if (n := rv._sentences(text)) > ANSWER_MAX_SENTENCES:
        out.append(f"답 문장 {n}개({ANSWER_MAX_SENTENCES}개 이하)")
    if not rv._HANGUL.search(text):
        out.append("답에 한글 없음")
    return rv._dedupe(out)


# ---------------------------------------------------------------- 대체 답(LLM 없이 코드가)


def _turn_sentences(info: dict) -> list[str]:
    """turn_info 결과 → 해라체 문장들(앱 review_sentence.dart 3-8 틀과 같은 뜻)."""
    n, d = info["turn"], bracket(info["discard"])
    status = info["status"]
    if status == "forced":
        return [f"{n}순은 리치 중이라 뽑은 패 {with_eul(d)} 그대로 버렸다."]
    if status == "defense":
        winds = " · ".join(info.get("riichi_winds") or []) or "다른"
        g = info.get("genbutsu")
        tail = "현물인지는 판단하지 않았다." if g is None else f"버린 {with_eun(d)} 현물{'이다' if g else '이 아니다'}."
        return [f"{n}순은 {winds} 자리가 리치 중이라 패효율 등급을 매기지 않았다.", tail]
    if status == "complete":
        return [f"{n}순은 버리기 전 손이 화료 형태였고 {with_eul(d)} 버렸다."]
    if status in NUMBERLESS or info.get("actual") is None:
        why = {"unknownHand": "손패 기록이 없어", "inconsistent": "손패 기록이 앞뒤가 맞지 않아",
               "guessed": "손패가 짐작이라"}.get(status, "손패를 몰라")
        return [f"{n}순은 {why} 숫자로 평가하지 않았다.", f"버린 패는 {with_ida(d)}."]
    a = info["actual"]
    grade = info.get("grade_name") or GRADE_KO.get(info.get("grade") or "", "")
    out = [f"{n}순에 {with_eul(d)} 버렸고 패효율 기준 {with_ida(grade)}." if grade else f"{n}순에 {with_eul(d)} 버렸다.",
           f"버린 뒤 {shanten_text(a['shanten'])}이고 {ukeire_word(a['shanten'])} {a['total']}장이다."]
    best = info.get("best") or []
    b = info.get("best_option")
    if b and info["discard"] not in best:
        names = bracket(best[0]) + (f"(또는 {bracket(best[1])})" if len(best) > 1 else "")
        out.append(f"최선은 {names} 버림으로 {shanten_text(b['shanten'])} · {ukeire_word(b['shanten'])} {b['total']}장이다.")
    elif b:
        out.append("최선 후보와 같은 버림이다.")
    return out


def fallback_answer(facts: dict, question: str) -> tuple[str, list[dict]]:
    """질문에서 이 판에 있는 차례 번호를 찾으면 그 차례의 turn_info 를 문장으로, 못 찾으면 판 요약. (글, 쓴 evidence)."""
    for n in question_turns(question):
        info = run_tool(facts, "turn_info", {"turn": n})
        if "error" not in info:
            ev = [{"tool": "turn_info", "args": {"turn": n}, "result": info}]
            return " ".join(["확인한 사실만 적는다.", *_turn_sentences(info)]), ev
    s = run_tool(facts, "game_summary", {})
    ev = [{"tool": "game_summary", "args": {}, "result": s}]
    parts = ["확인할 수 있는 사실로 답하지 못했다.",
             (f"이 판은 {s['round']}, " if s.get("round") else "") + f"내 버림은 {s['hero_discards']}번이다."]
    if s.get("efficiency_pct") is None:
        parts.append("등급을 매긴 버림이 없어 패효율이 없다.")
    else:
        grades = " · ".join(f"{k} {v}번" for k, v in s["grades"].items() if v)
        parts.append(f"패효율은 {s['efficiency_pct']}%이고, 등급은 {with_ida(grades)}.")
    return " ".join(parts), ev


# ---------------------------------------------------------------- 지시문 · 스키마

_ARG_PROPS = {
    "turn": {"type": "integer", "description": "몇 번째 버림인가(1부터)"},
    "tile": {"type": "string", "description": "패 코드(3m) 또는 이름(3만)"},
    "grade": {"type": "string", "description": "best · good · inaccuracy · mistake · blunder"},
    "status": {"type": "string", "description": "graded · forced · defense · complete · unknownHand · inconsistent · guessed"},
    "min_ukeire_loss": {"type": "integer"},
    "after_riichi": {"type": "string", "description": "자리 바람(동 · 남 · 서 · 북) — 그 자리가 리치한 뒤의 내 버림"},
    "name": {"type": "string", "description": "용어 이름"},
}


def action_schema(answer_only: bool = False) -> dict:
    """plan 의 JSON 스키마 응답. 도구 한도를 다 쓰면 answer 만 받는다."""
    if answer_only:
        return {"type": "object",
                "properties": {"action": {"type": "string", "enum": ["answer"]}, "text": {"type": "string"}},
                "required": ["action", "text"]}
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": ["tool", "answer"]},
                           "tool": {"type": "string", "enum": TOOL_NAMES},
                           "args": {"type": "object", "properties": _ARG_PROPS},
                           "text": {"type": "string", "description": "action 이 answer 일 때 답(3~5문장)"}},
            "required": ["action"]}


def _tool_lines() -> str:
    return "\n".join(f"- {n}({', '.join(d['args'])}): {d['desc']}" for n, d in TOOLS.items())


PROMPT_RULES = f"""너는 리치마작 패보 복기 도우미다. 사용자가 자기 패보 한 판을 보며 묻는 말에 답한다.
이 도우미는 **패효율(샹텐 · 유효패)만 평가한다**. 밀기 · 내리기 · 타점 · 역 · 상대 대기 읽기는 판단하지 않는다 — 그런 질문에는
현물 여부처럼 도구로 확인되는 사실까지만 말하고, 나머지는 판단하지 않는다고 밝힌다.

## 도구 — 사실은 도구로만 찾는다
숫자와 패는 프로그램이 미리 계산해 두었다. 너는 계산하지 않고 도구로 찾아 읽는다. 한 번에 도구 하나를 부르고, 결과를 본 뒤 다음을 정한다.
도구는 한 질문에 {Limits.tool_calls}번까지다. 필요한 것만 부른다.
{_tool_lines()}
- 차례 번호(turn)는 내(hero) 몇 번째 버림인가다(1부터). 사용자가 "6순" · "6번째" 라고 하면 turn 6 이다.
- 도구가 {{"error": "..."}} 를 돌려주면 그 내용을 읽고 다른 도구나 다른 인자를 고르거나, 확인할 수 없다고 답한다.
- 질문이 특정 차례를 물으면 그 차례를 turn_info · discard_options · what_if · safe_tiles 중 하나로 꼭 확인한 뒤 답한다.
- "만약 · 버렸으면 · 대신" 을 묻는 질문은 what_if 결과의 상태(status_name · can_choose · note)를 먼저 본다. 리치 중
  (can_choose 가 false)이면 다른 패를 고를 수 없던 차례다 — 숫자를 비교하기 전에 그것부터 말하고, 숫자는 참고로만 쓴다.
- 여러 차례에 걸친 질문("리치 받은 뒤에 어떻게 했어" · "실수한 곳 전부")은 차례마다 도구를 부르지 말고 find_turns 조건
  (after_riichi · status · grade 등)으로 판 전체를 먼저 한 번에 찾는다. 줄마다 현물 여부 · 상태가 있다.

## 답 규칙 — 프로그램이 도구 결과와 대조해 검사한다
- 숫자("N장" · "N종" · "N샹텐" · "N순" · "N번째" · "N%")는 이번에 부른 도구 결과에 있는 값만 쓴다. 더하거나 빼거나 나눠 새 숫자를 만들지 않는다.
  유효패 장수 차이는 what_if 의 total_diff 나 ukeire_loss(같은 샹텐일 때)만 쓴다. 샹텐 0 은 "텐파이", -1 은 "화료 형태"("0샹텐" 금지).
- 패는 대괄호 하나에 한 장씩 이름으로: [3만] · [5통] · [7삭] · [중]. 이름: {rv._NAME_TABLE}
  대괄호 밖에 "3만" · "3m" 처럼 쓰지 않는다. 도구 결과에 나온 패만 쓴다.
- 영어를 쓰지 않는다(도구의 칸 이름 best · shanten · ukeire 등을 옮기지 않는다). 등급은 최선 · 좋음 · 부정확 · 실수 · 큰 실수,
  상태는 등급 매김 · 리치 중 · 수비 · 화료 형태 · 손패 기록 없음 · 손패 기록 어긋남 · 짐작 손패.
- 해라체("…다")로 쓴다. 존댓말("…습니다" · "…해요")을 쓰지 않는다. "반드시" · "무조건" · "절대" 를 쓰지 않는다. 점수("N점")를 쓰지 않는다.
- 3~5문장, 500자 안. 숫자를 늘어놓지 말고 질문에 필요한 것만.
- 손패를 모르는 차례(손패 기록 없음 · 어긋남 · 짐작 손패)는 숫자가 없다 — 모른다고 답한다.
- 등급은 패효율만 본 것이라 수비 · 타점 때문에 일부러 고른 수일 수 있다. "실수" 라고 단정하기보다 "패효율 기준" 임을 밝힌다.
- 모양(양면 · 간짱 · 샤보 …)과 대기는 도구 결과에 없으니 단정하지 않는다. 손패 · 유효패 목록으로 말할 수 있는 것만 말한다.

## 응답 형식(JSON 하나)
- 도구를 부를 때: {{"action": "tool", "tool": "turn_info", "args": {{"turn": 6}}}}
- 답할 때: {{"action": "answer", "text": "…"}}
"""


def _render_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            lines.append(f"- 도구 {m['tool']} {json.dumps(m.get('args') or {}, ensure_ascii=False)}")
            lines.append(f"  결과: {json.dumps(m['result'], ensure_ascii=False, separators=(',', ':'))}")
        elif role == "answer":
            lines.append(f"- 내 답: {json.dumps(m.get('text'), ensure_ascii=False)}")
        elif role == "check":
            lines.append("- 검사에 걸림(이 위반을 고쳐 다시 답한다. 필요하면 도구를 더 불러도 된다):")
            lines += [f"  - 위반: {v}" for v in m["violations"]]
        elif role == "note":
            lines.append(f"- 알림: {m['text']}")
    return "\n".join(lines)


def build_prompt(state: dict, limits: Limits, answer_only: bool) -> str:
    facts = state["facts"]
    g = facts.get("game") or {}
    n_turns = len(facts.get("turns") or [])
    parts = [PROMPT_RULES,
             f"## 이 판\n- {g.get('round') or '국 모름'} · 내 자리 바람 {g.get('hero_wind')} · 내 버림 차례 번호 1~{n_turns}"]
    hist = state.get("history") or []
    if hist:
        parts.append("## 앞선 질문과 답(흐름 참고만 — 숫자 · 패의 근거로 쓰지 않는다)\n"
                     + "\n".join(f"- 질문: {h['q']}\n  답: {h['a']}" for h in hist))
    parts.append(f"## 질문\n{state['question']}")
    used = len(state.get("tool_calls") or [])
    parts.append(f"## 지금까지(도구 {used}/{limits.tool_calls}회)\n" + (_render_messages(state.get("messages") or [])
                                                                     or "- 아직 없음"))
    if answer_only:
        parts.append("## 다음\n도구는 더 부를 수 없다. 지금까지의 도구 결과만으로 답한다(action \"answer\").")
    else:
        parts.append("## 다음\n도구를 하나 부르거나(action \"tool\"), 충분하면 답한다(action \"answer\"). JSON 하나로.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 그래프


class AgentState(TypedDict, total=False):
    question: str
    history: list
    facts: dict
    messages: list          # LLM 과 주고받은 줄(도구 결과 · 답 · 검사 위반 · 알림)
    evidence: list          # 도구 결과 [{tool, args, result}] — 검증의 근거
    tool_calls: list        # [{tool, args}] — 횟수 = 길이
    retries: int
    answer: str | None
    violations: list
    outcome: str | None     # verified | fallback
    llm_calls: int
    tokens_in: int
    tokens_out: int
    pending: dict | None    # plan 이 고른 도구 호출
    route: str              # plan · verify 가 정한 다음 칸
    fallback_reason: str | None


def build_graph(generate: Generate, limits: Limits = Limits(), clock: Callable[[], float] = time.monotonic,
                deadline: float | None = None):
    """그래프를 만든다(질문마다 — generate · 시계 · 기한을 닫아 넣는다)."""

    def plan(state: AgentState) -> dict:
        if deadline is not None and clock() >= deadline:
            return {"route": "fallback", "fallback_reason": f"시간 한도({limits.seconds:g}초)를 넘김"}
        if state["llm_calls"] >= limits.llm_calls:
            return {"route": "fallback", "fallback_reason": f"LLM 호출 한도({limits.llm_calls}회)를 넘김"}
        answer_only = len(state["tool_calls"]) >= limits.tool_calls
        data, t_in, t_out = generate(build_prompt(state, limits, answer_only), action_schema(answer_only))
        upd: dict = {"llm_calls": state["llm_calls"] + 1, "tokens_in": state["tokens_in"] + t_in,
                     "tokens_out": state["tokens_out"] + t_out}
        msgs = list(state["messages"])
        action = data.get("action") if isinstance(data, dict) else None
        if action == "answer":
            text = data.get("text")
            text = text.strip() if isinstance(text, str) else ""
            msgs.append({"role": "answer", "text": text})
            return {**upd, "messages": msgs, "answer": text, "route": "verify"}
        if action == "tool":
            if answer_only:
                return {**upd, "route": "fallback",
                        "fallback_reason": f"도구 한도({limits.tool_calls}회)를 다 쓴 뒤에도 도구를 요청함"}
            return {**upd, "pending": {"tool": data.get("tool"), "args": data.get("args") or {}}, "route": "act"}
        msgs.append({"role": "note", "text": 'JSON 형식이 맞지 않았다. {"action": "tool", ...} 또는 {"action": "answer", ...} 로 답한다.'})
        return {**upd, "messages": msgs, "route": "plan"}

    def act(state: AgentState) -> dict:
        call = state["pending"] or {}
        tool, args = call.get("tool"), call.get("args")
        result = run_tool(state["facts"], tool, args)
        args_json = args if isinstance(args, dict) else {"_": args}
        entry = {"tool": tool, "args": args_json, "result": result}
        return {"evidence": [*state["evidence"], entry],
                "tool_calls": [*state["tool_calls"], {"tool": tool, "args": args_json}],
                "messages": [*state["messages"], {"role": "tool", **entry}],
                "pending": None, "route": "plan"}

    def verify(state: AgentState) -> dict:
        v = verify_answer(state.get("answer"), state["evidence"], state["question"])
        if not v:
            return {"violations": [], "outcome": "verified", "route": "end"}
        if state["retries"] < limits.retries:
            return {"violations": v, "retries": state["retries"] + 1, "route": "plan",
                    "messages": [*state["messages"], {"role": "check", "violations": v}]}
        return {"violations": v, "route": "fallback"}

    def fallback(state: AgentState) -> dict:
        text, ev = fallback_answer(state["facts"], state["question"])
        reason = state.get("fallback_reason")
        return {"answer": text, "outcome": "fallback", "evidence": [*state["evidence"], *ev],
                "violations": [*(state.get("violations") or []), *([reason] if reason else [])]}

    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("verify", verify)
    g.add_node("fallback", fallback)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: s["route"], {"act": "act", "verify": "verify", "plan": "plan",
                                                           "fallback": "fallback"})
    g.add_edge("act", "plan")
    g.add_conditional_edges("verify", lambda s: s["route"], {"end": END, "plan": "plan", "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


def run(req: dict, generate: Generate, limits: Limits = Limits(),
        clock: Callable[[], float] = time.monotonic) -> tuple[dict, float]:
    """검사를 거친 요청 → (그래프 끝 상태, 걸린 초). generate 가 던진 예외는 그대로 올린다(라우트가 502/429 로)."""
    t0 = clock()
    graph = build_graph(generate, limits, clock, t0 + limits.seconds)
    init: AgentState = {
        "question": req["question"].strip(), "history": list(req.get("history") or []), "facts": req["facts"],
        "messages": [], "evidence": [], "tool_calls": [], "retries": 0, "answer": None, "violations": [],
        "outcome": None, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "pending": None, "route": "plan",
        "fallback_reason": None,
    }
    # 걸음 상한: LLM 8회 × (plan + act/verify) + fallback 이 넉넉히 들어간다(넘으면 GraphRecursionError — 한도 고장).
    final = graph.invoke(init, config={"recursion_limit": 4 * limits.llm_calls + 8})
    return final, clock() - t0


def ask(req: dict, generate: Generate, model: str = "", limits: Limits = Limits(),
        clock: Callable[[], float] = time.monotonic) -> dict:
    """설계서 5절 응답."""
    s, elapsed = run(req, generate, limits, clock)
    return to_response(s, elapsed, model)


def to_response(s: dict, elapsed: float, model: str = "") -> dict:
    """그래프 끝 상태 → 응답. `retries`(검증 뒤 다시 받은 횟수)는 설계서 5절에 더한 칸 — 첫 통과 · 재시도 뒤 · 대체 비율을 잰다."""
    return {
        "answer": s["answer"], "outcome": s["outcome"], "tool_calls": s["tool_calls"], "violations": s["violations"],
        "retries": s["retries"], "prompt_version": PROMPT_VERSION, "model": model, "elapsed_s": round(elapsed, 3),
        "usage": {"input_tokens": s["tokens_in"], "output_tokens": s["tokens_out"], "llm_calls": s["llm_calls"]},
    }
