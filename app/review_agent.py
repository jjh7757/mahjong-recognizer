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

import copy
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

# a4(11-22, mahjong_calc docs/paifu_review_agent_defense_design_1122.md "a4"): 묻지 않은 "판정하지 않는다" 를 줄임(F2) ·
# 도구마다 따로인 스키마(F3) · 판정 말 검사의 부정문 오검출(F4) · 시도마다의 위반(F5).
# a5(11-22 아침, 같은 설계서 "a5"): what_if risk 에 실제 버린 패 · 가장 안전한 패마다의 단계(G1) · 자리 바람을 패 대괄호로
# 쓰지 않게(G2) · 판정 말 검사의 부정 문구에 "판정 정보는 없다" 꼴(G3).
# a6(11-23): 답의 말투 — stance 는 단정 사실문 하나로("사실상 밀었다" · "…로 기록되었다" 금지) · 평가 꾸밈말 금지(지시문 +
# 검사 8) · "판정하지 않는다" 는 남 리치가 있는 밀기 · 접기 질문에만(지시문 + 검사 9, 리치 없는 패효율 질문에 붙이던 것).
# a7(11-23 밤, mahjong_calc docs/review_agent_foldspot_scores_design_1123.md 3절): 사실표 3 의 "보통 접는 자리"(fold_spot 객체)와
# 국 시작 점수(game.scores)를 사실로 말함 — 판정 말은 fold_spot 이 있어도 위반(검사 7), "접는 자리" 말은 fold_spot 사실이 있을
# 때만(검사 10), "N점" 은 scores 의 score · gap 값만(검사 11).
# a8(11-23 밤, 같은 설계서 8-1절): a7 측정(표 15-12)의 흠 넷 — 묻지 않은 "접는 자리 판정이 없다"(검사 12) · 묻지 않은
# "점수 기록이 없다"(game_summary 알림 뺌 + 검사 13) · find_turns 결과의 "AI" 영어 재시도(fold_spot 객체면 허용) · 안전한 순
# 자리를 위험 순위처럼 씀(검사 14).
PROMPT_VERSION = "a8"
# 사실표 판: 1(설계서 1121) · 2(설계서 1122 — 차례마다 defense) · 3(설계서 1123 — fold_spot 객체 · game.scores).
# 1 은 defense 가 없는 것으로, 2 의 fold_spot(null · bool)은 null 로 본다(판정 말을 허락하지 않게).
FACTS_VERSION = 3
FACTS_VERSIONS = (1, 2, 3)

# 방총 위험 다섯 단계(안전한 순). 앱 lib/paifu/agent_facts.dart `riskLevels` 와 같은 말.
LEVELS = ["안전", "낮음", "보통", "높음", "매우 높음"]
LEVEL_INDEX = {x: i for i, x in enumerate(LEVELS)}
STANCES = ["pushed", "folded", "between"]
STANCE_KO = {"pushed": "밀었다", "folded": "접었다", "between": "어느 쪽도 아님"}
DEFENSE_KEYS = ("threat_winds", "tiles", "actual", "safest", "efficient", "stance", "fold_spot")
SCORES_BASIS = "국 시작 점수(이 국의 리치봉 · 결과 반영 전)"

# 요청 상한(설계서 5절). 사실표는 예시 11판에서 압축 2.3 ~ 33.2KB(11-22 v2, 엔진 켬).
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
    version = f.get("version")
    need(rv._int(version) and version in FACTS_VERSIONS, "facts.version",
         f"{' 또는 '.join(str(v) for v in FACTS_VERSIONS)} 이어야 함")
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
    if version >= 3:
        need("scores" in g, "facts.game", "scores 없음(사실표 3)")
        _validate_scores(g)

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
        if version >= 2:
            need("defense" in t, w, "defense 없음(사실표 2)")
            _validate_defense(t, w, version)

    others = f["others"]
    need(isinstance(others, list) and len(others) <= MAX_OTHERS, "facts.others", f"목록(최대 {MAX_OTHERS})")
    for i, o in enumerate(others):
        need(isinstance(o, dict) and int_in(o.get("step"), 1, 9999) and isinstance(o.get("kind"), str)
             and (o.get("wind") is None or o.get("wind") in WINDS), f"facts.others[{i}]", "{step, wind, kind}")


FOLD_SHANTEN = (3, 8)        # fold_spot.shanten — 규칙 R3 은 버리기 전 3샹텐 이상
FOLD_PCT = (1, 100)          # fold_spot.ai_fold_pct
SCORE_ABS_MAX = 1_000_000    # 점수 상한(값 검사용 — 실제 판은 한참 작다)
KYOTAKU_MAX = 99


def _validate_fold_spot(v: Any, dw: str, version: int) -> None:
    """판 2: null 또는 참 · 거짓(도구는 null 로 다룸). 판 3: null 또는 {shanten(3~8 정수), ai_fold_pct(1~100 정수)}."""
    need, int_in = rv._need, rv._int_in
    if version < 3:
        need(v is None or isinstance(v, bool), dw, "fold_spot 은 null 또는 참 · 거짓")
        return
    need(v is None or (isinstance(v, dict) and int_in(v.get("shanten"), *FOLD_SHANTEN)
                       and int_in(v.get("ai_fold_pct"), *FOLD_PCT)),
         dw, f"fold_spot 은 null 또는 {{shanten({FOLD_SHANTEN[0]}~{FOLD_SHANTEN[1]} 정수), "
             f"ai_fold_pct({FOLD_PCT[0]}~{FOLD_PCT[1]} 정수)}}")


def _score_ok(v: Any) -> bool:
    return rv._int_in(v, -SCORE_ABS_MAX, SCORE_ABS_MAX) and v % 100 == 0


def _validate_scores(g: dict) -> None:
    """사실표 3 의 `game.scores`(설계서 1123 2-2): null 또는
    {hero: {wind, score, rank}, others: [{wind, score, rank, gap, ahead}] × 3(동 · 남 · 서 · 북 순), kyotaku: null | 0 이상 정수}.
    점수는 100 의 배수 정수, rank 는 1 + (나보다 점수가 높은 자리 수), gap = |그 자리 − 나|, ahead = 그 자리가 높은가."""
    need, int_in = rv._need, rv._int_in
    s = g["scores"]
    if s is None:
        return
    w = "facts.game.scores"
    need(isinstance(s, dict) and all(k in s for k in ("hero", "others", "kyotaku")), w, "객체(hero · others · kyotaku)")
    hero, others = s["hero"], s["others"]
    need(isinstance(hero, dict) and hero.get("wind") in WINDS and _score_ok(hero.get("score"))
         and int_in(hero.get("rank"), 1, 4), f"{w}.hero", "{wind, score(100 의 배수 정수), rank(1~4)}")
    need(hero["wind"] == g["hero_wind"], f"{w}.hero", "wind 는 game.hero_wind 와 같아야 함")
    need(isinstance(others, list) and len(others) == 3, f"{w}.others", "3개 목록")
    for i, o in enumerate(others):
        need(isinstance(o, dict) and o.get("wind") in WINDS and _score_ok(o.get("score")) and int_in(o.get("rank"), 1, 4)
             and int_in(o.get("gap"), 0, 2 * SCORE_ABS_MAX) and isinstance(o.get("ahead"), bool), f"{w}.others[{i}]",
             "{wind, score(100 의 배수 정수), rank(1~4), gap(0 이상 정수), ahead(참 · 거짓)}")
    winds = [hero["wind"], *(o["wind"] for o in others)]
    need(len(set(winds)) == 4, w, "wind 넷이 겹침")
    need([o["wind"] for o in others] == [x for x in WINDS if x != hero["wind"]], f"{w}.others", "동 · 남 · 서 · 북 순")
    k = s["kyotaku"]
    need(k is None or int_in(k, 0, KYOTAKU_MAX), w, f"kyotaku 는 null 또는 0~{KYOTAKU_MAX} 정수")
    # 값끼리 맞나(앱이 계산한 값 — 에이전트는 이 값을 그대로 말하므로 어긋난 표는 받지 않는다).
    pts = [hero["score"], *(o["score"] for o in others)]
    ranks = [hero["rank"], *(o["rank"] for o in others)]
    need(all(r == 1 + sum(p > x for p in pts) for r, x in zip(ranks, pts)), w,
         "rank 는 1 + (점수가 더 높은 자리 수) — 같은 점수면 같은 순위")
    for i, o in enumerate(others):
        need(o["gap"] == abs(o["score"] - hero["score"]) and o["ahead"] == (o["score"] > hero["score"]),
             f"{w}.others[{i}]", "gap = |그 자리 점수 − 내 점수|, ahead = 그 자리가 나보다 높은가")


def _validate_defense(t: dict, w: str, version: int = 2) -> None:
    """사실표 2 · 3 의 `defense` 꼴(설계서 1122 2절 · 1123 2-1). null 이거나, 리치한 자리가 있고 손패를 아는 차례의 객체."""
    need, tile_ok, int_in = rv._need, rv._tile_ok, rv._int_in
    d = t["defense"]
    if d is None:
        return
    dw = f"{w}.defense"
    need(isinstance(d, dict) and all(k in d for k in DEFENSE_KEYS), dw, f"객체({' · '.join(DEFENSE_KEYS)})")
    need(bool(t["riichi_winds"]) and t["hand"] is not None, dw, "리치한 자리가 있고 손패를 아는 차례에만")
    tw = d["threat_winds"]
    need(isinstance(tw, list) and len(tw) <= 3 and all(x in WINDS for x in tw), dw, "threat_winds")
    tiles = d["tiles"]
    need(isinstance(tiles, list) and 1 <= len(tiles) <= 14, dw, "tiles 는 1~14개 목록")
    level_of: dict[str, str] = {}
    for j, x in enumerate(tiles):
        need(isinstance(x, dict) and tile_ok(x.get("tile")) and x.get("level") in LEVEL_INDEX
             and isinstance(x.get("genbutsu"), bool), f"{dw}.tiles[{j}]", "{tile, level, genbutsu}")
        need(x["tile"] not in level_of, f"{dw}.tiles[{j}]", "같은 패가 두 번")
        level_of[x["tile"]] = x["level"]
    need(set(level_of) == set(t["hand"]), dw, "tiles 는 손의 패 종류 전부")
    a = d["actual"]
    need(isinstance(a, dict) and a.get("tile") == t["discard"] and a.get("level") == level_of.get(a.get("tile"))
         and a.get("of") == len(tiles) and int_in(a.get("rank"), 1, len(tiles)), dw,
         "actual = {tile(버린 패), level(tiles 와 같음), rank(1~of), of(tiles 수)}")
    s = d["safest"]
    need(rv._tiles_ok(s, 14) and len(s) >= 1 and set(s) <= set(level_of), dw, "safest 는 tiles 의 패")
    e = d["efficient"]
    need(e is None or (isinstance(e, dict) and e.get("tile") in level_of and e.get("level") == level_of[e["tile"]]),
         dw, "efficient = null 또는 {tile(손의 패), level}")
    # 내가 리치 중(forced)인 차례는 고를 수 없었다 — stance · efficient 는 null(11-22 보완), 그 밖의 차례는 stance 가 있다.
    forced = t["status"] == "forced"
    need(d["stance"] is None if forced else d["stance"] in STANCES, dw,
         "stance 는 리치 중 차례면 null, 그 밖은 " + " · ".join(STANCES))
    need(not forced or e is None, dw, "리치 중 차례의 efficient 는 null")
    _validate_fold_spot(d["fold_spot"], dw, version)


# ---------------------------------------------------------------- 도구(사실표에서 찾기만)

TOOLS: dict[str, dict] = {
    "turn_info": {"args": ["turn"], "desc": "그 차례의 손패 · 뽑은 패 · 버린 패 · 상태 · 등급 · 샹텐 · 유효패(실제와 최선)"},
    "discard_options": {"args": ["turn"],
                        "desc": "그 차례 버림 후보 전부(버릴 패 · 버린 뒤 샹텐 · 유효패 장수 · 종류 수), 실제 · 최선 표시, "
                                "그 차례 상태 · 고를 수 있었나(can_choose) · 알림(note)"},
    "what_if": {"args": ["turn", "tile"],
                "desc": "그 차례에 tile 을 버렸다면: 샹텐 · 유효패 목록(패마다 남은 장수), 실제 버림과의 차이(한쪽에만 있는 유효패), "
                        "그 차례 상태 · 고를 수 있었나(can_choose, 리치 중이면 false — 숫자는 참고) · 리치한 자리가 있으면 그 패가 현물인가와 "
                        "방총 위험(risk: 그 패의 단계 · 안전한 순 자리 rank / of · 실제 버린 패의 단계와 자리(actual) · "
                        "가장 안전한 패마다의 단계(safest))"},
    "safe_tiles": {"args": ["turn"], "desc": "그 차례에 리치한 다른 자리 · 버린 패가 현물인가 · 손 안의 현물"},
    "defense_info": {"args": ["turn"],
                     "desc": "리치한 자리가 있는 그 차례의 방총 위험: 손의 패마다 위험 단계(안전한 순) · 버린 패의 단계와 안전한 순 자리"
                             "(rank / of, 1 = 가장 안전) · 가장 안전한 패 · 효율대로 버릴 패와 그 단계 · stance(버린 패가 가장 안전한 패보다 "
                             "얼마나 위험했나의 사실) · fold_spot(보통 접는 자리 사실 — 내 손 샹텐 shanten · 강한 AI 가 이런 자리에서 "
                             "접는 비율 ai_fold_pct, null 이면 접는 자리 판정이 없다) · 상태 · 알림"},
    "find_turns": {"args": ["grade?", "status?", "min_ukeire_loss?", "tile?", "after_riichi?", "stance?", "min_level?",
                            "fold_spot?"],
                   "desc": "조건에 맞는 내 버림 차례를 판 전체에서 한 번에: 줄마다 버린 패 · 상태 · 등급 · 쯔모기리 · 리치한 자리 · "
                           "현물인가 · 버린 패의 위험 단계(level) · stance · fold_spot(보통 접는 자리면 줄 끝에 '보통 접는 자리'). "
                           "after_riichi 에 자리 바람(동 · 남 · 서 · 북)을 주면 그 자리가 "
                           "리치한 뒤의 내 버림, stance(pushed · folded · between)는 그 사실인 차례(리치 중 차례는 stance 없음), min_level(안전 · 낮음 · 보통 · 높음 · "
                           "매우 높음)은 그 단계 이상 위험한 패를 버린 차례, fold_spot 이 true 면 보통 접는 자리인 차례"
                           "(조건은 모두 그리고, 없으면 전부)"},
    "game_summary": {"args": [], "desc": "판 요약: 국 · 패효율 · 등급 개수 · 상태 개수 · 결과 · 남의 리치 · 울기 · 화료 · "
                                         "국 시작 점수(scores: 내 점수 · 순위, 다른 자리마다 점수 · 순위 · 나와의 점수 차 gap · "
                                         "나보다 높은가 ahead, 공탁 리치봉 kyotaku — null 이면 점수 기록이 없다)"},
    "rule": {"args": ["name"], "desc": "용어 풀이: 유효패 · 샹텐 · 현물 · 쯔모기리 · 등급 · 패효율 · 방총 위험 · 밀기 접기 · 점수"},
}
TOOL_NAMES = list(TOOLS)
PER_TURN_TOOLS = {"turn_info", "discard_options", "what_if", "safe_tiles", "defense_info"}

# rule 도구의 글. 숫자는 등급 문턱(10 · 30%)만 — 지시문과 앱(paifu_review.dart ReviewThresholds)의 임시 값.
RULES: dict[str, str] = {
    "유효패": "버린 뒤의 손에서 샹텐을 하나 줄여 주는 패다. 장수는 그 패가 아직 몇 장 남았나를 더한 것이다"
            "(한 종류 넉 장에서 내 손 · 네 자리 강 · 울기 · 도라 표시패로 보이는 장수를 뺀다). 텐파이에서는 대기라고 부른다.",
    "샹텐": "텐파이까지 패를 몇 번 더 바꿔야 하나를 센 수다. 영은 텐파이, 마이너스 일은 화료 형태다. "
          "버린 뒤 샹텐이 최선 후보보다 크면 샹텐이 뒤로 갔다고 한다.",
    "현물": "리치한 자리가 이미 버린 패, 그리고 리치 뒤 다른 자리가 버렸는데 그 자리가 론하지 않은 패다. 그 자리에는 론 당하지 않는다. "
          "스지 · 벽 같은 것은 따로 풀이하지 않는다 — 현물이 아닌 패의 위험 크기는 '방총 위험' 의 다섯 단계로만 말한다.",
    "쯔모기리": "뽑은 패를 손에 넣지 않고 그대로 버리는 것이다. 리치 뒤에는 쯔모기리만 할 수 있다.",
    "등급": "패효율만 본 임시 기준이다. 버린 뒤 샹텐이 최선 후보보다 크면 큰 실수, 샹텐이 같고 유효패가 최선보다 줄지 않았으면 최선, "
          "줄어든 장수가 최선 유효패의 10% 이하면 좋음, 30% 이하면 부정확, 그보다 많으면 실수다. "
          "다른 자리가 리치 중이면 수비, 내가 리치 중이면 리치 중으로 보고 등급을 매기지 않는다.",
    "패효율": "등급을 매긴 차례마다 내 버림의 유효패 장수를 최선 후보의 유효패 장수로 나눠 평균하고 백을 곱한 값이다. "
            "큰 실수 차례는 영으로 친다. 수비 · 리치 중 · 손패 모름 차례는 넣지 않는다.",
    # 관측 비율: 설계서 1122 1절(metrics 표 16-11) — 천봉 개발 200판, 남이 리치한 차례의 내 버림. 정수로만 적는다.
    "방총 위험": "리치한 자리가 있는 차례에, 손의 패마다 엔진이 어림한 방총 위험을 다섯 단계로 나눈 것이다. "
             "'안전' 은 엔진이 모든 상대에게 안전패로 보는 패이고, 그 밖은 엔진 값이 작은 순으로 '낮음' · '보통' · '높음' · "
             "'매우 높음' 이다. 엔진의 어림이라 확률로 읽지 않는다. 천봉 고단자 판에서 리치한 상대가 있을 때 잰 실제 방총은 "
             "'안전' 2,499번 중 0번, '낮음' 1% 미만, '보통' 약 3%, '높음' 약 8%, '매우 높음' 약 15% 였다. "
             "리치하지 않은 상대는 보지 않는다 — 리치한 자리가 없는 차례에는 단계가 없다. 다만 리치한 자리가 있는 차례에 울어서 "
             "텐파이가 짙은 자리가 있으면 그 자리도 엔진 어림에 섞인다.",
    "밀기 접기": "stance 는 버린 패가 그 차례 손에서 가장 안전한 패보다 얼마나 위험했나를 나눈 사실이다(판정이 아니다). "
             # a4 F1: folded 는 같은 구간일 때만. 단계 말에 작은따옴표를 달지 않는다 — 달면 이 글을 읽은 답에서 모든 단계가
             # 근거가 된다(evidence_levels).
             "엔진 위험의 차이가 크면 밀었다(pushed), 차이가 거의 없고 두 패의 엔진 위험이 같은 구간(엔진 값으로 나눈 "
             "낮음 · 보통 · 높음 · 매우 높음 넷, 안전 표시는 따로 보지 않음)이면 접었다(folded), 그 밖은 어느 쪽도 "
             "아니다(between). 차이가 작아도 버린 패가 한 구간 더 위험하면 접었다고 하지 않는다. "
             # a7(설계서 1123 3절): 보통 접는 자리의 뜻 · 94% 의 출처 · 판정이 아님. 94 는 앱 pushFoldAiFoldPct(metrics 표 16-15).
             "보통 접는 자리(fold_spot)는 리치를 받았고, 버리기 전 내 손이 3샹텐 이상이며, 효율대로 버릴 패가 가장 안전한 패보다 "
             "뚜렷이 위험한 차례다. 천봉 고수 판에서 강한 AI 는 이런 자리의 약 94% 에서 접었다. 판정이 아니라 참고다 — "
             "이 도우미는 밀지 접을지의 옳고 그름을 판정하지 않는다. fold_spot 이 null 인 차례는 접는 자리 판정이 없는 것이고, "
             "접는 자리가 아니라는 뜻이 아니다.",
    "점수": "점수는 이 국을 시작할 때의 점수다(이 국의 리치봉 · 결과 반영 전). 순위는 그 점수로 센 것이고, 점수가 같으면 같은 "
          "순위다. 점수 차는 그 자리 점수와 내 점수의 차이, 공탁은 국 시작 때 걸려 있던 리치봉 수다. 이 도우미는 역전에 필요한 "
          "타점 · 순위 전망을 계산하지 않고, 점수로 밀지 접을지를 판단하지 않는다.",
}
_RULE_ALIASES = {
    "유효패": "유효패", "대기": "유효패", "ukeire": "유효패", "샹텐": "샹텐", "향청": "샹텐", "shanten": "샹텐",
    "텐파이": "샹텐", "현물": "현물", "genbutsu": "현물", "쯔모기리": "쯔모기리", "tsumogiri": "쯔모기리",
    "등급": "등급", "등급기준": "등급", "grade": "등급", "grades": "등급", "패효율": "패효율", "efficiency": "패효율",
    "방총위험": "방총 위험", "방총": "방총 위험", "위험": "방총 위험", "위험단계": "방총 위험", "단계": "방총 위험",
    "risk": "방총 위험", "level": "방총 위험", "밀기접기": "밀기 접기", "밀기": "밀기 접기", "접기": "밀기 접기",
    "밀기·접기": "밀기 접기", "stance": "밀기 접기", "pushfold": "밀기 접기", "fold_spot": "밀기 접기",
    "접는자리": "밀기 접기", "보통접는자리": "밀기 접기", "foldspot": "밀기 접기",
    "점수": "점수", "점수상황": "점수", "순위": "점수", "등수": "점수", "점수차": "점수", "공탁": "점수", "리치봉": "점수",
    "scores": "점수", "score": "점수", "gap": "점수", "kyotaku": "점수",
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


def _defense_of(facts: dict, t: dict) -> dict | None:
    """그 차례의 defense 칸. 사실표 1 은 defense 가 없는 것으로 본다. fold_spot 은 객체(사실표 3)일 때만 남기고, 그 밖
    (판 2 의 참 · 거짓 등)은 null 로 바꾼 사본을 돌려준다 — 판 2 의 참으로 판정 · 접는 자리 말이 허락되지 않게."""
    if facts.get("version") not in (2, 3):
        return None
    d = t.get("defense")
    if not isinstance(d, dict):
        return None
    if d.get("fold_spot") is not None and not isinstance(d.get("fold_spot"), dict):
        d = {**d, "fold_spot": None}
    return d


def _fold_spot_of(d: dict | None) -> dict | None:
    """defense 칸([_defense_of] 를 거친 것)의 fold_spot 객체 또는 None."""
    fs = d.get("fold_spot") if d else None
    return fs if isinstance(fs, dict) else None


def fold_spot_note(fs: dict) -> str:
    """fold_spot 객체 → defense_info 알림(설계서 1123 3절 글 그대로, 숫자만 사실표 값)."""
    return (f"보통 접는 자리다(리치를 받았고 내 손이 {fs['shanten']}샹텐, 효율대로 버릴 패가 가장 안전한 패보다 뚜렷이 위험) — "
            f"강한 AI 는 이런 자리의 약 {fs['ai_fold_pct']}% 에서 접는다. 판정이 아니라 참고다")


def _risk_of_tile(d: dict, code: str) -> dict | None:
    """defense 칸에서 패 하나의 단계 · 안전한 순 자리(찾기만, 계산 없음). what_if 의 risk 로만 쓴다.
    rank: 버린 패는 사실표의 actual.rank(같은 위험이면 같은 자리), 가장 안전한 패(safest)는 1, 그 밖은 tiles 줄의 자리
    (사실표에 % 가 없어 같은 위험끼리의 동률은 알 수 없다 — 줄은 위험 오름차순 · 같으면 패 번호 순).
    a5 G1: 실제 버린 패의 단계(actual — 사실표 defense.actual 그대로)와 가장 안전한 패마다의 단계(safest 를 [{tile, level}],
    단계는 tiles 줄에서 찾기만)를 같이 싣는다. 물어본 패만 단계가 있고 safest 가 코드 목록이던 a4 에서는, 같은 목록에 든
    실제 버린 패를 같은 단계로 짐작했다(du2b: '낮음' 인 [2만]을 '안전' 이라 함 — 설계서 1122 "a5")."""
    tiles = d.get("tiles") or []
    idx = next((i for i, x in enumerate(tiles) if x.get("tile") == code), None)
    if idx is None:
        return None
    a = d.get("actual") or {}
    safest = list(d.get("safest") or [])
    level_of = {x.get("tile"): x.get("level") for x in tiles}
    if code == a.get("tile"):
        rank = a.get("rank")
    elif code in safest:
        rank = 1
    else:
        rank = idx + 1
    return {"tile": code, "level": tiles[idx]["level"], "rank": rank, "of": len(tiles),
            "genbutsu": tiles[idx]["genbutsu"], "is_safest": code in safest,
            "actual": {k: a.get(k) for k in ("tile", "level", "rank", "of")},
            "safest": [{"tile": s, "level": level_of.get(s)} for s in safest]}


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
    # 리치한 자리가 있는 차례에만 방총 위험(사실표 2 의 defense). 엔진 값이 없으면 null.
    risk: dict = {}
    if t.get("riichi_winds"):
        d = _defense_of(facts, t)
        risk = {"risk": _risk_of_tile(d, code) if d else None}
    return {
        "turn": t["turn"], "tile": code, **_status_view(t, code), "tile_genbutsu": _tile_genbutsu(t, code), **risk,
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


def _defense_info(facts: dict, args: dict) -> dict:
    """그 차례의 defense 칸(사실표 그대로) + 상태 + 알림. 위험은 단계 말뿐(% 없음)."""
    t = _find_turn(facts, _turn_arg(args))
    s = t["status"]
    d = _defense_of(facts, t)
    out: dict = {"turn": t["turn"], "status": s, "status_name": STATUS_KO.get(s, s), "can_choose": s != "forced",
                 "riichi_winds": t.get("riichi_winds") or [], "discard": t["discard"], "genbutsu": t.get("genbutsu"),
                 "defense": d}
    notes = []
    if not t.get("riichi_winds"):
        notes.append("이 차례에는 리치한 다른 자리가 없다 — 방총 위험 단계를 내지 않는다")
    elif t.get("hand") is None:
        notes.append(f"{STATUS_KO.get(s, '손패 모름')} 차례라 손패를 몰라 패마다의 위험 단계를 알 수 없다")
    elif facts.get("version") not in (2, 3):
        notes.append("이 사실표(판 1)에는 방총 위험 단계가 없다")
    elif d is None:
        notes.append("이 차례에는 엔진 위험 값이 없다")
    if d is not None:
        if s == "forced":
            notes.append("리치 중이라 뽑은 패를 그대로 버려야 했다 — 다른 패는 고를 수 없었다(단계 · 순서는 참고)")
        elif s == "complete":
            notes.append("버리기 전 손이 이미 화료 형태였다(단계 · 순서는 참고)")
        if d.get("stance") is None:
            notes.append("리치 중이라 고를 수 없던 차례 — 밀었다 · 접었다를 말하지 않는다(버린 패의 위험 단계까지만)")
        # a4 F2: "이 도우미는 밀지 접을지를 판정하지 않는다" 를 차례마다 실으면 묻지 않은 답에도 옮겨 적었다 — 사실까지만.
        # a7: fold_spot 객체(사실표 3)면 "보통 접는 자리" 사실과 근거 숫자, null 이면 지금 알림 그대로.
        fs = _fold_spot_of(d)
        if fs is None:
            notes.append("이 차례의 밀기 · 접기 판정은 없다(fold_spot null)")
        else:
            notes.append(fold_spot_note(fs))
    if notes:
        out["note"] = ". ".join(notes)
    return out


def _turn_line(t: dict, fold_spot: bool = False) -> str:
    """find_turns 의 한 줄: "6순 [4만] 버림 · 실수 · 유효패 5장 손해". 보통 접는 자리(a7)면 끝에 " · 보통 접는 자리"."""
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
    if fold_spot:
        line += " · 보통 접는 자리"
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


_STANCE_ARG = {"pushed": "pushed", "push": "pushed", "밀었다": "pushed", "밀기": "pushed", "밂": "pushed", "밀었음": "pushed",
               "folded": "folded", "fold": "folded", "접었다": "folded", "접기": "folded", "접음": "folded",
               "between": "between", "사이": "between", "중간": "between", "어느쪽도아님": "between"}
# find_turns 의 fold_spot? 조건: 참만 뜻이 있다(null 은 "접는 자리가 아니다" 가 아니라 판정이 없다는 뜻이라 거짓으로 고르지 않음).
# 거짓은 조건 없음으로 본다(선택 인자를 모델이 false 로 채워 보내도 오류로 도구 한 번을 버리지 않게).
_BOOL_TRUE = {"true", "참", "yes", "1", "예", "네"}
_BOOL_FALSE = {"false", "거짓", "no", "0", "아니오", "아니요"}


def fold_spot_arg(v: Any) -> bool | None:
    """fold_spot 조건 인자 → 참 · 거짓(모르면 None)."""
    if isinstance(v, bool):
        return v
    if rv._int(v) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
    return None


def level_arg(v: Any) -> str | None:
    """위험 단계 인자: `높음` · `'높음'` · `매우높음` · 영어(low · high …). 모르면 None."""
    if not isinstance(v, str):
        return None
    s = v.strip().strip("'\"‘’“”").replace(" ", "")
    for x in LEVELS:
        if s == x.replace(" ", ""):
            return x
    return {"safe": "안전", "low": "낮음", "mid": "보통", "medium": "보통", "high": "높음",
            "veryhigh": "매우 높음", "very_high": "매우 높음"}.get(s.lower())


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
    if args.get("stance") not in (None, ""):
        st = _STANCE_ARG.get(str(args["stance"]).replace(" ", "").lower())
        if st is None:
            raise ToolError(f"모르는 stance: {args['stance']!r}(쓸 수 있는 값: {', '.join(STANCES)})")
        cond["stance"] = st
    if args.get("min_level") not in (None, ""):
        lv = level_arg(args["min_level"])
        if lv is None:
            raise ToolError(f"모르는 min_level: {args['min_level']!r}(쓸 수 있는 값: {' · '.join(LEVELS)})")
        cond["min_level"] = lv
    if args.get("fold_spot") not in (None, ""):
        fsv = fold_spot_arg(args["fold_spot"])
        if fsv is None:
            raise ToolError(f"모르는 fold_spot: {args['fold_spot']!r}(true 면 보통 접는 자리인 차례만)")
        if fsv:
            cond["fold_spot"] = True
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
        d = _defense_of(facts, t)
        level = (d.get("actual") or {}).get("level") if d else None
        if "stance" in cond and (d is None or d.get("stance") != cond["stance"]):
            continue
        if "min_level" in cond and (level is None or LEVEL_INDEX[level] < LEVEL_INDEX[cond["min_level"]]):
            continue
        if "fold_spot" in cond and _fold_spot_of(d) is None:
            continue
        rows.append(_find_row(facts, t))
    out: dict = {"conditions": cond, "count": len(rows), "turns": rows}
    if "after_riichi" in cond:
        out["riichi"] = event
    notes = [note] if note else []
    if "fold_spot" in cond and not rows:
        # 검토(11-23): 판에 접는 자리가 아예 없는 때(가)와, 있는데 다른 조건에서 다 걸러진 때(나)를 가른다 — (나)를 "판정이 없다"
        # 로 알리면 "접는 자리에서 민 곳 있어?" 의 답이 "판정이 없다" 로 샜다. (나)는 접는 자리 줄(fold_spot_turns)을 같이 싣는다.
        spots = [_find_row(facts, t) for t in facts.get("turns") or [] if _fold_spot_of(_defense_of(facts, t)) is not None]
        if spots:
            out["fold_spot_turns"] = spots
            notes.append(f"보통 접는 자리는 {' · '.join(str(r['turn']) + '순' for r in spots)}에 있지만 "
                         "다른 조건까지 맞는 차례는 없다")
        else:
            # 이 알림을 답에 옮겨도 검사 10 을 통과하는 꼴("판정이 있는 차례가 없다" — [_NO_VERDICT]).
            notes.append("이 판에는 보통 접는 자리 판정이 있는 차례가 없다(접는 자리가 아니라는 뜻이 아니다)")
    if notes:
        out["note"] = ". ".join(notes)
    return out


def _find_row(facts: dict, t: dict) -> dict:
    """find_turns 의 한 줄(turns · fold_spot_turns 같은 꼴)."""
    d = _defense_of(facts, t)
    fs = _fold_spot_of(d)
    return {"turn": t["turn"], "discard": t["discard"], "status": t["status"], "grade": t.get("grade"),
            # 고를 수 있었나(what_if · discard_options 와 같은 뜻, 리치 중이면 false) — min_level 줄에 리치 중 차례가 섞여도 드러나게.
            "can_choose": t["status"] != "forced", "tsumogiri": t.get("tsumogiri"), "riichi_winds": t.get("riichi_winds") or [],
            "genbutsu": t.get("genbutsu"),
            "shanten_loss": t.get("shanten_loss"), "ukeire_loss": t.get("ukeire_loss"),
            # 버린 패의 방총 위험 단계 · stance(사실표 2 의 defense 가 있는 차례만, 그 밖은 null).
            "level": (d.get("actual") or {}).get("level") if d else None, "stance": d.get("stance") if d else None,
            # 보통 접는 자리(사실표 3 의 fold_spot 객체, 그 밖은 null).
            "fold_spot": fs, "line": _turn_line(t, fs is not None)}


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
    out = {
        "round": g.get("round"), "honba": g.get("honba"), "hero_wind": g.get("hero_wind"),
        "dora_indicators": g.get("dora_indicators"), "efficiency_pct": g.get("efficiency_pct"),
        "grades": {GRADE_KO[k]: v for k, v in (g.get("grades") or {}).items() if k in GRADE_KO},
        "statuses": {STATUS_KO[k]: v for k, v in (g.get("statuses") or {}).items() if k in STATUS_KO},
        "hero_discards": len(turns), "result": result, "others": others,
    }
    # a7: 국 시작 점수(사실표 3 의 game.scores 그대로 + 기준). 판 1 · 2 와 null 은 점수 기록이 없는 것으로.
    scores = g.get("scores") if facts.get("version") == 3 else None
    if isinstance(scores, dict):
        out["scores"] = {**copy.deepcopy(scores), "basis": SCORES_BASIS}
    else:
        # a8: 알림("이 판에는 점수 기록이 없다")은 뺀다 — 총평 끝에 묻지 않은 그 말을 붙였다(a7 ug3 · g3 · t19).
        out["scores"] = None
    return out


def _rule(facts: dict, args: dict) -> dict:
    name = args.get("name")
    key = _RULE_ALIASES.get(str(name).replace(" ", "").lower()) if isinstance(name, str) else None
    if key is None:
        raise ToolError(f"모르는 용어: {name!r}(쓸 수 있는 이름: {', '.join(RULES)})")
    return {"name": key, "text": RULES[key]}


_TOOL_FNS: dict[str, Callable[[dict, dict], dict]] = {
    "turn_info": _turn_info, "discard_options": _discard_options, "what_if": _what_if, "safe_tiles": _safe_tiles,
    "defense_info": _defense_info, "find_turns": _find_turns, "game_summary": _game_summary, "rule": _rule,
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
    # 옛 한 객체 스키마(ACTION_SCHEMA_PER_TOOL False)는 모든 도구의 인자 칸을 한 객체에 둔다 — 빈 값(null · "")으로 온 남의
    # 칸은 넘어간다. a4 의 도구별 스키마가 막아도 이 방어는 남긴다.
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
        elif key in ("kinds", "of"):  # of = defense 의 손 패 종류 수("13종 중")
            a.jong.add(v)
        elif key == "rank":  # defense · what_if risk 의 안전한 순 자리("9위")
            a.rank.add(v)
        elif key in _SHANTEN_KEYS and v >= 1:
            a.shanten.add(v)
        elif key == "turn":
            a.jun.add(v)
        elif key in ("efficiency_pct", "ai_fold_pct"):  # ai_fold_pct: 사실표 3 fold_spot 의 "약 94%"
            pct.add(v)

    for e in evidence:
        walk(e, None, None)
    # 검토(11-23): rule '밀기 접기' 를 불렀으면 그 글의 "3샹텐 이상"(보통 접는 자리의 문턱)을 옮겨 쓸 수 있다.
    if evidence_has_pushfold_rule(evidence):
        a.shanten.add(FOLD_SHANTEN[0])
    return a, pct


def evidence_points(evidence: list[dict]) -> set[int]:
    """"N점" 으로 쓸 수 있는 값(a7 검사 11): evidence 결과의 `score` · `gap` 칸(game_summary 의 scores). 음수 점수는 부호 없는
    값도 넣는다("N점" 정규식은 부호를 보지 않는다)."""
    out: set[int] = set()

    def walk(v, key):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
        elif key in ("score", "gap") and rv._int(v):
            out.update({v, abs(v)})

    for e in evidence:
        walk(e.get("result"), None)
    return out


# 위험 단계 말(지시문 a3): 작은따옴표('높음' · ‘높음’) 안의 단계만 단계 말로 본다. 따옴표 밖 "낮음" · "높음" · "매우 높음" 은
# 위반("안전" · "보통" 은 흔한 말이라 따옴표 밖은 보지 않음).
_LEVEL_QUOTED = re.compile(r"['‘](" + "|".join(sorted(LEVELS, key=len, reverse=True)) + r")['’]")
_LEVEL_BARE = re.compile(r"매우\s*높음|높음|낮음")
# 밀기 · 접기 판정 말: a3 ~ a6 은 evidence 에 fold_spot 이 null 이 아닌 차례가 있으면 허락, a7 부터 늘 위반.
# "이 도우미는 밀지 접을지를 판정하지 않는다" 는 안 걸림.
VERDICT_WORDS = [
    (re.compile(r"접었어야"), "접었어야"),
    (re.compile(r"접는\s*게\s*맞"), "접는 게 맞"),
    (re.compile(r"접어야\s*했"), "접어야 했"),
    (re.compile(r"내렸어야"), "내렸어야"),
    (re.compile(r"밀면\s*안(?!전)"), "밀면 안"),
    (re.compile(r"밀지\s*말았어야"), "밀지 말았어야"),
    (re.compile(r"밀어도\s*됐"), "밀어도 됐"),
    (re.compile(r"밀어도\s*괜찮"), "밀어도 괜찮"),
    (re.compile(r"밀\s*만했"), "밀 만했"),
    # a7(11-23 검토): 평가 꼴 — fold_spot 이 있어도 위반.
    (re.compile(r"잘\s*접었"), "잘 접었"),
    (re.compile(r"잘\s*밀었"), "잘 밀었"),
    (re.compile(r"잘\s*내렸"), "잘 내렸"),
]
# a4 F4: 판정 말 바로 뒤(한글 0 ~ 3자 안)가 간접 의문 어미이고 같은 문장에 판정을 안 한다는 말이 있으면 판정 말이 아니다
# ("접었어야 했는지 여부를 판단하지 않는다" · "밀어도 됐는지는 판정하지 않는다"). 2단계 유도 질문 8개 중 3개가 이 오검출로
# 재시도했다(설계서 1122 "a4"). 다른 문장의 부정("접었어야 했다. 다만 판정하지 않는다")이나 뒤의 다른 판정 말은 그대로 걸린다.
_VERDICT_INDIRECT = re.compile(r"\s*(?:[가-힣]\s*){0,3}?(?:는지|을지|은지|던지|느냐|는가)")
# a5 G3: "…했는지에 대한 판정 정보는 없다" 꼴도 부정 문구로 본다(판정 정보 · 판정 기록 · 판정 + 조사 은 · 는 · 이 · 가 + 없).
# a4 측정에서 남은 판정 말 재시도 하나(ds1a)가 이 꼴이었다. "판정" 이 없는 "…했는지 정보가 없다" 는 그대로 위반.
_VERDICT_DISCLAIM = re.compile(r"판정하지\s*않|판단하지\s*않|단정할\s*수\s*없|단정하지\s*않"
                               r"|판정\s*(?:정보|기록)?\s*(?:은|는|이|가)\s*없")
# 문장 나누기: `. ! ?`(연달아 나오면 하나, 소수점은 빼고 — review.py 의 문장 세기와 같은 끝) 와 줄바꿈.
_SENTENCE_SPLIT = re.compile(r"[.!?。]+(?!\d)|\n+")


def split_sentences(text: str) -> list[str]:
    """`. ! ?` · 줄바꿈으로 끊은 조각 중 글자가 있는 것."""
    return [p for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def verdict_words(text: str) -> list[str]:
    """글 속 밀기 · 접기 판정 말(보고할 말, 나온 차례대로 · 같은 말은 한 번). 판정을 안 한다는 간접 의문(F4)은 빼고 센다."""
    out: list[str] = []
    for sentence in split_sentences(text):
        disclaim = _VERDICT_DISCLAIM.search(sentence) is not None
        for pattern, word in VERDICT_WORDS:
            for m in pattern.finditer(sentence):
                if disclaim and _VERDICT_INDIRECT.match(sentence, m.end()):
                    continue
                out.append(word)
    return list(dict.fromkeys(out))


def evidence_levels(evidence: list[dict]) -> set[str]:
    """evidence 에 나온 위험 단계: `level` · `min_level`(find_turns 조건) 칸의 값 + 글 속 작은따옴표 단계(rule '방총 위험' 글)."""
    out: set[str] = set()

    def walk(v, key):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
        elif isinstance(v, str):
            if key in ("level", "min_level") and v in LEVEL_INDEX:  # min_level: find_turns 조건(그 단계 이상이 없다는 사실)
                out.add(v)
            out.update(m.group(1) for m in _LEVEL_QUOTED.finditer(v))

    for e in evidence:
        walk(e.get("result"), None)
    return out


def evidence_has_fold_spot(evidence: list[dict]) -> bool:
    """evidence 안에 fold_spot 객체(사실표 3 — 보통 접는 자리 사실)가 있는 차례가 있나. 객체만 센다 — find_turns 의
    조건 칸 `{"fold_spot": true}` 나 판 2 의 참 · 거짓은 사실이 아니다(도구가 판 2 의 참 · 거짓은 null 로 바꿔 낸다)."""

    def walk(v) -> bool:
        if isinstance(v, dict):
            return any((k == "fold_spot" and isinstance(x, dict)) or walk(x) for k, x in v.items())
        if isinstance(v, list):
            return any(walk(x) for x in v)
        return False

    return any(walk(e.get("result")) for e in evidence)


def check_risk_words(text: str, evidence: list[dict]) -> list[str]:
    """a3 의 단계 말 · 판정 말 검사(설계서 1122 4절 검증 1 · 2). 판정 말은 a4 부터 [verdict_words] 로(부정문 간접 의문은 뺌).
    a7: 판정 말은 fold_spot 사실이 있어도 위반(위반 줄의 괄호 말만 다르다 — 재시도 지시에 까닭이 보이게)."""
    out: list[str] = []
    levels = evidence_levels(evidence)
    plain = rv._BRACKET.sub(" ", text)
    for m in _LEVEL_QUOTED.finditer(plain):
        if m.group(1) not in levels:
            out.append(f"도구로 찾은 사실표에 없는 위험 단계: '{m.group(1)}'")
    for m in _LEVEL_BARE.finditer(_LEVEL_QUOTED.sub(" ", plain)):
        out.append(f"작은따옴표 없이 쓴 위험 단계: {m.group(0)}")
    why = "보통 접는 자리여도 판정은 쓰지 않음" if evidence_has_fold_spot(evidence) else "fold_spot 사실 없음"
    for word in verdict_words(plain):
        out.append(f"밀기 · 접기 판정 말({why}): {word}")
    return out


# a7 검사 10 — "접는 자리" 말: evidence 에 fold_spot 객체가 있을 때만. 같은 문장에 "판정(정보 · 기록)(은 · 는 · 이 · 가) 없" 이
# 있으면 통과("이 차례에는 접는 자리 판정이 없다"). 검토(11-23): find_turns (가) 알림 꼴 "판정이 있는 차례가 없" 도 통과.
_FOLD_SPOT_WORD = re.compile(r"접는\s*자리")
_NO_VERDICT = re.compile(r"판정\s*(?:정보|기록)?\s*(?:은|는|이|가)\s*없|판정\s*(?:이|가)\s*있는\s*차례\s*(?:가|는|도)?\s*없")


def evidence_has_pushfold_rule(evidence: list[dict]) -> bool:
    """이번 질문에서 rule 도구를 '밀기 접기'(별칭 포함)로 불렀나 — 용어 풀이("보통 접는 자리가 뭐야?")의 근거."""
    return any(e.get("tool") == "rule" and isinstance((e.get("args") or {}).get("name"), str)
               and _RULE_ALIASES.get(e["args"]["name"].replace(" ", "").lower()) == "밀기 접기" for e in evidence)


def check_fold_spot_words(text: str, evidence: list[dict]) -> list[str]:
    """검사 10. 위반 줄 목록(한 줄 — 몇 번 나와도). fold_spot 객체가 있거나 rule '밀기 접기' 를 불렀으면 통과(검토 11-23)."""
    if evidence_has_fold_spot(evidence) or evidence_has_pushfold_rule(evidence):
        return []
    for s in split_sentences(rv._BRACKET.sub(" ", text)):
        if _FOLD_SPOT_WORD.search(s) and not _NO_VERDICT.search(s):
            return ["접는 자리 말(fold_spot 사실 없음)"]
    return []


# a7 검사 11 — 점수: "N점"(쉼표 허용, review.py 의 _N_POINT 와 같은 꼴)은 evidence 의 score · gap 값만. 한글 수사 점수
# ("3만 점" · "2만 5천 점" · "8천점")는 늘 위반. 한글 수사 점수는 공용 검사 전에 지운다 — "3만" 이 대괄호 없는 패로도
# 잡혀 위반 줄이 엉뚱해지지 않게.
_KO_POINT = re.compile(r"\d[\d,]*\s?(?:만|천|백)(?:\s?\d[\d,]*\s?(?:천|백))*(?:\s?\d[\d,]*)?\s?점")
_POINT_COMMA_OK = re.compile(r"\d+|\d{1,3}(?:,\d{3})+")
_RV_POINT_LINE = "점수 말: "


def check_points(text: str, evidence: list[dict]) -> list[str]:
    """검사 11. [text] 는 대괄호 밖 글."""
    out: list[str] = []
    allowed = evidence_points(evidence)
    for m in _KO_POINT.finditer(text):
        out.append(f"한글 수로 쓴 점수(숫자 그대로 써야 함): {m.group(0).strip()}")
    for m in rv._N_POINT.finditer(_KO_POINT.sub(" ", text)):
        raw = m.group(0).strip()
        digits = re.sub(r"\s?점$", "", raw)
        if not _POINT_COMMA_OK.fullmatch(digits) or int(digits.replace(",", "")) not in allowed:
            out.append(f"도구로 찾은 점수 기록에 없는 점수: {raw}")
    return out


# 영어 검사의 예외: "AI" — fold_spot 알림 · rule '밀기 접기' 글의 "강한 AI" 를 옮긴 답(evidence 글에 "AI" 가 있을 때만).
_EN_OK = "AI"


def _evidence_texts(evidence: list[dict]) -> str:
    parts: list[str] = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            parts.append(v)

    for e in evidence:
        walk(e.get("result"))
    return "\n".join(parts)


# a6 검사 8 — 평가 말: 좋고 나쁨을 꾸미는 낱말과 "…고 · 라고 · 로 볼 수 있다" 꼴의 둘러대기. 대괄호 밖 답 글에서만 찾는다
# (질문 글에 같은 낱말이 있어도 답에 옮길 까닭은 없어 위반으로 둔다). 11-22 앱 질문 답의 "사실상 밀었다" · "무난하게 진행한 판" 꼴.
EVAL_WORDS = re.compile(r"사실상|비교적|무난|깔끔|성공적|승리|훌륭|현명|안타깝")
_EVAL_HEDGE = re.compile(r"(?:고|라고|로)\s*볼\s*수\s*있")
# a6 검사 9 — 밀기 · 접기 부정 문구: 한 문장 안에 "밀지 접을지 · 밀기 · 접기" 와 판정 부정 문구([_VERDICT_DISCLAIM])가 같이 있는 것.
_PUSHFOLD = re.compile(r"밀지\s*접을지|밀기\s*·?\s*접기")


def eval_words(text: str) -> list[str]:
    """대괄호 밖 글의 평가 말(나온 차례대로 · 같은 말은 한 번, 둘러대기는 공백을 하나로)."""
    plain = rv._BRACKET.sub(" ", text)
    found = [(m.start(), m.group(0)) for m in EVAL_WORDS.finditer(plain)]
    found += [(m.start(), re.sub(r"\s+", " ", m.group(0))) for m in _EVAL_HEDGE.finditer(plain)]
    return list(dict.fromkeys(w for _, w in sorted(found)))


def has_pushfold_disclaimer(text: str) -> bool:
    """답에 밀기 · 접기 부정 문구("이 도우미는 밀지 접을지를 판정하지 않는다" 꼴)가 있나 — 한 문장 안에 [_PUSHFOLD] 와
    [_VERDICT_DISCLAIM] 이 같이 있으면 참."""
    return any(_PUSHFOLD.search(s) and _VERDICT_DISCLAIM.search(s) for s in split_sentences(rv._BRACKET.sub(" ", text)))


# 남 리치 맥락이 되는 도구 · 칸. 도구: defense_info · safe_tiles(부른 것만으로), find_turns 의 after_riichi · stance ·
# min_level 조건(리치한 자리가 있는 차례에만 뜻이 있는 조건), rule 의 '밀기 접기'(밀기 · 접기 뜻을 물은 것).
_RIICHI_TOOLS = {"defense_info", "safe_tiles"}
_RIICHI_FIND_ARGS = ("after_riichi", "stance", "min_level", "fold_spot")
_RIICHI_KEYS = ("stance", "fold_spot", "defense", "risk", "riichi")


def evidence_has_riichi(evidence: list[dict]) -> bool:
    """이번 질문의 evidence 에 남 리치(밀기 · 접기) 맥락이 하나라도 있나(a6 검사 9).

    참이 되는 것: (1) defense_info · safe_tiles 를 불렀다. (2) find_turns 를 after_riichi · stance · min_level · fold_spot(참)
    조건으로 불렀다.
    (3) rule 을 '밀기 접기'(별칭 포함)로 불렀다. (4) 도구 결과 어디든 `riichi_winds` 가 빈 목록이 아니다(turn_info ·
    what_if · discard_options · find_turns 줄 — 남이 리치한 자리만 든 칸), `stance` · `fold_spot` · `defense` · `risk` ·
    `riichi`(find_turns 의 리치 사건)가 null 이 아니다, game_summary 의 `others` 에 kind 가 riichi 인 줄이 있다.
    내 리치(`riichi_declared`)는 남 리치가 아니라 보지 않는다. 오류 결과도 부른 도구 · 인자로는 센다."""

    def walk(v, key) -> bool:
        if key in _RIICHI_KEYS and v is not None:
            return True
        if isinstance(v, dict):
            if key == "others" and v.get("kind") == "riichi":
                return True
            return any(walk(x, k) for k, x in v.items())
        if isinstance(v, list):
            if key == "riichi_winds" and v:
                return True
            return any(walk(x, key) for x in v)
        return False

    for e in evidence:
        tool, args = e.get("tool"), e.get("args") or {}
        if tool in _RIICHI_TOOLS:
            return True
        if tool == "find_turns" and any(args.get(k) not in (None, "", False) for k in _RIICHI_FIND_ARGS):
            return True
        if tool == "rule" and isinstance(args.get("name"), str) \
                and _RULE_ALIASES.get(args["name"].replace(" ", "").lower()) == "밀기 접기":
            return True
        if walk(e.get("result"), None):
            return True
    return False


def check_tone(text: str, evidence: list[dict]) -> list[str]:
    """a6 의 말투 검사(검사 8 · 9). 위반 줄 목록."""
    out = [f"평가 말: {w}" for w in eval_words(text)]
    if has_pushfold_disclaimer(text) and not evidence_has_riichi(evidence):
        out.append("리치 없는 질문에 밀기 · 접기 부정 문구")
    return out


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


def verify_answer(text: Any, evidence: list[dict], question: str, history: list | None = None) -> list[str]:
    """답 검증(설계서 4절 verify). 위반마다 사람이 읽는 한 줄, 통과면 빈 목록.

    1 패: 대괄호 `[3만]` 만 패로 보고, evidence 에 나온 패여야 한다. 대괄호 밖 `3만` · `3m` 은 위반.
    2 숫자: `N장 · N종 · N샹텐 · N순(N번째) · N위 · N%` 는 evidence 안의 값(`0샹텐` 은 늘 위반 — 텐파이). `N점` 은 a6 까지
      늘 위반, a7 부터 검사 11.
    3 영어(대괄호 밖 영문자 2자 이상) · 금지어(반드시 · 무조건 · 절대) · 존댓말(해라체로).
    4 질문이 물은 차례(`N순 · N번째 · N차례`)를 차례 도구로 확인하지 않고 답함.
    5 길이 1~500자 · 문장 5개 이하 · 한글 한 글자 이상.
    6 (a3) 위험 단계: 작은따옴표 안의 단계('안전' · '낮음' · '보통' · '높음' · '매우 높음')는 evidence 에 나온 단계여야 한다.
      따옴표 밖 "낮음" · "높음" · "매우 높음" 은 위반.
    7 (a3) 밀기 · 접기 판정 말([VERDICT_WORDS])은 evidence 에 fold_spot 이 null 이 아닌 차례가 있을 때만.
      (a4) 판정 말 바로 뒤가 간접 의문(…는지 · …을지 …)이고 같은 문장에 "판정하지 않 · 판단하지 않 · 단정할 수 없 ·
      단정하지 않" 이 있으면 판정 말로 보지 않는다([verdict_words]). (a5) 그 부정 문구에 "판정 정보(기록)는 없 · 판정은 없 ·
      판정이 없" 을 더한다.
    8 (a6) 평가 말: 대괄호 밖에 [EVAL_WORDS](사실상 · 비교적 · 무난 · 깔끔 · 성공적 · 승리 · 훌륭 · 현명 · 안타깝)나
      "…고 · 라고 · 로 볼 수 있" 이 있으면 위반 "평가 말: …"(질문에 같은 말이 있어도).
    9 (a6) 한 문장에 "밀지 접을지 · 밀기 · 접기" 와 판정 부정 문구가 같이 있는데 evidence 에 남 리치 맥락이 없으면
      위반 "리치 없는 질문에 밀기 · 접기 부정 문구"(맥락의 정의는 [evidence_has_riichi]).
    8 · 9 는 앞 검사들의 위반 뒤에 붙는다(앞 위반 줄은 a5 와 같다).
    (a7) 7 의 판정 말은 fold_spot 사실이 있어도 위반(부정문 간접 의문 예외는 그대로).
    10 (a7) "접는 자리" 는 evidence 에 fold_spot 객체가 있거나 rule '밀기 접기'(별칭 포함)를 불렀을 때만. 같은 문장에 "판정(정보 · 기록)(은 · 는 · 이 · 가) 없" 이
      있으면 통과. 위반 "접는 자리 말(fold_spot 사실 없음)".
    11 (a7) "N점"(쉼표 허용)은 evidence 의 score · gap 값만(2 의 "N점 위반" 을 대신함). "N천 점 · N만 점" 같은 한글 수사
      점수는 늘 위반. "N위" 는 2 그대로(scores 의 rank 도 근거), "N%" 에는 fold_spot 의 ai_fold_pct 가 들어간다.
      "AI" 는 evidence 글에 "AI" 가 있으면 영어 위반으로 보지 않는다(fold_spot 알림 · rule '밀기 접기' 의 "강한 AI").
    10 · 11 은 9 뒤에 붙는다.
    (a8) [history] 는 요청의 history([{q, a}] — 앞 질문만 본다). 12 · 13 · 14 는 11 뒤에 붙는다([check_unasked_disclaimers]).
    12 한 문장에 "접는 자리" 와 판정 없음 꼴([_NO_VERDICT])이 있는데 질문(+ 앞 질문)에 접 · 밀 · 내리 · 내릴 · 내렸 · 판정 이 없으면 위반.
    13 "점수 기록(은 · 이 · 도) 없" 이 있는데 질문(+ 앞 질문)에 점수 · 몇 점 · 점 차 · 몇 등 · 등수 · 순위 · 몇 위 · 공탁 · 리치봉 이
      없으면 위반. 14 "위험 순위" · "N위의 위험" · "N위로 위험" 은 위반. "AI" 는 fold_spot 객체가 있어도 허용."""
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return ["답 없음"]
    allow, pct = evidence_allow(evidence)
    plain = rv._BRACKET.sub(" ", text)
    # 한글 수사 점수는 지운 글로 공용 검사(검사 11 이 따로 잡는다). 공용 검사의 "점수 말" 줄은 검사 11 이 대신한다.
    # a8: fold_spot 객체가 있으면 글에 "AI" 가 없어도(find_turns 줄) 허용.
    en_ok = evidence_has_fold_spot(evidence) or _EN_OK in _evidence_texts(evidence)
    out = [v for v in rv._check_tiles_numbers(_KO_POINT.sub(" ", text), allow, "도구로 찾은")
           if not v.startswith(_RV_POINT_LINE) and not (en_ok and v == f"영어(한국어로 써야 함): {_EN_OK}")]
    for m in _PCT.finditer(plain):
        if int(m.group(1)) not in pct:
            out.append(f"도구로 찾은 사실표에 없는 숫자: {m.group(0).strip()}")
    for w in _polite(plain):
        out.append(f"존댓말(해라체로 써야 함): {w}")
    out += check_risk_words(text, evidence)
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
    out += check_tone(text, evidence)
    out += check_fold_spot_words(text, evidence)
    out += check_points(plain, evidence)
    out += check_unasked_disclaimers(text, question, history)
    return rv._dedupe(out)


# a8 검사 12 · 13 · 14(설계서 1123 8-1). 질문 글 = 이번 질문 + history 의 앞 질문들(이어 묻기).
_ASKED_PUSHFOLD = re.compile(r"접|밀|내리|내릴|내렸|판정")
# "등급" 의 "등" 에 걸리지 않게 낱말을 한정한다.
_ASKED_SCORE = re.compile(r"점수|몇\s*점|점\s*차|몇\s*등|등수|순위|몇\s*위|공탁|리치봉")
_NO_SCORE_RECORD = re.compile(r"점수\s*기록\s*(?:은|이|도)?\s*없")
_RISK_RANK = re.compile(r"위험\s*순위|\d+\s*위의\s*위험|\d+\s*위로\s*위험")


def _asked_texts(question: str, history: list | None) -> str:
    qs = [question or ""]
    for h in history or []:
        q = h.get("q") if isinstance(h, dict) else h
        if isinstance(q, str):
            qs.append(q)
    return "\n".join(qs)


def check_unasked_disclaimers(text: str, question: str, history: list | None = None) -> list[str]:
    """a8 검사 12(묻지 않은 접는 자리 부정 문구) · 13(묻지 않은 점수 기록 부정 문구) · 14(안전한 순 자리를 위험 순위처럼)."""
    out: list[str] = []
    asked = _asked_texts(question, history)
    plain = rv._BRACKET.sub(" ", text)
    if not _ASKED_PUSHFOLD.search(asked) and any(
            _FOLD_SPOT_WORD.search(s) and _NO_VERDICT.search(s) for s in split_sentences(plain)):
        out.append("묻지 않은 접는 자리 부정 문구")
    if not _ASKED_SCORE.search(asked) and _NO_SCORE_RECORD.search(plain):
        out.append("묻지 않은 점수 기록 부정 문구")
    if _RISK_RANK.search(plain):
        out.append("안전한 순 자리를 위험 순위처럼 씀")
    return out


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
    "stance": {"type": "string", "description": "pushed · folded · between — 그 사실인 차례"},
    "min_level": {"type": "string", "description": "안전 · 낮음 · 보통 · 높음 · 매우 높음 — 그 단계 이상 위험한 패를 버린 차례"},
    "fold_spot": {"type": "boolean", "description": "true 면 보통 접는 자리인 차례만(false 는 조건 없음과 같다)"},
    "name": {"type": "string", "description": "용어 이름"},
}


# a4 F3: True 면 도구마다 따로인 스키마(최상위 anyOf — 변형마다 도구 이름 하나 + 그 도구의 인자만). 2단계 실제 호출 48개 중
# 6개가 첫 도구 호출에 없는 인자 · turn 빠짐(모두 what_if)이었다 — 한 객체에 모든 인자 칸을 두던 옛 꼴 탓으로 본다.
# API 가 anyOf 를 거절하면 False 로 되돌린다(옛 한 객체 꼴 그대로).
ACTION_SCHEMA_PER_TOOL = True

_ANSWER_TEXT = {"type": "string", "description": "action 이 answer 일 때 답(3~5문장)"}


def _answer_variant() -> dict:
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": ["answer"]}, "text": dict(_ANSWER_TEXT)},
            "required": ["action", "text"], "propertyOrdering": ["action", "text"]}


def _tool_variant(name: str) -> dict:
    """도구 하나의 변형. 인자 목록 · 필수 여부는 TOOLS[name]["args"](물음표 = 선택), 설명 · 타입은 _ARG_PROPS.
    인자가 없는 도구는 args 칸 자체를 뺀다(Gemini 는 properties 가 빈 object 를 거절한다). 인자가 모두 선택이면 args 는
    required 에 넣지 않는다."""
    specs = TOOLS[name]["args"]
    names = [a.rstrip("?") for a in specs]
    need = [a for a in specs if not a.endswith("?")]
    props: dict = {"action": {"type": "string", "enum": ["tool"]}, "tool": {"type": "string", "enum": [name]}}
    required = ["action", "tool"]
    if names:
        args: dict = {"type": "object", "properties": {a: dict(_ARG_PROPS[a]) for a in names}, "propertyOrdering": names}
        if need:
            args["required"] = need
            required.append("args")
        props["args"] = args
    return {"type": "object", "properties": props, "required": required, "propertyOrdering": list(props)}


def action_schema(answer_only: bool = False) -> dict:
    """plan 의 JSON 스키마 응답. 도구 한도를 다 쓰면 answer 만 받는다.

    기본([ACTION_SCHEMA_PER_TOOL])은 최상위 `anyOf`: 도구마다 변형 하나 + 답 변형. 스위치가 False 면 옛 한 객체 꼴."""
    if answer_only:
        return {"type": "object",
                "properties": {"action": {"type": "string", "enum": ["answer"]}, "text": {"type": "string"}},
                "required": ["action", "text"], "propertyOrdering": ["action", "text"]}
    if ACTION_SCHEMA_PER_TOOL:
        return {"anyOf": [*(_tool_variant(n) for n in TOOL_NAMES), _answer_variant()]}
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": ["tool", "answer"]},
                           "tool": {"type": "string", "enum": TOOL_NAMES},
                           "args": {"type": "object", "properties": _ARG_PROPS},
                           "text": dict(_ANSWER_TEXT)},
            "required": ["action"]}


def schema_answer_only(schema: dict) -> bool:
    """[action_schema] 가 낸 스키마가 답만 받는 꼴인가(도구 한도를 다 쓴 뒤). 도구 변형이 하나라도 있으면 False."""
    variants = schema.get("anyOf") or [schema]
    return all(((v.get("properties") or {}).get("action") or {}).get("enum") == ["answer"] for v in variants)


def _tool_lines() -> str:
    return "\n".join(f"- {n}({', '.join(d['args'])}): {d['desc']}" for n, d in TOOLS.items())


PROMPT_RULES = f"""너는 리치마작 패보 복기 도우미다. 사용자가 자기 패보 한 판을 보며 묻는 말에 답한다.
이 도우미는 **패효율(샹텐 · 유효패)과, 리치한 자리에 대한 방총 위험의 크기까지 말한다**. 밀지 접을지의 옳고 그름 · 타점 · 역 ·
상대 대기 읽기는 판단하지 않는다 — 그런 판단을 묻는 질문에는 현물 여부 · 위험 단계처럼 도구로 확인되는 사실까지만 말하고,
판단하지 않는다고 한 번 밝힌다(밀기 · 접기는 아래 수비 절의 때와 꼴을 따른다). 판단을 묻지 않은 질문에는 그 말을 붙이지 않는다.

## 도구 — 사실은 도구로만 찾는다
숫자와 패는 프로그램이 미리 계산해 두었다. 너는 계산하지 않고 도구로 찾아 읽는다. 한 번에 도구 하나를 부르고, 결과를 본 뒤 다음을 정한다.
도구는 한 질문에 {Limits.tool_calls}번까지다. 필요한 것만 부른다.
{_tool_lines()}
- 차례 번호(turn)는 내(hero) 몇 번째 버림인가다(1부터). 사용자가 "6순" · "6번째" 라고 하면 turn 6 이다.
- 도구가 {{"error": "..."}} 를 돌려주면 그 내용을 읽고 다른 도구나 다른 인자를 고르거나, 확인할 수 없다고 답한다.
- 질문이 특정 차례를 물으면 그 차례를 turn_info · discard_options · what_if · safe_tiles · defense_info 중 하나로 꼭 확인한 뒤 답한다.
- "만약 · 버렸으면 · 대신" 을 묻는 질문은 what_if 결과의 상태(status_name · can_choose · note)를 먼저 본다. 리치 중
  (can_choose 가 false)이면 다른 패를 고를 수 없던 차례다 — 숫자를 비교하기 전에 그것부터 말하고, 숫자는 참고로만 쓴다.
- 여러 차례에 걸친 질문("리치 받은 뒤에 어떻게 했어" · "실수한 곳 전부")은 차례마다 도구를 부르지 말고 find_turns 조건
  (after_riichi · status · grade 등)으로 판 전체를 먼저 한 번에 찾는다. 줄마다 현물 여부 · 상태가 있다.

## 수비 질문 — 방총 위험은 다섯 단계 말로만
- 수비 질문("리치 받고 뭘 버렸어야 했어" · "이 패 위험했어" · "그때 뭘 버리면 안전했어")은 그 차례를 defense_info 로 확인한 뒤
  답한다. 여러 차례에 걸친 질문("리치 받은 뒤 위험하게 버린 곳 있어")은 find_turns 의 after_riichi · stance · min_level 조건으로.
- 위험은 다섯 단계 '안전' · '낮음' · '보통' · '높음' · '매우 높음' 으로만 말한다. 단계는 작은따옴표 하나로 감싼다: '높음'
  (패를 대괄호로 쓰는 것과 같은 약속 — 검사가 이것을 본다). 도구 결과에 나온 단계만 쓰고, 따옴표 없이 쓰지 않는다.
- 위험 % 를 지어내지 않는다. 사실표의 위험에는 % 가 없다(쓸 수 있는 % 는 rule("방총 위험") 글의 관측 비율과 fold_spot 의 ai_fold_pct 뿐이다). 단계의 뜻 · 관측 비율은 rule("방총 위험")에 있다 — 그 글을 읽었을 때만 그 % 를 옮긴다.
- 순서는 defense_info 의 rank / of(1 = 가장 안전)를 "안전한 순으로 손의 N종 중 R위" 처럼 그대로 쓴다. "위험 순위 R위 · R위의 위험도" 처럼
  위험한 순으로 읽히게 쓰지 않는다. 가장 안전한 패는 safest, 효율대로 버릴 패는 efficient.
- 위험 단계는 패마다 따로 적힌 값만 말한다. 같은 목록(가장 안전한 패 · 현물)에 있다고 같은 단계로 말하지 않는다 — what_if 면
  물어본 패는 risk 의 level, 실제 버린 패는 risk.actual, 가장 안전한 패는 risk.safest 의 패마다 level.
- stance(pushed · folded · between)는 버린 패가 가장 안전한 패보다 얼마나 위험했나의 사실이지 판정이 아니다. 말할 때는
  단정하는 사실문 하나로 쓴다. pushed: "가장 안전한 [백] 대신 '높음' 인 [5통]을 버려 밀었다". folded: 버린 패가 가장 안전한
  패면 "가장 안전한 패를 버려 접었다", 아니면 "가장 안전한 [백]과 위험 차이가 거의 없는 [5통]을 버려 접었다". between:
  "가장 안전한 [백] 대신 [5통]을 버렸고, 밀었다 · 접었다 어느 쪽도 아니다"(뜻은 rule("밀기 접기") 글 그대로).
  "사실상" · "…라고 볼 수 있다" · "…로 기록되었다" · "…셈이다" 를 붙이지 않는다.
- 리치 중 차례(can_choose 가 false, stance 가 null)는 버린 패의 위험 단계까지만 말하고 밀었다 · 접었다고 하지 않는다.
- fold_spot 이 있는(null 이 아닌) 차례를 말할 때는 "이 차례는 보통 접는 자리다" 와 근거(내 손 N샹텐 · 강한 AI 는 이런 자리의
  약 94% 에서 접는다 — 숫자는 fold_spot 의 shanten · ai_fold_pct 값 그대로)를 사실로 말하고, 플레이어가 한 것은 stance 사실문으로
  잇는다. "접었어야 했다 · 밀면 안 됐다 · 밀어도 됐다 · 잘 접었다" 같은 판정 · 평가 말은 fold_spot 이 있어도 쓰지 않는다.
- "접었어야 했어?" 같은 판정 질문에는 "이 도우미는 밀지 접을지를 판정하지 않는다" 를 한 번 쓰고, fold_spot 이 있으면 보통 접는
  자리 사실을 더한다.
- fold_spot 이 null 인 차례는 단계 · 순서 · 가장 안전한 패 같은 사실까지만 말한다. "접는 자리가 아니다" 라고 하지 않는다 —
  접는 자리 · 밀기 · 접기를 물었을 때만 "이 차례에는 접는 자리 판정이 없다" 고 답한다. 묻지 않았으면 그 말을 붙이지 않는다.
- "판정하지 않는다" 는 말은 질문이 옳고 그름을 물었을 때만, 그것도 리치한 자리가 있는 차례(또는 판)의 밀기 · 접기를 물었을 때만
  ("접었어야 했어?" · "밀어도 됐어?" · 리치 받은 차례의 "그게 맞았어?") 답에 한 번 쓴다. 꼴은 "이 도우미는 밀지 접을지를 판정하지 않는다".
  리치한 자리가 없는 차례나 손패를 모르는 차례의 "잘한 거야?" · "실수야?" 는 패효율 질문이다 — 밀기 · 접기 말을 꺼내지 않고
  "패효율 기준" 등급까지, 손패를 모르면 "모른다" 까지만 말한다. 그 밖의 질문("위험했어?" · "뭐가 안전했어?" ·
  "몇 번째로 위험했어?")에는 붙이지 않는다.

## 답 규칙 — 프로그램이 도구 결과와 대조해 검사한다
- 숫자("N장" · "N종" · "N샹텐" · "N순" · "N번째" · "N%")는 이번에 부른 도구 결과에 있는 값만 쓴다. 더하거나 빼거나 나눠 새 숫자를 만들지 않는다.
  유효패 장수 차이는 what_if 의 total_diff 나 ukeire_loss(같은 샹텐일 때)만 쓴다. 샹텐 0 은 "텐파이", -1 은 "화료 형태"("0샹텐" 금지).
- 패는 대괄호 하나에 한 장씩 이름으로: [3만] · [5통] · [7삭] · [중]. 이름: {rv._NAME_TABLE}
  대괄호 밖에 "3만" · "3m" 처럼 쓰지 않는다. 도구 결과에 나온 패만 쓴다.
- 자리(동 · 남 · 서 · 북)는 대괄호 없이 "남 자리" 처럼 쓴다. 대괄호는 패에만 쓴다 — [남] 은 바람패 남을 말할 때만.
- 영어를 쓰지 않는다(도구의 칸 이름 best · shanten · ukeire 등을 옮기지 않는다). 등급은 최선 · 좋음 · 부정확 · 실수 · 큰 실수,
  상태는 등급 매김 · 리치 중 · 수비 · 화료 형태 · 손패 기록 없음 · 손패 기록 어긋남 · 짐작 손패, stance 는 밀었다 · 접었다 · 어느 쪽도 아니다.
- 해라체("…다")로 쓴다. 존댓말("…습니다" · "…해요")을 쓰지 않는다. "반드시" · "무조건" · "절대" 를 쓰지 않는다.
- 점수는 game_summary 의 scores 값만 "N점"(숫자 그대로, 쉼표는 써도 됨: 25000점 · 25,000점)으로 말한다. 더하거나 빼 새 점수를
  만들지 않는다(점수 차는 gap 값). "2만 5천 점" 처럼 한글 수로 쓰지 않는다. 순위는 rank 값을 "N위" 로 쓴다. scores 가 null 이면,
  점수 · 순위를 물었을 때만 "이 판에는 점수 기록이 없다" 고 답한다. 묻지 않았으면 그 말을 붙이지 않는다. 점수로 밀기 · 접기 · 필요한 타점 · 순위 전망을 판단하지 않는다.
- 3~5문장, 500자 안. 숫자를 늘어놓지 말고 질문에 필요한 것만.
- 좋고 나쁨 · 잘하고 못함을 꾸미는 말을 쓰지 않는다: "사실상" · "비교적" · "무난" · "깔끔" · "성공적" · "승리" · "훌륭" · "현명" ·
  "안타깝", 그리고 "…할 수 있었던 선택" 처럼 다른 선택을 권하는 말. 총평을 물으면 결과 · 등급 개수 · 아까운 수 같은 사실로 답하고, 판 전체를 좋다 · 나쁘다로 꾸미는 맺음 문장은 붙이지 않는다. 화료는 "화료했다"
  (론 · 쯔모)로 쓴다. 등급 말(최선 · 좋음 · 부정확 · 실수 · 큰 실수)은 도구 결과의 등급으로만 쓴다.
- 손패를 모르는 차례(손패 기록 없음 · 어긋남 · 짐작 손패)는 숫자가 없다 — 모른다고 답한다.
- 등급은 패효율만 본 것이라 수비 · 타점 때문에 일부러 고른 수일 수 있다. "실수" 라고 단정하기보다 "패효율 기준" 임을 밝힌다.
- 모양(양면 · 간짱 · 샤보 …)과 대기는 도구 결과에 없으니 단정하지 않는다. 손패 · 유효패 목록으로 말할 수 있는 것만 말한다.

## 응답 형식(JSON 하나)
- 도구를 부를 때: {{"action": "tool", "tool": "turn_info", "args": {{"turn": 6}}}}
- 인자가 둘인 도구: {{"action": "tool", "tool": "what_if", "args": {{"turn": 6, "tile": "3m"}}}}
- args 에는 그 도구의 인자만 넣는다(위 도구 목록의 괄호 안, 물음표는 빼도 되는 인자).
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
    retry_violations: list  # 재시도를 부른 위반 목록들(시도 순, a4 F5) — verify 가 plan 으로 돌려보낼 때마다 하나
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
        v = verify_answer(state.get("answer"), state["evidence"], state["question"], state.get("history"))
        if not v:
            return {"violations": [], "outcome": "verified", "route": "end"}
        if state["retries"] < limits.retries:
            return {"violations": v, "retries": state["retries"] + 1, "route": "plan",
                    "retry_violations": [*(state.get("retry_violations") or []), list(v)],
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
        "retry_violations": [], "outcome": None, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "pending": None, "route": "plan",
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
    """그래프 끝 상태 → 응답. `retries`(검증 뒤 다시 받은 횟수)는 설계서 5절에 더한 칸 — 첫 통과 · 재시도 뒤 · 대체 비율을 잰다.
    `retry_violations`(a4 F5)는 재시도를 부른 위반 목록들(시도 순, 재시도가 없으면 빈 목록). `violations` 는 마지막 상태
    (통과면 빈 목록, 대체 답이면 마지막 검증의 위반 + 대체 까닭)."""
    return {
        "answer": s["answer"], "outcome": s["outcome"], "tool_calls": s["tool_calls"], "violations": s["violations"],
        "retries": s["retries"], "retry_violations": [list(v) for v in s.get("retry_violations") or []],
        "prompt_version": PROMPT_VERSION, "model": model, "elapsed_s": round(elapsed, 3),
        "usage": {"input_tokens": s["tokens_in"], "output_tokens": s["tokens_out"], "llm_calls": s["llm_calls"]},
    }
