"""패보 리뷰 해설(POST /v2/review/explain) — 지시문 · 응답 스키마 · 검증 · 재시도 · 대체.

규격: mahjong_calc/docs/paifu_review_llm_design_1118.md 부록 A — v1.1(A-8 · A-7 이 A-5 보다 먼저, 요청 `version` 은 1 그대로).
숫자는 전부 앱이 계산해 보낸 사실표에 있다.
LLM 은 그 숫자 · 패를 말로 풀기만 하고, 여기서 그 글을 사실표와 대조해 검증한다(기계적).
걸린 항목(차례 · 총평)만 위반 목록을 붙여 한 번 더 받고, 또 걸리면 그 항목은 fallback(앱이 지금 카드 문구를 쓴다).

LLM 호출은 주입 함수 `generate(prompt, schema) -> (파싱한 dict, 입력 토큰, 출력 토큰)` — 운영은 GeminiTextGenerator,
테스트는 가짜. 프레임워크 없이 함수 몇 개로 둔다(설계서 6절).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter
from collections.abc import Callable

from .recognizer import rate_limited, retryable

log = logging.getLogger("recognizer")

PROMPT_VERSION = "r2"
SPEC_VERSION = 1
# 생각 수준 기본값(REVIEW_THINKING 이 없을 때). 첫 실제 호출에서 low 가 default 보다 8배 빨랐다(규격 A-8).
DEFAULT_THINKING = "low"

# 요청 상한. hero 버림은 한 판에 많아야 20개 남짓(울기 · 깡 포함) — 넉넉히 잡는다.
MAX_TIMELINE = 40
MAX_TURNS = 30
# 차례 사실 하나가 압축 JSON 으로 1.5KB 안팎(30차례 ≈ 45KB). 라우트는 본문을 이 크기까지만 읽는다.
MAX_REQUEST_BYTES = 128 * 1024

# ---------------------------------------------------------------- 패 표기(A-1)

SUITS = {"m": "만", "p": "통", "s": "삭"}
HONORS = ["동", "남", "서", "북", "백", "발", "중"]
TILE_CODES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [f"{n}z" for n in range(1, 8)]
_TILE_SET = set(TILE_CODES)
WINDS = {"동", "남", "서", "북"}


def tile_name(code: str) -> str:
    """`3s` → `3삭`, `7z` → `중`."""
    n, s = int(code[0]), code[1]
    return HONORS[n - 1] if s == "z" else f"{n}{SUITS[s]}"


NAME_TO_CODE = {tile_name(c): c for c in TILE_CODES}


def _tile_key(code: str) -> tuple[int, int]:
    return "mpsz".index(code[1]), int(code[0])


# guessed: 앱 11-21 짐작 손패 차례(설계서 paifu_review_ui_design_1121 3-6) — 등급 · 숫자 없음.
STATUSES = {"graded", "forced", "defense", "complete", "unknownHand", "inconsistent", "guessed"}
GRADES = {"best", "good", "inaccuracy", "mistake", "blunder"}
EXPLAINED_GRADES = {"inaccuracy", "mistake", "blunder"}
SHAPE_KINDS = ["triplet", "pair", "sequence", "ryanmen", "penchan", "kanchan", "isolated"]
# shape 의 목록: 버린 패 · 최선 후보 패 · 뽑은 패(drawn 은 A-8 에서 더함). 모양 말 근거는 셋 모두에서 찾는다.
SHAPE_SIDES = ("discard", "best", "drawn")

# ---------------------------------------------------------------- 검증 규칙(A-5)

_BRACKET = re.compile(r"\[([^\[\]]*)\]")
# 대괄호 밖의 수패 이름(N = 1~9, 앞에 숫자가 붙은 "10만" 등은 패가 아님). 규격 A-5 의 1.
_BARE_NAME = re.compile(r"(?<![0-9])[1-9](?:만|통|삭)")
# 대괄호 밖의 패 코드(3s · 7z). 규격에는 없지만 1 의 "대괄호를 안 씀" 과 같은 실수라 같이 잡는다(보고에 적음).
_BARE_CODE = re.compile(r"(?<![0-9A-Za-z])[1-9][mpsz](?![A-Za-z0-9])")
_N_JANG = re.compile(r"(\d+)\s?장")
_N_JONG = re.compile(r"(\d+)\s?종")
_N_SHANTEN = re.compile(r"(\d+)\s?(?:샹텐|향청)")
_N_JUN = re.compile(r"(\d+)\s?(?:순(?!위)|번째)")
# 후보 순위("7위" · "7순위"): 규격의 N순 · N번째(= turn)와 섞이지 않게 따로 세고 rank · top 자리(1~3)만 허용(보고에 적음).
_N_RANK = re.compile(r"(\d+)\s?(?:순위|위)")
_N_POINT = re.compile(r"\d[\d,]*\s?점")
_EFFICIENCY = re.compile(r"패효율\s*(?:은|는|이|가|:)?\s*(\d+)")
# 영어(A-8 의 3): 대괄호 토큰을 지운 글에 영문자 2자 이상이 이어지면 위반(best · blunder · shanten 같은 칸 이름이 샘).
_ENGLISH = re.compile(r"[A-Za-z]{2,}")
# "유지" 말(A-8 의 4). 규격 문자열(텐파이를 유지 · 텐파이 유지 · 텐파이를 지키 / N샹텐을 유지 · N샹텐 유지)에
# 조사 · 띄어쓰기 · 지키다 활용(지켰 · 지킨)이 바뀐 같은 뜻의 말도 같이 잡는다(보고에 적음).
_KEEP = r"(?:을|를|은|는|이|가|도)?\s*(?:유지|지키|지켰|지킨)"
_KEEP_TENPAI = re.compile(r"텐파이" + _KEEP)
_KEEP_SHANTEN = re.compile(r"(\d+)\s?(?:샹텐|향청)" + _KEEP)
_HANGUL = re.compile(r"[가-힣]")
_SENTENCE_END = re.compile(r"[.!?。]+(?!\d)")

# 모양 말 → kind(A-5 의 3). 흔한 다른 표기도 같은 kind 로 본다.
SHAPE_WORDS: list[tuple[str, str, str]] = [  # (정규식, 보고할 말, kind)
    (r"양면", "양면", "ryanmen"), (r"료멘", "료멘", "ryanmen"),
    (r"간짱", "간짱", "kanchan"), (r"칸짱", "칸짱", "kanchan"),
    (r"변짱", "변짱", "penchan"), (r"펜짱", "펜짱", "penchan"),
    (r"또이쯔", "또이쯔", "pair"), (r"또이츠", "또이츠", "pair"), (r"토이츠", "토이츠", "pair"),
    (r"커쯔", "커쯔", "triplet"), (r"커츠", "커츠", "triplet"), (r"코츠", "코츠", "triplet"),
    (r"순자", "순자", "sequence"), (r"슌츠", "슌츠", "sequence"),
    (r"뜬\s?패", "뜬 패", "isolated"), (r"고립", "고립", "isolated"),
]
# 대기 모양 말: 늘 위반(대기 모양 힌트가 없음). "다면" 은 "버렸다면" 같은 조건 어미와 겹쳐, 앞이 한글이 아닐 때만 모양 말로 본다.
WAIT_WORDS: list[tuple[str, str]] = [
    (r"샤보", "샤보"), (r"샨폰", "샨폰"), (r"단기", "단기"), (r"탕키", "탕키"), (r"노베탄", "노베탄"),
    (r"(?<![가-힣])다면", "다면"),
]
STATUS_WORDS = ["실수", "잘못", "틀렸", "틀린"]
FORBIDDEN_WORDS = ["반드시", "무조건", "절대"]

TITLE_MAX = 20
TEXT_MAX = 200
TEXT_SENTENCES = 3
SUMMARY_MAX = 300
SUMMARY_SENTENCES = 4

MISSING = "빠짐 — 응답에 이 차례가 없음"


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _sentences(text: str) -> int:
    """문장 수: `. ! ?`(연달아 나오면 하나, 소수점은 빼고)로 끊은 조각 중 글자가 있는 것. 끝에 마침표가 없는 조각도 한 문장."""
    return sum(1 for piece in _SENTENCE_END.split(text) if piece.strip())


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class _Allow:
    """한 차례(또는 판 전체)에서 글에 나와도 되는 패 · 숫자."""

    def __init__(self):
        self.tiles: set[str] = set()
        self.jang: set[int] = set()
        self.jong: set[int] = set()
        self.shanten: set[int] = set()
        self.jun: set[int] = set()
        self.rank: set[int] = set()
        self.shapes: set[str] = set()

    def merge(self, other: _Allow) -> None:
        for k in ("tiles", "jang", "jong", "shanten", "jun", "rank", "shapes"):
            getattr(self, k).update(getattr(other, k))


def _turn_allow(fact: dict) -> _Allow:
    a = _Allow()
    actual = fact.get("actual") or {}
    best = fact.get("best") or {}
    top = [t for t in fact.get("top") or [] if isinstance(t, dict)]
    hand = [t for t in fact.get("hand") or [] if isinstance(t, str)]
    tiles = [*hand, fact.get("drawn"), fact.get("discard"), actual.get("discard"), best.get("discard"),
             *(t.get("discard") for t in top), *(fact.get("safe_tiles") or [])]
    for opt in (actual, best):
        for u in opt.get("tiles") or []:
            if isinstance(u, dict):
                tiles.append(u.get("tile"))
                if _int(u.get("left")):
                    a.jang.add(u["left"])
    a.tiles = {t for t in tiles if isinstance(t, str)}
    for opt in (actual, best, *top):
        if _int(opt.get("total")):
            a.jang.add(opt["total"])
        if _int(opt.get("kinds")):
            a.jong.add(opt["kinds"])
        if _int(opt.get("shanten")) and opt["shanten"] >= 1:
            a.shanten.add(opt["shanten"])
    # |ukeire_loss| 는 같은 샹텐에서 덜 받은 장수일 때만(A-8 의 1). 샹텐이 뒤로 간 차례의 유효패 차이는 손실이 아니다.
    if fact.get("shanten_loss") == 0 and _int(fact.get("ukeire_loss")) and fact["ukeire_loss"] > 0:
        a.jang.add(fact["ukeire_loss"])
    a.jang.update(Counter(hand).values())
    for k in ("shanten_before", "shanten_prev"):  # shanten_prev(뽑기 전 샹텐)는 A-8 의 2
        if _int(fact.get(k)) and fact[k] >= 1:
            a.shanten.add(fact[k])
    if _int(fact.get("turn")):
        a.jun.add(fact["turn"])
    a.rank.update(range(1, len(top) + 1))
    if _int(fact.get("rank")):
        a.rank.add(fact["rank"])
    shape = fact.get("shape") or {}
    for side in SHAPE_SIDES:
        for blk in shape.get(side) or []:
            if isinstance(blk, dict) and isinstance(blk.get("kind"), str):
                a.shapes.add(blk["kind"])
    return a


def _game_allow(req: dict) -> _Allow:
    """총평: 모든 차례 허용값의 합집합 + timeline 의 버림 · total · shanten_after, 순은 timeline 의 turn 만."""
    a = _Allow()
    for fact in req.get("turns") or []:
        a.merge(_turn_allow(fact))
    a.jun = set()
    for t in req.get("timeline") or []:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("discard"), str):
            a.tiles.add(t["discard"])
        if _int(t.get("total")):
            a.jang.add(t["total"])
        if _int(t.get("shanten_after")) and t["shanten_after"] >= 1:
            a.shanten.add(t["shanten_after"])
        if _int(t.get("turn")):
            a.jun.add(t["turn"])
    return a


def _check_tiles_numbers(text: str, allow: _Allow, scope: str) -> list[str]:
    """규칙 1(패) · 2(숫자) · 5(금지어) · A-8 의 3(영어). [scope] 는 위반 문구의 "이 차례" / "이 판"."""
    out: list[str] = []
    for m in _BRACKET.finditer(text):
        name = m.group(1)
        code = NAME_TO_CODE.get(name)
        if code is None:
            out.append(f"패 이름이 아님: [{name}]")
        elif code not in allow.tiles:
            out.append(f"{scope} 사실표에 없는 패: [{name}]")
    plain = _BRACKET.sub(" ", text)
    for m in _BARE_NAME.finditer(plain):
        out.append(f"대괄호 없이 쓴 패: {m.group(0)}")
    for m in _BARE_CODE.finditer(plain):
        out.append(f"대괄호 없이 쓴 패 코드: {m.group(0)}")
    for m in _ENGLISH.finditer(plain):
        out.append(f"영어(한국어로 써야 함): {m.group(0)}")
    for pattern, values, unit in ((_N_JANG, allow.jang, "장"), (_N_JONG, allow.jong, "종"),
                                  (_N_SHANTEN, allow.shanten, "샹텐"), (_N_JUN, allow.jun, "순"),
                                  (_N_RANK, allow.rank, "위")):
        for m in pattern.finditer(plain):
            if int(m.group(1)) not in values:
                out.append(f"{scope} 사실표에 없는 숫자: {m.group(0).strip()}")
    for m in _N_POINT.finditer(plain):
        out.append(f"점수 말: {m.group(0).strip()}")
    for w in FORBIDDEN_WORDS:
        if w in text:
            out.append(f"금지어: {w}")
    return out


def _check_keep_words(plain: str, prev) -> list[str]:
    """A-8 의 4: "텐파이를 유지" 류는 shanten_prev == 0, "N샹텐을 유지" 류는 N == shanten_prev 일 때만. null 이면 늘 위반.
    (첫 실제 호출에서 뽑기 전 1샹텐 → 텐파이가 되는 수를 "텐파이를 유지" 로 썼다.) 총평에는 쓰지 않는다(차례별 근거 없음)."""
    out: list[str] = []
    known = _int(prev)
    was = ("텐파이" if prev == 0 else "화료형" if prev < 0 else f"{prev}샹텐") if known else ""
    for pattern, ok in ((_KEEP_TENPAI, lambda m: prev == 0), (_KEEP_SHANTEN, lambda m: int(m.group(1)) == prev)):
        for m in pattern.finditer(plain):
            if not known:
                out.append(f"뽑기 전 샹텐을 모르는 차례에 유지 말: {m.group(0)}")
            elif not ok(m):
                out.append(f"뽑기 전 손은 {was}인데 유지 말: {m.group(0)}")
    return out


def verify_turn(fact: dict, title: str | None, text: str | None) -> list[str]:
    """차례 글(title + text)을 그 차례 사실만 가지고 검증(A-5 의 1~6 + A-8). 위반마다 사람이 읽는 한 줄, 통과면 빈 목록."""
    out: list[str] = []
    title = title.strip() if isinstance(title, str) else ""
    text = text.strip() if isinstance(text, str) else ""
    allow = _turn_allow(fact)
    both = f"{title}\n{text}"
    out += _check_tiles_numbers(both, allow, "이 차례")
    plain = _BRACKET.sub(" ", both)
    out += _check_keep_words(plain, fact.get("shanten_prev"))
    for pattern, word, kind in SHAPE_WORDS:
        if re.search(pattern, plain) and kind not in allow.shapes:
            out.append(f"모양 말 근거 없음(shape 에 {kind} 없음): {word}")
    for pattern, word in WAIT_WORDS:
        if re.search(pattern, plain):
            out.append(f"대기 모양 말(쓰지 않음): {word}")
    if fact.get("status") in ("defense", "complete"):
        for w in STATUS_WORDS:
            if w in plain:
                out.append(f"{fact['status']} 차례에 평가 말: {w}")
    if not title:
        out.append("제목 없음")
    elif len(title) > TITLE_MAX:
        out.append(f"제목 {len(title)}자(1~{TITLE_MAX}자)")
    if not text:
        out.append("본문 없음")
    else:
        if len(text) > TEXT_MAX:
            out.append(f"본문 {len(text)}자(1~{TEXT_MAX}자)")
        if (n := _sentences(text)) > TEXT_SENTENCES:
            out.append(f"본문 문장 {n}개({TEXT_SENTENCES}개 이하)")
    for field, value in (("제목", title), ("본문", text)):
        if value and not _HANGUL.search(value):
            out.append(f"{field}에 한글 없음")
    return _dedupe(out)


def verify_summary(req: dict, text: str | None) -> list[str]:
    """총평 검증(A-5 끝 문단): 패 · 숫자는 판 전체 합집합, 순은 timeline, 금지어 · 점, 1~300자 · 4문장, 패효율 N."""
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return ["총평 없음"]
    out = _check_tiles_numbers(text, _game_allow(req), "이 판")
    eff = (req.get("game") or {}).get("efficiency_pct")
    for m in _EFFICIENCY.finditer(_BRACKET.sub(" ", text)):
        if eff is None:
            out.append(f"패효율 숫자가 없는 판인데 씀: {m.group(0)}")
        elif int(m.group(1)) != eff:
            out.append(f"패효율이 다름: {m.group(0)}(사실표 {eff})")
    if len(text) > SUMMARY_MAX:
        out.append(f"총평 {len(text)}자(1~{SUMMARY_MAX}자)")
    if (n := _sentences(text)) > SUMMARY_SENTENCES:
        out.append(f"총평 문장 {n}개({SUMMARY_SENTENCES}개 이하)")
    if not _HANGUL.search(text):
        out.append("총평에 한글 없음")
    return _dedupe(out)


# ---------------------------------------------------------------- 요청 검사(A-2)


class _Bad(ValueError):
    pass


def _need(cond: bool, where: str, what: str) -> None:
    if not cond:
        raise _Bad(f"{where}: {what}")


def _tile_ok(v) -> bool:
    return isinstance(v, str) and v in _TILE_SET


def _int_in(v, lo: int, hi: int) -> bool:
    return _int(v) and lo <= v <= hi


def _tiles_ok(v, max_len: int = 34) -> bool:
    return isinstance(v, list) and len(v) <= max_len and all(_tile_ok(t) for t in v)


def _check_option(o, where: str, with_tiles: bool) -> None:
    _need(isinstance(o, dict), where, "객체가 아님")
    _need(_tile_ok(o.get("discard")), where, "discard")
    _need(_int_in(o.get("shanten"), -1, 8), where, "shanten")
    _need(_int_in(o.get("total"), 0, 136), where, "total")
    _need(_int_in(o.get("kinds"), 0, 34), where, "kinds")
    if with_tiles:
        tiles = o.get("tiles")
        _need(isinstance(tiles, list) and len(tiles) <= 34, where, "tiles")
        for u in tiles:
            _need(isinstance(u, dict) and _tile_ok(u.get("tile")) and _int_in(u.get("left"), 0, 4), where,
                  "tiles[] = {tile, left}")


def _check_turn(t, i: int, timeline_turns: set[int]) -> None:
    w = f"turns[{i}]"
    _need(isinstance(t, dict), w, "객체가 아님")
    for k in ("turn", "status", "grade", "hand", "drawn", "discard", "melds", "riichi_declared", "shanten_prev",
              "shanten_before", "actual", "best", "top", "rank", "shanten_loss", "ukeire_loss", "only_best", "only_actual", "shape",
              "riichi_winds", "genbutsu", "safe_tiles"):
        _need(k in t, w, f"{k} 없음")
    _need(_int_in(t["turn"], 1, 999), w, "turn")
    _need(t["turn"] in timeline_turns, w, f"turn {t['turn']} 이 timeline 에 없음")
    status, grade = t["status"], t["grade"]
    if status == "graded":
        _need(grade in EXPLAINED_GRADES, w, "graded 차례는 grade 가 inaccuracy · mistake · blunder")
    else:
        _need(status in ("defense", "complete"), w, "status 는 graded · defense · complete")
        _need(grade is None, w, f"{status} 차례의 grade 는 null")
    _need(_tiles_ok(t["hand"], 14) and len(t["hand"]) >= 2, w, "hand")
    _need(t["drawn"] is None or _tile_ok(t["drawn"]), w, "drawn")
    _need(_tile_ok(t["discard"]) and t["discard"] in t["hand"], w, "discard 가 hand 에 없음")
    _need(_int_in(t["melds"], 0, 4), w, "melds")
    _need(isinstance(t["riichi_declared"], bool), w, "riichi_declared")
    # shanten_prev(A-8): 뽑기 전 샹텐, 정수 또는 null. drawn 이 null 이면 null 이어야 한다(뽑은 패를 모르면 뽑기 전도 모름).
    _need(t["shanten_prev"] is None or _int_in(t["shanten_prev"], -1, 8), w, "shanten_prev")
    _need(t["drawn"] is not None or t["shanten_prev"] is None, w, "drawn 이 null 인데 shanten_prev 가 있음")
    _need(_int_in(t["shanten_before"], -1, 8), w, "shanten_before")
    _check_option(t["actual"], f"{w}.actual", True)
    _need(t["actual"]["discard"] == t["discard"], w, "actual.discard 가 discard 와 다름")
    _check_option(t["best"], f"{w}.best", True)
    _need(isinstance(t["top"], list) and 1 <= len(t["top"]) <= 34, w, "top")
    for j, o in enumerate(t["top"]):
        _check_option(o, f"{w}.top[{j}]", False)
    _need(_int_in(t["rank"], 1, 34), w, "rank")
    _need(_int_in(t["shanten_loss"], -8, 8), w, "shanten_loss")
    _need(_int_in(t["ukeire_loss"], -136, 136), w, "ukeire_loss")
    _need(_tiles_ok(t["only_best"]), w, "only_best")
    _need(_tiles_ok(t["only_actual"]), w, "only_actual")
    shape = t["shape"]
    _need(isinstance(shape, dict), w, "shape")
    for side in SHAPE_SIDES:  # shape.drawn 은 A-8 에서 더함(필수, 비어도 됨)
        _need(side in shape, w, f"shape.{side} 없음")
        blocks = shape[side]
        _need(isinstance(blocks, list) and len(blocks) <= 16, w, f"shape.{side}")
        for b in blocks:
            _need(isinstance(b, dict) and b.get("kind") in SHAPE_KINDS and _tiles_ok(b.get("tiles"), 3)
                  and 1 <= len(b["tiles"]) <= 3, w, f"shape.{side}[] = {{kind, tiles}}")
    _need(t["drawn"] is not None or not shape["drawn"], w, "drawn 이 null 인데 shape.drawn 이 비어 있지 않음")
    _need(isinstance(t["riichi_winds"], list) and len(t["riichi_winds"]) <= 3
          and all(x in WINDS for x in t["riichi_winds"]), w, "riichi_winds")
    _need(t["genbutsu"] is None or isinstance(t["genbutsu"], bool), w, "genbutsu")
    _need(_tiles_ok(t["safe_tiles"]), w, "safe_tiles")


def validate_request(req: dict) -> None:
    """규격 v1.1(A-2 · A-7 · A-8)에 안 맞거나 너무 큰 요청이면 ValueError. 모르는 칸은 무시한다(앞으로 더해도 깨지지 않게)."""
    _need(isinstance(req, dict), "요청", "JSON 객체가 아님")
    size = len(json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode())
    _need(size <= MAX_REQUEST_BYTES, "요청", f"너무 큼({size} > {MAX_REQUEST_BYTES} bytes)")
    _need(req.get("version") == SPEC_VERSION, "version", f"{SPEC_VERSION} 이어야 함")
    for k in ("game", "timeline", "turns"):
        _need(k in req, "요청", f"{k} 없음")
    g = req["game"]
    _need(isinstance(g, dict), "game", "객체가 아님")
    for k in ("round", "honba", "hero_wind", "dora_indicators", "efficiency_pct", "grades", "statuses"):
        _need(k in g, "game", f"{k} 없음")
    _need(isinstance(g["round"], str) and 1 <= len(g["round"]) <= 16, "game", "round")
    _need(_int_in(g["honba"], 0, 99), "game", "honba")
    _need(g["hero_wind"] in WINDS, "game", "hero_wind")
    _need(_tiles_ok(g["dora_indicators"], 10), "game", "dora_indicators")
    _need(g["efficiency_pct"] is None or _int_in(g["efficiency_pct"], 0, 100), "game", "efficiency_pct")
    for k, keys in (("grades", GRADES), ("statuses", STATUSES)):
        d = g[k]
        _need(isinstance(d, dict) and all(x in keys and _int_in(v, 0, 999) for x, v in d.items()), "game", k)

    tl = req["timeline"]
    _need(isinstance(tl, list), "timeline", "목록이 아님")
    _need(len(tl) <= MAX_TIMELINE, "timeline", f"너무 김({len(tl)} > {MAX_TIMELINE})")
    seen: set[int] = set()
    for i, t in enumerate(tl):
        w = f"timeline[{i}]"
        _need(isinstance(t, dict), w, "객체가 아님")
        for k in ("turn", "discard", "status", "grade", "shanten_after", "total", "riichi"):
            _need(k in t, w, f"{k} 없음")
        _need(_int_in(t["turn"], 1, 999) and t["turn"] not in seen, w, "turn(1 이상, 겹치지 않게)")
        seen.add(t["turn"])
        _need(_tile_ok(t["discard"]), w, "discard")
        _need(t["status"] in STATUSES, w, "status")
        _need(t["grade"] is None or t["grade"] in GRADES, w, "grade")
        _need(t["shanten_after"] is None or _int_in(t["shanten_after"], -1, 8), w, "shanten_after")
        _need(t["total"] is None or _int_in(t["total"], 0, 136), w, "total")
        _need(isinstance(t["riichi"], bool), w, "riichi")

    turns = req["turns"]
    _need(isinstance(turns, list), "turns", "목록이 아님")
    _need(len(turns) <= MAX_TURNS, "turns", f"너무 많음({len(turns)} > {MAX_TURNS})")
    for i, t in enumerate(turns):
        _check_turn(t, i, seen)
    nums = [t["turn"] for t in turns]
    _need(len(set(nums)) == len(nums), "turns", "turn 이 겹침")


# ---------------------------------------------------------------- 지시문 · 스키마


def response_schema(summary: bool = True) -> dict:
    """Gemini 구조화 출력 스키마. 다시 받을 때 총평이 통과했으면 summary 를 빼고 받는다."""
    props: dict = {}
    if summary:
        props["summary"] = {"type": "string", "description": f"판 총평. {SUMMARY_SENTENCES}문장 안, {SUMMARY_MAX}자 안"}
    props["turns"] = {
        "type": "array",
        "description": "해설할 차례마다 하나(사실표 turns 의 turn 번호 그대로)",
        "items": {
            "type": "object",
            "properties": {
                "turn": {"type": "integer"},
                "title": {"type": "string", "description": f"대괄호 포함 {TITLE_MAX}자 안"},
                "text": {"type": "string", "description": f"{TEXT_SENTENCES}문장 안, {TEXT_MAX}자 안"},
            },
            "required": ["turn", "title", "text"],
        },
    }
    return {"type": "object", "properties": props, "required": list(props)}


RESPONSE_SCHEMA = response_schema()

_NAME_TABLE = (
    "`1m`~`9m` = 1만~9만, `1p`~`9p` = 1통~9통, `1s`~`9s` = 1삭~9삭, "
    "`1z` 동 · `2z` 남 · `3z` 서 · `4z` 북 · `5z` 백 · `6z` 발 · `7z` 중. 적5 도 5만 · 5통 · 5삭으로 쓴다."
)

PROMPT_RULES = f"""너는 리치마작 패효율 해설자다. 아래 사실표(JSON)의 숫자와 패는 프로그램이 계산한 값이다.
너는 사실표에 있는 숫자 · 패만 사람 말로 풀어 쓴다. 샹텐 · 유효패 · 장수 · 대기를 스스로 계산하거나 추측하지 않는다.
글은 프로그램이 사실표와 하나하나 대조해 검사한다. 아래 규칙을 어긴 항목은 버려진다.

## 1. 패 이름
- 사실표의 패 코드 → 이름: {_NAME_TABLE}
- 글 안의 패는 꼭 대괄호 하나에 한 장씩 이름으로 쓴다: [3삭] · [중] · [5만]. 여러 장이면 [3만] · [8삭] 처럼 따로 묶는다.
- 대괄호 안에는 패 이름 하나만 넣는다([3s] 같은 코드, [3삭 2장] 같은 말 넣기 금지). 대괄호 밖에 "3삭" 이나 "3s" 처럼 쓰지 않는다.
- 차례 글에는 그 차례의 hand · drawn · discard · actual.tiles · best.tiles · best.discard · top[].discard · safe_tiles 에 있는 패만 쓴다.
  아래 "쓸 수 있는 패" 표에 차례마다 모아 두었다. 총평에는 "총평에 쓸 수 있는 패" 만.
- 자리 바람(리치한 자리 등)은 패가 아니므로 대괄호 없이 "남 자리" 처럼 쓴다. 도라 표시패는 언급하지 않는다.

## 2. 숫자
- "N장" 은 그 차례의 actual.total · best.total · top[].total · 유효패 한 종류의 left · 손패(hand) 안 한 종류의 장수에만 쓴다.
  ukeire_loss 는 shanten_loss 가 0 이고 ukeire_loss 가 0 보다 클 때만 "N장" 으로 쓴다(같은 샹텐에서 덜 받은 장수).
  샹텐이 뒤로 간 차례(shanten_loss 1 이상)에서는 ukeire_loss 를 장수로 쓰지 않는다 — 검사에 걸린다.
- "N종" 은 actual.kinds · best.kinds · top[].kinds 에만.
- "N샹텐" 은 1 이상인 샹텐(shanten_prev · shanten_before · actual.shanten · best.shanten · top[].shanten)에만. 샹텐 0 은 "텐파이", -1 은 "화료형" 이라고 쓴다("0샹텐" 금지).
- "N순" · "N번째" 는 그 차례의 turn(몇 번째 버림인가)에만 쓴다.
- 후보 안 순위(rank)는 굳이 쓰지 않는다. 꼭 쓸 때만 "N위" 로, rank 나 top 의 자리(1위 = top[0])만.
- 점수("N점")는 쓰지 않는다. 그 밖의 숫자도 사실표에 있는 값만 쓴다. 숫자는 아라비아 숫자로 쓴다.

## 3. 샹텐 — 뽑기 전 · 버리기 전 · 버린 뒤
- shanten_prev = 뽑기 전 손(hand 에서 drawn 한 장을 뺀 것)의 샹텐. null 이면 뽑기 전 샹텐을 모르는 차례다(울기 뒤 버림 · 뽑은 패 모름).
- shanten_before = 뽑은 패까지 넣은 손(hand)에서 가장 좋은 버림을 했을 때의 샹텐. shanten_prev 보다 작으면 뽑은 패로 한 걸음 나아갈 수 있었다는 뜻이고, 같으면 그대로였다는 뜻이다.
- actual.shanten · best.shanten 은 버린 뒤의 샹텐이다.
- "유지" · "지키다" 말은 shanten_prev 로만 쓴다: "텐파이를 유지" · "텐파이 유지" · "텐파이를 지키다" 는 shanten_prev 가 0 일 때만,
  "N샹텐을 유지" · "N샹텐 유지" 는 N 이 shanten_prev 일 때만. shanten_prev 가 null 이면 쓰지 않는다. 어기면 검사에 걸린다.
  예) shanten_prev 1 · best.shanten 0 이면 "[3삭]을 버리면 텐파이가 됐다" 라고 쓴다("텐파이를 유지" 가 아니다).
- 샹텐이 뒤로 간 차례(shanten_loss 1 이상)에서는 유효패 장수가 늘거나 준 것을 손실 · 이득으로 말하지 않는다(ukeire_loss 가 음수일 수 있다). 샹텐 차이가 먼저다.

## 4. 모양 말
- 양면 · 간짱 · 변짱 · 또이쯔 · 커쯔 · 순자 · 뜬 패(고립)는 그 차례의 shape.discard · shape.best · shape.drawn 중 하나에 그 kind 가 있을 때만 쓴다
  (ryanmen = 양면, kanchan = 간짱, penchan = 변짱, pair = 또이쯔, triplet = 커쯔, sequence = 순자, isolated = 뜬 패). 다른 표기(료멘 · 칸짱 · 펜짱 · 또이츠 · 커츠 · 슌츠)도 같다.
- shape.discard 는 버린 패, shape.best 는 최선 후보 패, shape.drawn 은 뽑은 패(drawn)가 버리기 전 손(hand)에서 낄 수 있는 덩어리 후보를 다 적은 것이다
  — 그중 하나로 단정하지 않는다. shape 에 없으면 모양 말을 하지 않는다.
- 뽑은 패가 덩어리를 완성했는지는 shape.drawn 과 shape.discard 로만 말한다. 둘에 kind · tiles 가 같은 덩어리가 있을 때만
  "뽑은 [4통]으로 된 순자를 [2통]을 버려 헐었다" 처럼 쓴다(예: 두 목록에 다 {{"kind": "sequence", "tiles": ["2p", "3p", "4p"]}} 가 있을 때). 그런 덩어리가 없으면 완성했다 · 헐었다고 말하지 않는다.
- 대기 모양 말(샤보 · 샨폰 · 단기 · 탕키 · 노베탄 · 다면)은 쓰지 않는다 — 대기 모양 정보가 없다. "단기적" 같은 말도 쓰지 않는다.

## 5. 말투 · 금지
- 해라체("…다")로 쓴다. 존댓말("…습니다" · "…해요")을 쓰지 않는다.
- 영어를 쓰지 않는다. 대괄호 밖에 영문자가 두 자 이상 이어지면 검사에 걸린다 — 사실표의 칸 이름(best · blunder · shanten · ukeire 등)을 글에 옮기지 않는다.
  등급은 한국어 이름만: best = 최선, good = 좋음, inaccuracy = 부정확, mistake = 실수, blunder = 큰 실수.
  상태도 한국어 이름만: graded = 등급 매김, forced = 고를 수 없는 버림, defense = 수비, complete = 화료형, unknownHand = 손패 기록 없음, inconsistent = 손패 기록 어긋남.
- "반드시" · "무조건" · "절대" 같은 단정. 플레이어 이름(사실표에 없다). 결과(누가 화료했나)로 그 수를 평가하기.
- status 가 defense(다른 자리 리치 중) · complete(버리기 전 화료형) 인 차례에는 "실수" · "잘못" · "틀렸다" · "틀린" 을 쓰지 않는다.

## 6. 차례 해설(turns) — 사실표 turns 의 차례마다 하나, 같은 turn 번호로
- title: 대괄호 포함 {TITLE_MAX}자 안. text: {TEXT_SENTENCES}문장 안 · {TEXT_MAX}자 안, 문장은 마침표로 끝낸다. 한국어로.
- graded 차례의 뼈대(한 문장에 하나씩, 숫자를 다 늘어놓지 말고 까닭에 필요한 것만 고른다):
  ① 무엇이 문제였나 — "패효율 기준" 임을 밝히고 grade 이름과 함께, 샹텐이 뒤로 갔나(shanten_loss 1 이상), 같은 샹텐에서 받는 패가 줄었나(shanten_loss 0).
  ② 최선(best)과 숫자 비교 — 버린 뒤 샹텐 · 유효패 장수(actual 대 best).
  ③ 모양으로 본 까닭 — 4절의 근거가 shape 에 있을 때만(예: "뽑은 [4통]으로 된 순자를 [2통]을 버려 헐었다"). 근거가 없으면 ③ 은 쓰지 않는다.
- only_best · only_actual(한쪽 유효패에만 있는 패)은 까닭을 말하는 데 도움이 될 때만 쓴다. 버린 패 자신이 유효패 목록에 있어도 그 사실은 말하지 않는다.
- defense 차례: 다른 자리 리치(riichi_winds)가 있어 패효율 등급을 매기지 않았다는 점, 버린 패가 현물이었는지
  (genbutsu true = 현물, false = 현물 아님, null = 판단 없음), 손 안의 현물(safe_tiles)만 사실대로 쓴다. 좋다 · 나쁘다 평하지 않는다.
- complete 차례: 버리기 전 손이 화료형이었다는 사실과 버린 패만 쓴다. 평가하지 않는다.

## 7. 총평(summary)
- {SUMMARY_SENTENCES}문장 안 · {SUMMARY_MAX}자 안. timeline(hero 버림 전부, 순서대로)의 흐름 — 초반 · 중반 · 리치 여부 — 과 요약 숫자(game.grades · game.efficiency_pct).
- game.grades 는 한국어 등급 이름으로 쓴다(예: "최선 5번, 큰 실수 1번"). 영어 칸 이름을 옮기지 않는다.
- 패효율은 "패효율 N%" 로 game.efficiency_pct 값 그대로 쓴다(null 이면 패효율 숫자를 쓰지 않는다). "N순" 은 timeline 의 turn 만.
- 등급은 패효율만 본 것이라 수비 · 타점 · 역 때문에 일부러 물러난 수일 수 있다는 단서를 총평에서 한 번 붙인다(차례 글마다 되풀이하지 않는다).
- 총평에는 모양 말과 "유지" · "지키다" 말을 쓰지 않는다(뽑기 전 샹텐은 차례마다 달라 총평에서 맞출 근거가 없다).

## 사실표 칸
- game: round 국 · honba 본장 · hero_wind 내 자리 바람 · efficiency_pct 패효율(0~100) · grades 등급별 개수(best 최선 · good 좋음 · inaccuracy 부정확 · mistake 실수 · blunder 큰 실수) · statuses 상태별 개수.
- timeline[]: turn 몇 번째 버림 · discard 버린 패 · status · grade · shanten_after 버린 뒤 샹텐 · total 버린 뒤 유효패 장수 · riichi 리치 선언.
- turns[]: hand 버리기 전 손패(뽑은 패 포함) · drawn 뽑은 패 · discard 버린 패 · melds 울기 수 · shanten_prev 뽑기 전 샹텐 ·
  shanten_before 버리기 전(뽑은 패 포함) 샹텐 · actual 실제 버림의 결과(shanten 버린 뒤 샹텐 · total 유효패 장수 · kinds 남은 종류 수 ·
  tiles 유효패와 남은 장수 left) · best 최선 후보 · top 후보 위에서 3개 · rank 실제 버림의 순위 · shanten_loss(actual.shanten − best.shanten) ·
  ukeire_loss(best.total − actual.total, 샹텐이 다르면 음수일 수 있다) · only_best · only_actual ·
  shape(discard 버린 패 · best 최선 후보 패 · drawn 뽑은 패가 낄 수 있는 덩어리) · riichi_winds 리치 중인 다른 자리 · genbutsu · safe_tiles.

## 답 형식
JSON: {{"summary": "…", "turns": [{{"turn": 5, "title": "…", "text": "…"}}]}}
"""


def _names_line(codes) -> str:
    return " ".join(f"{c}=[{tile_name(c)}]" for c in sorted(codes, key=_tile_key))


def build_prompt(req: dict, retry: dict | None = None) -> str:
    """지시문(r2) + 쓸 수 있는 패 표(덧붙인 것) + [retry] 면 걸린 항목의 앞선 글 · 위반 + 사실표 JSON(맨 끝, 그대로).

    [retry] = {"summary": {"text", "violations"} | None, "turns": [{"turn", "title", "text", "violations"}]}.
    다시 받을 때는 걸린 차례의 사실만 붙인다(총평도 다시 받으면 총평에 필요한 모든 차례)."""
    want_summary = retry is None or retry.get("summary") is not None
    redo = {t["turn"] for t in (retry or {}).get("turns") or []}
    turns = [t for t in req.get("turns") or [] if retry is None or want_summary or t["turn"] in redo]
    facts = {**req, "turns": turns}
    parts = [PROMPT_RULES]

    lines = ["## 쓸 수 있는 패(코드=글에 쓸 이름)"]
    for t in turns:
        if retry is None or t["turn"] in redo:
            lines.append(f"- 차례 {t['turn']}: {_names_line(_turn_allow(t).tiles)}")
    if want_summary:
        lines.append(f"- 총평: {_names_line(_game_allow(facts).tiles)}")
    parts.append("\n".join(lines))

    if retry is not None:
        r = ["## 다시 쓰기",
             "앞선 답에서 아래 항목이 검사에 걸렸다. 이 항목만 위반을 고쳐 다시 쓴다. 통과한 항목은 답에 넣지 않는다."]
        for t in retry.get("turns") or []:
            r.append(f"- 차례 {t['turn']}: 앞선 title {json.dumps(t.get('title'), ensure_ascii=False)} · "
                     f"앞선 text {json.dumps(t.get('text'), ensure_ascii=False)}")
            r += [f"  - 위반: {v}" for v in t["violations"]]
        if want_summary:
            s = retry["summary"]
            r.append(f"- 총평: 앞선 글 {json.dumps(s.get('text'), ensure_ascii=False)}")
            r += [f"  - 위반: {v}" for v in s["violations"]]
        else:
            r.append("총평(summary)은 이번 답에 넣지 않는다.")
        if not redo:
            r.append("다시 쓸 차례가 없으니 turns 는 빈 목록으로 둔다.")
        parts.append("\n".join(r))

    parts.append("## 사실표(JSON)\n" + json.dumps(facts, ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 생성 → 검증 → 다시 → 대체(A-6)

Generate = Callable[[str, dict], tuple[dict, int, int]]


def _parse(data, wanted: set[int]) -> tuple[str | None, dict[int, tuple[str | None, str | None]], int]:
    """LLM 답 → (총평, {turn: (title, text)}, 버린 차례 수). 요청하지 않은 차례 · 겹친 차례(첫 것만 씀)는 버린다."""
    if not isinstance(data, dict):
        return None, {}, 0
    summary = data.get("summary") if isinstance(data.get("summary"), str) else None
    turns: dict[int, tuple[str | None, str | None]] = {}
    dropped = 0
    for item in data.get("turns") if isinstance(data.get("turns"), list) else []:
        n = item.get("turn") if isinstance(item, dict) else None
        if isinstance(n, str) and n.strip().isdigit():
            n = int(n)
        if not _int(n) or n not in wanted or n in turns:
            dropped += 1
            continue
        title, text = item.get("title"), item.get("text")
        turns[n] = (title.strip() if isinstance(title, str) else None, text.strip() if isinstance(text, str) else None)
    return summary.strip() if summary is not None else None, turns, dropped


def explain(req: dict, generate: Generate, model: str = "") -> dict:
    """사실표(검사를 거친 요청) → A-4 응답. generate 가 던진 예외는 그대로 올린다(라우트가 502/429 로)."""
    t0 = time.perf_counter()
    facts = {t["turn"]: t for t in req["turns"]}
    order = [t["turn"] for t in req["turns"]]

    data, tok_in, tok_out = generate(build_prompt(req), RESPONSE_SCHEMA)
    summary, got, dropped = _parse(data, set(facts))
    summary_v = verify_summary(req, summary)
    texts = {n: got.get(n, (None, None)) for n in order}
    viol = {n: verify_turn(facts[n], *got[n]) if n in got else [MISSING] for n in order}
    attempts = 1

    redo = [n for n in order if viol[n]]
    if redo or summary_v:
        attempts = 2
        retry = {"summary": {"text": summary, "violations": summary_v} if summary_v else None,
                 "turns": [{"turn": n, "title": texts[n][0], "text": texts[n][1], "violations": viol[n]} for n in redo]}
        data, i2, o2 = generate(build_prompt(req, retry), response_schema(summary=bool(summary_v)))
        tok_in, tok_out = tok_in + i2, tok_out + o2
        summary2, got2, dropped2 = _parse(data, set(redo))
        dropped += dropped2
        if summary_v:
            summary, summary_v = summary2, verify_summary(req, summary2)
        for n in redo:
            texts[n] = got2.get(n, (None, None))
            viol[n] = verify_turn(facts[n], *got2[n]) if n in got2 else [MISSING]

    out_turns = []
    for n in order:
        ok = not viol[n]
        out_turns.append({"turn": n, "title": texts[n][0] if ok else None, "text": texts[n][1] if ok else None,
                          "source": "llm" if ok else "fallback", "violations": viol[n]})
    if dropped:
        log.info("review: dropped %d unrequested/duplicate turn(s) from LLM answer", dropped)
    return {
        "version": SPEC_VERSION, "model": model, "prompt_version": PROMPT_VERSION,
        "summary": {"text": summary if not summary_v else None, "source": "fallback" if summary_v else "llm",
                    "violations": summary_v},
        "turns": out_turns,
        "attempts": attempts, "input_tokens": tok_in, "output_tokens": tok_out,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


# ---------------------------------------------------------------- 실제 LLM 호출(글만)


class GeminiTextGenerator:
    """google-genai 로 글만 보내 JSON 스키마 응답을 받는 `generate`. (GeminiRecognizer 는 사진 전용.)

    시간 초과 · 5xx · 429 면 한 번 더 보낸다(429 는 [rate_backoff_s] 쉬고). 그래도 실패하면 마지막 예외를 그대로 던진다.
    [timeout_s] 는 호출 하나의 HTTP 기한."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", thinking: str = DEFAULT_THINKING,
                 timeout_s: float = 60.0, attempts: int = 2, rate_backoff_s: tuple[float, float] = (1.0, 3.0),
                 client=None):
        from google.genai import types

        self._types = types
        self.model = model
        self.thinking = thinking
        self.attempts = max(1, attempts)
        self.rate_backoff_s = rate_backoff_s
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.client = client

    def _config(self, schema: dict):
        types = self._types
        kwargs = dict(response_mime_type="application/json", response_schema=schema)
        if self.thinking == "off":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif self.thinking == "low":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        return types.GenerateContentConfig(**kwargs)

    def __call__(self, prompt: str, schema: dict) -> tuple[dict, int, int]:
        for i in range(self.attempts):
            try:
                response = self.client.models.generate_content(model=self.model, contents=[prompt],
                                                               config=self._config(schema))
                break
            except Exception as e:  # noqa: BLE001
                if not retryable(e) or i == self.attempts - 1:
                    raise
                if rate_limited(e):
                    pause = random.uniform(*self.rate_backoff_s)
                    log.warning("review gemini 429: %.1fs 쉬고 다시 보낸다", pause)
                    time.sleep(pause)
                else:
                    log.warning("review gemini retry %d/%d: %s", i + 2, self.attempts, type(e).__name__)
        um = response.usage_metadata
        inp = (um.prompt_token_count or 0) if um else 0
        out = ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
        if not response.text:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            raise RuntimeError(f"empty response ({reason})")
        return json.loads(response.text), inp, out
