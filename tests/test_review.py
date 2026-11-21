"""패보 리뷰 해설(app/review.py · /v2/review/explain). LLM 은 전부 가짜(주입 함수).

사실표 견본은 mahjong_calc 설계서 1118 부록 A-2 예시를 바탕으로 손으로 만든 것(숫자는 서로 맞게만, 엔진 계산값 아님)."""

import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from app import main as server
from app import review as rv
from app.review import explain, validate_request, verify_summary, verify_turn

# ---------------------------------------------------------------- 견본


def _blunder_turn() -> dict:
    """5번째 버림: 뽑기 전 1샹텐, [4통]을 뽑아 텐파이가 될 수 있었다. [2통]을 버려 1샹텐(유효패 10장 4종),
    최선은 [3삭] 버려 텐파이(4장 2종). 뽑은 [4통]과 버린 [2통]이 같은 순자 [2통][3통][4통]에 낀다."""
    return {
        "turn": 5, "status": "graded", "grade": "blunder",
        "hand": ["3m", "3m", "7m", "8m", "9m", "2p", "3p", "4p", "3s", "3s", "4s", "5s", "8s", "8s"],
        "drawn": "4p", "discard": "2p", "melds": 0, "riichi_declared": False,
        "shanten_prev": 1, "shanten_before": 0,
        "actual": {"discard": "2p", "shanten": 1, "total": 10, "kinds": 4,
                   "tiles": [{"tile": "3m", "left": 2}, {"tile": "2p", "left": 3}, {"tile": "5p", "left": 3},
                             {"tile": "8s", "left": 2}]},
        "best": {"discard": "3s", "shanten": 0, "total": 4, "kinds": 2,
                 "tiles": [{"tile": "3m", "left": 2}, {"tile": "8s", "left": 2}]},
        "top": [{"discard": "3s", "shanten": 0, "total": 4, "kinds": 2},
                {"discard": "8s", "shanten": 0, "total": 4, "kinds": 2},
                {"discard": "9m", "shanten": 1, "total": 16, "kinds": 5}],
        "rank": 7, "shanten_loss": 1, "ukeire_loss": -6,
        "only_best": [], "only_actual": ["2p", "5p"],
        "shape": {"discard": [{"kind": "sequence", "tiles": ["2p", "3p", "4p"]},
                              {"kind": "ryanmen", "tiles": ["2p", "3p"]},
                              {"kind": "kanchan", "tiles": ["2p", "4p"]}],
                  "best": [{"kind": "pair", "tiles": ["3s", "3s"]},
                           {"kind": "sequence", "tiles": ["3s", "4s", "5s"]},
                           {"kind": "ryanmen", "tiles": ["3s", "4s"]},
                           {"kind": "kanchan", "tiles": ["3s", "5s"]}],
                  "drawn": [{"kind": "sequence", "tiles": ["2p", "3p", "4p"]},
                            {"kind": "ryanmen", "tiles": ["3p", "4p"]},
                            {"kind": "kanchan", "tiles": ["2p", "4p"]}]},
        "riichi_winds": [], "genbutsu": None, "safe_tiles": [],
    }


def _mistake_turn() -> dict:
    """4번째 버림(같은 샹텐에서 덜 받음): 뽑기 전 2샹텐, [3통]을 뽑아 1샹텐. [1삭]을 버려 1샹텐 12장,
    최선은 [9만] 버려 1샹텐 18장 → ukeire_loss 6(다른 칸에 6 이 없다). 양면은 shape.drawn 에만 있다."""
    return {
        "turn": 4, "status": "graded", "grade": "mistake",
        "hand": ["3m", "3m", "7m", "8m", "9m", "2p", "3p", "1s", "3s", "3s", "4s", "5s", "8s", "8s"],
        "drawn": "3p", "discard": "1s", "melds": 0, "riichi_declared": False,
        "shanten_prev": 2, "shanten_before": 1,
        "actual": {"discard": "1s", "shanten": 1, "total": 12, "kinds": 4,
                   "tiles": [{"tile": "3m", "left": 2}, {"tile": "1p", "left": 4}, {"tile": "4p", "left": 4},
                             {"tile": "8s", "left": 2}]},
        "best": {"discard": "9m", "shanten": 1, "total": 18, "kinds": 6,
                 "tiles": [{"tile": "3m", "left": 2}, {"tile": "1p", "left": 4}, {"tile": "4p", "left": 4},
                           {"tile": "2s", "left": 3}, {"tile": "6s", "left": 3}, {"tile": "8s", "left": 2}]},
        "top": [{"discard": "9m", "shanten": 1, "total": 18, "kinds": 6},
                {"discard": "7m", "shanten": 1, "total": 14, "kinds": 5},
                {"discard": "1s", "shanten": 1, "total": 12, "kinds": 4}],
        "rank": 3, "shanten_loss": 0, "ukeire_loss": 6,
        "only_best": ["2s", "6s"], "only_actual": [],
        "shape": {"discard": [{"kind": "kanchan", "tiles": ["1s", "3s"]}],
                  "best": [{"kind": "sequence", "tiles": ["7m", "8m", "9m"]},
                           {"kind": "penchan", "tiles": ["8m", "9m"]},
                           {"kind": "kanchan", "tiles": ["7m", "9m"]}],
                  "drawn": [{"kind": "ryanmen", "tiles": ["2p", "3p"]}]},
        "riichi_winds": [], "genbutsu": None, "safe_tiles": [],
    }


def _defense_turn() -> dict:
    """9번째 버림: 남 자리 리치 중, 현물 [6만]을 버림. 손에 [중] 2장 · [동] 1장."""
    return {
        "turn": 9, "status": "defense", "grade": None,
        "hand": ["1m", "2m", "3m", "6m", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "1z", "7z", "7z"],
        "drawn": "1z", "discard": "6m", "melds": 0, "riichi_declared": False,
        "shanten_prev": 1, "shanten_before": 1,
        "actual": {"discard": "6m", "shanten": 1, "total": 8, "kinds": 3,
                   "tiles": [{"tile": "4p", "left": 3}, {"tile": "7p", "left": 2}, {"tile": "1z", "left": 3}]},
        "best": {"discard": "1z", "shanten": 1, "total": 11, "kinds": 4,
                 "tiles": [{"tile": "3p", "left": 3}, {"tile": "8p", "left": 3}, {"tile": "6m", "left": 3},
                           {"tile": "7z", "left": 2}]},
        "top": [{"discard": "1z", "shanten": 1, "total": 11, "kinds": 4},
                {"discard": "6m", "shanten": 1, "total": 8, "kinds": 3},
                {"discard": "1m", "shanten": 2, "total": 20, "kinds": 6}],
        "rank": 2, "shanten_loss": 0, "ukeire_loss": 3,
        "only_best": ["3p", "6m", "7z", "8p"], "only_actual": ["1z", "4p", "7p"],
        "shape": {"discard": [{"kind": "isolated", "tiles": ["6m"]}],
                  "best": [{"kind": "isolated", "tiles": ["1z"]}],
                  "drawn": [{"kind": "isolated", "tiles": ["1z"]}]},
        "riichi_winds": ["남"], "genbutsu": True, "safe_tiles": ["6m", "1z"],
    }


def _timeline() -> list[dict]:
    rows = [(1, "9m", "graded", "best", 3, 20), (2, "1z", "graded", "best", 2, 30), (3, "8p", "graded", "best", 2, 28),
            (4, "1s", "graded", "mistake", 1, 12), (5, "2p", "graded", "blunder", 1, 10),
            (6, "4z", "graded", "best", 1, 14), (7, "9s", "graded", "best", 1, 15), (8, "5z", "graded", "best", 1, 18),
            (9, "6m", "defense", None, 1, 8)]
    return [{"turn": n, "discard": d, "status": s, "grade": g, "shanten_after": sh, "total": tot, "riichi": False}
            for n, d, s, g, sh, tot in rows]


def _request(turns=None) -> dict:
    return {
        "version": 1,
        "game": {"round": "남4국", "honba": 0, "hero_wind": "서", "dora_indicators": ["3z", "2m"],
                 "efficiency_pct": 80,
                 "grades": {"best": 6, "good": 0, "inaccuracy": 0, "mistake": 1, "blunder": 1},
                 "statuses": {"graded": 8, "forced": 0, "defense": 1, "complete": 0, "unknownHand": 0,
                              "inconsistent": 0}},
        "timeline": _timeline(),
        "turns": [_blunder_turn(), _defense_turn()] if turns is None else turns,
    }


def _empty_request() -> dict:
    """해설할 차례가 없는 판(총평만)."""
    req = _request(turns=[])
    req["game"]["efficiency_pct"] = None
    return req


GOOD_TITLE_5 = "[2통] 대신 [3삭]"
GOOD_TEXT_5 = ("패효율 기준으로 큰 실수다. [2통]을 버려 1샹텐이 됐지만 [3삭]을 버리면 텐파이였다. "
               "유효패는 10장 4종 대 4장 2종이었다.")
GOOD_TITLE_9 = "리치 뒤 현물 [6만]"
GOOD_TEXT_9 = "남 자리 리치가 있어 패효율 등급은 매기지 않았다. 버린 [6만]은 현물이었다. 손 안의 현물은 [6만]과 [동]이었다."
GOOD_SUMMARY = ("초반에는 넓게 받았지만 5순에 텐파이를 놓쳤다. 패효율 80%로 큰 실수 1번, 실수 1번이 있었다. "
                "9순에는 남 자리 리치에 현물로 물러났다. 등급은 패효율만 본 것이라 일부러 물러난 수일 수 있다.")


def _good_answer() -> dict:
    return {"summary": GOOD_SUMMARY,
            "turns": [{"turn": 5, "title": GOOD_TITLE_5, "text": GOOD_TEXT_5},
                      {"turn": 9, "title": GOOD_TITLE_9, "text": GOOD_TEXT_9}]}


def _v5(text, title=GOOD_TITLE_5):
    return verify_turn(_blunder_turn(), title, text)


def _v9(text, title=GOOD_TITLE_9):
    return verify_turn(_defense_turn(), title, text)


def _has(violations, word):
    return any(word in v for v in violations)


# ---------------------------------------------------------------- verify_turn


def test_good_texts_pass():
    assert _v5(GOOD_TEXT_5) == []
    assert _v9(GOOD_TEXT_9) == []
    assert verify_summary(_request(), GOOD_SUMMARY) == []


def test_tile_in_brackets_must_be_in_this_turn():
    assert _v5("[2통]과 [3삭]과 [9만]은 이 차례 사실표에 있다.") == []      # hand · best.discard · top 의 9m
    v = _v5("[9삭]을 남겼어야 했다.")
    assert _has(v, "사실표에 없는 패: [9삭]")


def test_bracket_must_hold_a_tile_name():
    for bad in ("[3s]", "[삼삭]", "[3삭 2장]", "[]", "[ 3삭]"):
        v = _v5(f"{bad}을 버렸다.")
        assert _has(v, "패 이름이 아님"), bad


def test_tile_outside_brackets_is_a_violation():
    v = _v5("3삭을 버렸어야 했다.")
    assert _has(v, "대괄호 없이 쓴 패: 3삭")
    assert _has(_v5("[2통] 대신 8통을 남겼다."), "대괄호 없이 쓴 패: 8통")
    assert _has(_v5("[2통] 대신 3s 를 버렸다."), "대괄호 없이 쓴 패 코드: 3s")


def test_honor_name_only_counts_inside_brackets():
    # [중] 은 패(7z): 방어 차례 손에 있으면 통과, 큰 실수 차례에는 없으니 위반
    assert _v9("남 자리 리치 중이라 등급은 매기지 않았다. [중] 2장은 남겼다.") == []
    assert _has(_v5("[중]을 버렸다."), "사실표에 없는 패: [중]")
    # 대괄호 밖 "리치 중" · "동 자리" 는 보통 말
    assert _v5("남 자리 리치 중이 아니었다. 동 자리도 조용했다.") == []


def test_numbers_jang():
    assert _v5("유효패는 10장이었다.") == []
    # |ukeire_loss| 6 은 샹텐 손실 차례(shanten_loss 1)라 허용값이 아니다(v1.1, A-8 의 1 — v1 에서는 통과였다)
    assert _has(_v5("유효패 차이는 6장이다."), "사실표에 없는 숫자: 6장")
    assert _v5("[3만]은 2장 남았다.") == []                 # left
    assert _v5("후보 [9만]은 16장이다.") == []              # top[].total
    assert _has(_v5("유효패는 11장이었다."), "사실표에 없는 숫자: 11장")
    assert _has(_v5("유효패는 11 장이었다."), "11 장")


def test_numbers_jong_shanten_jun():
    assert _v5("유효패는 2종이었다.") == []
    assert _has(_v5("유효패는 3종이었다."), "3종")
    assert _v5("버린 뒤 1샹텐이었다.") == []
    v = _v5("버리기 전엔 0샹텐이었다.")                      # 0 은 허용값이 아님(텐파이라고 써야)
    assert _has(v, "0샹텐")
    assert _has(_v5("버린 뒤 2샹텐이었다."), "2샹텐")
    assert _v5("5순에 버렸다. 5번째 버림이다.") == []
    assert _has(_v5("6순에 버렸다."), "6순")
    assert _has(_v5("3번째 버림이다."), "3번째")


def test_rank_is_not_turn():
    # "7순위" · "7위" 는 순(turn)이 아니라 후보 순위: rank(7) · top 자리(1~3)만
    assert _v5("실제 버림은 후보 중 7위였다.") == []
    assert _v5("실제 버림은 7순위였다. [3삭]은 1위였다.") == []
    assert _has(_v5("실제 버림은 9위였다."), "9위")


def test_points_are_a_violation():
    assert _has(_v5("1000점 손해였다."), "점수 말")
    assert _has(_v5("8,000점을 놓쳤다."), "점수 말")
    assert _v5("장점이 있었다.") == []                       # 숫자 없는 "점" 은 상관없음


def test_shape_words_need_shape_hint():
    assert _v5("[2통]은 양면의 일부였다.") == []              # ryanmen 이 shape.discard 에
    assert _v5("[3삭]은 또이쯔이기도 했다.") == []           # pair 가 shape.best 에
    assert _v5("간짱도 순자도 될 수 있었다.") == []
    assert _has(_v5("[2통]은 변짱이었다."), "변짱")            # penchan 없음
    assert _has(_v5("[2통]은 뜬 패였다."), "뜬 패")            # isolated 없음
    assert _has(_v5("[2통]은 커쯔였다."), "커쯔")
    assert _v9("버린 [6만]은 뜬 패였다.") == []               # 방어 차례엔 isolated 가 있다
    assert _has(_v9("[중]은 또이쯔였다."), "또이쯔")
    assert _has(_v9("[중]은 또이츠였다."), "또이츠")           # 다른 표기도 같다


def test_wait_shape_words_always_fail():
    assert _has(_v5("[3만] · [8삭] 샤보 대기였다."), "샤보")   # 실제로 샤보여도 대기 힌트가 없어 위반
    assert _has(_v5("단기 대기였다."), "단기")
    assert _has(_v5("노베탄이었다."), "노베탄")
    assert _has(_v5("다면 대기를 놓쳤다."), "다면")
    assert _v5("[3삭]을 버렸다면 텐파이였다.") == []           # 조건 어미 "~다면" 은 모양 말이 아님


def test_status_words_on_defense():
    assert _has(_v9("현물을 버렸으니 실수가 아니다."), "defense 차례에 평가 말: 실수")
    assert _has(_v9("버린 [6만]이 잘못은 아니다."), "잘못")
    assert _has(_v9("틀린 수는 아니다."), "틀린")
    assert _v5(GOOD_TEXT_5 + "") == []                        # graded 차례는 "실수" 를 써도 된다
    complete = {**_blunder_turn(), "status": "complete", "grade": None, "shanten_before": -1}
    assert _has(verify_turn(complete, "화료형", "버리기 전 손이 화료형이었다. 틀렸다."), "complete 차례에 평가 말: 틀렸")


def test_forbidden_words():
    for w in ("반드시", "무조건", "절대"):
        assert _has(_v5(f"{w} [3삭]을 버려야 했다."), f"금지어: {w}")


def test_length_and_sentences():
    assert _has(_v5(GOOD_TEXT_5, title="가" * 21), "제목 21자")
    assert _v5(GOOD_TEXT_5, title="가" * 20) == []
    assert _has(_v5(GOOD_TEXT_5, title=""), "제목 없음")
    assert _has(_v5(GOOD_TEXT_5, title=None), "제목 없음")
    assert _has(_v5(None), "본문 없음")
    assert _has(_v5("가" * 201), "본문 201자")
    assert _has(_v5("하나다. 둘이다. 셋이다. 넷이다."), "문장 4개")
    assert _v5("하나다. 둘이다! 셋이다?") == []
    assert _v5("하나다... 둘이다. 셋이다") == []               # 연달은 마침표는 하나, 끝 조각도 한 문장
    assert _has(_v5("하나다. 둘이다. 셋이다. 넷이다"), "문장 4개")
    assert _has(_v5("abc", title="xyz"), "한글 없음")


# ---------------------------------------------------------------- verify_summary


def test_summary_numbers_are_the_union_of_all_turns():
    req = _request()
    assert verify_summary(req, "방어 차례의 최선은 11장이었다.") == []    # 차례 9 의 best.total
    assert _has(_v5("최선은 11장이었다."), "11장")                     # 차례 5 에서는 위반
    assert verify_summary(req, "두 번째 버림 뒤 30장이었다.") == []     # timeline total(차례 사실에는 없음)
    assert verify_summary(req, "한때 3샹텐이었다.") == []               # timeline shanten_after
    assert _has(verify_summary(req, "한때 4샹텐이었다."), "4샹텐")
    assert _has(verify_summary(req, "유효패 99장."), "99장")


def test_summary_efficiency_and_turns():
    req = _request()
    assert verify_summary(req, "패효율 80%였다.") == []
    assert verify_summary(req, "패효율은 80이었다.") == []
    assert _has(verify_summary(req, "패효율 90%였다."), "패효율이 다름")
    assert _has(verify_summary(_empty_request(), "패효율 80%였다."), "패효율 숫자가 없는 판")
    assert verify_summary(req, "9순에 물러났다.") == []
    assert _has(verify_summary(req, "12순에 물러났다."), "12순")      # timeline 에 없는 순


def test_summary_tiles_length_and_words():
    req = _request()
    assert verify_summary(req, "[1동]") != []
    assert verify_summary(req, "[북]을 먼저 버렸다.") == []           # timeline discard 4z
    assert verify_summary(req, "[중]을 지켰다.") == []                 # 차례 9 의 hand
    assert _has(verify_summary(req, "[7삭]을 버렸다."), "이 판 사실표에 없는 패: [7삭]")
    assert _has(verify_summary(req, "반드시 이겼다."), "금지어")
    assert _has(verify_summary(req, "1000점."), "점수 말")
    assert _has(verify_summary(req, "가. 나. 다. 라. 마."), "문장 5개")
    assert _has(verify_summary(req, "가" * 301), "301자")
    assert verify_summary(req, None) == ["총평 없음"]
    assert verify_summary(req, "  ") == ["총평 없음"]


# ---------------------------------------------------------------- 규격 v1.1(A-8) 검증 고침


def _v4(text, title="[1삭] 대신 [9만]"):
    return verify_turn(_mistake_turn(), title, text)


def test_ukeire_loss_counts_only_on_same_shanten():
    # 같은 샹텐(shanten_loss 0)에서 덜 받은 장수(ukeire_loss 6, 다른 칸엔 6 없음)는 통과
    assert _v4("패효율 기준 실수다. [9만]을 버렸으면 6장 더 받았다.") == []
    # 샹텐이 뒤로 간 차례(shanten_loss 1, ukeire_loss −6)에서는 6장이 위반 — 첫 실제 호출 뒤 찾은 빈틈("대기 6장")
    assert _has(_v5("[3삭]을 버렸으면 대기 6장이었다."), "사실표에 없는 숫자: 6장")
    # 같은 샹텐이어도 ukeire_loss 가 0 이하이면 넣지 않는다
    assert _has(verify_turn({**_mistake_turn(), "ukeire_loss": -6}, "제목이다", "6장 차이다."), "6장")
    # 총평의 합집합도 같은 기준: 견본 판(차례 5 · 9)엔 6 이 없고, 차례 4 를 넣으면 생긴다
    assert _has(verify_summary(_request(), "유효패 6장 차이가 났다."), "이 판 사실표에 없는 숫자: 6장")
    three = _request([_mistake_turn(), _blunder_turn(), _defense_turn()])
    assert verify_summary(three, "유효패 6장 차이가 났다.") == []


def test_shanten_prev_is_an_allowed_shanten():
    assert _v4("뽑기 전에는 2샹텐이었다.") == []                       # shanten_prev 2(다른 칸엔 2 없음)
    assert _has(verify_turn({**_mistake_turn(), "shanten_prev": None}, "제목이다", "뽑기 전에는 2샹텐이었다."),
                "사실표에 없는 숫자: 2샹텐")
    assert _has(_v4("뽑기 전에는 3샹텐이었다."), "3샹텐")
    three = _request([_mistake_turn(), _blunder_turn(), _defense_turn()])
    assert verify_summary(three, "한때 2샹텐이었다.") == []              # 총평 합집합에도 들어간다


def test_english_is_a_violation():
    assert _has(_v5("패효율 기준 best 5회 중 하나가 아니었다."), "영어(한국어로 써야 함): best")
    assert _has(_v5(GOOD_TEXT_5, title="blunder [2통]"), "영어(한국어로 써야 함): blunder")
    assert _has(_v5("버린 뒤 shanten 이 늘었다."), "shanten")
    v = verify_summary(_request(), "패효율 80%로 best 5회, mistake 1회, blunder 1회였다.")   # 첫 실제 호출(low)의 꼴
    assert _has(v, "영어(한국어로 써야 함): best") and _has(v, "mistake") and _has(v, "blunder")
    # 대괄호 안 이름 · 한글만 있는 글은 통과, 영문 한 자는 셈하지 않는다
    assert _v5("[3삭]을 버렸으면 텐파이였다. [2통] [3통] [4통]은 이어져 있었다.") == []
    assert _v5(GOOD_TEXT_5) == [] and verify_summary(_request(), GOOD_SUMMARY) == []
    assert _v5("A 쪽보다 [3삭] 쪽이 나았다.") == []
    # 대괄호 안의 코드 · 영어는 "패 이름이 아님" 으로만, 대괄호 밖 패 코드(영문 한 자)는 "패 코드" 로만 잡힌다
    v = _v5("[3s]을 버렸다. [best]였다.")
    assert _has(v, "패 이름이 아님") and not _has(v, "영어")
    v = _v5("3s 를 버렸다.")
    assert _has(v, "대괄호 없이 쓴 패 코드: 3s") and not _has(v, "영어")


def test_keep_words_follow_shanten_prev():
    # 첫 실제 호출의 틀림: 뽑기 전 1샹텐(shanten_prev 1)인데 "텐파이를 유지"
    assert _has(_v5("[3삭]을 버리면 텐파이를 유지하며 4장이었다."), "뽑기 전 손은 1샹텐인데 유지 말: 텐파이를 유지")
    for text in ("텐파이 유지가 나았다.", "텐파이를 지키는 수는 [3삭]이었다.", "텐파이를 지켰어야 했다."):
        assert _has(_v5(text), "유지 말"), text
    tenpai = {**_blunder_turn(), "shanten_prev": 0}
    for text in ("[3삭]을 버리면 텐파이를 유지했다.", "텐파이 유지가 나았다.", "텐파이를 지키는 수였다."):
        assert verify_turn(tenpai, GOOD_TITLE_5, text) == [], text
    unknown = {**_blunder_turn(), "shanten_prev": None}
    assert _has(verify_turn(unknown, GOOD_TITLE_5, "텐파이를 유지했다."), "뽑기 전 샹텐을 모르는 차례에 유지 말: 텐파이를 유지")
    assert _has(verify_turn(unknown, GOOD_TITLE_5, "1샹텐을 유지했다."), "뽑기 전 샹텐을 모르는 차례에 유지 말: 1샹텐을 유지")
    # "N샹텐을 유지" 는 N == shanten_prev 일 때만
    assert _v5("[2통]을 버려 1샹텐을 유지했다.") == []
    assert _v5("1샹텐 유지에 그쳤다.") == []
    assert _has(_v4("1샹텐을 유지했다."), "뽑기 전 손은 2샹텐인데 유지 말: 1샹텐을 유지")
    assert _has(verify_turn(tenpai, GOOD_TITLE_5, "1샹텐을 유지했다."), "뽑기 전 손은 텐파이인데 유지 말")
    # 텐파이 · 샹텐에 붙지 않은 "유지" 는 상관없다
    assert _v5("[3삭]을 버리면 텐파이가 됐다. 형태 유지는 [2통] 쪽이었다.") == []
    # 총평에는 이 규칙을 쓰지 않는다(차례별 근거가 없다 — 지시문에서 막는다)
    assert verify_summary(_request(), "5순에 텐파이를 유지하지 못했다.") == []


def test_shape_words_may_rest_on_shape_drawn():
    assert _v4("뽑은 [3통]은 양면이 됐다.") == []                     # ryanmen 은 shape.drawn 에만 있다
    no_drawn = _mistake_turn()
    no_drawn["shape"]["drawn"] = []
    assert _has(verify_turn(no_drawn, "제목이다", "뽑은 [3통]은 양면이 됐다."), "모양 말 근거 없음(shape 에 ryanmen 없음): 양면")
    assert _v5("뽑은 [4통]으로 된 순자를 [2통]을 버려 헐었다.") == []  # 지시문 r2 의 예시 꼴


# ---------------------------------------------------------------- explain


class FakeGen:
    """답을 차례로 돌려주는 가짜 generate. 받은 prompt · schema 를 적어 둔다. 예외를 넣으면 던진다."""

    def __init__(self, *answers, tokens=(100, 20)):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.schemas: list[dict] = []
        self.tokens = tokens
        self.model = "fake-model"

    def __call__(self, prompt, schema):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return copy.deepcopy(a), *self.tokens


def _facts_in_prompt(prompt: str) -> dict:
    return json.loads(prompt.split("## 사실표(JSON)\n", 1)[1])


def test_explain_first_try_passes():
    gen = FakeGen(_good_answer())
    out = explain(_request(), gen, "m1")
    assert out["attempts"] == 1 and len(gen.prompts) == 1
    assert out["version"] == 1 and out["model"] == "m1" and out["prompt_version"] == "r2"
    assert out["summary"] == {"text": GOOD_SUMMARY, "source": "llm", "violations": []}
    assert [t["turn"] for t in out["turns"]] == [5, 9]
    assert out["turns"][0] == {"turn": 5, "title": GOOD_TITLE_5, "text": GOOD_TEXT_5, "source": "llm", "violations": []}
    assert out["input_tokens"] == 100 and out["output_tokens"] == 20 and out["elapsed_ms"] >= 0
    assert gen.schemas[0] == rv.RESPONSE_SCHEMA
    assert _facts_in_prompt(gen.prompts[0]) == _request()      # 사실표는 JSON 그대로


def test_explain_retries_only_the_failed_turn():
    bad = _good_answer()
    bad["turns"][1]["text"] = "현물을 버렸으니 실수가 아니다."
    fixed = {"turns": [{"turn": 9, "title": GOOD_TITLE_9, "text": GOOD_TEXT_9}]}
    gen = FakeGen(bad, fixed, tokens=(50, 10))
    out = explain(_request(), gen, "m")
    assert out["attempts"] == 2
    assert out["turns"][1]["source"] == "llm" and out["turns"][1]["text"] == GOOD_TEXT_9
    assert out["turns"][0]["text"] == GOOD_TEXT_5 and out["summary"]["source"] == "llm"
    retry = gen.prompts[1]
    assert [t["turn"] for t in _facts_in_prompt(retry)["turns"]] == [9]   # 걸린 차례의 사실만
    assert "차례 9:" in retry and "차례 5:" not in retry
    assert "현물을 버렸으니 실수가 아니다." in retry and "평가 말: 실수" in retry   # 앞선 글 · 위반
    assert "summary" not in gen.schemas[1]["properties"]                  # 총평은 통과했으니 안 받는다
    assert out["input_tokens"] == 100 and out["output_tokens"] == 20       # 두 번 합산


def test_explain_falls_back_after_two_failures():
    bad = _good_answer()
    bad["turns"][0]["text"] = "유효패는 11장이었다."
    again = {"turns": [{"turn": 5, "title": GOOD_TITLE_5, "text": "반드시 [3삭]을 버려야 했다."}]}
    out = explain(_request(), FakeGen(bad, again), "m")
    assert out["attempts"] == 2
    t5 = out["turns"][0]
    assert t5["source"] == "fallback" and t5["title"] is None and t5["text"] is None
    assert t5["violations"] and _has(t5["violations"], "금지어: 반드시")      # 마지막 시도의 위반
    assert not _has(t5["violations"], "11장")
    assert out["turns"][1]["source"] == "llm"


def test_explain_summary_fallback_and_retry_prompt_keeps_all_turns():
    bad = _good_answer()
    bad["summary"] = "패효율 90%였다."
    gen = FakeGen(bad, {"summary": "패효율 95%였다.", "turns": []})
    out = explain(_request(), gen, "m")
    assert out["summary"] == {"text": None, "source": "fallback", "violations": out["summary"]["violations"]}
    assert _has(out["summary"]["violations"], "95")
    assert [t["source"] for t in out["turns"]] == ["llm", "llm"]
    assert "summary" in gen.schemas[1]["properties"]
    assert len(_facts_in_prompt(gen.prompts[1])["turns"]) == 2              # 총평을 다시 쓰려면 모든 차례가 필요


def test_explain_missing_turn_is_retried_as_missing():
    first = _good_answer()
    del first["turns"][0]
    gen = FakeGen(first, {"turns": [{"turn": 5, "title": GOOD_TITLE_5, "text": GOOD_TEXT_5}]})
    out = explain(_request(), gen, "m")
    assert out["attempts"] == 2 and out["turns"][0]["source"] == "llm"
    assert "빠짐" in gen.prompts[1]
    # 다시 받아도 빠지면 fallback, 위반은 "빠짐"
    out = explain(_request(), FakeGen(first, {"turns": []}), "m")
    assert out["turns"][0]["source"] == "fallback" and _has(out["turns"][0]["violations"], "빠짐")


def test_explain_drops_unrequested_and_duplicate_turns():
    ans = _good_answer()
    ans["turns"].append({"turn": 7, "title": "요청 안 한 차례", "text": "버린다."})
    ans["turns"].append({"turn": 5, "title": "겹친 것", "text": "반드시 버린다."})
    gen = FakeGen(ans)
    out = explain(_request(), gen, "m")
    assert out["attempts"] == 1 and [t["turn"] for t in out["turns"]] == [5, 9]
    assert out["turns"][0]["title"] == GOOD_TITLE_5


def test_explain_summary_only_game():
    gen = FakeGen({"summary": "패효율 숫자는 없다. 2순에 [동]을 먼저 버렸다.", "turns": []})
    out = explain(_empty_request(), gen, "m")
    assert out["turns"] == [] and out["summary"]["source"] == "llm" and out["attempts"] == 1


def test_explain_malformed_answer_falls_back_without_raising():
    out = explain(_request(), FakeGen(["not", "a", "dict"], {"turns": "x"}), "m")
    assert out["attempts"] == 2 and out["summary"]["source"] == "fallback"
    assert all(t["source"] == "fallback" for t in out["turns"])


def test_explain_propagates_generate_errors():
    with pytest.raises(RuntimeError, match="boom"):
        explain(_request(), FakeGen(RuntimeError("boom")), "m")
    bad = _good_answer()
    bad["summary"] = "반드시."
    with pytest.raises(TimeoutError):                                  # 다시 받는 호출이 실패해도 그대로
        explain(_request(), FakeGen(bad, TimeoutError("slow")), "m")


def test_build_prompt_lists_names_and_rules():
    p = rv.build_prompt(_request())
    assert "차례 5: 3m=[3만]" in p and "7z=[중]" in p and "총평:" in p
    assert "반드시" in p and "샤보" in p and "패효율 기준" in p           # 금지 · 대기 말 · 단서를 지시문에 적음
    assert p.index("## 사실표(JSON)") > p.index("## 답 형식")


def test_build_prompt_r2_matches_new_rules():
    """지시문 r2 가 새 검증 규칙(장 허용값 · 유지 · 영어 · shape.drawn)과 말투 · 뼈대를 적었나."""
    assert rv.PROMPT_VERSION == "r2"
    p = rv.build_prompt(_request())
    for s in ("ukeire_loss 는 shanten_loss 가 0 이고 ukeire_loss 가 0 보다 클 때만",
              "shanten_prev = 뽑기 전 손", "\"텐파이를 유지\" · \"텐파이 유지\" · \"텐파이를 지키다\" 는 shanten_prev 가 0 일 때만",
              "N 이 shanten_prev 일 때만", "손실 · 이득으로 말하지 않는다",
              "shape.discard · shape.best · shape.drawn", "shape.drawn 과 shape.discard 로만",
              "뽑은 [4통]으로 된 순자를 [2통]을 버려 헐었다",
              "해라체", "영어를 쓰지 않는다", "blunder = 큰 실수", "defense = 수비",
              "순위(rank)는 굳이 쓰지 않는다", "버린 패 자신이 유효패 목록에 있어도",
              "① 무엇이 문제였나", "② 최선(best)과 숫자 비교", "③ 모양으로 본 까닭",
              "총평에는 모양 말과 \"유지\""):
        assert s in p, s
    assert "ukeire_loss 의 절댓값" not in p                              # r1 의 장 허용값 문구는 빠졌다


# ---------------------------------------------------------------- validate_request


def test_validate_accepts_samples():
    validate_request(_request())
    validate_request(_empty_request())
    validate_request(_request([_mistake_turn(), _blunder_turn(), _defense_turn()]))


def test_validate_accepts_null_shanten_prev():
    req = _request()
    req["turns"][0].update(drawn=None, shanten_prev=None)          # 울기 뒤 버림 · 뽑은 패 모름(A-8)
    req["turns"][0]["shape"]["drawn"] = []
    validate_request(req)
    req = _request()
    req["turns"][1]["shanten_prev"] = None                         # drawn 은 있어도 hand 에 없으면 null(A-8)
    validate_request(req)


@pytest.mark.parametrize("mutate, word", [
    (lambda r: r.update(version=2), "version"),
    (lambda r: r.update(turns={"turn": 5}), "turns"),
    (lambda r: r.update(timeline="x"), "timeline"),
    (lambda r: r.pop("game"), "game"),
    (lambda r: r["turns"][0].pop("shape"), "shape 없음"),
    (lambda r: r["turns"][0].pop("actual"), "actual 없음"),
    (lambda r: r["turns"][0].update(hand=["0m"] * 14), "hand"),
    (lambda r: r["turns"][0].update(status="forced"), "status"),
    (lambda r: r["turns"][0].update(grade="good"), "grade"),
    (lambda r: r["turns"][1].update(grade="blunder"), "grade 는 null"),
    (lambda r: r["turns"][0].update(turn=30), "timeline 에 없음"),
    (lambda r: r["turns"][0]["actual"].update(total="10"), "total"),
    (lambda r: r["turns"][0].update(riichi_winds=["east"]), "riichi_winds"),
    (lambda r: r["game"].update(hero_wind="west"), "hero_wind"),
    (lambda r: r["timeline"][0].pop("riichi"), "riichi 없음"),
    (lambda r: r["turns"].append(_blunder_turn()), "겹침"),
    # 규격 v1.1(A-8): 새 칸은 필수
    (lambda r: r["turns"][0].pop("shanten_prev"), "shanten_prev 없음"),
    (lambda r: r["turns"][0]["shape"].pop("drawn"), "shape.drawn 없음"),
    (lambda r: r["turns"][0].update(shanten_prev="1"), "shanten_prev"),
    (lambda r: r["turns"][0].update(shanten_prev=True), "shanten_prev"),
    (lambda r: r["turns"][0]["shape"].update(drawn=None), "shape.drawn"),
    (lambda r: r["turns"][0]["shape"].update(drawn=[{"kind": "sequence"}]), "shape.drawn"),
    (lambda r: r["turns"][0].update(drawn=None), "drawn 이 null 인데 shanten_prev"),
    (lambda r: r["turns"][0].update(drawn=None, shanten_prev=None), "drawn 이 null 인데 shape.drawn"),
])
def test_validate_rejects(mutate, word):
    req = _request()
    mutate(req)
    with pytest.raises(ValueError, match=word):
        validate_request(req)


def test_validate_rejects_too_large():
    with pytest.raises(ValueError, match="dict|객체"):
        validate_request([1, 2])
    req = _request()
    req["padding"] = "x" * (rv.MAX_REQUEST_BYTES + 1)               # 모르는 칸이어도 크기는 센다
    with pytest.raises(ValueError, match="너무 큼"):
        validate_request(req)
    req = _request()
    req["timeline"] = [{**req["timeline"][0], "turn": n} for n in range(1, rv.MAX_TIMELINE + 2)]
    req["turns"] = []
    with pytest.raises(ValueError, match="너무 김"):
        validate_request(req)
    req = _request(turns=[])
    req["timeline"] = [{**req["timeline"][0], "turn": n} for n in range(1, rv.MAX_TIMELINE + 1)]
    req["turns"] = [{**_blunder_turn(), "turn": n} for n in range(1, rv.MAX_TURNS + 2)]
    with pytest.raises(ValueError, match="너무 많음"):
        validate_request(req)


# ---------------------------------------------------------------- GeminiTextGenerator(가짜 client)


class _Usage:
    prompt_token_count = 1200
    candidates_token_count = 300
    thoughts_token_count = 500


class _Resp:
    def __init__(self, text='{"summary": "가", "turns": []}'):
        self.text = text
        self.usage_metadata = _Usage()
        self.candidates = []


class _Models:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def _gen(behaviors, thinking="default"):
    client = type("C", (), {"models": _Models(behaviors)})()
    return rv.GeminiTextGenerator("unused", model="gm", thinking=thinking, rate_backoff_s=(0.01, 0.01), client=client)


def _err(code):
    body = {"error": {"code": code, "message": "x", "status": "X"}}
    return errors.ClientError(code, body) if code < 500 else errors.ServerError(code, body)


def test_text_generator_sends_text_only_and_counts_tokens():
    g = _gen([_Resp()])
    data, inp, out = g("지시문", rv.RESPONSE_SCHEMA)
    assert data == {"summary": "가", "turns": []} and inp == 1200 and out == 800
    model, contents, config = g.client.models.calls[0]
    assert model == "gm" and contents == ["지시문"]
    assert config.response_mime_type == "application/json" and config.thinking_config is None
    assert _gen([_Resp()], "low")._config({}).thinking_config.thinking_level is not None
    assert _gen([_Resp()], "off")._config({}).thinking_config.thinking_budget == 0


def test_thinking_defaults_to_low():
    assert rv.DEFAULT_THINKING == "low"
    g = rv.GeminiTextGenerator("unused", model="gm", client=type("C", (), {"models": _Models([])})())
    assert g.thinking == "low" and g._config({}).thinking_config.thinking_level is not None
    # 라우트(app/main.py): REVIEW_THINKING 이 없으면 low, 주면 그 값 — 새 프로세스에서 모듈을 읽어 본다
    root = Path(__file__).resolve().parents[1]
    code = "import app.main as m; print(m.REVIEW_THINKING)"
    env = {k: v for k, v in os.environ.items() if k != "REVIEW_THINKING"}
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "low"
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env={**env, "REVIEW_THINKING": "default"},
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "default"


def test_text_generator_retries_once_on_429_and_5xx():
    g = _gen([_err(429), _Resp()])
    assert g("p", {})[0]["summary"] == "가" and len(g.client.models.calls) == 2
    g = _gen([_err(503), _Resp()])
    assert g("p", {})[1] == 1200
    g = _gen([_err(429), _err(429)])
    with pytest.raises(errors.ClientError) as e:
        g("p", {})
    assert e.value.code == 429 and len(g.client.models.calls) == 2
    g = _gen([_err(400)])
    with pytest.raises(errors.ClientError):
        g("p", {})
    assert len(g.client.models.calls) == 1                           # 400 은 다시 보내지 않는다
    with pytest.raises(RuntimeError, match="empty response"):
        _gen([_Resp(text="")])("p", {})


# ---------------------------------------------------------------- 라우트


@pytest.fixture
def client(monkeypatch):
    """키 없이, 새 해설 제한기, 가짜 generator. 테스트가 server.review_generator 를 바꿔 끼울 수 있다."""
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "review_generator", FakeGen(*[_good_answer()] * 5))
    monkeypatch.setattr(server, "_review_limiter", server.RateLimiter(60, 10))
    with TestClient(server.app) as c:
        yield c


PATH = "/v2/review/explain"


def test_route_200_and_shape(client, caplog):
    with caplog.at_level(logging.INFO, logger="recognizer"):
        r = client.post(PATH, json=_request())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"version", "model", "prompt_version", "summary", "turns", "attempts", "input_tokens",
                         "output_tokens", "elapsed_ms"}
    assert body["model"] == "fake-model" and body["attempts"] == 1
    assert [t["source"] for t in body["turns"]] == ["llm", "llm"] and body["summary"]["source"] == "llm"
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "review/explain turns=2 attempts=1 fallback=0" in logs
    assert "3m" not in logs and GOOD_TEXT_5 not in logs and "남4국" not in logs   # 사실표 · 해설 본문은 안 찍는다


def test_route_is_listed_in_health(client):
    assert PATH in client.get("/").json()["endpoints"]


def test_route_401_with_wrong_key(client, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "secret")
    assert client.post(PATH, json=_request()).status_code == 401
    assert client.post(PATH, json=_request(), headers={"X-Api-Key": "nope"}).status_code == 401
    assert client.post(PATH, json=_request(), headers={"X-Api-Key": "secret"}).status_code == 200


def test_route_422_out_of_spec(client):
    req = _request()
    req["version"] = 2
    r = client.post(PATH, json=req)
    assert r.status_code == 422 and "version" in r.json()["detail"]
    assert client.post(PATH, content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post(PATH, content="[]".encode()).status_code == 422


def test_route_413_too_large_body(client):
    big = b'{"version": 1, "pad": "' + b"x" * rv.MAX_REQUEST_BYTES + b'"}'
    r = client.post(PATH, content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_route_429_own_limiter(client, monkeypatch):
    monkeypatch.setattr(server, "_review_limiter", server.RateLimiter(per_min=1, burst=1))
    assert client.post(PATH, json=_request()).status_code == 200
    r = client.post(PATH, json=_request())
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # 사진 판독 제한기와는 따로 센다
    assert server._limiter.tokens == server.RATE_BURST


def test_route_422_does_not_spend_limit(client, monkeypatch):
    monkeypatch.setattr(server, "_review_limiter", server.RateLimiter(per_min=1, burst=1))
    assert client.post(PATH, json={"version": 9}).status_code == 422
    assert client.post(PATH, json=_request()).status_code == 200


def test_route_502_and_upstream_429(client, monkeypatch):
    monkeypatch.setattr(server, "review_generator", FakeGen(RuntimeError("empty response")))
    r = client.post(PATH, json=_request())
    assert r.status_code == 502 and "RuntimeError" in r.json()["detail"]
    monkeypatch.setattr(server, "review_generator", FakeGen(TimeoutError("slow")))
    assert client.post(PATH, json=_request()).status_code == 502
    monkeypatch.setattr(server, "review_generator", FakeGen(_err(429)))
    r = client.post(PATH, json=_request())
    assert r.status_code == 429 and r.headers["Retry-After"] == str(server.UPSTREAM_429_RETRY_S)


def test_route_503_without_gemini_key(client, monkeypatch):
    monkeypatch.setattr(server, "review_generator", None)
    assert client.post(PATH, json=_request()).status_code == 503

def test_guessed_status_accepted():
    """앱 11-21 부터 statuses 에 guessed 가 늘 들어온다 — 거절하면 안 된다."""
    req = _request()
    req["game"]["statuses"]["guessed"] = 1
    req["timeline"][0]["status"] = "guessed"
    req["timeline"][0]["grade"] = None
    validate_request(req)
