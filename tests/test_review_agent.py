"""패보 복기 에이전트(app/review_agent.py · /v2/review/ask). LLM 은 전부 대본대로 답하는 가짜(돈 · 네트워크 0).

사실표 견본은 tests/fixtures/agent_facts/ — 앱의 `dart run tool/paifu_agent_facts.dart --compact` 로 예시 11판에서 만든 것.
기대값은 설계서(mahjong_calc docs/paifu_review_agent_design_1121.md) · 사실표 파일 · 손으로 센 값에서 온다.
"""

import copy
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from app import main as server
from app import review_agent as ra
from app.review_agent import Limits, ask, run, run_tool, validate_request, verify_answer

FIXTURES = Path(__file__).parent / "fixtures" / "agent_facts"
GAMES = sorted(p.stem for p in FIXTURES.glob("*.json"))
ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def r19() -> dict:
    return load("r19_142013")


@pytest.fixture(scope="module")
def r11() -> dict:
    return load("r11_130921")


@pytest.fixture(scope="module")
def r12() -> dict:
    return load("r12_131655")


def _turn(facts: dict, n: int) -> dict:
    return next(t for t in facts["turns"] if t["turn"] == n)


def _req(facts: dict, question: str, history=None) -> dict:
    r = {"facts": facts, "question": question}
    if history is not None:
        r["history"] = history
    return r


class Script:
    """대본대로 답하는 가짜 generate. 항목: dict(그대로 돌려줌) · 예외(던짐) · 함수(prompt, schema → dict).
    받은 prompt · schema 를 적어 둔다. 대본이 떨어지면 AssertionError(호출이 너무 많음)."""

    def __init__(self, *answers, tokens=(100, 20)):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.schemas: list[dict] = []
        self.tokens = tokens
        self.model = "fake-model"

    def __call__(self, prompt, schema):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        assert self.answers, "대본보다 LLM 호출이 많다"
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        if callable(a):
            a = a(prompt, schema)
        return copy.deepcopy(a), *self.tokens


def tool(tool_name: str, /, **args) -> dict:
    return {"action": "tool", "tool": tool_name, "args": args}


def answer(text: str) -> dict:
    return {"action": "answer", "text": text}


# r19 6번째(설계서 예시 · 앱 시험의 손 계산): [4만] 쯔모기리, 버린 뒤 1샹텐 10장, 최선 [3만] · [3삭] 15장, 손실 5장.
GOOD_6 = "6순에 [4만]을 버린 뒤 유효패는 10장이었다. 최선 [3만]을 버렸다면 유효패 15장이었다. 패효율 기준 실수다."
BAD_6 = "6순에 [4만]을 버린 뒤 유효패는 12장이었다. 최선 [3만]을 버렸다면 유효패 15장이었다."


# ---------------------------------------------------------------- 사실표 견본(앱이 만든 11판)


def test_fixtures_are_the_11_example_games():
    assert len(GAMES) == 11
    assert {g[:3] for g in GAMES} == {"L16", "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20"}


@pytest.mark.parametrize("name", GAMES)
def test_fixture_passes_request_check_and_size(name):
    facts = load(name)
    validate_request(_req(facts, "6순에 왜?"))
    # 설계서 2절 목표 60KB(압축). 라우트 상한 200KB 보다 한참 작다.
    assert len(json.dumps(facts, ensure_ascii=False, separators=(",", ":")).encode()) <= 60 * 1024


@pytest.mark.parametrize("name", GAMES)
def test_every_turn_tools_and_fallback(name):
    """모든 차례에서 차례 도구가 사실표 값 그대로 돌려주거나(숫자 없는 차례는) 오류를 돌려주고,
    대체 답이 자기 검증을 통과한다."""
    facts = load(name)
    for t in facts["turns"]:
        n = t["turn"]
        info = run_tool(facts, "turn_info", {"turn": n})
        for k in ("step", "status", "grade", "hand", "drawn", "discard", "tsumogiri", "melds", "riichi_declared",
                  "shanten_prev", "shanten_before", "shanten_loss", "ukeire_loss", "best", "riichi_winds", "genbutsu"):
            assert info[k] == t[k], (name, n, k)
        opts = run_tool(facts, "discard_options", {"turn": n})
        if t["options"]:
            assert [o["discard"] for o in opts["options"]] == [o["discard"] for o in t["options"]]
            assert [o["discard"] for o in opts["options"] if o["best"]] == t["best"]
            assert [o["discard"] for o in opts["options"] if o["actual"]] == [t["discard"]]
            assert info["actual"]["total"] == next(o["total"] for o in t["options"] if o["discard"] == t["discard"])
        else:
            assert set(opts) == {"error"} and info["actual"] is None and "note" in info
        safe = run_tool(facts, "safe_tiles", {"turn": n})
        assert (safe["genbutsu"], safe["safe_tiles"], safe["riichi_winds"]) == (t["genbutsu"], t["safe_tiles"],
                                                                                t["riichi_winds"])
        q = f"{n}순은 어땠어?"
        text, ev = ra.fallback_answer(facts, q)
        assert verify_answer(text, ev, q) == [], (name, n, text)
    text, ev = ra.fallback_answer(facts, "이 판 어땠어?")
    assert text.startswith("확인할 수 있는 사실로 답하지 못했다.")
    assert verify_answer(text, ev, "이 판 어땠어?") == []


# ---------------------------------------------------------------- 도구


def test_turn_info_r19_turn6_by_hand(r19):
    info = run_tool(r19, "turn_info", {"turn": 6})
    assert info["hand"] == ["3m", "3m", "4m", "7m", "8m", "9m", "3p", "4p", "5p", "3s", "3s", "5s", "8s", "8s"]
    assert (info["drawn"], info["discard"], info["tsumogiri"]) == ("4m", "4m", True)
    assert (info["grade"], info["grade_name"], info["status_name"]) == ("mistake", "실수", "등급 매김")
    assert (info["shanten_loss"], info["ukeire_loss"]) == (0, 5)
    assert info["best"] == ["3m", "3s"]
    a, b = info["actual"], info["best_option"]
    # 4m 버림: 33m 789m 345p 33s 5s 88s → 3m · 3s · 4s · 8s 4종 10장.
    assert (a["discard"], a["shanten"], a["total"], a["kinds"]) == ("4m", 1, 10, 4)
    assert [u["tile"] for u in a["tiles"]] == ["3m", "3s", "4s", "8s"]
    # 3m 버림: 34m 789m 345p 33s 5s 88s → 2m · 5m · 3s · 4s · 8s 15장.
    assert (b["discard"], b["shanten"], b["total"]) == ("3m", 1, 15)
    assert [u["tile"] for u in b["tiles"]] == ["2m", "5m", "3s", "4s", "8s"]
    # 유효패 목록은 사실표 맵을 [{tile, left}] 목록으로 옮긴 것.
    assert {u["tile"]: u["left"] for u in b["tiles"]} == _turn(r19, 6)["options"][0]["tiles"]


def test_turn_info_numberless_turn_has_note(r19):
    info = run_tool(r19, "turn_info", {"turn": 1})
    assert info["status"] == "guessed" and info["hand"] is None and info["actual"] is None and info["best"] == []
    assert "숫자가 없다" in info["note"]


def test_discard_options_marks_actual_and_best(r19):
    out = run_tool(r19, "discard_options", {"turn": 6})
    assert out["actual"] == "4m" and out["best"] == ["3m", "3s"]
    rows = {o["discard"]: o for o in out["options"]}
    assert rows["4m"]["actual"] and not rows["4m"]["best"]
    assert rows["3m"]["best"] and rows["3s"]["best"] and not rows["3m"]["actual"]
    assert "tiles" not in rows["3m"]
    for o in _turn(r19, 6)["options"]:
        assert {k: rows[o["discard"]][k] for k in ("shanten", "total", "kinds")} == \
            {k: o[k] for k in ("shanten", "total", "kinds")}


def test_what_if_by_hand(r19):
    out = run_tool(r19, "what_if", {"turn": 6, "tile": "3m"})
    assert (out["tile"], out["shanten"], out["total"], out["kinds"]) == ("3m", 1, 15, 5)
    assert out["is_best"] and not out["is_actual"]
    assert out["actual"] == {"discard": "4m", "shanten": 1, "total": 10, "kinds": 4}
    assert (out["shanten_diff"], out["total_diff"]) == (0, 5)
    # 3m 을 버리면 34m 양면이 남아 2m · 5m 을 받고, 4m 을 버리면 33m 또이쯔라 3m 을 받는다.
    assert out["only_this"] == ["2m", "5m"] and out["only_actual"] == ["3m"]
    # 이름 · 대괄호 이름 · 적5 코드도 받는다.
    assert run_tool(r19, "what_if", {"turn": 6, "tile": "3만"}) == out
    assert run_tool(r19, "what_if", {"turn": 6, "tile": "[3만]"}) == out
    assert run_tool(r19, "what_if", {"turn": "6", "tile": "3m"}) == out


def test_what_if_shanten_change_has_no_total_diff(r19):
    t = _turn(r19, 6)
    worse = next(o for o in t["options"] if o["shanten"] > 1)
    out = run_tool(r19, "what_if", {"turn": 6, "tile": worse["discard"]})
    assert out["shanten_diff"] == worse["shanten"] - 1 and out["total_diff"] is None


def test_what_if_actual_tile_is_itself(r19):
    out = run_tool(r19, "what_if", {"turn": 6, "tile": "4m"})
    assert out["is_actual"] and out["total_diff"] == 0 and out["only_this"] == [] and out["only_actual"] == []


@pytest.mark.parametrize("name,args,word", [
    ("turn_info", {"turn": 99}, "없다"),
    ("turn_info", {}, "turn"),
    ("turn_info", {"turn": True}, "turn"),
    ("discard_options", {"turn": 1}, "짐작 손패"),
    ("what_if", {"turn": 6, "tile": "7z"}, "손에 없는 패"),
    ("what_if", {"turn": 6, "tile": "칠만"}, "tile"),
    ("what_if", {"turn": 6}, "tile"),
    ("what_if", {"turn": 1, "tile": "3m"}, "후보 숫자가 없다"),
    ("find_turns", {"grade": "terrible"}, "grade"),
    ("find_turns", {"status": "?"}, "status"),
    ("find_turns", {"min_ukeire_loss": "many"}, "정수"),
    ("rule", {"name": "역"}, "모르는 용어"),
    ("rule", {}, "모르는 용어"),
    ("calc", {}, "없는 도구"),
    (None, {}, "없는 도구"),
    ("game_summary", {"turn": 3}, "받지 않는 인자"),
    ("turn_info", [6], "객체"),
])
def test_tool_errors_are_results_not_exceptions(r19, name, args, word):
    out = run_tool(r19, name, args)
    assert set(out) == {"error"} and word in out["error"]


def test_empty_foreign_args_are_ignored(r19):
    # Gemini 스키마는 모든 도구의 인자 칸을 한 객체에 둔다 — 빈 값으로 온 남의 칸은 오류가 아니다.
    assert run_tool(r19, "turn_info", {"turn": 6, "tile": "", "name": None})["turn"] == 6


def test_safe_tiles_r11_defense(r11):
    for n in (8, 9):
        t = _turn(r11, n)
        out = run_tool(r11, "safe_tiles", {"turn": n})
        assert out["riichi_winds"] == ["서"] and out["hand_known"]
        assert out["genbutsu"] == t["genbutsu"] and out["safe_tiles"] == t["safe_tiles"] and "note" not in out
    assert run_tool(r11, "safe_tiles", {"turn": 5})["note"] == "이 차례에는 리치한 다른 자리가 없다"


def test_find_turns(r19, r11):
    out = run_tool(r19, "find_turns", {"grade": "mistake"})
    assert [r["turn"] for r in out["turns"]] == [6] and out["count"] == 1
    assert out["turns"][0]["line"] == "6순 [4만] 버림 · 실수 · 유효패 5장 손해"
    assert run_tool(r19, "find_turns", {"grade": "실수"})["turns"] == out["turns"]
    assert [r["turn"] for r in run_tool(r19, "find_turns", {"status": "forced"})["turns"]] == [9, 10]
    assert [r["turn"] for r in run_tool(r19, "find_turns", {"status": "리치 중"})["turns"]] == [9, 10]
    want = [t["turn"] for t in r19["turns"] if t["ukeire_loss"] is not None and t["ukeire_loss"] >= 2]
    assert [r["turn"] for r in run_tool(r19, "find_turns", {"min_ukeire_loss": 2})["turns"]] == want
    want = [t["turn"] for t in r19["turns"] if t["discard"] == "4m"]
    assert [r["turn"] for r in run_tool(r19, "find_turns", {"tile": "4만"})["turns"]] == want
    # 조건 없음 = 전부. 조건은 모두 그리고.
    assert run_tool(r19, "find_turns", {})["count"] == len(r19["turns"])
    assert run_tool(r19, "find_turns", {"grade": "mistake", "tile": "3m"})["count"] == 0
    # r11 3 · 4번째 큰 실수: 샹텐 후퇴 줄.
    rows = run_tool(r11, "find_turns", {"grade": "blunder"})["turns"]
    assert [r["turn"] for r in rows] == [3, 4] and all("샹텐" in r["line"] and "후퇴" in r["line"] for r in rows)


# r12(동2국, 내 자리 남): 7번째에 [3통] 리치 선언 → 8~14번째 리치 중(forced). 북이 46걸음에 [9통] 리치 → 10~14번째의
# riichi_winds = [북], 현물은 13번째 [서] · 14번째 [6만]만(사실표 genbutsu false · false · false · true · true).
# 첫 실제 호출(metrics 15-3)의 f4 · g4 실패를 고친 것.


def test_what_if_forced_turn_has_status_and_note_r12(r12):
    # f4 "14순에 3만을 버렸어도 같았어?": 14번째는 리치 중이라 [3만]을 고를 수 없다 — 숫자(같은 텐파이 5장)는 참고.
    out = run_tool(r12, "what_if", {"turn": 14, "tile": "3m"})
    assert (out["status"], out["status_name"], out["can_choose"]) == ("forced", "리치 중", False)
    assert out["note"].startswith("리치 중이라 뽑은 패를 그대로 버려야 한다 — 다른 패는 고를 수 없다(숫자는 참고)")
    assert out["riichi_winds"] == ["북"] and "북 자리도 리치 중" in out["note"]
    # 숫자는 그대로 둔다.
    assert (out["shanten"], out["total"], out["total_diff"], out["actual"]["discard"]) == (0, 5, 0, "6m")
    # [3만]은 14번째 손 안의 현물이 아니다(safe_tiles 2m · 4m · 6m · 4p · 9p).
    assert out["tile_genbutsu"] is False
    opts = run_tool(r12, "discard_options", {"turn": 14})
    assert (opts["status"], opts["can_choose"], opts["note"]) == ("forced", False, out["note"])
    assert all("genbutsu" not in o for o in opts["options"])  # 고를 수 없는 차례라 후보별 현물은 넣지 않는다


def test_what_if_graded_turn_has_status_no_note(r19):
    out = run_tool(r19, "what_if", {"turn": 6, "tile": "3m"})
    assert (out["status"], out["status_name"], out["can_choose"], out["riichi_winds"]) == ("graded", "등급 매김", True, [])
    assert "note" not in out and out["tile_genbutsu"] is None
    opts = run_tool(r19, "discard_options", {"turn": 6})
    assert (opts["status"], opts["can_choose"]) == ("graded", True) and "note" not in opts


def test_what_if_defense_turn_note_has_genbutsu_r11(r11):
    # r11 8번째: 서 리치 중 수비 차례, 손 안 현물은 [6만] 하나.
    t = _turn(r11, 8)
    assert t["status"] == "defense" and t["safe_tiles"] == ["6m"]
    out = run_tool(r11, "what_if", {"turn": 8, "tile": "6m"})
    assert (out["status"], out["can_choose"], out["tile_genbutsu"]) == ("defense", True, True)
    assert out["note"] == "서 자리가 리치 중이라 패효율 등급을 매기지 않은 수비 차례다(숫자는 패효율만 본 참고). [6만]은 현물이다"
    out = run_tool(r11, "what_if", {"turn": 8, "tile": "1m"})
    assert out["tile_genbutsu"] is False and out["note"].endswith("[1만]은 현물이 아니다")
    opts = run_tool(r11, "discard_options", {"turn": 8})
    assert opts["status"] == "defense" and "서 자리가 리치 중" in opts["note"]
    assert [o["discard"] for o in opts["options"] if o["genbutsu"]] == ["6m"]


def test_numberless_turn_error_still_names_status(r12):
    out = run_tool(r12, "what_if", {"turn": 1, "tile": "3m"})
    assert set(out) == {"error"} and "짐작 손패" in out["error"] and "알 수 없다" in out["error"]


def test_find_turns_after_riichi_r12(r12):
    # g4 "북이 리치한 뒤에 나는 어떻게 했어?": 한 번에 10~14번째와 현물 여부.
    out = run_tool(r12, "find_turns", {"after_riichi": "북"})
    rows = out["turns"]
    assert [r["turn"] for r in rows] == [10, 11, 12, 13, 14] and out["count"] == 5
    assert [r["genbutsu"] for r in rows] == [False, False, False, True, True]
    assert [r["discard"] for r in rows] == ["8m", "4s", "5s", "3z", "6m"]
    assert all(r["status"] == "forced" and r["tsumogiri"] is True and r["riichi_winds"] == ["북"] for r in rows)
    assert rows[0]["line"] == "10순 [8만] 버림 · 리치 중 · 북 리치에 현물 아님"
    assert rows[3]["line"] == "13순 [서] 버림 · 리치 중 · 북 리치에 현물"
    assert out["conditions"] == {"after_riichi": "북"}
    assert out["riichi"] == {"wind": "북", "step": 46, "tile": "9p", "hero": False} and "note" not in out
    # 사실표의 riichi_winds 와 같은 답(others 의 리치 걸음 46 뒤 내 버림 = 49걸음의 10번째부터).
    assert [t["turn"] for t in r12["turns"] if "북" in t["riichi_winds"]] == [10, 11, 12, 13, 14]
    assert [t["turn"] for t in r12["turns"] if t["step"] > 46] == [10, 11, 12, 13, 14]
    # 자리 바람은 이름 · 대괄호 · 패 코드 · "자리" 를 붙여도 받는다.
    for arg in ("[북]", "4z", "북 자리", "north"):
        assert run_tool(r12, "find_turns", {"after_riichi": arg}) == out, arg
    # 다른 조건과 그리고.
    assert run_tool(r12, "find_turns", {"after_riichi": "북", "status": "forced"})["turns"] == rows
    assert run_tool(r12, "find_turns", {"after_riichi": "북", "tile": "6m"})["count"] == 1


def test_find_turns_after_riichi_hero_and_none(r12, r19):
    # 내 자리(남) 리치: 7번째 선언 다음 8~14번째.
    out = run_tool(r12, "find_turns", {"after_riichi": "남"})
    assert [r["turn"] for r in out["turns"]] == list(range(8, 15))
    assert out["riichi"] == {"wind": "남", "step": 34, "tile": "3p", "hero": True, "turn": 7}
    # 리치하지 않은 자리: 빈 목록 + 알림.
    out = run_tool(r12, "find_turns", {"after_riichi": "동"})
    assert out["count"] == 0 and out["riichi"] is None and out["note"] == "동 자리는 이 판에서 리치하지 않았다"
    out = run_tool(r12, "find_turns", {"after_riichi": "?"})
    assert set(out) == {"error"} and "after_riichi" in out["error"]
    # 남의 리치가 없는 판의 리치 없는 줄은 예전 꼴 그대로(현물 말 없음).
    assert "리치에" not in run_tool(r19, "find_turns", {"grade": "mistake"})["turns"][0]["line"]


@pytest.mark.parametrize("name", GAMES)
def test_find_turns_all_fits_whole_game(name):
    # 줄 수 상한 없음 — 판 전체(사실표 최대 40차례)가 한 번에, 줄마다 현물 칸이 사실표와 같다.
    facts = load(name)
    out = run_tool(facts, "find_turns", {})
    assert out["count"] == len(facts["turns"])
    for r, t in zip(out["turns"], facts["turns"]):
        assert (r["turn"], r["discard"], r["status"], r["tsumogiri"], r["riichi_winds"], r["genbutsu"]) == \
            (t["turn"], t["discard"], t["status"], t["tsumogiri"], t["riichi_winds"], t["genbutsu"])


def test_game_summary(r19):
    s = run_tool(r19, "game_summary", {})
    g = r19["game"]
    assert (s["round"], s["hero_wind"], s["efficiency_pct"]) == (g["round"], g["hero_wind"], g["efficiency_pct"])
    assert s["grades"] == {"최선": g["grades"]["best"], "좋음": g["grades"]["good"], "부정확": g["grades"]["inaccuracy"],
                           "실수": g["grades"]["mistake"], "큰 실수": g["grades"]["blunder"]}
    assert sum(s["statuses"].values()) == len(r19["turns"]) == s["hero_discards"]
    assert s["result"]["type"] == g["result"]["type"] and s["result"]["type_name"] == "론"
    assert s["result"]["hero_won"] == (g["result"]["winner"] == g["hero_wind"])
    assert [o["step"] for o in s["others"]] == [o["step"] for o in r19["others"]]
    assert s["others"][0]["kind_name"] == "퐁"


def test_rule():
    out = run_tool({"game": {}, "turns": []}, "rule", {"name": "등급 기준"})
    assert out["name"] == "등급" and "10%" in out["text"] and "30%" in out["text"]
    for name in ("유효패", "샹텐", "현물", "쯔모기리", "패효율", "ukeire", "Shanten"):
        assert run_tool({}, "rule", {"name": name})["text"]


# ---------------------------------------------------------------- 검증


def _ev(facts, *calls) -> list[dict]:
    return [{"tool": n, "args": a, "result": run_tool(facts, n, a)} for n, a in calls]


def test_verify_good_answer_passes(r19):
    ev = _ev(r19, ("turn_info", {"turn": 6}))
    assert verify_answer(GOOD_6, ev, "6순에 왜 4만이 실수야?") == []


@pytest.mark.parametrize("text,word", [
    (BAD_6, "12장"),                                                          # evidence 에 없는 장수
    ("6순에 [4만]을 버려 3샹텐이 됐다.", "3샹텐"),
    ("6순에 [4만]을 버려 0샹텐이 됐다.", "0샹텐"),                                # 텐파이라고 써야 함
    ("7순에 [4만]을 버렸다.", "7순"),
    ("6순에 [중]을 버렸다.", "[중]"),                                           # evidence 에 없는 패
    ("6순에 [4m]을 버렸다.", "패 이름이 아님"),
    ("6순에 4만을 버렸다.", "대괄호 없이"),
    ("6순에 [4만]을 버렸고 grade 는 mistake 다.", "영어"),
    ("6순에 [4만]을 버렸습니다.", "존댓말"),
    ("6순에 [4만]을 버렸어요.", "존댓말"),
    ("6순에 [4만]을 반드시 버리지 말았어야 했다.", "금지어"),
    ("6순 패효율은 50%다.", "50%"),
    ("6순에 [4만]을 버려 8000점을 잃었다.", "점수"),
    ("6순이다. 둘이다. 셋이다. 넷이다. 다섯이다. 여섯이다.", "문장 6개"),
    ("", "답 없음"),
    ("6m 4m", "한글 없음"),
])
def test_verify_catches(r19, text, word):
    ev = _ev(r19, ("turn_info", {"turn": 6}))
    v = verify_answer(text, ev, "6순에 왜?")
    assert any(word in x for x in v), v


def test_verify_question_turn_must_be_checked(r19):
    # 도구 없이 답함 → 질문한 차례 위반(숫자 · 패도 근거가 없다).
    v = verify_answer(GOOD_6, [], "6순에 왜 4만이 실수야?")
    assert "질문한 6번째 차례를 도구로 확인하지 않고 답함" in v and any("10장" in x for x in v)
    # 판 요약만 불러도 6번째를 확인한 것은 아니다.
    v = verify_answer("이 판 패효율은 93%다.", _ev(r19, ("game_summary", {})), "6번째는?")
    assert v == ["질문한 6번째 차례를 도구로 확인하지 않고 답함"]
    # 없는 차례를 물어 오류를 받았으면 확인한 것으로 친다(인자의 30 도 근거).
    assert verify_answer("30순은 이 판에 없다.", _ev(r19, ("turn_info", {"turn": 30})), "30순은?") == []


def test_verify_plain_korean_is_not_polite_or_english(r11):
    # "아니다"(해라체) · "리치 중" · 대괄호 안 이름은 통과. r11 8번째: 서 자리 리치 중, 버린 [9통]은 현물 아님.
    assert (_turn(r11, 8)["discard"], _turn(r11, 8)["genbutsu"]) == ("9p", False)
    ev = _ev(r11, ("safe_tiles", {"turn": 8}))
    assert verify_answer("8순에는 서 자리가 리치 중이었다. 버린 [9통]은 현물이 아니다.", ev, "8순은?") == []


def test_verify_percent_from_rule_and_summary(r19):
    ev = _ev(r19, ("rule", {"name": "등급"}), ("game_summary", {}))
    assert verify_answer("부정확은 최선의 30% 이하로 준 경우다. 이 판 패효율은 93%다.", ev, "등급 기준은?") == []
    assert r19["game"]["efficiency_pct"] == 93


def test_verify_ukeire_loss_only_on_same_shanten():
    # 샹텐이 뒤로 간 차례의 유효패 차이는 손실이 아니다 — ukeire_loss 를 장수로 허용하지 않는다(해설 규격 A-8 과 같은 규칙).
    def ev(shanten_loss, ukeire_loss):
        return [{"tool": "turn_info", "args": {"turn": 3},
                 "result": {"turn": 3, "shanten_loss": shanten_loss, "ukeire_loss": ukeire_loss}}]
    assert 7 in ra.evidence_allow(ev(0, 7))[0].jang
    assert 7 not in ra.evidence_allow(ev(1, 7))[0].jang
    assert 7 not in ra.evidence_allow(ev(1, -7))[0].jang
    assert verify_answer("3순에 유효패 7장을 손해 봤다.", ev(1, -7), "3순은?") == ["도구로 찾은 사실표에 없는 숫자: 7장"]


G4_Q = "북이 리치한 뒤에 나는 어떻게 했어?"
G4_A = ("북 자리가 리치한 뒤 나는 리치 중이라 10순부터 14순까지 뽑은 패를 그대로 버렸다. "
        "13순 [서]와 14순 [6만]은 현물이었고, 10순 [8만] · 11순 [4삭] · 12순 [5삭]은 현물이 아니었다.")
F4_Q = "14순에 3만을 버렸어도 같았어?"
F4_A = "14순은 리치 중이라 뽑은 [6만]을 그대로 버려야 했다. [3만]을 고를 수는 없었고, 숫자로는 둘 다 텐파이 대기 5장이다."


def test_verify_find_turns_rows_are_evidence(r12):
    # find_turns 줄의 버린 패 · 차례 번호 · 리치 패가 근거가 된다.
    ev = _ev(r12, ("find_turns", {"after_riichi": "북"}))
    assert verify_answer(G4_A, ev, G4_Q) == []
    assert verify_answer("북은 [9통]으로 리치했다. 나는 10순부터 리치 중이었다.", ev, G4_Q) == []
    # 줄에 없는 패 · 차례는 여전히 걸린다(규칙을 풀지 않음).
    v = verify_answer("9순 [1통]은 현물이었다.", ev, G4_Q)
    assert any("9순" in x for x in v) and any("[1통]" in x for x in v)
    # find_turns 는 차례 도구가 아니다 — 질문이 특정 차례를 물으면 여전히 그 차례를 확인해야 한다.
    assert "질문한 13번째 차례를 도구로 확인하지 않고 답함" in verify_answer("13순 [서]는 현물이었다.", ev, "13순은?")


def test_verify_what_if_forced_answer(r12):
    ev = _ev(r12, ("what_if", {"turn": 14, "tile": "3m"}))
    assert verify_answer(F4_A, ev, F4_Q) == []


# ---------------------------------------------------------------- 그래프(대본 가짜 LLM)


def test_graph_tool_then_answer(r19):
    gen = Script(tool("turn_info", turn=6), answer(GOOD_6))
    out = ask(_req(r19, "6순에 왜 4만이 실수야?"), gen, "m")
    assert out["outcome"] == "verified" and out["answer"] == GOOD_6 and out["violations"] == []
    assert out["tool_calls"] == [{"tool": "turn_info", "args": {"turn": 6}}]
    assert out["retries"] == 0 and out["usage"] == {"input_tokens": 200, "output_tokens": 40, "llm_calls": 2}
    assert out["prompt_version"] == "a2" and out["model"] == "m"
    # 첫 지시문: 규칙 · 도구 목록 · 질문, 아직 도구 없음. 둘째: 도구 결과가 붙음.
    p1, p2 = gen.prompts
    assert "## 질문\n6순에 왜 4만이 실수야?" in p1 and "도구 0/6회" in p1 and "- 아직 없음" in p1
    for name in ra.TOOL_NAMES:
        assert f"- {name}(" in p1
    assert "도구 1/6회" in p2 and '- 도구 turn_info {"turn": 6}' in p2 and '"best":["3m","3s"]' in p2
    assert gen.schemas[0]["properties"]["action"]["enum"] == ["tool", "answer"]


def test_graph_verify_fail_then_retry_passes(r19):
    gen = Script(tool("turn_info", turn=6), answer(BAD_6), answer(GOOD_6))
    s, _ = run(_req(r19, "6순에 왜?"), gen)
    assert s["outcome"] == "verified" and s["retries"] == 1 and s["llm_calls"] == 3 and s["answer"] == GOOD_6
    assert "- 검사에 걸림" in gen.prompts[2] and "12장" in gen.prompts[2] and BAD_6 in gen.prompts[2]
    assert [m["role"] for m in s["messages"]] == ["tool", "answer", "check", "answer"]


def test_graph_retry_may_call_more_tools(r19):
    gen = Script(tool("turn_info", turn=6), answer(BAD_6.replace("15장", "5장 더 많은 15장")),
                 tool("what_if", turn=6, tile="3m"), answer("6순 [4만] 대신 [3만]을 버렸다면 유효패가 5장 더 많았다."))
    out = ask(_req(r19, "6순에 왜?"), gen)
    assert out["outcome"] == "verified" and out["retries"] == 1 and len(out["tool_calls"]) == 2


def test_graph_fails_twice_then_fallback(r19):
    gen = Script(tool("turn_info", turn=6), answer(BAD_6), answer(BAD_6))
    out = ask(_req(r19, "6순에 왜 4만이 실수야?"), gen)
    assert out["outcome"] == "fallback" and out["retries"] == 1 and out["usage"]["llm_calls"] == 3
    assert any("12장" in v for v in out["violations"])
    # 대체 답: 6번째 turn_info 를 코드 문장으로(앱 3-8 틀과 같은 뜻).
    assert out["answer"] == ("확인한 사실만 적는다. 6순에 [4만]을 버렸고 패효율 기준 실수다. 버린 뒤 1샹텐이고 유효패 10장이다. "
                             "최선은 [3만](또는 [3삭]) 버림으로 1샹텐 · 유효패 15장이다.")
    # 대체 답이 부른 도구는 LLM 의 도구 호출 목록에 넣지 않는다.
    assert out["tool_calls"] == [{"tool": "turn_info", "args": {"turn": 6}}]


def test_fallback_without_turn_is_game_summary(r19):
    gen = Script(answer("모르겠다 99장."), answer("모르겠다 99장."))
    out = ask(_req(r19, "이 판에서 제일 아까운 수는?"), gen)
    assert out["outcome"] == "fallback"
    assert out["answer"] == ("확인할 수 있는 사실로 답하지 못했다. 이 판은 남4국, 내 버림은 10번이다. "
                             "패효율은 93%이고, 등급은 최선 5번 · 부정확 1번 · 실수 1번이다.")


def test_graph_tool_limit_forces_answer(r19):
    gen = Script(*[tool("turn_info", turn=n) for n in range(1, 7)], answer(GOOD_6))
    out = ask(_req(r19, "6순에 왜?"), gen)
    assert out["outcome"] == "verified" and len(out["tool_calls"]) == 6
    # 7번째 호출은 답만 받는 스키마 · 지시문.
    assert gen.schemas[6]["properties"]["action"]["enum"] == ["answer"] and gen.schemas[6]["required"] == ["action", "text"]
    assert "도구는 더 부를 수 없다" in gen.prompts[6] and "도구 6/6회" in gen.prompts[6]
    assert all(s["properties"]["action"]["enum"] == ["tool", "answer"] for s in gen.schemas[:6])


def test_graph_tool_after_limit_goes_to_fallback(r19):
    gen = Script(*[tool("game_summary") for _ in range(7)])
    out = ask(_req(r19, "6순에 왜?"), gen)
    assert out["outcome"] == "fallback" and len(out["tool_calls"]) == 6 and out["usage"]["llm_calls"] == 7
    assert out["violations"] == ["도구 한도(6회)를 다 쓴 뒤에도 도구를 요청함"]
    assert out["answer"].startswith("확인한 사실만 적는다. 6순에 [4만]")


def test_graph_missing_turn_error_and_continue(r19):
    gen = Script(tool("turn_info", turn=30), answer("질문한 30순은 이 판에 없다. 다른 차례를 물어 주면 확인한다."))
    s, _ = run(_req(r19, "30순에 뭐 버렸어?"), gen)
    assert s["outcome"] == "verified"
    assert "error" in s["evidence"][0]["result"] and "1~10번째" in s["evidence"][0]["result"]["error"]
    assert '"error":"30번째 버림은 이 판에 없다' in gen.prompts[1]


def test_graph_bad_tool_calls_are_errors_and_graph_goes_on(r19):
    gen = Script(tool("calc", turn=6), tool("what_if", turn=6, tile="중"), tool("find_turns", grade="bad"),
                 tool("turn_info", turn=6), answer(GOOD_6))
    out = ask(_req(r19, "6순에 왜?"), gen)
    assert out["outcome"] == "verified" and len(out["tool_calls"]) == 4
    assert [c["tool"] for c in out["tool_calls"]] == ["calc", "what_if", "find_turns", "turn_info"]


def test_graph_answer_without_tool_is_retried(r19):
    gen = Script(answer(GOOD_6), tool("turn_info", turn=6), answer(GOOD_6))
    out = ask(_req(r19, "6순에 왜 4만이 실수야?"), gen)
    assert out["outcome"] == "verified" and out["retries"] == 1 and len(out["tool_calls"]) == 1
    assert "질문한 6번째 차례를 도구로 확인하지 않고 답함" in gen.prompts[1]


def test_graph_malformed_output_counts_and_hits_llm_limit(r19):
    gen = Script(*([{"foo": 1}, "not a dict", {"action": "dance"}] * 3))
    out = ask(_req(r19, "6순에 왜?"), gen)
    assert out["outcome"] == "fallback" and out["usage"]["llm_calls"] == 8 and len(gen.answers) == 1
    assert out["violations"] == ["LLM 호출 한도(8회)를 넘김"]
    assert "JSON 형식이 맞지 않았다" in gen.prompts[1]


def test_graph_llm_limit_counts_tools_and_retries(r19):
    # 도구 5 + 틀린 답 + 다시 도구 1(한도 6) + 틀린 답 = 8회 → 대체.
    gen = Script(*[tool("rule", name="유효패")] * 5, answer(BAD_6), tool("turn_info", turn=6), answer(BAD_6))
    out = ask(_req(r19, "6순에 왜?"), gen)
    assert out["outcome"] == "fallback" and out["usage"]["llm_calls"] == 8 and out["retries"] == 1


def test_graph_time_limit(r19):
    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()

    def slow(prompt, schema):
        clock.t += 31.0
        return tool("turn_info", turn=6)

    gen = Script(slow, slow, slow)
    out = ask(_req(r19, "6순에 왜?"), gen, clock=clock)
    # 0초 · 31초에 부르고, 62초에는 부르지 않고 대체.
    assert out["outcome"] == "fallback" and out["usage"]["llm_calls"] == 2 and out["elapsed_s"] == 62.0
    assert out["violations"] == ["시간 한도(60초)를 넘김"]


def test_graph_custom_limits(r19):
    gen = Script(tool("turn_info", turn=6), answer(BAD_6))
    out = ask(_req(r19, "6순에 왜?"), gen, limits=Limits(retries=0))
    assert out["outcome"] == "fallback" and out["retries"] == 0


def test_graph_generate_errors_propagate(r19):
    with pytest.raises(TimeoutError):
        ask(_req(r19, "6순에 왜?"), Script(tool("turn_info", turn=6), TimeoutError("slow")))


def test_history_goes_into_prompt_but_not_evidence(r19):
    hist = [{"q": "이 판 패효율은?", "a": "93%다."}]
    gen = Script(answer("앞에서 말한 패효율은 93%다."), answer("앞에서 말한 패효율은 93%다."))
    out = ask(_req(r19, "다시 말해 줘", hist), gen)
    assert "## 앞선 질문과 답" in gen.prompts[0] and "이 판 패효율은?" in gen.prompts[0]
    # 앞선 답의 숫자는 근거가 아니다 — 도구 없이 93% 를 쓰면 걸린다.
    assert out["outcome"] == "fallback" and any("93%" in v for v in out["violations"])


def test_graph_f4_forced_what_if(r12):
    # f4 다시: what_if 한 번이면 결과에 "고를 수 없다" 가 보이고, 그것을 말한 답이 통과한다.
    gen = Script(tool("what_if", turn=14, tile="3m"), answer(F4_A))
    out = ask(_req(r12, F4_Q), gen)
    assert out["outcome"] == "verified" and out["retries"] == 0
    assert '"can_choose":false' in gen.prompts[1] and "다른 패는 고를 수 없다" in gen.prompts[1]


def test_graph_g4_one_find_turns_call(r12):
    # g4 다시: 도구 한 번으로 10~14번째 전부와 현물 여부(첫 실제 호출은 6회를 쓰고도 10 · 13 · 14번째를 빠뜨림).
    gen = Script(tool("find_turns", after_riichi="북"), answer(G4_A))
    out = ask(_req(r12, G4_Q), gen)
    assert out["outcome"] == "verified" and out["tool_calls"] == [{"tool": "find_turns", "args": {"after_riichi": "북"}}]
    for n in range(10, 15):
        assert f'"turn":{n},' in gen.prompts[1]


def test_prompt_rules_a2():
    p = ra.PROMPT_RULES
    for word in ("패효율", "대괄호", "해라체", "3~5문장", "도구 결과에 있는 값만", "0샹텐", "6번까지"):
        assert word in p, word
    # a2: "만약" 질문은 차례 상태부터, 여러 차례 질문은 find_turns 조건부터.
    for word in ("can_choose", "고를 수 없던 차례", "차례마다 도구를 부르지 말고", "after_riichi"):
        assert word in p, word
    assert "- find_turns(grade?, status?, min_ukeire_loss?, tile?, after_riichi?)" in p
    assert "after_riichi" in ra.action_schema()["properties"]["args"]["properties"]
    assert ra.PROMPT_VERSION == "a2"


# ---------------------------------------------------------------- 요청 검사


def test_validate_rejects(r19):
    ok = _req(r19, "6순에 왜?")
    validate_request(ok)
    bad = []
    r = copy.deepcopy(ok); r["facts"]["version"] = 2; bad.append((r, "version"))
    r = copy.deepcopy(ok); r["question"] = ""; bad.append((r, "question"))
    r = copy.deepcopy(ok); r["question"] = "가" * 301; bad.append((r, "question"))
    r = copy.deepcopy(ok); r["history"] = [{"q": "a", "a": "b"}] * 5; bad.append((r, "history"))
    r = copy.deepcopy(ok); r["history"] = [{"q": "a"}]; bad.append((r, "history"))
    r = copy.deepcopy(ok); del r["facts"]["others"]; bad.append((r, "others"))
    r = copy.deepcopy(ok); r["facts"]["turns"][5]["turn"] = 7; bad.append((r, "turn"))
    r = copy.deepcopy(ok); r["facts"]["turns"][5]["options"][0]["tiles"] = {"2m": 9}; bad.append((r, "tiles"))
    r = copy.deepcopy(ok); r["facts"]["turns"][5]["actual"] = "9z"; bad.append((r, "actual"))
    r = copy.deepcopy(ok); r["facts"]["turns"][5]["best"] = ["1z"]; bad.append((r, "best"))
    r = copy.deepcopy(ok); r["facts"]["game"]["statuses"]["weird"] = 1; bad.append((r, "statuses"))
    r = copy.deepcopy(ok); r["facts"]["game"]["hero_wind"] = "west"; bad.append((r, "hero_wind"))
    r = copy.deepcopy(ok); r["facts"] = []; bad.append((r, "facts"))
    r = copy.deepcopy(ok); r["facts"]["pad"] = "x" * (ra.MAX_REQUEST_BYTES + 1); bad.append((r, "너무 큼"))
    for r, word in bad:
        with pytest.raises(ValueError, match=word):
            validate_request(r)
    validate_request({**ok, "history": [{"q": "a", "a": "b"}] * 4})
    validate_request({**ok, "history": None})


# ---------------------------------------------------------------- 라우트


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "review_generator", Script(*[tool("turn_info", turn=6), answer(GOOD_6)] * 5))
    monkeypatch.setattr(server, "_review_limiter", server.RateLimiter(60, 10))
    with TestClient(server.app) as c:
        yield c


PATH = "/v2/review/ask"


def test_route_200_and_shape(client, caplog, r19):
    with caplog.at_level(logging.INFO, logger="recognizer"):
        r = client.post(PATH, json=_req(r19, "6순에 왜 4만이 실수야?"))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "outcome", "tool_calls", "violations", "retries", "prompt_version", "model",
                         "elapsed_s", "usage"}
    assert body["outcome"] == "verified" and body["answer"] == GOOD_6 and body["model"] == "fake-model"
    assert body["tool_calls"] == [{"tool": "turn_info", "args": {"turn": 6}}]
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "review/ask tools=1 outcome=verified retries=0 llm=2" in logs
    assert "6순에 왜" not in logs and GOOD_6 not in logs and "남4국" not in logs   # 질문 · 답 · 사실표는 안 찍는다


def test_route_is_listed_in_health(client):
    eps = client.get("/").json()["endpoints"]
    assert PATH in eps and "/v2/review/explain" in eps


def test_route_401(client, monkeypatch, r19):
    monkeypatch.setattr(server, "API_KEY", "secret")
    assert client.post(PATH, json=_req(r19, "6순?")).status_code == 401
    assert client.post(PATH, json=_req(r19, "6순?"), headers={"X-Api-Key": "secret"}).status_code == 200


def test_route_422(client, r19):
    r = client.post(PATH, json=_req(r19, ""))
    assert r.status_code == 422 and "question" in r.json()["detail"]
    assert client.post(PATH, content=b"{nope", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post(PATH, json={"facts": {"version": 9}, "question": "a"}).status_code == 422


def test_route_413_over_200kb(client):
    big = b'{"question": "a", "pad": "' + b"x" * ra.MAX_REQUEST_BYTES + b'"}'
    assert client.post(PATH, content=big, headers={"Content-Type": "application/json"}).status_code == 413


def test_route_body_between_explain_and_ask_caps_is_read(client, r19):
    # 해설 상한(128KiB)과 에이전트 상한(200KB) 사이 본문은 413 이 아니라 읽은 뒤 규격 검사(여기선 크기)로 간다.
    req = copy.deepcopy(_req(r19, "6순?"))
    req["history"] = [{"q": "a", "a": "b"}]
    req["facts"]["pad"] = "x" * 150_000
    r = client.post(PATH, json=req)
    assert r.status_code == 200


def test_route_429_shares_review_limiter(client, monkeypatch, r19):
    monkeypatch.setattr(server, "_review_limiter", server.RateLimiter(per_min=1, burst=1))
    assert client.post(PATH, json=_req(r19, "6순?")).status_code == 200
    r = client.post(PATH, json=_req(r19, "6순?"))
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert client.post("/v2/review/explain", json={"version": 1}).status_code == 422  # 같은 제한기라도 422 가 먼저


def test_route_502_429_503(client, monkeypatch, r19):
    monkeypatch.setattr(server, "review_generator", Script(RuntimeError("empty response")))
    r = client.post(PATH, json=_req(r19, "6순?"))
    assert r.status_code == 502 and "RuntimeError" in r.json()["detail"]
    monkeypatch.setattr(server, "review_generator", Script(tool("turn_info", turn=6),
                                                           errors.ClientError(429, {"error": {"code": 429, "message": "x", "status": "X"}})))
    r = client.post(PATH, json=_req(r19, "6순?"))
    assert r.status_code == 429 and r.headers["Retry-After"] == str(server.UPSTREAM_429_RETRY_S)
    monkeypatch.setattr(server, "review_generator", None)
    assert client.post(PATH, json=_req(r19, "6순?")).status_code == 503


# ---------------------------------------------------------------- 스크립트(--fake)


def test_script_fake_runs_one_game(tmp_path):
    out = tmp_path / "out.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "review_ask.py"), str(FIXTURES / "r19_142013.json"),
                        "6순에 왜 4만이 실수야?", "--fake", "--out", str(out)],
                       capture_output=True, cwd=ROOT, timeout=120)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    text = p.stdout.decode("utf-8")
    assert "검증 실패 → 다시" in text and "결과: verified" in text
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["outcome"] == "verified" and body["retries"] == 1 and body["model"] == "fake"
    assert [c["tool"] for c in body["tool_calls"]] == ["turn_info", "discard_options"]
