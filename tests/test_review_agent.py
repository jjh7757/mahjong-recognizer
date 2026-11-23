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


def _variant(tool_name: str) -> dict:
    """a4 도구별 스키마(최상위 anyOf)에서 그 도구의 변형."""
    return next(v for v in ra.action_schema()["anyOf"] if v["properties"]["tool"]["enum"] == [tool_name])


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
    assert out["prompt_version"] == "a8" and out["model"] == "m"   # 11-22 a2 → a3(수비 확장) → a4(F1 ~ F5) → a5(G1 ~ G3) → a6(말투)
    assert out["retry_violations"] == []
    # 첫 지시문: 규칙 · 도구 목록 · 질문, 아직 도구 없음. 둘째: 도구 결과가 붙음.
    p1, p2 = gen.prompts
    assert "## 질문\n6순에 왜 4만이 실수야?" in p1 and "도구 0/6회" in p1 and "- 아직 없음" in p1
    for name in ra.TOOL_NAMES:
        assert f"- {name}(" in p1
    assert "도구 1/6회" in p2 and '- 도구 turn_info {"turn": 6}' in p2 and '"best":["3m","3s"]' in p2
    assert "anyOf" in gen.schemas[0] and not ra.schema_answer_only(gen.schemas[0])   # a4: 도구마다 따로인 스키마


def test_graph_verify_fail_then_retry_passes(r19):
    gen = Script(tool("turn_info", turn=6), answer(BAD_6), answer(GOOD_6))
    s, _ = run(_req(r19, "6순에 왜?"), gen)
    assert s["outcome"] == "verified" and s["retries"] == 1 and s["llm_calls"] == 3 and s["answer"] == GOOD_6
    assert "- 검사에 걸림" in gen.prompts[2] and "12장" in gen.prompts[2] and BAD_6 in gen.prompts[2]
    assert [m["role"] for m in s["messages"]] == ["tool", "answer", "check", "answer"]
    # a4 F5: 재시도를 부른 위반이 응답에 남는다(마지막 violations 는 통과라 빈 목록).
    out = ra.to_response(s, 0.0)
    assert out["violations"] == [] and len(out["retry_violations"]) == 1
    assert out["retry_violations"][0] == s["messages"][2]["violations"] and any("12장" in v for v in out["retry_violations"][0])


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
    # a4 F5: 첫 시도의 위반은 retry_violations 에, 대체로 끝난 마지막 시도의 위반은 지금처럼 violations 에.
    assert len(out["retry_violations"]) == 1 and any("12장" in v for v in out["retry_violations"][0])
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
    assert gen.schemas[6]["propertyOrdering"] == ["action", "text"] and ra.schema_answer_only(gen.schemas[6])
    assert "도구는 더 부를 수 없다" in gen.prompts[6] and "도구 6/6회" in gen.prompts[6]
    assert all(s == ra.action_schema() and not ra.schema_answer_only(s) for s in gen.schemas[:6])


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
    # 11-22 a3: find_turns 에 stance? · min_level? 이 더해짐(설계서 1122 4절) — 앞 다섯 인자는 그대로.
    # 11-23 a7: fold_spot? 이 끝에 더해짐(설계서 1123 3절).
    assert "- find_turns(grade?, status?, min_ukeire_loss?, tile?, after_riichi?, stance?, min_level?, fold_spot?)" in p
    assert "after_riichi" in _variant("find_turns")["properties"]["args"]["properties"]
    assert ra.PROMPT_VERSION == "a8"


# ---------------------------------------------------------------- 요청 검사


def test_validate_rejects(r19):
    ok = _req(r19, "6순에 왜?")
    validate_request(ok)
    bad = []
    r = copy.deepcopy(ok); r["facts"]["version"] = 4; bad.append((r, "version"))  # 11-23: 3 도 받는 판(1 · 2 · 3)
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
    assert set(body) == {"answer", "outcome", "tool_calls", "violations", "retries", "retry_violations",
                         "prompt_version", "model", "elapsed_s", "usage"}
    assert body["retry_violations"] == []
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


# ---------------------------------------------------------------- 수비 확장(설계서 mahjong_calc docs/paifu_review_agent_defense_design_1122.md)
#
# 사실표 2 의 defense: 리치한 다른 자리가 있고 손패를 알고 엔진 위험표가 있는 차례만. 예시 11판에서 L16 17 · 18번째(수비),
# r11 8 · 9번째(수비) · 10~16번째(리치 중), r12 10~14번째(리치 중) = 16차례. 값은 앱 tool/paifu_agent_facts.dart 가 낸 그대로.


def _v1(facts: dict) -> dict:
    """사실표 2 → 1(defense 칸을 뺌)."""
    f = copy.deepcopy(facts)
    f["version"] = 1
    for t in f["turns"]:
        t.pop("defense", None)
    return f


def _scalars(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _scalars(x)
    elif isinstance(v, list):
        for x in v:
            yield from _scalars(x)
    else:
        yield v


def test_fixtures_are_v2_with_16_defense_turns():
    got = {g[:3]: [t["turn"] for t in load(g)["turns"] if t.get("defense")] for g in GAMES}
    assert {k: v for k, v in got.items() if v} == {"L16": [17, 18], "r11": list(range(8, 17)), "r12": list(range(10, 15))}
    for g in GAMES:
        facts = load(g)
        assert facts["version"] == 2
        for t in facts["turns"]:
            assert "defense" in t
            if t["defense"]:
                # defense 는 리치한 자리가 있고 손패를 아는 차례에만. 엔진 % 값(실수)은 없다.
                assert t["riichi_winds"] and t["hand"] is not None
                assert not any(isinstance(x, float) for x in _scalars(t["defense"]))


@pytest.mark.parametrize("name", GAMES)
def test_defense_info_returns_facts_as_is(name):
    facts = load(name)
    for t in facts["turns"]:
        out = run_tool(facts, "defense_info", {"turn": t["turn"]})
        assert out["defense"] == t["defense"], (name, t["turn"])
        assert (out["turn"], out["status"], out["riichi_winds"], out["discard"], out["genbutsu"]) == \
            (t["turn"], t["status"], t["riichi_winds"], t["discard"], t["genbutsu"])
        assert out["can_choose"] == (t["status"] != "forced")
        note = out.get("note", "")
        if not t["riichi_winds"]:
            assert "리치한 다른 자리가 없다" in note
        elif t["defense"] is None:
            assert "손패를 몰라" in note or "엔진 위험 값이 없다" in note
        else:
            # fold_spot 이 늘 null(1단계). a4 F2: note 는 "판정 없음" 까지만 — "판정하지 않는다" 문장을 싣지 않는다.
            assert "이 차례의 밀기 · 접기 판정은 없다(fold_spot null)" in note
            assert "판정하지 않는다" not in note and "이 도우미는" not in note
            assert ("고를 수 없었다" in note) == (t["status"] == "forced")
            # 리치 중 차례: stance · efficient 는 null, 밀었다 · 접었다를 말하지 않는다는 알림.
            assert (t["defense"]["stance"] is None) == (t["status"] == "forced")
            assert ("밀었다 · 접었다를 말하지 않는다" in note) == (t["status"] == "forced")
            if t["status"] == "forced":
                assert t["defense"]["efficient"] is None


def test_defense_info_r11_turn8_values(r11):
    # r11 8번째(수비, 서 리치): 손 13종, 버린 [9통]은 '높음' · 6위, 가장 안전한 패는 현물 [6만]('안전').
    out = run_tool(r11, "defense_info", {"turn": 8})
    d = out["defense"]
    assert d["threat_winds"] == ["서"] and d["stance"] == "pushed" and d["fold_spot"] is None
    assert d["actual"] == {"tile": "9p", "level": "높음", "rank": 6, "of": 13}
    assert d["safest"] == ["6m"] and d["tiles"][0] == {"tile": "6m", "level": "안전", "genbutsu": True}
    assert d["efficient"] == {"tile": "1m", "level": "높음"}
    assert [x["tile"] for x in d["tiles"] if x["genbutsu"]] == _turn(r11, 8)["safe_tiles"]
    assert out["status"] == "defense" and out["can_choose"]


def test_defense_info_notes_v1_missing_unknown(r11, r19):
    # 리치 없음.
    out = run_tool(r19, "defense_info", {"turn": 6})
    assert out["defense"] is None and "리치한 다른 자리가 없다" in out["note"]
    # 사실표 1: defense 없는 것으로.
    out = run_tool(_v1(r11), "defense_info", {"turn": 8})
    assert out["defense"] is None and "판 1" in out["note"]
    # 사실표 1 에 defense 칸이 남아 있어도 보지 않는다.
    v1b = copy.deepcopy(r11)
    v1b["version"] = 1
    assert run_tool(v1b, "defense_info", {"turn": 8})["defense"] is None
    # 엔진 값 없음(리치 · 손패는 있는데 defense null).
    f = copy.deepcopy(r11)
    _turn(f, 8)["defense"] = None
    assert "엔진 위험 값이 없다" in run_tool(f, "defense_info", {"turn": 8})["note"]
    # 손패 모름.
    _turn(f, 8)["hand"] = None
    assert "손패를 몰라" in run_tool(f, "defense_info", {"turn": 8})["note"]
    # 없는 차례 · 인자 오류.
    assert set(run_tool(r11, "defense_info", {"turn": 99})) == {"error"}
    assert set(run_tool(r11, "defense_info", {})) == {"error"}


@pytest.mark.parametrize("name", GAMES)
def test_what_if_risk_matches_defense_tiles(name):
    facts = load(name)
    for t in facts["turns"]:
        for o in t["options"]:
            out = run_tool(facts, "what_if", {"turn": t["turn"], "tile": o["discard"]})
            if not t["riichi_winds"]:
                assert "risk" not in out
                continue
            d = t["defense"]
            if d is None:
                assert out["risk"] is None
                continue
            r = out["risk"]
            row = next(x for x in d["tiles"] if x["tile"] == o["discard"])
            assert (r["tile"], r["level"], r["genbutsu"], r["of"]) == \
                (o["discard"], row["level"], row["genbutsu"], len(d["tiles"]))
            assert r["is_safest"] == (o["discard"] in d["safest"])
            # a5 G1: 실제 버린 패는 사실표 defense.actual 그대로, 가장 안전한 패는 패마다 tiles 줄의 단계(찾기만).
            assert r["actual"] == d["actual"]
            level_of = {x["tile"]: x["level"] for x in d["tiles"]}
            assert r["safest"] == [{"tile": s, "level": level_of[s]} for s in d["safest"]]
            if o["discard"] == d["actual"]["tile"]:
                assert r["rank"] == d["actual"]["rank"]
            elif r["is_safest"]:
                assert r["rank"] == 1
            else:
                assert r["rank"] == [x["tile"] for x in d["tiles"]].index(o["discard"]) + 1


def test_what_if_risk_r11_turn8_by_hand(r11):
    out = run_tool(r11, "what_if", {"turn": 8, "tile": "6m"})
    assert out["risk"] == {"tile": "6m", "level": "안전", "rank": 1, "of": 13, "genbutsu": True, "is_safest": True,
                           "actual": {"tile": "9p", "level": "높음", "rank": 6, "of": 13},
                           "safest": [{"tile": "6m", "level": "안전"}]}
    asked = run_tool(r11, "what_if", {"turn": 8, "tile": "9p"})["risk"]
    assert asked["rank"] == 6       # 버린 패 = actual.rank
    assert asked["actual"] == {"tile": "9p", "level": "높음", "rank": 6, "of": 13}     # 물어본 패가 버린 패여도 actual 은 둔다
    last = run_tool(r11, "what_if", {"turn": 8, "tile": "4m"})["risk"]
    assert (last["level"], last["rank"]) == ("높음", 13)
    # 사실표 1 이면 risk 는 null(리치한 자리는 있으니 칸은 둔다).
    assert run_tool(_v1(r11), "what_if", {"turn": 8, "tile": "6m"})["risk"] is None
    # 기존 칸은 그대로(note · tile_genbutsu).
    assert out["note"].endswith("[6만]은 현물이다") and out["tile_genbutsu"] is True


@pytest.mark.parametrize("name", GAMES)
def test_find_turns_stance_and_min_level_are_facts(name):
    facts = load(name)
    for st in ra.STANCES:
        rows = run_tool(facts, "find_turns", {"stance": st})["turns"]
        assert [r["turn"] for r in rows] == [t["turn"] for t in facts["turns"]
                                             if t["defense"] and t["defense"]["stance"] == st]
        assert all(r["stance"] == st for r in rows)
    for i, lv in enumerate(ra.LEVELS):
        rows = run_tool(facts, "find_turns", {"min_level": lv})["turns"]
        assert [r["turn"] for r in rows] == [t["turn"] for t in facts["turns"]
                                             if t["defense"] and ra.LEVELS.index(t["defense"]["actual"]["level"]) >= i]
    # 줄마다 버린 패의 단계 · stance(defense 없는 차례는 null).
    for r, t in zip(run_tool(facts, "find_turns", {})["turns"], facts["turns"]):
        d = t["defense"]
        assert (r["level"], r["stance"]) == ((d["actual"]["level"], d["stance"]) if d else (None, None))


def test_find_turns_defense_conditions_r11(r11):
    out = run_tool(r11, "find_turns", {"min_level": "높음"})
    assert [r["turn"] for r in out["turns"]] == [8, 9, 11, 14] and out["conditions"] == {"min_level": "높음"}
    assert [r["level"] for r in out["turns"]] == ["높음", "높음", "매우 높음", "매우 높음"]
    for arg in ("'높음'", " 높음 ", "high"):
        assert run_tool(r11, "find_turns", {"min_level": arg}) == out, arg
    assert run_tool(r11, "find_turns", {"min_level": "매우높음"})["count"] == 2
    # min_level 은 리치 중 차례(11 · 14)도 고르되 줄에 고를 수 없던 차례임이 드러난다.
    assert [(r["status"], r["can_choose"], r["stance"]) for r in out["turns"]] ==         [("defense", True, "pushed"), ("defense", True, "pushed"), ("forced", False, None), ("forced", False, None)]
    # stance 조건은 stance 가 null 인 리치 중 차례를 고르지 않는다(12번째는 리치 중이라 folded 가 아님).
    assert [r["turn"] for r in run_tool(r11, "find_turns", {"stance": "pushed"})["turns"]] == [8, 9]
    folded = run_tool(r11, "find_turns", {"stance": "folded"})
    assert folded["count"] == 0
    assert run_tool(r11, "find_turns", {"stance": "접었다"}) == folded
    for name in GAMES:
        f = load(name)
        for st in ra.STANCES:
            assert all(r["status"] != "forced" for r in run_tool(f, "find_turns", {"stance": st})["turns"])
    # 다른 조건과 그리고: 서 리치 뒤 수비 차례 중 밀었다.
    rows = run_tool(r11, "find_turns", {"after_riichi": "서", "status": "defense", "stance": "pushed"})["turns"]
    assert [r["turn"] for r in rows] == [8, 9]
    # 줄 글(line)은 a2 그대로 — 단계는 level 칸에만.
    assert rows[0]["line"] == "8순 [9통] 버림 · 수비 · 서 리치에 현물 아님"
    for args, word in (({"stance": "x"}, "stance"), ({"min_level": "?"}, "min_level")):
        bad = run_tool(r11, "find_turns", args)
        assert set(bad) == {"error"} and word in bad["error"]


def test_rule_defense_texts():
    risk = run_tool({}, "rule", {"name": "방총 위험"})
    assert risk["name"] == "방총 위험"
    for word in ("2,499번 중 0번", "'낮음' 1% 미만", "'보통' 약 3%", "'높음' 약 8%", "'매우 높음' 약 15%", "천봉 고단자 판",
                 "리치한 상대", "리치하지 않은 상대는 보지 않는다", "엔진의 어림"):
        assert word in risk["text"], word
    pf = run_tool({}, "rule", {"name": "밀기 접기"})
    assert pf["name"] == "밀기 접기" and "판정이 아니다" in pf["text"] and "fold_spot" in pf["text"]
    for alias, key in (("방총위험", "방총 위험"), ("위험 단계", "방총 위험"), ("stance", "밀기 접기"), ("밀기접기", "밀기 접기")):
        assert run_tool({}, "rule", {"name": alias})["name"] == key


# ---- 요청 검사(사실표 1 · 2)


def test_validate_accepts_v1_and_v2(r11, r19):
    validate_request(_req(r11, "8순에 위험했어?"))
    validate_request(_req(_v1(r11), "8순에 위험했어?"))
    validate_request(_req(_v1(r19), "6순에 왜?"))
    v1b = copy.deepcopy(r11)
    v1b["version"] = 1                       # 판 1 에 남은 defense 칸은 검사하지 않는다(무시)
    _turn(v1b, 8)["defense"] = "아무거나"
    validate_request(_req(v1b, "8순?"))


def test_validate_rejects_bad_defense(r11):
    ok = _req(r11, "8순에 위험했어?")
    bad = []

    def mut(fn, word):
        r = copy.deepcopy(ok)
        fn(_turn(r["facts"], 8))
        bad.append((r, word))

    mut(lambda t: t.pop("defense"), "defense 없음")
    mut(lambda t: t.update(defense=[]), "객체")
    mut(lambda t: t["defense"].pop("stance"), "객체")
    mut(lambda t: t["defense"]["tiles"][0].update(level="조금"), "tiles")
    mut(lambda t: t["defense"]["tiles"].pop(), "손의 패 종류 전부")
    mut(lambda t: t["defense"]["tiles"].append(dict(t["defense"]["tiles"][0])), "두 번")
    mut(lambda t: t["defense"]["actual"].update(tile="6m"), "actual")
    mut(lambda t: t["defense"]["actual"].update(rank=14), "actual")
    mut(lambda t: t["defense"]["actual"].update(level="안전"), "actual")
    mut(lambda t: t["defense"].update(safest=["1z"]), "safest")
    mut(lambda t: t["defense"].update(efficient={"tile": "1z", "level": "낮음"}), "efficient")
    mut(lambda t: t["defense"].update(stance="maybe"), "stance")
    mut(lambda t: t["defense"].update(stance=None), "stance")                     # 수비 차례는 stance 가 있어야
    bad.append((_bad_forced(ok, stance="pushed"), "stance"))                   # 리치 중 차례는 stance null
    bad.append((_bad_forced(ok, efficient={"tile": "1m", "level": "높음"}), "efficient"))
    mut(lambda t: t["defense"].update(fold_spot="yes"), "fold_spot")
    mut(lambda t: t["defense"].update(threat_winds=["east"]), "threat_winds")
    mut(lambda t: t.update(riichi_winds=[]), "리치한 자리가 있고")
    for r, word in bad:
        with pytest.raises(ValueError, match=word):
            validate_request(r)


# ---- 검증: 단계 말 · 판정 말


DEF_8 = "8순에 버린 [9통]은 위험 '높음'으로 손의 13종 중 6위였다. 가장 안전한 패는 현물 [6만]으로 '안전'이었다."


def test_verify_level_words_pass(r11, r19):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    assert verify_answer(DEF_8, ev, "8순에 버린 9통 위험했어?") == []
    # 따옴표 밖 "안전" · "보통" 은 흔한 말이라 보지 않는다. 굽은 따옴표도 받는다.
    assert verify_answer("8순에 버린 [9통]은 ‘높음’이었다. 보통은 안전한 패를 고른다.", ev, "8순?") == []
    # 판정하지 않는다는 문장은 판정 말이 아니다.
    assert verify_answer("8순 [9통]은 위험 '높음'이었다. 이 도우미는 밀지 접을지를 판정하지 않는다.", ev,
                         "8순에 접었어야 했어?") == []
    # what_if 의 risk · find_turns 줄의 level 도 근거.
    ev2 = _ev(r11, ("what_if", {"turn": 8, "tile": "6m"}))
    assert verify_answer("8순에 [6만]을 버렸다면 위험 '안전'으로 손의 13종 중 1위였다.", ev2, "8순에 6만은?") == []
    ev3 = _ev(r11, ("find_turns", {"min_level": "높음"}))
    assert verify_answer("11순 [5삭]은 '매우 높음'이었다.", ev3, "위험한 버림?") == []
    # 그 단계 이상이 없다는 답(min_level 조건 값이 근거).
    ev4 = _ev(r19, ("find_turns", {"min_level": "높음"}))
    assert ev4[0]["result"]["count"] == 0
    assert verify_answer("'높음' 이상 위험한 패를 버린 차례는 없다.", ev4, "위험한 버림?") == []
    # rule '방총 위험' 을 읽으면 다섯 단계 · 관측 비율 % 가 근거가 된다.
    ev5 = _ev(r11, ("rule", {"name": "방총 위험"}))
    assert verify_answer("'매우 높음' 단계의 실제 방총은 약 15%였다.", ev5, "매우 높음은 얼마나 위험해?") == []


@pytest.mark.parametrize("text,word", [
    ("8순에 버린 [9통]은 위험 '매우 높음'이었다.", "사실표에 없는 위험 단계: '매우 높음'"),   # evidence 에는 안전 · 보통 · 높음뿐
    ("8순에 버린 [9통]은 위험 '낮음'이었다.", "사실표에 없는 위험 단계: '낮음'"),
    ("8순에 버린 [9통]은 위험이 높음이었다.", "작은따옴표 없이 쓴 위험 단계: 높음"),
    ("8순에 버린 [9통]은 위험이 매우 높음이었다.", "작은따옴표 없이 쓴 위험 단계: 매우 높음"),
    ("8순에 [6만]은 위험이 낮음이었다.", "작은따옴표 없이 쓴 위험 단계: 낮음"),
    ("8순에 버린 [9통]의 방총 위험은 12%였다.", "12%"),                                   # % 를 지어냄
    ("8순에 버린 [9통]은 손의 13종 중 7위였다.", "7위"),
])
def test_verify_level_words_catch(r11, text, word):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    v = verify_answer(text, ev, "8순?")
    assert any(word in x for x in v), v


@pytest.mark.parametrize("phrase", ["접었어야 했다", "접는 게 맞았다", "접어야 했다", "내렸어야 했다", "밀면 안 됐다",
                                    "밀지 말았어야 했다", "밀어도 됐다", "밀어도 괜찮았다", "밀 만했다", "접는게 맞다"])
def test_verify_verdict_words_need_fold_spot(r11, phrase):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    text = f"8순 [9통]은 위험 '높음'이었다. 여기서는 {phrase}."
    assert any("판정 말" in x for x in verify_answer(text, ev, "8순?"))
    # a7(11-23): 판정 말은 fold_spot 이 있어도 위반 — 전에는 fold_spot 이 참이면 통과였다. 판 2 의 참은 도구가 null 로 다루고,
    # 판 3 의 객체(보통 접는 자리 사실)여도 판정 말은 쓰지 않는다(위반 줄의 까닭만 다름).
    f = copy.deepcopy(r11)
    _turn(f, 8)["defense"]["fold_spot"] = True
    assert f"밀기 · 접기 판정 말(fold_spot 사실 없음): {ra.verdict_words(text)[0]}" in \
        verify_answer(text, _ev(f, ("defense_info", {"turn": 8})), "8순?")
    f3 = _v3(r11, fold={8: FOLD_8})
    v = verify_answer(text, _ev(f3, ("defense_info", {"turn": 8})), "8순?")
    assert f"밀기 · 접기 판정 말(보통 접는 자리여도 판정은 쓰지 않음): {ra.verdict_words(text)[0]}" in v, v


def test_verify_verdict_not_triggered_by_ordinary_words(r11):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    for text in ("8순 [9통]은 '높음'이었다. 이 도우미는 밀지 접을지를 판정하지 않는다.",
                 "8순 [6만]을 밀면 안전하다는 뜻은 아니다. 현물이라 '안전'이었다.",
                 "8순에는 [9통]을 밀었다. 위험은 '높음'이었다."):
        assert verify_answer(text, ev, "8순?") == [], text


# ---- 그래프(가짜 LLM): 수비 질문


DEF_Q = "8순에 버린 9통 위험했어?"
VERDICT_8 = DEF_8 + " 여기서는 접었어야 했다."


def test_graph_defense_question_verified(r11):
    gen = Script(tool("defense_info", turn=8), answer(DEF_8))
    out = ask(_req(r11, DEF_Q), gen)
    assert out["outcome"] == "verified" and out["retries"] == 0 and out["prompt_version"] == "a8"
    assert out["tool_calls"] == [{"tool": "defense_info", "args": {"turn": 8}}]
    # 지시문: 수비 규칙 · 새 도구, 둘째 지시문에 defense 칸이 그대로.
    p1, p2 = gen.prompts
    assert "- defense_info(turn):" in p1 and "'높음'" in p1
    assert '"actual":{"tile":"9p","level":"높음","rank":6,"of":13}' in p2


def test_graph_defense_verdict_retry_then_pass(r11):
    gen = Script(tool("defense_info", turn=8), answer(VERDICT_8), answer(DEF_8))
    s, _ = run(_req(r11, DEF_Q), gen)
    assert s["outcome"] == "verified" and s["retries"] == 1 and s["answer"] == DEF_8
    assert "판정 말(fold_spot 사실 없음): 접었어야" in gen.prompts[2]


def test_graph_defense_verdict_twice_falls_back(r11):
    gen = Script(tool("defense_info", turn=8), answer(VERDICT_8), answer(VERDICT_8))
    out = ask(_req(r11, DEF_Q), gen)
    assert out["outcome"] == "fallback" and out["retries"] == 1
    assert any("판정 말" in v for v in out["violations"])
    # 대체 답은 8번째 turn_info 문장(수비 차례 · 현물 아님) — 판정 말 없이 자기 검증을 통과.
    assert out["answer"] == ("확인한 사실만 적는다. 8순은 서 자리가 리치 중이라 패효율 등급을 매기지 않았다. "
                             "버린 [9통]은 현물이 아니다.")


def test_graph_defense_whole_game_find_turns(r11):
    a = ("리치한 자리가 있을 때 '높음' 이상 위험한 패를 버린 차례는 8순 [9통] · 9순 [4만] · 11순 [5삭] · 14순 [6통]이다. "
         "11순과 14순은 '매우 높음'이었다. 이 도우미는 밀지 접을지를 판정하지 않는다.")
    gen = Script(tool("find_turns", min_level="높음"), answer(a))
    out = ask(_req(r11, "리치 받은 뒤 위험하게 버린 곳 있어?"), gen)
    assert out["outcome"] == "verified", out["violations"]


def test_graph_v1_request_runs_as_before(r19, r11):
    # 사실표 1 요청이 그대로 돈다(a2 때의 흐름 · 답).
    gen = Script(tool("turn_info", turn=6), answer(GOOD_6))
    out = ask(_req(_v1(r19), "6순에 왜 4만이 실수야?"), gen)
    assert out["outcome"] == "verified" and out["answer"] == GOOD_6
    # 판 1 의 수비 질문: 단계가 없다고 알리고, 단계 말을 쓰면 걸린다.
    gen = Script(tool("defense_info", turn=8), answer(DEF_8),
                 answer("8순 [9통]은 현물이 아니었다. 이 사실표에는 위험 단계가 없다."))
    s, _ = run(_req(_v1(r11), DEF_Q), gen)
    assert s["outcome"] == "verified" and s["retries"] == 1
    assert s["evidence"][0]["result"]["defense"] is None and "판 1" in s["evidence"][0]["result"]["note"]


def test_prompt_rules_a3():
    p = ra.PROMPT_RULES
    for word in ("리치한 자리에 대한 방총 위험의 크기까지 말한다", "밀지 접을지의 옳고 그름", "다섯 단계",
                 "'안전' · '낮음' · '보통' · '높음' · '매우 높음'", "작은따옴표", "defense_info 로 확인한 뒤",
                 "위험 % 를 지어내지 않는다", "이 도우미는 밀지 접을지를 판정하지 않는다"):
        assert word in p, word
    props = _variant("find_turns")["properties"]["args"]["properties"]
    assert "- defense_info(turn):" in p and "stance" in props and "min_level" in props
    assert ra.TOOL_NAMES == ["turn_info", "discard_options", "what_if", "safe_tiles", "defense_info", "find_turns",
                             "game_summary", "rule"]
    # 지시문의 단계 말 예시가 검증 규칙과 같은 꼴(작은따옴표)이다.
    assert ra._LEVEL_QUOTED.search("'높음'") and not ra._LEVEL_QUOTED.search("높음")


def test_script_fake_defense_question(tmp_path):
    out = tmp_path / "out.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "review_ask.py"), str(FIXTURES / "r11_130921.json"),
                        DEF_Q, "--fake", "--out", str(out)], capture_output=True, cwd=ROOT, timeout=120)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    text = p.stdout.decode("utf-8")
    assert "판정 말(fold_spot 사실 없음)" in text and "결과: verified" in text
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["outcome"] == "verified" and body["retries"] == 1
    assert [c["tool"] for c in body["tool_calls"]] == ["defense_info"]
    assert "'높음'" in body["answer"] and "접었어야" not in body["answer"]


def _bad_forced(ok: dict, **change) -> dict:
    """r11 10번째(리치 중) defense 를 바꾼 요청."""
    r = copy.deepcopy(ok)
    t = _turn(r["facts"], 10)
    assert t["status"] == "forced"
    if "efficient" in change:
        change["efficient"]["tile"] = t["defense"]["tiles"][0]["tile"]
        change["efficient"]["level"] = t["defense"]["tiles"][0]["level"]
    t["defense"].update(change)
    return r


def test_defense_info_forced_turn_r12(r12):
    # r12 12번째: 내가 리치 중 · 북 리치. 버린 [5삭] 의 단계는 남고, stance · efficient 는 null.
    out = run_tool(r12, "defense_info", {"turn": 12})
    d = out["defense"]
    assert (out["status"], out["can_choose"]) == ("forced", False)
    assert d["actual"] == {"tile": "5s", "level": "매우 높음", "rank": 11, "of": 11}
    assert d["stance"] is None and d["efficient"] is None and d["safest"] == ["2m", "4p"]
    assert "리치 중이라 고를 수 없던 차례 — 밀었다 · 접었다를 말하지 않는다" in out["note"]


def test_graph_whole_game_forced_turns_not_pushed(r11):
    # "리치 받은 뒤 위험하게 버린 곳": min_level 줄에 리치 중 차례(11 · 14)가 섞여도 stance 는 없고 고를 수 없던 차례로 보인다.
    a = ("리치 받은 뒤 '높음' 이상 위험한 패를 버린 차례는 8순 [9통] · 9순 [4만]이고, 둘 다 가장 안전한 패보다 위험한 패를 밀었다. "
         "11순 [5삭] · 14순 [6통]도 '매우 높음'이었지만 리치 중이라 고를 수 없던 차례다. 이 도우미는 밀지 접을지를 판정하지 않는다.")
    gen = Script(tool("find_turns", min_level="높음"), answer(a))
    s, _ = run(_req(r11, "리치 받은 뒤 위험하게 버린 곳 있어?"), gen)
    assert s["outcome"] == "verified", s["violations"]
    rows = {r["turn"]: r for r in s["evidence"][0]["result"]["turns"]}
    assert all(rows[n]["stance"] is None and rows[n]["can_choose"] is False for n in (11, 14))
    assert rows[8]["stance"] == rows[9]["stance"] == "pushed"
    # LLM 이 본 지시문: 리치 중 줄은 can_choose false · stance null, 규칙 줄도 들어 있다.
    p2 = gen.prompts[1]
    assert '"turn":11,"discard":"5s","status":"forced","grade":null,"can_choose":false' in p2
    assert '"level":"매우 높음","stance":null' in p2
    assert "리치 중 차례(can_choose 가 false, stance 가 null)는 버린 패의 위험 단계까지만" in gen.prompts[0]


def test_script_fake_whole_game_forced_not_pushed(tmp_path):
    out = tmp_path / "out.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "review_ask.py"), str(FIXTURES / "r11_130921.json"),
                        "리치 받은 뒤 위험하게 버린 곳 있어?", "--fake", "--out", str(out)],
                       capture_output=True, cwd=ROOT, timeout=120)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["outcome"] == "verified" and [c["tool"] for c in body["tool_calls"]] == ["find_turns"]
    assert "밀었다" not in body["answer"] and "접었다" not in body["answer"]
    # a4 F2: 옳고 그름을 묻지 않은 질문이라 "판정하지 않는다" 를 붙이지 않는다. F5: 첫 시도의 판정 말 위반이 남는다.
    assert "판정하지 않는다" not in body["answer"]
    assert len(body["retry_violations"]) == 1 and any("판정 말" in v for v in body["retry_violations"][0])


# ---------------------------------------------------------------- a4(설계서 1122 "a4" F1 ~ F5)


def test_a4_schema_per_tool_variants():
    s = ra.action_schema()
    assert set(s) == {"anyOf"}
    variants = s["anyOf"]
    assert len(variants) == len(ra.TOOLS) + 1
    tools = [v for v in variants if v["properties"]["action"]["enum"] == ["tool"]]
    assert [v["properties"]["tool"]["enum"] for v in tools] == [[n] for n in ra.TOOL_NAMES]   # 변형마다 도구 이름 하나
    for v in tools:
        assert v["type"] == "object" and v["propertyOrdering"] == list(v["properties"])
        assert v["required"][:2] == ["action", "tool"]
        name = v["properties"]["tool"]["enum"][0]
        names = [a.rstrip("?") for a in ra.TOOLS[name]["args"]]
        if "args" in v["properties"]:
            a = v["properties"]["args"]
            assert a["type"] == "object" and list(a["properties"]) == names == a["propertyOrdering"]
            assert all(a["properties"][k] == ra._ARG_PROPS[k] for k in names)        # 설명 · 타입은 한 곳(_ARG_PROPS)
            assert ("args" in v["required"]) == bool(a.get("required"))
        else:
            assert names == []
    # what_if: 둘 다 꼭. game_summary: args 칸 자체가 없다. find_turns: 모두 선택이라 args 가 required 가 아님.
    w = _variant("what_if")
    assert w["required"] == ["action", "tool", "args"] and w["properties"]["args"]["required"] == ["turn", "tile"]
    assert w["propertyOrdering"] == ["action", "tool", "args"]
    g = _variant("game_summary")
    assert "args" not in g["properties"] and g["required"] == ["action", "tool"] and g["propertyOrdering"] == ["action", "tool"]
    f = _variant("find_turns")
    assert f["required"] == ["action", "tool"] and "required" not in f["properties"]["args"]
    assert list(f["properties"]["args"]["properties"]) == ["grade", "status", "min_ukeire_loss", "tile", "after_riichi",
                                                            "stance", "min_level", "fold_spot"]   # a7: fold_spot 더함
    assert _variant("rule")["properties"]["args"]["required"] == ["name"]
    assert _variant("defense_info")["properties"]["args"]["required"] == ["turn"]
    # 빈 properties 를 가진 object 가 어디에도 없다(Gemini 가 거절).
    def walk(x):
        if isinstance(x, dict):
            assert not (x.get("type") == "object" and x.get("properties") == {}), x
            for y in x.values():
                walk(y)
        elif isinstance(x, list):
            for y in x:
                walk(y)
    walk(s)
    ans = variants[-1]
    assert ans["properties"]["action"]["enum"] == ["answer"] and ans["required"] == ["action", "text"]
    assert ans["propertyOrdering"] == ["action", "text"]
    assert not ra.schema_answer_only(s) and ra.schema_answer_only(ra.action_schema(answer_only=True))
    json.dumps(s, ensure_ascii=False)


def test_a4_schema_switch_off_is_old_shape(monkeypatch):
    monkeypatch.setattr(ra, "ACTION_SCHEMA_PER_TOOL", False)
    s = ra.action_schema()
    assert s == {"type": "object",
                 "properties": {"action": {"type": "string", "enum": ["tool", "answer"]},
                                "tool": {"type": "string", "enum": ra.TOOL_NAMES},
                                "args": {"type": "object", "properties": ra._ARG_PROPS},
                                "text": {"type": "string", "description": "action 이 answer 일 때 답(3~5문장)"}},
                 "required": ["action"]}
    assert not ra.schema_answer_only(s)
    # answer_only 는 스위치와 상관없이 같은 꼴.
    assert ra.action_schema(answer_only=True)["propertyOrdering"] == ["action", "text"]


def test_a4_run_tool_still_rejects_extra_and_missing_turn(r19):
    # 스키마가 막아도 방어는 남는다.
    assert "받지 않는 인자" in run_tool(r19, "what_if", {"turn": 6, "tile": "3m", "name": "x"})["error"]
    assert set(run_tool(r19, "what_if", {"tile": "3m"})) == {"error"}


def test_a4_prompt_rules():
    p = ra.PROMPT_RULES
    # F3: 응답 형식에 what_if 예시 · args 는 그 도구의 인자만.
    assert '{"action": "tool", "tool": "what_if", "args": {"turn": 6, "tile": "3m"}}' in p
    assert "args 에는 그 도구의 인자만 넣는다" in p
    # F2: "판정하지 않는다" 는 옳고 그름을 물었을 때만 한 번, 그 밖에는 붙이지 않는다. 권하는 꼴은 그대로.
    assert "질문이 옳고 그름을 물었을 때만" in p and "붙이지 않는다" in p
    assert "이 도우미는 밀지 접을지를 판정하지 않는다" in p
    assert "판단을 묻지 않은 질문에는 그 말을 붙이지 않는다" in p
    # 지시문의 예시 줄 자체가 판정 말 검사에 걸리지 않는다(위험한 말을 예시로 권하지 않음).
    assert ra.verdict_words("이 도우미는 밀지 접을지를 판정하지 않는다.") == []


def test_a4_rule_push_fold_text_same_band():
    pf = run_tool({}, "rule", {"name": "밀기 접기"})["text"]
    assert "같은 구간" in pf and "한 구간 더 위험하면 접었다고 하지 않는다" in pf
    # 단계 말에 작은따옴표가 없다 — 이 글을 읽어도 위험 단계가 근거로 늘지 않는다.
    assert ra.evidence_levels([{"tool": "rule", "args": {"name": "밀기 접기"}, "result": {"text": pf}}]) == set()


@pytest.mark.parametrize("text", [
    "이 도우미는 접었어야 했는지 여부를 직접 판단하지 않는다.",
    "밀어도 됐는지는 판정하지 않는다.",
    "접는 게 맞았는지는 단정할 수 없다.",
    "내렸어야 했을지는 판단하지 않는다.",
    "8순 [9통]은 '높음'이었다. 밀어도 됐는지는 판정하지 않는다.",
])
def test_a4_verdict_disclaimer_with_indirect_question_passes(r11, text):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    assert ra.check_risk_words(text, ev) == [], text
    assert verify_answer(text, ev, "8순에 접었어야 했어?") == [], text


@pytest.mark.parametrize("text,word", [
    ("접었어야 했다.", "접었어야"),
    ("접었어야 했다. 다만 이 도우미는 판정하지 않는다.", "접었어야"),          # 다른 문장의 부정
    ("접었어야 했는지는 판정하지 않지만 접는 게 맞았다.", "접는 게 맞"),        # 뒤의 판정 말
    ("밀어도 됐다고 본다.", "밀어도 됐"),
    ("접었어야 했는지 모르겠지만 위험했다.", "접었어야"),                     # 부정 문구 없음
    ("접었어야 했는지는\n판정하지 않는다.", "접었어야"),                       # 줄바꿈은 문장 끝
])
def test_a4_verdict_words_still_caught(r11, text, word):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    v = ra.check_risk_words(text, ev)
    assert f"밀기 · 접기 판정 말(fold_spot 사실 없음): {word}" in v, (text, v)
    if "접는 게 맞" in word:
        assert not any(x.endswith(": 접었어야") for x in v)       # 앞의 간접 의문은 넘어갔다


def test_a4_split_sentences():
    assert ra.split_sentences("가. 나! 다?\n라\n\n마 1.5장") == ["가", " 나", " 다", "라", "마 1.5장"]
    assert ra.split_sentences("") == []


def test_a4_retry_violations_per_attempt(r11):
    # 판정 말 → 재시도 → 또 판정 말 → 대체: 재시도를 부른 위반은 첫 시도의 것 하나, 마지막 위반은 violations 에.
    gen = Script(tool("defense_info", turn=8), answer(VERDICT_8), answer(DEF_8 + " 밀어도 됐다."))
    out = ask(_req(r11, DEF_Q), gen)
    assert out["outcome"] == "fallback" and out["retries"] == 1
    assert out["retry_violations"] == [["밀기 · 접기 판정 말(fold_spot 사실 없음): 접었어야"]]
    assert "밀기 · 접기 판정 말(fold_spot 사실 없음): 밀어도 됐" in out["violations"]
    # 재시도 둘(한도 2): 시도 순으로 둘이 쌓인다.
    gen = Script(tool("defense_info", turn=8), answer(VERDICT_8), answer(DEF_8 + " 밀어도 됐다."), answer(DEF_8))
    out = ask(_req(r11, DEF_Q), gen, limits=Limits(retries=2))
    assert out["outcome"] == "verified" and out["retries"] == 2 and out["violations"] == []
    assert [v[0].rsplit(": ", 1)[1] for v in out["retry_violations"]] == ["접었어야", "밀어도 됐"]


# ---------------------------------------------------------------- a5(설계서 1122 "a5" G1 ~ G3)

# du2b 꼴: 실제 버린 [2만]은 '낮음', 물어본 [백]만 '안전'(가장 안전한 패). 손 셋은 시험용으로 줄인 것.
_DU2B_DEFENSE = {"tiles": [{"tile": "5z", "level": "안전", "genbutsu": True},
                           {"tile": "2m", "level": "낮음", "genbutsu": True},
                           {"tile": "8s", "level": "보통", "genbutsu": False}],
                 "actual": {"tile": "2m", "level": "낮음", "rank": 2, "of": 3}, "safest": ["5z"]}


def _du2b_ev(risk: dict) -> list[dict]:
    return [{"tool": "what_if", "args": {"turn": 9, "tile": "5z"}, "result": {"turn": 9, "tile": "5z", "risk": risk}}]


def test_a5_what_if_risk_actual_and_safest_levels():
    r = ra._risk_of_tile(_DU2B_DEFENSE, "5z")
    assert r["actual"] == {"tile": "2m", "level": "낮음", "rank": 2, "of": 3}
    assert r["safest"] == [{"tile": "5z", "level": "안전"}]
    # 사실표 쪽 defense.safest(코드 목록)는 그대로 — 찾기만 했다.
    assert _DU2B_DEFENSE["safest"] == ["5z"]


def test_a5_du2b_answer_grounded_by_what_if_only():
    ev = _du2b_ev(ra._risk_of_tile(_DU2B_DEFENSE, "5z"))
    q = "9순에 백을 버렸으면 어땠어?"
    assert verify_answer("9순에 실제 버린 [2만]은 '낮음', [백]은 '안전' 이다.", ev, q) == []
    # 순서 숫자도 risk.actual 에서: 버린 패의 "3종 중 2위".
    assert verify_answer("9순에 실제 버린 [2만]은 '낮음'으로 손의 3종 중 2위였다.", ev, q) == []
    # 도구 결과에 없는 단계는 그대로 위반.
    v = verify_answer("9순에 실제 버린 [2만]은 '매우 높음' 이었다.", ev, q)
    assert "도구로 찾은 사실표에 없는 위험 단계: '매우 높음'" in v
    # a4 꼴(actual 없음 · safest 는 코드 목록)이었다면 [2만]의 '낮음' · "2위" 는 근거가 없었다.
    old = {k: v for k, v in ra._risk_of_tile(_DU2B_DEFENSE, "5z").items() if k != "actual"}
    old["safest"] = ["5z"]
    v = verify_answer("9순에 실제 버린 [2만]은 '낮음'으로 손의 3종 중 2위였다.", _du2b_ev(old), q)
    assert "도구로 찾은 사실표에 없는 위험 단계: '낮음'" in v and any("2위" in x for x in v), v


def test_a5_evidence_from_new_risk_shape():
    ev = _du2b_ev(ra._risk_of_tile(_DU2B_DEFENSE, "8s"))
    # 물어본 [8삭] · 실제 버린 [2만] · 가장 안전한 [백] 셋 다 패 근거, 단계도 셋 다.
    allow, _ = ra.evidence_allow(ev)
    assert {"8s", "2m", "5z"} <= allow.tiles
    assert {3, 2} <= allow.rank and 3 in allow.jong      # 물어본 패 3위 · 버린 패 2위(safest 줄에는 rank 가 없다)
    assert ra.evidence_levels(ev) == {"보통", "낮음", "안전"}


def test_a5_what_if_risk_real_game_actual_differs_from_safest():
    # L16 17번째: 버린 [4만]은 '높음'(9종 중 4위), 가장 안전한 [7만]은 '낮음'. what_if(7만) 하나로 둘 다 말할 수 있다.
    facts = load("L16_135321")
    ev = _ev(facts, ("what_if", {"turn": 17, "tile": "7m"}))
    r = ev[0]["result"]["risk"]
    assert r["actual"] == {"tile": "4m", "level": "높음", "rank": 4, "of": 9}
    assert r["safest"] == [{"tile": "7m", "level": "낮음"}]
    text = "17순에 [7만]을 버렸다면 '낮음'이었다. 실제 버린 [4만]은 '높음'으로 손의 9종 중 4위였다."
    q = "17순에 7만을 버렸으면 어땠어?"
    assert verify_answer(text, ev, q) == []
    assert "도구로 찾은 사실표에 없는 위험 단계: '안전'" in verify_answer("17순에 실제 버린 [4만]은 '안전'이었다.", ev, q)


def test_a5_risk_only_in_what_if():
    # risk 의 새 꼴은 what_if 에만. defense_info 의 defense 는 사실표 그대로(safest 는 코드 목록, actual 은 사실표 칸).
    facts = load("L16_135321")
    d = run_tool(facts, "defense_info", {"turn": 17})["defense"]
    assert d == _turn(facts, 17)["defense"] and all(isinstance(x, str) for x in d["safest"])
    # 사실표 1 · defense 없음 · 리치한 자리 없음은 그대로(null · null · 칸 없음).
    r11 = load("r11_130921")
    assert run_tool(_v1(r11), "what_if", {"turn": 8, "tile": "6m"})["risk"] is None
    f = copy.deepcopy(r11)
    _turn(f, 8)["defense"] = None
    assert run_tool(f, "what_if", {"turn": 8, "tile": "6m"})["risk"] is None


def test_a5_prompt_rules():
    p = ra.PROMPT_RULES
    # G1: 도구 설명 · 수비 절.
    assert "실제 버린 패의 단계와 자리(actual)" in ra.TOOLS["what_if"]["desc"]
    assert "가장 안전한 패마다의 단계(safest)" in ra.TOOLS["what_if"]["desc"]
    assert "위험 단계는 패마다 따로 적힌 값만 말한다. 같은 목록(가장 안전한 패 · 현물)에 있다고 같은 단계로 말하지 않는다" in p
    # G2: 자리는 대괄호 없이.
    assert '자리(동 · 남 · 서 · 북)는 대괄호 없이 "남 자리" 처럼 쓴다. 대괄호는 패에만 쓴다 — [남] 은 바람패 남을 말할 때만.' in p
    assert ra.PROMPT_VERSION == "a8"


def test_a5_seat_wind_bracket_still_checked(r11):
    # G2 는 지시문만 — 도구 결과에 없는 바람패 대괄호는 지금처럼 걸린다.
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    v = verify_answer(DEF_8 + " [남] 자리가 리치 중이었다.", ev, DEF_Q)
    assert any("[남]" in x for x in v), v


@pytest.mark.parametrize("text", [
    "해당 차례에 접었어야 했는지에 대한 판정 정보는 없다.",
    "밀어도 됐는지에 대한 판정 기록은 없다.",
    "접었어야 했는지 판정이 없다.",
    "접었어야 했는지는 판정은 없다.",
    "밀어도 됐는지에 대한 판정정보가 없다.",
])
def test_a5_verdict_no_verdict_info_passes(r11, text):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    assert ra.verdict_words(text) == [], text
    assert ra.check_risk_words(text, ev) == [], text
    assert verify_answer(text, ev, "8순에 접었어야 했어?") == [], text


@pytest.mark.parametrize("text,word", [
    ("접었어야 했다. 판정 정보는 없다.", "접었어야"),             # 다른 문장의 부정
    ("판정 정보는 없지만 접었어야 했다.", "접었어야"),             # 간접 의문이 아님
    ("접었어야 했는지 정보가 없다.", "접었어야"),                 # 부정 문구에 "판정" 이 없음
    ("밀어도 됐다. 판정 기록은 없다.", "밀어도 됐"),
])
def test_a5_verdict_words_still_caught(r11, text, word):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    v = ra.check_risk_words(text, ev)
    assert f"밀기 · 접기 판정 말(fold_spot 사실 없음): {word}" in v, (text, v)


# ---------------------------------------------------------------- a6(11-23 말투: stance 단정문 · 평가 말 · 엉뚱한 부정 문구)

# 11-22 앱 질문 실제 답(mahjong_calc experiments/paifu_review/results/agent_1122app · agent_1122app_old_b). 틀린 사실은 없고 말투만.
B3A = ("17순에 버린 [4만]은 서 자리가 리치한 상황에서 현물이 아니며 위험 단계는 '높음'이었다. 당시 손에 있던 패 9종 가운데 "
       "안전한 순서로는 4위였다. 가장 안전했던 패는 '낮음' 단계인 [7만]이었다. 버린 패가 가장 안전한 패보다 위험했던 사실상 밀었던 차례다.")
B4A = ("서 자리가 리치한 뒤 내 버림 차례는 17순과 18순의 2번 있었다. 17순에는 현물이 아닌 [4만]을 버렸고, 방총 위험은 '높음' "
       "단계였으며 가장 안전한 패보다 위험한 패를 버려 사실상 밀었다. 18순에도 [4만]을 버렸는데, 이때는 서 자리의 현물이었으나 "
       "위험 단계는 '보통'이었고 사실상 밀었다. 두 차례 모두 수비 상태에서 비교적 위험한 패를 버려 나아간 진행이었다.")
UG4_DISCLAIM = "이 도우미는 밀지 접을지를 판정하지 않으며, 손패 기록이 없어 패효율 판단도 불가능하다."
NO_RIICHI = "리치 없는 질문에 밀기 · 접기 부정 문구"


def test_a6_prompt_rules():
    p = ra.PROMPT_RULES
    assert ra.PROMPT_VERSION == "a8"
    # stance 는 단정 사실문 하나로 — 셋의 예시와 붙이지 않을 말.
    assert "단정하는 사실문 하나로 쓴다" in p
    assert "pushed: \"가장 안전한 [백] 대신 '높음' 인 [5통]을 버려 밀었다\"" in p
    assert "\"가장 안전한 패를 버려 접었다\"" in p and "어느 쪽도 아니다\"(뜻은 rule(\"밀기 접기\") 글 그대로)" in p
    assert '"사실상" · "…라고 볼 수 있다" · "…로 기록되었다" · "…셈이다" 를 붙이지 않는다' in p
    # 평가 말 금지 · 화료 · 등급 말.
    assert "좋고 나쁨 · 잘하고 못함을 꾸미는 말을 쓰지 않는다" in p and "\"…할 수 있었던 선택\"" in p
    assert "화료는 \"화료했다\"" in p and "등급 말(최선 · 좋음 · 부정확 · 실수 · 큰 실수)은 도구 결과의 등급으로만" in p
    # "판정하지 않는다" 는 리치가 있는 밀기 · 접기 질문에만, 패효율 질문에는 밀기 · 접기 말을 꺼내지 않는다.
    assert "리치한 자리가 있는 차례(또는 판)의 밀기 · 접기를 물었을 때만" in p
    assert "패효율 질문이다 — 밀기 · 접기 말을 꺼내지 않고" in p
    # 지시문의 stance 예시 문장 자체는 새 검사(평가 말 · 판정 말)에 걸리지 않는다.
    for example in ("가장 안전한 [백] 대신 '높음' 인 [5통]을 버려 밀었다", "가장 안전한 패를 버려 접었다",
                    "가장 안전한 [백]과 위험 차이가 거의 없는 [5통]을 버려 접었다",
                    "가장 안전한 [백] 대신 [5통]을 버렸고, 밀었다 · 접었다 어느 쪽도 아니다"):
        assert ra.eval_words(example) == [] and ra.verdict_words(example) == [], example


def test_a6_response_prompt_version(r19):
    out = ask(_req(r19, "6순에 왜 4만이 실수야?"), Script(tool("turn_info", turn=6), answer(GOOD_6)))
    assert out["outcome"] == "verified" and out["prompt_version"] == "a8"


def test_a6_real_answers_caught_only_by_eval_words():
    # b3a · b4a 전문은 a5 검사를 모두 통과했고, a6 에서는 평가 말만 걸린다(다른 위반 없음).
    facts = load("L16_135321")
    assert verify_answer(B3A, _ev(facts, ("defense_info", {"turn": 17})), "17순에 버린 4만 위험했어?") == ["평가 말: 사실상"]
    assert verify_answer(B4A, _ev(facts, ("find_turns", {"after_riichi": "서"})), "서 리치 뒤에 나는 어떻게 했어?") == \
        ["평가 말: 사실상", "평가 말: 비교적"]


@pytest.mark.parametrize("text,words", [
    ("버린 패가 가장 안전한 패보다 위험했던 사실상 밀었던 차례다.", ["사실상"]),                              # b3a
    ("가장 안전한 패 대신 위험 단계가 더 높은 패를 버려 수비 사실상 밀었다로 기록되었다.", ["사실상"]),         # ax4a
    ("가장 안전한 패보다 위험한 패를 버려 사실상 밀었다고 볼 수 있다.", ["사실상", "고 볼 수 있"]),
    ("두 차례 모두 수비 상태에서 비교적 위험한 패를 버려 나아간 진행이었다.", ["비교적"]),                     # b4a
    ("전반적으로 큰 실수 없이 높은 패효율을 유지하며 무난하게 진행한 판이다.", ["무난"]),                       # ax3c
    ("실수나 큰 실수 없이 패효율 기준 깔끔하게 조패를 마친 판이다.", ["깔끔"]),                                 # ug3
    ("[4만]을 쯔모해 성공적으로 화료했다.", ["성공적"]),
    ("[4만] 쯔모로 승리를 거두었다.", ["승리"]),
    ("훌륭한 수비였다. 현명한 선택이다. 안타깝게도 방총했다.", ["훌륭", "현명", "안타깝"]),
    ("8순은 밀었다라고 볼 수 있다.", ["라고 볼 수 있"]),
    ("8순은 수비로 볼 수 있다.", ["로 볼 수 있"]),
    ("8순은 밀었다고 볼  수 있다.", ["고 볼 수 있"]),                                                          # 공백 둘도
])
def test_a6_eval_words_caught(r11, text, words):
    assert ra.eval_words(text) == words
    v = verify_answer(text, _ev(r11, ("defense_info", {"turn": 8})), "8순 어땠어?")
    assert [x for x in v if x.startswith("평가 말: ")] == [f"평가 말: {w}" for w in words], v


def test_a6_eval_words_pass_and_question_words(r11):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    for text in (DEF_8, "8순에 가장 안전한 [6만] 대신 '높음' 인 [9통]을 버려 밀었다.", "8순은 가장 안전한 패를 버려 접었다.",
                 "8순 [9통]은 현물로 보지 않는다.", "볼 수 없다.", "승부를 걸었는지는 모른다."):
        assert ra.eval_words(text) == [], text
    assert verify_answer("8순에 가장 안전한 [6만] 대신 '높음' 인 [9통]을 버려 밀었다.", ev, "8순 어땠어?") == []
    # 질문에만 있으면 통과, 답에 옮기면 질문에 있어도 위반.
    assert verify_answer(DEF_8, ev, "8순 무난했어?") == []
    assert "평가 말: 무난" in verify_answer(DEF_8 + " 무난한 버림이었다.", ev, "8순 무난했어?")
    # 대괄호 안은 보지 않는다.
    assert ra.eval_words("[사실상]") == []


def test_a6_new_checks_append_after_old_violations(r19):
    ev = _ev(r19, ("turn_info", {"turn": 6}))
    q = "6순에 왜 4만이 실수야?"
    old = verify_answer(BAD_6, ev, q)
    assert old and not any(x.startswith("평가 말") for x in old)
    assert verify_answer(BAD_6 + " 무난하지 않았다.", ev, q) == old + ["평가 말: 무난"]
    assert verify_answer(BAD_6 + " 이 도우미는 밀지 접을지를 판정하지 않는다.", ev, q) == old + [NO_RIICHI]


def test_a6_pushfold_disclaimer_without_riichi_caught(r19):
    # ug4 꼴: 리치한 자리가 없는 손패 모름 차례(r19 1번째 — 짐작 손패)의 "잘한 거야?" 에 밀기 · 접기 부정 문구.
    ev = _ev(r19, ("turn_info", {"turn": 1}))
    assert not ra.evidence_has_riichi(ev)
    d = ra.bracket(_turn(r19, 1)["discard"])
    text = f"1순은 손패가 짐작이라 숫자로 평가하지 않았다. 버린 패는 {ra.with_ida(d)}. {UG4_DISCLAIM}"
    assert verify_answer(text, ev, "1순에 버린 건 잘한 거야?") == [NO_RIICHI]
    # 등급 매긴 차례(6번째)의 "실수야?" 도 패효율 질문 — 꼴이 달라도(밀기 · 접기 · 판정은 없다) 걸린다.
    ev6 = _ev(r19, ("turn_info", {"turn": 6}))
    for tail in (" 이 도우미는 밀지 접을지를 판정하지 않는다.", " 밀기 · 접기는 판정하지 않는다.", " 밀기접기 판정은 없다."):
        assert verify_answer(GOOD_6 + tail, ev6, "6순에 왜 4만이 실수야?") == [NO_RIICHI], tail
    # game_summary · 조건 없는 find_turns 도 리치가 없는 판이면 맥락이 없다.
    assert not ra.evidence_has_riichi(_ev(r19, ("game_summary", {}), ("find_turns", {})))


def test_a6_pushfold_disclaimer_with_riichi_passes(r11, r19):
    facts = load("L16_135321")
    cases = [
        (r11, ("defense_info", {"turn": 8})),                 # 수비 도구
        (r19, ("safe_tiles", {"turn": 6})),                    # 부른 것만으로(리치 없다는 결과여도)
        (r11, ("find_turns", {"min_level": "높음"})),          # 수비 조건
        (r19, ("find_turns", {"after_riichi": "동"})),         # after_riichi 조건(리치가 없던 자리여도)
        (r19, ("rule", {"name": "밀기 접기"})),               # 밀기 · 접기 뜻을 찾음
        (facts, ("turn_info", {"turn": 17})),                  # 결과의 riichi_winds
        (facts, ("what_if", {"turn": 17, "tile": "7m"})),      # risk
        (r11, ("game_summary", {})),                           # others 의 남 리치
        (r11, ("find_turns", {})),                             # 줄의 riichi_winds · stance
    ]
    for f, call in cases:
        ev = _ev(f, call)
        assert ra.evidence_has_riichi(ev), call
        assert ra.check_tone("이 도우미는 밀지 접을지를 판정하지 않는다.", ev) == [], call
    # a5 의 시험 답 꼴(리치 맥락이 있는 부정 문구)은 그대로 통과.
    assert verify_answer(DEF_8 + " 이 도우미는 밀지 접을지를 판정하지 않는다.", _ev(r11, ("defense_info", {"turn": 8})),
                         "8순에 접었어야 했어?") == []


def test_a6_pushfold_disclaimer_needs_one_sentence():
    assert ra.has_pushfold_disclaimer(UG4_DISCLAIM)
    assert ra.has_pushfold_disclaimer("밀기·접기 판정 정보는 없다.")
    # 밀기 · 접기 말만, 부정 문구만, 다른 문장에 나뉘면 부정 문구가 아니다.
    assert not ra.has_pushfold_disclaimer("이 차례는 밀기 · 접기와 상관없다.")
    assert not ra.has_pushfold_disclaimer("타점은 판정하지 않는다.")
    assert not ra.has_pushfold_disclaimer("밀기 · 접기 질문이다. 판정하지 않는다.")


def test_a6_graph_disclaimer_retry_then_pass(r19):
    d = ra.bracket(_turn(r19, 1)["discard"])
    clean = f"1순은 손패가 짐작이라 숫자로 평가하지 않았다. 버린 패는 {ra.with_ida(d)}. 손패를 몰라 패효율 기준 등급도 모른다."
    gen = Script(tool("turn_info", turn=1), answer(f"{clean} {UG4_DISCLAIM}"), answer(clean))
    out = ask(_req(r19, "1순에 버린 건 잘한 거야?"), gen)
    assert out["outcome"] == "verified" and out["retries"] == 1 and out["answer"] == clean
    assert out["retry_violations"] == [[NO_RIICHI]] and out["prompt_version"] == "a8"
    assert f"위반: {NO_RIICHI}" in gen.prompts[2]


def test_a6_graph_eval_words_twice_falls_back(r19):
    gen = Script(tool("turn_info", turn=6), answer(GOOD_6 + " 무난한 선택은 아니었다."),
                 answer(GOOD_6 + " 사실상 실수다."))
    out = ask(_req(r19, "6순에 왜 4만이 실수야?"), gen)
    assert out["outcome"] == "fallback" and out["retries"] == 1
    assert out["retry_violations"] == [["평가 말: 무난"]] and out["violations"] == ["평가 말: 사실상"]
    assert ra.eval_words(out["answer"]) == []                 # 대체 답(turn_info 문장)은 평가 말이 없다


# ---------------------------------------------------------------- a7(설계서 mahjong_calc docs/review_agent_foldspot_scores_design_1123.md)
#
# 사실표 3: turns[i].defense.fold_spot = null | {shanten, ai_fold_pct}, game.scores = null | 2-2 꼴. 예시 11판에는 접는 자리가
# 없어(설계서 6절) 시험은 사실표 2 견본을 판 3 으로 올리고 fold_spot · scores 를 손으로 넣는다.

FOLD_8 = {"shanten": 3, "ai_fold_pct": 94}
# 설계서 2-2 예시 그대로(내 자리 서 — r19 의 hero_wind).
DESIGN_SCORES = {"hero": {"wind": "서", "score": 22000, "rank": 3},
                 "others": [{"wind": "동", "score": 25000, "rank": 2, "gap": 3000, "ahead": True},
                            {"wind": "남", "score": 31000, "rank": 1, "gap": 9000, "ahead": True},
                            {"wind": "북", "score": 22000, "rank": 3, "gap": 0, "ahead": False}],
                 "kyotaku": 1}
NO_SCORE_NOTE = "이 판에는 점수 기록이 없다"


def _v3(facts: dict, fold: dict | None = None, scores=None) -> dict:
    """사실표 2 → 3: version 3, game.scores, 차례마다 fold_spot(손으로 넣은 것 말고는 null)."""
    f = copy.deepcopy(facts)
    f["version"] = 3
    f["game"]["scores"] = copy.deepcopy(scores)
    for t in f["turns"]:
        if t["defense"] is not None:
            t["defense"]["fold_spot"] = copy.deepcopy((fold or {}).get(t["turn"]))
    return f


def _scores_of(hero_wind: str, pts: dict, kyotaku=0) -> dict:
    """{바람: 점수} → scores(앱 설계서 2-2 의 계산: rank = 1 + 더 높은 자리 수, gap = |차|, ahead = 더 높은가)."""
    def rank(w):
        return 1 + sum(p > pts[w] for p in pts.values())
    me = pts[hero_wind]
    return {"hero": {"wind": hero_wind, "score": me, "rank": rank(hero_wind)},
            "others": [{"wind": w, "score": pts[w], "rank": rank(w), "gap": abs(pts[w] - me), "ahead": pts[w] > me}
                       for w in ra.WINDS if w != hero_wind],
            "kyotaku": kyotaku}


@pytest.fixture(scope="module")
def r11f() -> dict:
    """r11 판 3: 8번째(수비, 서 리치)만 보통 접는 자리, 점수 없음."""
    return _v3(load("r11_130921"), fold={8: FOLD_8})


@pytest.fixture(scope="module")
def r19s() -> dict:
    """r19 판 3: 설계서 예시 점수(내 자리 서), 접는 자리 없음(리치 없는 판)."""
    return _v3(load("r19_142013"), scores=DESIGN_SCORES)


# 앱이 보낼 사실표 3 의 최소 꼴(검사만 통과하면 되는 손 — 실제 앱은 손 14장 · 후보 · 사건을 다 싣는다).
MINIMAL_V3 = {
    "version": 3,
    "game": {"round": "남2국", "honba": 0, "hero_wind": "서", "dora_indicators": ["3p"], "efficiency_pct": None,
             "grades": {}, "statuses": {"defense": 1}, "result": None,
             "scores": {"hero": {"wind": "서", "score": 22000, "rank": 3},
                        "others": [{"wind": "동", "score": 25000, "rank": 2, "gap": 3000, "ahead": True},
                                   {"wind": "남", "score": 31000, "rank": 1, "gap": 9000, "ahead": True},
                                   {"wind": "북", "score": 22000, "rank": 3, "gap": 0, "ahead": False}],
                        "kyotaku": 1}},
    "turns": [{"turn": 1, "step": 30, "status": "defense", "grade": None, "hand": ["1m", "9p", "5z"], "drawn": "5z",
               "discard": "9p", "tsumogiri": False, "melds": 0, "riichi_declared": False, "shanten_prev": None,
               "shanten_before": 3, "shanten_loss": None, "ukeire_loss": None, "options": [], "actual": None, "best": [],
               "riichi_winds": ["남"], "genbutsu": False, "safe_tiles": ["5z"],
               "defense": {"threat_winds": ["남"],
                           "tiles": [{"tile": "5z", "level": "안전", "genbutsu": True},
                                     {"tile": "1m", "level": "보통", "genbutsu": False},
                                     {"tile": "9p", "level": "높음", "genbutsu": False}],
                           "actual": {"tile": "9p", "level": "높음", "rank": 3, "of": 3},
                           "safest": ["5z"], "efficient": {"tile": "9p", "level": "높음"}, "stance": "pushed",
                           "fold_spot": {"shanten": 3, "ai_fold_pct": 94}}}],
    "others": [{"step": 27, "wind": "남", "kind": "riichi", "tile": "4s"}],
}


# ---- 요청 검사(사실표 3)


def test_a7_validate_v3_accepts(r11f, r19s):
    validate_request(_req(r11f, "8순에 접었어야 했어?"))
    validate_request(_req(r19s, "이 판 시작할 때 점수 상황은 어땠어?"))
    validate_request(_req(MINIMAL_V3, "1순 위험했어?"))
    # 같은 점수는 같은 순위(셋이 같아도), 공탁 모름(null), 음수 점수(100 의 배수).
    for pts, k in (({"동": 25000, "남": 25000, "서": 25000, "북": 25000}, 0),
                   ({"동": 41200, "남": 30000, "서": -1200, "북": 30000}, None),
                   ({"동": 25000, "남": 30000, "서": 30000, "북": 15000}, 3)):
        validate_request(_req(_v3(load("r19_142013"), scores=_scores_of("서", pts, k)), "점수?"))
    assert _scores_of("서", {"동": 25000, "남": 30000, "서": 30000, "북": 15000})["hero"]["rank"] == 1
    assert ra.FACTS_VERSION == 3 and ra.FACTS_VERSIONS == (1, 2, 3)


def _mut_scores(fn):
    r = _req(_v3(load("r19_142013"), scores=DESIGN_SCORES), "점수?")
    fn(r["facts"]["game"]["scores"])
    return r


def _mut_fold(value):
    r = _req(_v3(load("r11_130921"), fold={8: FOLD_8}), "8순?")
    _turn(r["facts"], 8)["defense"]["fold_spot"] = value
    return r


def _no_scores_key():
    r = _req(_v3(load("r19_142013")), "점수?")
    del r["facts"]["game"]["scores"]
    return r


@pytest.mark.parametrize("make,word", [
    (_no_scores_key, "scores 없음"),                                # 판 3 은 scores 칸이 꼭(null 이어도)
    (lambda: _mut_fold(True), "fold_spot"),                         # 판 3 에 참 · 거짓은 안 됨
    (lambda: _mut_fold({"shanten": 2, "ai_fold_pct": 94}), "fold_spot"),
    (lambda: _mut_fold({"shanten": 9, "ai_fold_pct": 94}), "fold_spot"),
    (lambda: _mut_fold({"shanten": 3, "ai_fold_pct": 0}), "fold_spot"),
    (lambda: _mut_fold({"shanten": 3, "ai_fold_pct": 101}), "fold_spot"),
    (lambda: _mut_fold({"shanten": 3, "ai_fold_pct": 94.0}), "fold_spot"),
    (lambda: _mut_fold({"shanten": 3, "ai_fold_pct": "94"}), "fold_spot"),
    (lambda: _mut_fold({"shanten": True, "ai_fold_pct": 94}), "fold_spot"),
    (lambda: _mut_fold({"shanten": 3}), "fold_spot"),
    (lambda: _mut_fold([3, 94]), "fold_spot"),
    (lambda: _mut_scores(lambda s: s.clear()), "객체"),
    (lambda: _mut_scores(lambda s: s.pop("kyotaku")), "객체"),
    (lambda: _mut_scores(lambda s: s["hero"].update(score=22050)), "hero"),         # 100 의 배수가 아님
    (lambda: _mut_scores(lambda s: s["hero"].update(score=22000.0)), "hero"),
    (lambda: _mut_scores(lambda s: s["hero"].update(rank=5)), "hero"),
    (lambda: _mut_scores(lambda s: s["hero"].update(rank=0)), "hero"),
    (lambda: _mut_scores(lambda s: s["hero"].update(wind="북")), "hero_wind"),     # game.hero_wind(서)와 다름
    (lambda: _mut_scores(lambda s: s["others"].pop()), "3개"),
    (lambda: _mut_scores(lambda s: s["others"][1].update(wind="동")), "겹침"),
    (lambda: _mut_scores(lambda s: s["others"].reverse()), "순"),                   # 동 · 남 · 서 · 북 순이 아님
    (lambda: _mut_scores(lambda s: s["others"][0].update(gap=-3000)), "others\\[0\\]"),
    (lambda: _mut_scores(lambda s: s["others"][0].update(ahead="yes")), "others\\[0\\]"),
    (lambda: _mut_scores(lambda s: s["others"][0].update(score=25010)), "others\\[0\\]"),
    (lambda: _mut_scores(lambda s: s.update(kyotaku=-1)), "kyotaku"),
    (lambda: _mut_scores(lambda s: s.update(kyotaku=1.5)), "kyotaku"),
    (lambda: _mut_scores(lambda s: s["hero"].update(rank=2)), "rank"),            # 값끼리 어긋남
    (lambda: _mut_scores(lambda s: s["others"][2].update(rank=4)), "rank"),        # 같은 점수는 같은 순위
    (lambda: _mut_scores(lambda s: s["others"][0].update(gap=3100)), "gap ="),
    (lambda: _mut_scores(lambda s: s["others"][2].update(ahead=True)), "ahead"),   # 같은 점수면 false
])
def test_a7_validate_v3_rejects(make, word):
    r = make()
    with pytest.raises(ValueError, match=word):
        validate_request(r)


def test_a7_validate_v1_v2_still_pass():
    # 예전 견본(판 2)은 그대로 — scores 칸 없이, fold_spot null.
    for name in GAMES:
        f = load(name)
        assert f["version"] == 2 and "scores" not in f["game"]
        validate_request(_req(f, "6순에 왜?"))
        validate_request(_req(_v1(f), "6순에 왜?"))
    # 판 2 의 참 · 거짓 fold_spot 은 받는다(도구가 null 로 다룸). 판 2 에 객체는 안 된다.
    f = copy.deepcopy(load("r11_130921"))
    _turn(f, 8)["defense"]["fold_spot"] = True
    validate_request(_req(f, "8순?"))
    _turn(f, 8)["defense"]["fold_spot"] = dict(FOLD_8)
    with pytest.raises(ValueError, match="참 · 거짓"):
        validate_request(_req(f, "8순?"))
    # 판 2 에 scores 칸이 있어도 보지 않는다(무시).
    f = copy.deepcopy(load("r19_142013"))
    f["game"]["scores"] = "아무거나"
    validate_request(_req(f, "점수?"))


# ---- 도구


def test_a7_defense_info_fold_spot_note(r11f):
    out = run_tool(r11f, "defense_info", {"turn": 8})
    assert out["defense"]["fold_spot"] == FOLD_8 and out["defense"] == _turn(r11f, 8)["defense"]
    want = ("보통 접는 자리다(리치를 받았고 내 손이 3샹텐, 효율대로 버릴 패가 가장 안전한 패보다 뚜렷이 위험) — "
            "강한 AI 는 이런 자리의 약 94% 에서 접는다. 판정이 아니라 참고다")
    assert out["note"] == want
    assert "판정은 없다" not in out["note"] and "판정하지 않는다" not in out["note"]
    # 숫자는 사실표 값 그대로.
    f = _v3(load("r11_130921"), fold={8: {"shanten": 4, "ai_fold_pct": 90}})
    note = run_tool(f, "defense_info", {"turn": 8})["note"]
    assert "내 손이 4샹텐" in note and "약 90% 에서" in note
    # null 인 차례는 지금 알림 그대로.
    out9 = run_tool(r11f, "defense_info", {"turn": 9})
    assert out9["defense"]["fold_spot"] is None and "이 차례의 밀기 · 접기 판정은 없다(fold_spot null)" in out9["note"]
    # 판 2 의 참은 null 로(사실표는 그대로 두고 사본으로).
    v2 = copy.deepcopy(load("r11_130921"))
    _turn(v2, 8)["defense"]["fold_spot"] = True
    o2 = run_tool(v2, "defense_info", {"turn": 8})
    assert o2["defense"]["fold_spot"] is None and "판정은 없다(fold_spot null)" in o2["note"]
    assert _turn(v2, 8)["defense"]["fold_spot"] is True
    assert {k: v for k, v in o2["defense"].items() if k != "fold_spot"} == \
        {k: v for k, v in _turn(v2, 8)["defense"].items() if k != "fold_spot"}


def test_a7_find_turns_fold_spot(r11f, r19):
    out = run_tool(r11f, "find_turns", {"fold_spot": True})
    assert out["conditions"] == {"fold_spot": True} and out["count"] == 1 and "note" not in out
    row = out["turns"][0]
    assert row["turn"] == 8 and row["fold_spot"] == FOLD_8
    assert row["line"] == "8순 [9통] 버림 · 수비 · 서 리치에 현물 아님 · 보통 접는 자리"
    for arg in ("true", "참", " True "):
        assert run_tool(r11f, "find_turns", {"fold_spot": arg}) == out, arg
    # 다른 조건과 그리고.
    assert [r["turn"] for r in run_tool(r11f, "find_turns", {"after_riichi": "서", "fold_spot": True})["turns"]] == [8]
    assert run_tool(r11f, "find_turns", {"stance": "folded", "fold_spot": True})["count"] == 0
    # 조건 없음: 줄마다 fold_spot 칸, 접는 자리 줄에만 꼬리말.
    rows = run_tool(r11f, "find_turns", {})["turns"]
    assert [r["fold_spot"] for r in rows] == [FOLD_8 if r["turn"] == 8 else None for r in rows]
    assert [r["turn"] for r in rows if r["line"].endswith(" · 보통 접는 자리")] == [8]
    # 거짓은 조건 없음과 같다(참만 뜻이 있음). 모르는 값은 오류.
    assert run_tool(r11f, "find_turns", {"fold_spot": False}) == run_tool(r11f, "find_turns", {})
    assert run_tool(r11f, "find_turns", {"fold_spot": "false"})["conditions"] == {}
    bad = run_tool(r11f, "find_turns", {"fold_spot": "maybe"})
    assert set(bad) == {"error"} and "fold_spot" in bad["error"]
    # 접는 자리가 없는 판(리치 없음 · 판 2): 빈 목록 + "판정이 없다" 알림(접는 자리가 아니라는 말이 아님).
    for f in (r19, load("r11_130921")):
        none = run_tool(f, "find_turns", {"fold_spot": True})
        assert none["count"] == 0
        assert none["note"] == NOTE_NO_SPOT and "fold_spot_turns" not in none
    # 판 2 의 참은 접는 자리로 고르지 않는다.
    v2 = copy.deepcopy(load("r11_130921"))
    _turn(v2, 8)["defense"]["fold_spot"] = True
    assert run_tool(v2, "find_turns", {"fold_spot": True})["count"] == 0
    assert _turn_line_tail(run_tool(v2, "find_turns", {})) == []
    # after_riichi 알림과 겹치면 둘 다.
    both = run_tool(load("r12_131655"), "find_turns", {"after_riichi": "동", "fold_spot": True})
    assert both["note"] == "동 자리는 이 판에서 리치하지 않았다. " + NOTE_NO_SPOT


NOTE_NO_SPOT = "이 판에는 보통 접는 자리 판정이 있는 차례가 없다(접는 자리가 아니라는 뜻이 아니다)"


def _folded_spot_10() -> dict:
    """접는 자리가 10순 하나(stance folded)인 판: r11 판 3, 10번째(리치 중)의 defense 를 수비 · folded 로 바꿈(도구 시험용 —
    요청 검사는 거치지 않는다). 10번째 버린 [중]은 '보통'."""
    f = _v3(load("r11_130921"), fold={10: dict(FOLD_8)})
    t = _turn(f, 10)
    t["status"] = "defense"
    t["defense"]["stance"] = "folded"
    return f


def test_a7_find_turns_zero_rows_two_notes(r19):
    # (가) 판 전체에 접는 자리가 없다: "이 판에는 … 판정이 있는 차례가 없다", fold_spot_turns 없음.
    a = run_tool(r19, "find_turns", {"fold_spot": True, "stance": "pushed"})
    assert a["count"] == 0 and a["note"] == NOTE_NO_SPOT and "fold_spot_turns" not in a
    ev_a = [{"tool": "find_turns", "args": {"fold_spot": True}, "result": a}]
    assert verify_answer(NOTE_NO_SPOT + ".", ev_a, "접는 자리에서 민 곳 있어?") == []       # 알림을 옮긴 답은 검사 10 통과
    # (나) 접는 자리는 있는데 다른 조건(stance)에서 다 걸러졌다: 알림 + 접는 자리 줄.
    f = _folded_spot_10()
    out = run_tool(f, "find_turns", {"fold_spot": True, "stance": "pushed"})
    assert out["count"] == 0 and out["turns"] == []
    assert out["note"] == "보통 접는 자리는 10순에 있지만 다른 조건까지 맞는 차례는 없다"
    spots = out["fold_spot_turns"]
    assert spots == run_tool(f, "find_turns", {"fold_spot": True})["turns"]      # turns 줄과 같은 꼴
    s = spots[0]
    assert (s["turn"], s["discard"], s["level"], s["stance"], s["fold_spot"]) == (10, "7z", "보통", "folded", FOLD_8)
    assert s["line"].endswith(" · 보통 접는 자리")
    # 근거: 순 번호 · 패 · 단계 · fold_spot(검사 10).
    ev = [{"tool": "find_turns", "args": {"fold_spot": True, "stance": "pushed"}, "result": out}]
    allow, pct = ra.evidence_allow(ev)
    assert 10 in allow.jun and "7z" in allow.tiles and 3 in allow.shanten and 94 in pct
    assert "보통" in ra.evidence_levels(ev) and ra.evidence_has_fold_spot(ev)
    q = "리치 받은 뒤에 접는 자리에서 민 곳 있어?"
    for text in ("없다. 보통 접는 자리는 10순 하나이고, 가장 안전한 패를 버려 접었다.",
                 "보통 접는 자리는 10순 하나이고, 그때 '보통' 인 [중]을 버려 접었다. 접는 자리에서 민 차례는 없다.",
                 out["note"] + "."):
        assert verify_answer(text, ev, q) == [], text
    # 줄에 없는 순 · 패는 그대로 위반.
    v = verify_answer("보통 접는 자리는 12순 [9통]이었다.", ev, q)
    assert any("12순" in x for x in v) and any("[9통]" in x for x in v)
    # 접는 자리가 둘이면 순 번호를 모두.
    f2 = _v3(load("r11_130921"), fold={8: dict(FOLD_8), 9: dict(FOLD_8)})
    two = run_tool(f2, "find_turns", {"fold_spot": True, "stance": "folded"})
    assert two["note"] == "보통 접는 자리는 8순 · 9순에 있지만 다른 조건까지 맞는 차례는 없다"
    assert [r["turn"] for r in two["fold_spot_turns"]] == [8, 9]
    # after_riichi 알림과 겹치면 둘 다(앞이 after_riichi). 조건 없이 줄이 있으면 fold_spot_turns 는 싣지 않는다.
    both = run_tool(f, "find_turns", {"after_riichi": "동", "fold_spot": True})
    assert both["note"] == "동 자리는 이 판에서 리치하지 않았다. 보통 접는 자리는 10순에 있지만 다른 조건까지 맞는 차례는 없다"
    assert "fold_spot_turns" not in run_tool(f, "find_turns", {"fold_spot": True})
    # 검사 10 의 새 통과 꼴은 "판정이 있는 차례가 없" 까지만 — "판정이 있는 차례다" 는 아님.
    assert ra.check_fold_spot_words("9순은 접는 자리 판정이 있는 차례다.", []) == ["접는 자리 말(fold_spot 사실 없음)"]


def _turn_line_tail(out: dict) -> list[int]:
    return [r["turn"] for r in out["turns"] if r["line"].endswith("보통 접는 자리")]


def test_a7_game_summary_scores(r19s, r19):
    s = run_tool(r19s, "game_summary", {})
    assert s["scores"] == {**DESIGN_SCORES, "basis": "국 시작 점수(이 국의 리치봉 · 결과 반영 전)"} and "note" not in s
    assert "basis" not in r19s["game"]["scores"]                  # 사실표는 그대로
    # 다른 칸은 그대로.
    base = run_tool(r19, "game_summary", {})
    assert {k: v for k, v in s.items() if k != "scores"} == {k: v for k, v in base.items() if k not in ("scores", "note")}
    # 점수 없음: 판 3 의 null · 판 2 · 판 1.
    for f in (_v3(r19), r19, _v1(r19)):
        out = run_tool(f, "game_summary", {})
        assert out["scores"] is None and "note" not in out      # a8: 점수 없음 알림은 뺌(scores: null 만)
    # 판 2 에 scores 칸이 남아 있어도 보지 않는다.
    v2 = copy.deepcopy(r19)
    v2["game"]["scores"] = copy.deepcopy(DESIGN_SCORES)
    assert run_tool(v2, "game_summary", {})["scores"] is None


def test_a7_rule_scores_and_push_fold():
    sc = run_tool({}, "rule", {"name": "점수"})
    assert sc["name"] == "점수"
    for word in ("이 국을 시작할 때의 점수", "리치봉 · 결과 반영 전", "점수가 같으면 같은", "필요한 타점", "계산하지 않"):
        assert word in sc["text"], word
    for alias in ("점수 상황", "점수상황", "순위", "scores", "Score", "공탁", "점수 차"):
        assert run_tool({}, "rule", {"name": alias})["name"] == "점수", alias
    pf = run_tool({}, "rule", {"name": "밀기 접기"})["text"]
    for word in ("보통 접는 자리", "3샹텐 이상", "약 94%", "판정이 아니라 참고다", "판정하지 않는다",
                 "접는 자리가 아니라는 뜻이 아니다", "fold_spot"):
        assert word in pf, word
    assert run_tool({}, "rule", {"name": "접는 자리"})["name"] == "밀기 접기"
    # 새 글에도 작은따옴표 단계 말 · 영어 칸 이름(AI 말고) · "N점" 이 없다.
    for text in (pf, sc["text"]):
        assert ra.evidence_levels([{"tool": "rule", "args": {}, "result": {"text": text}}]) == set()
    assert not ra.rv._N_POINT.search(sc["text"])
    assert "점수" in ra.TOOLS["rule"]["desc"] and "scores" in ra.TOOLS["game_summary"]["desc"]
    assert "fold_spot" in ra.TOOLS["find_turns"]["desc"] and "보통 접는 자리" in ra.TOOLS["defense_info"]["desc"]


def test_a7_evidence_helpers(r11f, r19):
    ev = _ev(r11f, ("defense_info", {"turn": 8}))
    assert ra.evidence_has_fold_spot(ev) and ra.evidence_has_riichi(ev)
    assert not ra.evidence_has_fold_spot(_ev(r11f, ("defense_info", {"turn": 9})))
    # find_turns 의 조건 칸 {"fold_spot": true} 는 사실이 아니다(0 줄).
    ev0 = _ev(r19, ("find_turns", {"fold_spot": True}))
    assert not ra.evidence_has_fold_spot(ev0)
    assert ra.evidence_has_riichi(ev0)                       # 접는 자리 조건은 남 리치 맥락
    assert not ra.evidence_has_riichi(_ev(r19, ("find_turns", {"fold_spot": False})))
    # 판 2 의 참: 도구가 null 로 바꾸니 사실이 아니다.
    v2 = copy.deepcopy(load("r11_130921"))
    _turn(v2, 8)["defense"]["fold_spot"] = True
    assert not ra.evidence_has_fold_spot(_ev(v2, ("defense_info", {"turn": 8}), ("find_turns", {})))
    # 손으로 만든 evidence 에 참이 있어도 객체만 센다.
    assert not ra.evidence_has_fold_spot([{"tool": "x", "args": {}, "result": {"fold_spot": True}}])
    # ai_fold_pct 는 % 근거, rank 는 위 근거, score · gap 은 점 근거.
    _, pct = ra.evidence_allow(_ev(_v3(load("r11_130921"), fold={8: {"shanten": 3, "ai_fold_pct": 90}}),
                                   ("find_turns", {"fold_spot": True})))
    assert 90 in pct
    sev = _ev(_v3(r19, scores=DESIGN_SCORES), ("game_summary", {}))
    allow, _ = ra.evidence_allow(sev)
    assert {1, 2, 3} <= allow.rank
    assert ra.evidence_points(sev) == {22000, 25000, 31000, 3000, 9000, 0}


# ---- 검증: 판정 말 · 접는 자리 말 · 점수


FOLD_8_A = ("8순은 보통 접는 자리다. 내 손이 3샹텐이었고 강한 AI 는 이런 자리의 약 94% 에서 접는다. "
            "가장 안전한 [6만] 대신 '높음' 인 [9통]을 버려 밀었다.")


def test_a7_fold_spot_answer_passes(r11f):
    ev = _ev(r11f, ("defense_info", {"turn": 8}))
    assert verify_answer(FOLD_8_A, ev, "8순에 버린 9통 위험했어?") == []
    # 판정 질문: 부정 문구 한 번 + 접는 자리 사실.
    a = "이 도우미는 밀지 접을지를 판정하지 않는다. " + FOLD_8_A
    assert verify_answer(a, ev, "8순에 접었어야 했어?") == []
    # find_turns 줄로도.
    ev2 = _ev(r11f, ("find_turns", {"fold_spot": True}))
    assert verify_answer("보통 접는 자리는 8순 하나였고, 그때 [9통]을 버려 밀었다.", ev2, "접는 자리에서 민 곳 있어?") == []


@pytest.mark.parametrize("phrase", ["접었어야 했다", "밀면 안 됐다", "밀어도 됐다", "접는 게 맞았다", "밀지 말았어야 했다"])
def test_a7_verdict_words_caught_even_with_fold_spot(r11f, phrase):
    ev = _ev(r11f, ("defense_info", {"turn": 8}))
    v = verify_answer(f"{FOLD_8_A} 여기서는 {phrase}.", ev, "8순에 접었어야 했어?")
    assert len(v) == 1 and v[0].startswith("밀기 · 접기 판정 말(보통 접는 자리여도 판정은 쓰지 않음): "), v
    # 부정문 간접 의문 예외는 그대로.
    assert verify_answer(f"{FOLD_8_A} 접었어야 했는지는 판정하지 않는다.", ev, "8순에 접었어야 했어?") == []


@pytest.mark.parametrize("text,ok", [
    ("9순은 보통 접는 자리다.", False),
    ("9순은 접는 자리가 아니다.", False),
    ("9순에는 접는 자리 판정이 없다.", True),
    ("9순은 접는 자리 판정은 없다.", True),
    ("9순에 대한 접는 자리 판정 정보는 없다.", True),
    ("9순은 접는자리 판정이 없다.", True),
    ("9순에는 접는 자리 판정이 없다. 접는 자리가 아니다.", False),   # 다른 문장
    ("9순은 접는 자리인지 모른다.", False),                           # "판정 … 없" 꼴이 아님
    ("9순에는 판정이 없다. 접어도 된다.", True),                      # "접는 자리" 말이 없다(검사 10 밖)
])
def test_a7_fold_spot_words_need_fold_spot(r11f, text, ok):
    ev = _ev(r11f, ("defense_info", {"turn": 9}))        # 9번째는 fold_spot null
    v = verify_answer(text, ev, "9순은 접는 자리였어?")
    assert (v == []) == ok, v
    if not ok:
        assert v == ["접는 자리 말(fold_spot 사실 없음)"]


def test_a7_fold_spot_words_with_zero_row_find_turns(r19):
    # 접는 자리 조건으로 찾았는데 0 줄: "없다" 고 단정하면 위반, "판정이 없다" 는 통과.
    ev = _ev(r19, ("find_turns", {"fold_spot": True}))
    assert verify_answer("이 판에는 보통 접는 자리가 없다.", ev, "접는 자리 있었어?") == ["접는 자리 말(fold_spot 사실 없음)"]
    assert verify_answer("이 판에는 접는 자리 판정이 없다.", ev, "접는 자리 있었어?") == []
    # 도구 알림을 그대로 옮긴 답도 통과(알림이 검사 10 의 통과 꼴로 쓰여 있다).
    assert verify_answer(ev[0]["result"]["note"] + ".", ev, "접는 자리 있었어?") == []


def test_a7_percent_needs_fold_spot(r11f, r11):
    q = "8순에 접었어야 했어?"
    assert verify_answer("8순은 보통 접는 자리다. 강한 AI 는 이런 자리의 약 94% 에서 접는다.",
                         _ev(r11f, ("defense_info", {"turn": 8})), q) == []
    # fold_spot 이 없는 판 2 evidence: 94% · 접는 자리 · AI 모두 근거 없음.
    v = verify_answer("8순은 보통 접는 자리다. 강한 AI 는 이런 자리의 약 94% 에서 접는다.",
                      _ev(r11, ("defense_info", {"turn": 8})), q)
    assert "도구로 찾은 사실표에 없는 숫자: 94%" in v and "접는 자리 말(fold_spot 사실 없음)" in v
    assert "영어(한국어로 써야 함): AI" in v
    # 사실표 값이 90 이면 94% 는 근거가 없다.
    f = _v3(load("r11_130921"), fold={8: {"shanten": 3, "ai_fold_pct": 90}})
    ev90 = _ev(f, ("defense_info", {"turn": 8}))
    assert verify_answer("8순은 보통 접는 자리다. 이런 자리의 약 90% 에서 접는다.", ev90, q) == []
    assert "도구로 찾은 사실표에 없는 숫자: 94%" in verify_answer("8순은 보통 접는 자리다. 약 94% 에서 접는다.", ev90, q)
    # rule '밀기 접기' 를 읽었으면 94% · AI 는 근거(방총 위험 % 와 같은 규칙). 검토(11-23) 뒤로 "접는 자리" 말도 근거가 된다
    # (용어 풀이 — 전에는 이 답이 위반이었다).
    evr = _ev(r11, ("defense_info", {"turn": 8}), ("rule", {"name": "밀기 접기"}))
    assert verify_answer("강한 AI 는 이런 자리의 약 94% 에서 접었다.", evr, q) == []
    assert verify_answer("8순은 보통 접는 자리다.", evr, q) == []


# ---- 검토(11-23) 고침: 검사 10 의 rule '밀기 접기' 예외 · 평가 꼴 판정 말 · 지시문 % 줄

TERM_Q = "보통 접는 자리가 뭐야?"
# "3샹텐 이상" 은 rule 글에만 있어 숫자 근거가 아니다(샹텐 수는 도구 결과의 칸에서만) — 용어 답에는 숫자 없이.
TERM_A = ("보통 접는 자리는 리치를 받았고 내 손이 텐파이에서 멀며, 효율대로 버릴 패가 가장 안전한 패보다 뚜렷이 "
          "위험한 차례다. 강한 AI 는 이런 자리의 약 94% 에서 접었다. 판정이 아니라 참고다.")


@pytest.mark.parametrize("name", ["밀기 접기", "밀기접기", "접는 자리", "보통 접는 자리", "fold_spot", "stance"])
def test_a7_fold_spot_words_ok_after_push_fold_rule(r19, name):
    ev = _ev(r19, ("rule", {"name": name}))
    assert ev[0]["result"]["name"] == "밀기 접기" and ra.evidence_has_pushfold_rule(ev)
    assert verify_answer(TERM_A, ev, TERM_Q) == [], name


def test_a7_fold_spot_words_need_rule_or_fold_spot(r19, r11):
    # rule 을 안 불렀거나 다른 용어(방총 위험 · 점수)를 불렀으면 "접는 자리" 말은 위반(94% · AI 도 근거 없음).
    for ev in ([], _ev(r19, ("rule", {"name": "방총 위험"})), _ev(r19, ("rule", {"name": "점수"})),
               _ev(r11, ("defense_info", {"turn": 8}))):
        assert not ra.evidence_has_pushfold_rule(ev)
        assert "접는 자리 말(fold_spot 사실 없음)" in verify_answer("보통 접는 자리는 리치를 받은 차례다.", ev, TERM_Q)
    # 모르는 이름으로 부른 rule(오류 결과)도 '밀기 접기' 가 아니다.
    bad = _ev(r19, ("rule", {"name": "접기 자리"}))
    assert "error" in bad[0]["result"] and not ra.evidence_has_pushfold_rule(bad)


@pytest.mark.parametrize("phrase,word", [("잘 접었다", "잘 접었"), ("잘접었다", "잘 접었"), ("잘 밀었다", "잘 밀었"),
                                         ("잘  밀었다", "잘 밀었"), ("잘 내렸다", "잘 내렸")])
def test_a7_eval_verdict_words_caught_even_with_fold_spot(r11f, r11, phrase, word):
    q = "8순에 접었어야 했어?"
    ev = _ev(r11f, ("defense_info", {"turn": 8}))
    assert verify_answer(f"{FOLD_8_A} 여기서는 {phrase}.", ev, q) == \
        [f"밀기 · 접기 판정 말(보통 접는 자리여도 판정은 쓰지 않음): {word}"]
    assert f"밀기 · 접기 판정 말(fold_spot 사실 없음): {word}" in \
        verify_answer(f"8순 [9통]은 '높음'이었다. {phrase}.", _ev(r11, ("defense_info", {"turn": 8})), q)
    # 부정문 간접 의문 예외는 그대로.
    stem = phrase.removesuffix("다.").removesuffix("다")
    assert verify_answer(f"{FOLD_8_A} {stem}는지는 판정하지 않는다.", ev, q) == []


def test_a7_rule_push_fold_allows_3_shanten(r19):
    a = ("보통 접는 자리는 리치를 받았고 버리기 전 내 손이 3샹텐 이상이며, 효율대로 버릴 패가 가장 안전한 패보다 뚜렷이 "
         "위험한 차례다. 판정이 아니라 참고다.")
    ev = _ev(r19, ("rule", {"name": "밀기 접기"}))
    assert "3샹텐 이상" in ev[0]["result"]["text"] and 3 in ra.evidence_allow(ev)[0].shanten
    assert verify_answer(a, ev, TERM_Q) == []
    # rule 을 안 불렀거나 다른 rule 이면 3샹텐 · 접는 자리 말 모두 근거 없음. 다른 샹텐 수는 rule 을 불러도 위반.
    for other in ([], _ev(r19, ("rule", {"name": "샹텐"}))):
        v = verify_answer(a, other, TERM_Q)
        assert "도구로 찾은 사실표에 없는 숫자: 3샹텐" in v and "접는 자리 말(fold_spot 사실 없음)" in v
    assert verify_answer(a.replace("3샹텐", "4샹텐"), ev, TERM_Q) == ["도구로 찾은 사실표에 없는 숫자: 4샹텐"]


def test_a7_prompt_percent_line():
    p = ra.PROMPT_RULES
    assert ('- 위험 % 를 지어내지 않는다. 사실표의 위험에는 % 가 없다(쓸 수 있는 % 는 rule("방총 위험") 글의 관측 비율과 '
            'fold_spot 의 ai_fold_pct 뿐이다). 단계의 뜻 · 관측 비율은 rule("방총 위험")에 있다 — 그 글을 읽었을 때만 그 % 를 옮긴다.\n') in p
    assert "사실표에는 % 가 없다" not in p


SCORE_A = ("이 판 시작할 때 내 점수는 22000점으로 3위였다. 남 자리가 31,000점으로 1위였고 나와 9,000점 차였다. "
           "동 자리는 25000점으로 3000점 앞서 있었고, 공탁 리치봉은 1개였다.")


def test_a7_points_from_scores(r19s):
    ev = _ev(r19s, ("game_summary", {}))
    q = "이 판 시작할 때 점수 상황은 어땠어?"
    assert verify_answer(SCORE_A, ev, q) == []
    assert verify_answer("북 자리와는 0점 차로 같은 3위였다.", ev, q) == []
    for text in ("남 자리는 31000점이었다.", "남 자리는 31,000점이었다.", "남 자리는 31,000 점이었다."):
        assert verify_answer(text, ev, q) == [], text


@pytest.mark.parametrize("text,line", [
    ("남 자리는 30000점이었다.", "도구로 찾은 점수 기록에 없는 점수: 30000점"),             # 사실표에 없는 값
    ("남 자리와는 6,000점 차였다.", "도구로 찾은 점수 기록에 없는 점수: 6,000점"),          # 계산해 만든 차(31000 − 25000)
    ("1등이 되려면 9100점이 필요하다.", "도구로 찾은 점수 기록에 없는 점수: 9100점"),
    ("남 자리는 3,1000점이었다.", "도구로 찾은 점수 기록에 없는 점수: 3,1000점"),           # 쉼표 자리가 틀림
    ("남 자리는 3만 1천 점이었다.", "한글 수로 쓴 점수(숫자 그대로 써야 함): 3만 1천 점"),
    ("내 점수는 2만 2천점이었다.", "한글 수로 쓴 점수(숫자 그대로 써야 함): 2만 2천점"),
    ("동 자리와 3천 점 차였다.", "한글 수로 쓴 점수(숫자 그대로 써야 함): 3천 점"),
    ("남 자리는 3만점을 넘었다.", "한글 수로 쓴 점수(숫자 그대로 써야 함): 3만점"),
    ("나는 4위였다.", "도구로 찾은 사실표에 없는 숫자: 4위"),                                # rank 는 1 · 2 · 3 뿐
])
def test_a7_points_caught(r19s, text, line):
    v = verify_answer(text, _ev(r19s, ("game_summary", {})), "점수 상황은?")
    assert v == [line], v          # 한글 수사 점수는 "대괄호 없이 쓴 패(3만)" 로 따로 걸리지 않는다


def test_a7_points_without_scores(r19):
    # 점수 기록이 없는 판(판 2)과 점수 도구를 안 부른 답: "N점" 은 근거가 없다.
    ev = _ev(r19, ("game_summary", {}))
    assert ev[0]["result"]["scores"] is None and "note" not in ev[0]["result"]          # a8: 알림 없음
    assert verify_answer("이 판에는 점수 기록이 없다.", ev, "내가 몇 등이었어?") == []
    assert verify_answer("내 점수는 25000점이었다.", ev, "점수?") == ["도구로 찾은 점수 기록에 없는 점수: 25000점"]
    assert verify_answer("나는 1위였다.", ev, "내가 몇 등이었어?") == ["도구로 찾은 사실표에 없는 숫자: 1위"]
    # 차례 도구의 숫자(장 · 순)는 점이 아니다.
    ev6 = _ev(r19, ("turn_info", {"turn": 6}))
    assert verify_answer("6순에 [4만]을 버려 10점을 잃었다.", ev6, "6순?") == ["도구로 찾은 점수 기록에 없는 점수: 10점"]


def test_a7_negative_score_and_ai_word(r19):
    f = _v3(r19, scores=_scores_of("서", {"동": 41200, "남": 30000, "서": -1200, "북": 30000}, None))
    ev = _ev(f, ("game_summary", {}))
    assert verify_answer("내 점수는 -1,200점으로 4위였다. 동 자리와는 42,400점 차였다.", ev, "점수?") == []
    # "AI" 는 evidence 글에 있을 때만 영어 예외.
    assert verify_answer("강한 AI 는 모른다.", ev, "점수?") == ["영어(한국어로 써야 함): AI"]
    assert verify_answer("강한 AI 가 접었다.", _ev(f, ("rule", {"name": "밀기 접기"})), "?") == []
    assert verify_answer("강한 AIX 가 접었다.", _ev(f, ("rule", {"name": "밀기 접기"})), "?") == ["영어(한국어로 써야 함): AIX"]


# ---- 그래프 · 지시문


def test_a7_graph_fold_spot_verdict_retry_then_fact(r11f):
    q = "8순에 접었어야 했어?"
    first = FOLD_8_A + " 여기서는 접었어야 했다."
    good = "이 도우미는 밀지 접을지를 판정하지 않는다. " + FOLD_8_A
    gen = Script(tool("defense_info", turn=8), answer(first), answer(good))
    out = ask(_req(r11f, q), gen)
    assert out["outcome"] == "verified" and out["answer"] == good and out["prompt_version"] == "a8"
    assert out["retry_violations"] == [["밀기 · 접기 판정 말(보통 접는 자리여도 판정은 쓰지 않음): 접었어야"]]
    p2 = gen.prompts[1]
    assert '"fold_spot":{"shanten":3,"ai_fold_pct":94}' in p2 and "보통 접는 자리다(리치를 받았고 내 손이 3샹텐" in p2


def test_a7_graph_scores_question(r19s, r19):
    gen = Script(tool("game_summary"), answer(SCORE_A))
    out = ask(_req(r19s, "이 판 시작할 때 점수 상황은 어땠어?"), gen)
    assert out["outcome"] == "verified" and out["tool_calls"] == [{"tool": "game_summary", "args": {}}]
    assert '"basis":"국 시작 점수(이 국의 리치봉 · 결과 반영 전)"' in gen.prompts[1]
    # 점수 없는 판에 점수를 지어내면 재시도 → 그래도 지어내면 대체(판 요약 — 점수 말 없음).
    gen = Script(tool("game_summary"), answer("내 점수는 25000점이었다."), answer("내 점수는 25000점이었다."))
    out = ask(_req(r19, "내 점수는 몇 점이었어?"), gen)
    assert out["outcome"] == "fallback" and out["retry_violations"] == [["도구로 찾은 점수 기록에 없는 점수: 25000점"]]
    assert "점" not in out["answer"].replace("확인할 수 있는", "")


def test_a7_prompt_rules():
    p = ra.PROMPT_RULES
    assert ra.PROMPT_VERSION == "a8"
    # 없어진 줄.
    assert "fold_spot 이 null 이 아닌 차례에서만" not in p and '점수("N점")를 쓰지 않는다' not in p
    # a7 네 항목.
    for word in ('"이 차례는 보통 접는 자리다"', "강한 AI 는 이런 자리의", "약 94% 에서 접는다", "stance 사실문으로",
                 '"접었어야 했다 · 밀면 안 됐다 · 밀어도 됐다 · 잘 접었다"', "fold_spot 이 있어도 쓰지 않는다",
                 "fold_spot 이 있으면 보통 접는", '"접는 자리가 아니다" 라고 하지 않는다', '"이 차례에는 접는 자리 판정이 없다"',
                 "game_summary 의 scores 값만", "쉼표는 써도 됨", "점수 차는 gap 값", '"2만 5천 점" 처럼 한글 수로 쓰지 않는다',
                 '"이 판에는 점수 기록이 없다"', "필요한 타점 · 순위 전망을 판단하지 않는다"):
        assert word in p, word
    # 지키는 줄(a6 말투 측정 결과): 바뀌지 않음.
    assert '- 해라체("…다")로 쓴다. 존댓말("…습니다" · "…해요")을 쓰지 않는다. "반드시" · "무조건" · "절대" 를 쓰지 않는다.\n' in p
    assert "- find_turns(grade?, status?, min_ukeire_loss?, tile?, after_riichi?, stance?, min_level?, fold_spot?)" in p
    # 지시문이 권하는 말 자체는 검사에 걸리지 않는다.
    for s in ("이 차례는 보통 접는 자리다.", "이 차례에는 접는 자리 판정이 없다.", "이 판에는 점수 기록이 없다."):
        assert ra.eval_words(s) == [] and ra.verdict_words(s) == [], s
    assert ra.check_fold_spot_words("이 차례에는 접는 자리 판정이 없다.", []) == []
    assert ra.check_points("25000점 · 25,000점", [{"tool": "game_summary", "args": {},
                                                   "result": {"scores": {"hero": {"score": 25000}}}}]) == []
    # find_turns 스키마: fold_spot 은 boolean.
    assert _variant("find_turns")["properties"]["args"]["properties"]["fold_spot"]["type"] == "boolean"


# ---------------------------------------------------------------- a8(설계서 1123 8-1절 — a7 측정 표 15-12 의 흠 넷)

UNASKED_FOLD = "묻지 않은 접는 자리 부정 문구"
UNASKED_SCORE = "묻지 않은 점수 기록 부정 문구"
RISK_RANK = "안전한 순 자리를 위험 순위처럼 씀"
# a7 실제 답 꼴(ax1a · ax2c · ax4a): 사실 뒤에 묻지 않은 "접는 자리 판정이 없다".
A12 = "8순에 가장 안전한 [6만] 대신 '높음' 인 [9통]을 버려 밀었다. 이 차례에는 접는 자리 판정이 없다."


def test_a8_check12_unasked_fold_spot_disclaimer(r11):
    ev = _ev(r11, ("defense_info", {"turn": 8}))
    assert verify_answer(A12, ev, "8순에 버린 9통 위험했어?") == [UNASKED_FOLD]
    # 접 · 밀 · 내리 · 판정 을 물었으면 통과.
    for q in ("8순에 대해: 여기서 접었어야 했어?", "8순은 접는 자리였어?", "8순에 밀어도 됐어?", "8순에 내렸어야 했어?",
              "8순 판정은?"):
        assert verify_answer(A12, ev, q) == [], q
    # 이어 묻기: 앞 질문에 "접" 이 있으면 통과(답 글은 보지 않는다).
    hist = [{"q": "8순에 접었어야 했어?", "a": "이 도우미는 밀지 접을지를 판정하지 않는다."}]
    assert verify_answer(A12, ev, "8순에 버린 9통 위험했어?", hist) == []
    assert verify_answer(A12, ev, "8순에 버린 9통 위험했어?", [{"q": "8순 어땠어?", "a": "접었다."}]) == [UNASKED_FOLD]
    # 실제 질문 꼴 그대로(검사 함수만).
    assert ra.check_unasked_disclaimers("이 차례에는 접는 자리 판정이 없다.", "11순에 버린 6삭 위험했어?") == [UNASKED_FOLD]
    assert ra.check_unasked_disclaimers("이 차례에는 접는 자리 판정이 없다.", "9순에 대해: 여기서 접었어야 했어?") == []
    # "접는 자리" 와 판정 없음이 다른 문장이거나, 판정 없음만 있으면 검사 12 밖.
    assert ra.check_unasked_disclaimers("접는 자리는 아래다. 판정이 없다.", "위험했어?") == []
    assert ra.check_unasked_disclaimers("이 차례의 판정은 없다.", "위험했어?") == []


def test_a8_check12_graph_passes_history(r11):
    hist = [{"q": "8순에 접었어야 했어?", "a": "이 도우미는 밀지 접을지를 판정하지 않는다."}]
    out = ask(_req(r11, "8순에 버린 9통 위험했어?", hist), Script(tool("defense_info", turn=8), answer(A12)))
    assert out["outcome"] == "verified" and out["retries"] == 0 and out["prompt_version"] == "a8"
    clean = A12.split(" 이 차례에는")[0]
    out = ask(_req(r11, "8순에 버린 9통 위험했어?"), Script(tool("defense_info", turn=8), answer(A12), answer(clean)))
    assert out["outcome"] == "verified" and out["retry_violations"] == [[UNASKED_FOLD]] and out["answer"] == clean


def test_a8_check13_unasked_score_disclaimer(r19):
    ev = _ev(r19, ("game_summary", {}))
    text = "이 판은 남4국이고 패효율은 93%다. 이 판에는 점수 기록이 없다."
    assert verify_answer(text, ev, "이 판 총평 해 줘") == [UNASKED_SCORE]
    assert verify_answer(text, ev, "등급은 어땠어?") == [UNASKED_SCORE]          # "등급" 의 "등" 은 점수 질문이 아니다
    for q in ("이 판 시작할 때 점수 상황은 어땠어?", "내가 몇 등이었어?", "몇 점 차였어?", "1등이랑 점 차는?", "등수는?",
              "순위는?", "몇 위였어?", "공탁은?", "리치봉 몇 개였어?"):
        assert verify_answer(text, ev, q) == [], q
    assert verify_answer(text, ev, "총평 해 줘", [{"q": "내 점수는?", "a": "이 판에는 점수 기록이 없다."}]) == []
    for t in ("점수 기록은 없다.", "점수 기록도 없다.", "점수기록이 없다."):
        assert ra.check_unasked_disclaimers(t, "총평 해 줘") == [UNASKED_SCORE], t


@pytest.mark.parametrize("text,bad", [
    ("8순에 버린 [9통]은 위험 순위 6위였다.", True),                    # ax4b 꼴
    ("8순에 버린 [9통]은 6위의 위험도였다.", True),                     # fa2c 꼴
    ("8순에 버린 [9통]은 6위로 위험했다.", True),
    ("8순에 버린 [9통]은 안전한 순으로 13종 중 6위였다.", False),
    ("8순에 버린 [9통]은 위험 단계 '높음'이었다.", False),
])
def test_a8_check14_risk_rank(r11, text, bad):
    v = verify_answer(text, _ev(r11, ("defense_info", {"turn": 8})), "8순에 버린 9통 위험했어?")
    assert v == ([RISK_RANK] if bad else []), v
    assert ra.check_unasked_disclaimers("안전한 순으로 9종 중 3위였다.", "?") == []


def test_a8_ai_word_with_fold_spot_rows_only(r11f, r19):
    ev = _ev(r11f, ("find_turns", {"fold_spot": True}))
    assert "AI" not in ra._evidence_texts(ev) and ra.evidence_has_fold_spot(ev)
    text = "보통 접는 자리는 8순이고, 강한 AI 는 이런 자리의 약 94% 에서 접는다."
    assert verify_answer(text, ev, "접는 자리 있었어?") == []
    # fold_spot 객체도 "AI" 글도 없으면 그대로 위반.
    assert "영어(한국어로 써야 함): AI" in verify_answer("강한 AI 는 모른다.", _ev(r19, ("find_turns", {})), "?")


def test_a8_game_summary_no_score_note(r19, r19s):
    for f in (r19, _v3(r19), _v1(r19)):
        s = run_tool(f, "game_summary", {})
        assert s["scores"] is None and "note" not in s
        assert "점수 기록" not in json.dumps(s, ensure_ascii=False)
    assert run_tool(r19s, "game_summary", {})["scores"]["basis"] == ra.SCORES_BASIS


@pytest.mark.parametrize("name", GAMES)
def test_a8_fallback_never_trips_new_checks(name):
    # 대체 답(코드 글)은 검사 12 · 13 · 14 에 걸리는 말을 내지 않는다 — 판 2 견본과 판 3(점수 없음)으로.
    for facts in (load(name), _v3(load(name))):
        qs = [f"{t['turn']}순은 어땠어?" for t in facts["turns"]] + ["이 판 총평 해 줘"]
        for q in qs:
            text, ev = ra.fallback_answer(facts, q)
            assert ra.check_unasked_disclaimers(text, q) == [], (name, q, text)
            assert verify_answer(text, ev, q) == [], (name, q, text)


def test_a8_prompt_rules():
    p = ra.PROMPT_RULES
    assert ra.PROMPT_VERSION == "a8"
    assert ('접는 자리 · 밀기 · 접기를 물었을 때만 "이 차례에는 접는 자리 판정이 없다" 고 답한다. 묻지 않았으면 그 말을 붙이지 '
            '않는다.') in p
    assert '점수 · 순위를 물었을 때만 "이 판에는 점수 기록이 없다" 고 답한다. 묻지 않았으면 그 말을 붙이지 않는다.' in p
    assert ('"안전한 순으로 손의 N종 중 R위" 처럼 그대로 쓴다. "위험 순위 R위 · R위의 위험도" 처럼\n  위험한 순으로 읽히게 쓰지 '
            '않는다.') in p
    assert "접는 자리였냐고 물으면" not in p and '"손의 N종 중 R위" 처럼 그대로 쓴다' not in p
    # 지시문이 권하는 꼴("안전한 순으로 … R위")은 검사 14 에 안 걸린다.
    assert ra.check_unasked_disclaimers("안전한 순으로 손의 9종 중 3위였다.", "?") == []
