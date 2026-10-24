"""관찰 전용 판독(/v2/*). 모델은 보이는 패만 고정된 JSON 으로 적고, 판단은 앱 코드가 한다.

- 손패 크롭 → 깐 손패, 떨어뜨려 놓은 한 장, 울기 묶음(패·옆으로 눕힌 위치·뒤집힌 위치)
- 강 크롭(그 자리 사람이 보는 방향으로 세운 것) → 버린 순서대로 패와 가로 여부, 리치봉
- 가운데·왕패 크롭 → 앞면 표시패 전부, 자리별 리치봉

울기 종류(치·퐁·깡), 리치, 강 끝 패, 도라·뒷도라 구분, 론/쯔모는 여기서 정하지 않는다.
"""

from __future__ import annotations

from .prompts import NOTATION, SIDE_NOTE
from .schemas import SIDES, TILE_CODES, _NOTES, _tile

# ---------------------------------------------------------------- 스키마

_INDEXES = {"type": "array", "items": {"type": "integer"}}

HAND_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "tiles": {
            "type": "array",
            "description": "앞면으로 깐 손패 줄의 패를 왼쪽부터. 따로 떨어진 한 장과 울기 묶음은 제외",
            "items": _tile(),
        },
        "separate_tile": {
            **_tile("손패 줄 끝에서 조금 떨어뜨려 놓은 한 장. 없으면 null"),
            "nullable": True,
        },
        "melds": {
            "type": "array",
            "description": "손패와 따로 놓인 울기 묶음(앞면 3~4장). 테이블 모서리까지 떨어져 있을 수 있음. 없으면 빈 배열",
            "items": {
                "type": "object",
                "properties": {
                    "tiles": {
                        "type": "array",
                        "description": "묶음의 패를 왼쪽부터. 뒤집혀 뒷면만 보이는 패는 같은 묶음의 앞면 패와 같은 것으로 적음",
                        "items": {"type": "string", "enum": TILE_CODES},
                    },
                    "sideways": {**_INDEXES, "description": "옆으로(90도) 눕혀 놓은 패의 위치(0부터)"},
                    "face_down": {**_INDEXES, "description": "뒤집혀 뒷면만 보이는 패의 위치(0부터)"},
                    "confidence": {"type": "number"},
                },
                "required": ["tiles", "sideways", "face_down", "confidence"],
            },
        },
        "notes": _NOTES,
    },
    "required": ["tiles", "separate_tile", "melds", "notes"],
}

RIVER_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "tiles": {
            "type": "array",
            "description": "버림패를 버린 순서대로: 윗줄부터, 각 줄은 왼쪽에서 오른쪽",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "enum": TILE_CODES},
                    "sideways": {"type": "boolean", "description": "같은 줄의 다른 패와 방향이 90도 다르게 놓였으면 true"},
                    "confidence": {"type": "number"},
                },
                "required": ["code", "sideways", "confidence"],
            },
        },
        "riichi_stick": {"type": "boolean", "description": "버림패 줄 앞(사진 아래쪽)에 리치봉이 놓여 있으면 true"},
        "notes": _NOTES,
    },
    "required": ["tiles", "riichi_stick", "notes"],
}

CENTER_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "indicators": {
            "type": "array",
            "description": "패산·손패·울기·강 어디에도 속하지 않고 따로 놓인 앞면 패(도라·뒷도라 표시패) 전부. 왼쪽부터",
            "items": _tile(),
        },
        "riichi_sticks": {
            "type": "array",
            "description": "자리별 리치봉 여부. 네 자리 모두",
            "items": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "enum": SIDES},
                    "present": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": ["side", "present", "confidence"],
            },
        },
        "notes": _NOTES,
    },
    "required": ["indicators", "riichi_sticks", "notes"],
}

# ---------------------------------------------------------------- 프롬프트

_ONLY_SEE = "보이는 것만 적습니다. 리치·론·도라 여부 같은 판단은 하지 않습니다."

HAND_PROMPT_V2 = f"""이 사진은 리치마작 한 플레이어 앞의 패를 그 사람 쪽에서 본 방향으로 세운 것입니다. {_ONLY_SEE}

- tiles: 앞면으로 깐 손패 줄. 왼쪽부터.
- separate_tile: 손패 줄 끝에서 조금 떨어뜨려 놓은 한 장(없으면 null).
- melds: 손패 줄과 따로 놓인 앞면 3~4장 묶음. 오른쪽 끝, 테이블 모서리까지 멀리 떨어져 있을 수 있습니다.
  묶음마다 패를 왼쪽부터 적고, 옆으로 눕힌 패의 위치(sideways)와 뒤집힌 패의 위치(face_down)를 0부터 적습니다.
- 앞면 1~2장만 따로 놓인 패(도라 표시패)와 버림패, 패산(뒷면 벽패)은 적지 않습니다.

{NOTATION}

보이는 패 수를 먼저 세고 그 수만큼 정확히 나열하세요. 각 패의 확신도(0~1)를 적어 주세요."""

RIVER_PROMPT_V2 = f"""이 사진은 리치마작 한 플레이어의 버림패 줄(강)을 그 사람 쪽에서 본 방향으로 세운 것입니다. {_ONLY_SEE}

- tiles: 버린 순서대로 적습니다. 윗줄(가운데 쪽)부터, 각 줄은 왼쪽에서 오른쪽입니다. 보통 6장마다 다음 줄입니다.
  같은 줄의 다른 패와 방향이 90도 다르게(가로로) 놓인 패는 sideways 를 true 로 합니다.
- riichi_stick: 버림패 줄 앞에 1000점 리치봉(가늘고 긴 막대)이 놓여 있으면 true.
- 사진 가장자리에 걸친 다른 자리의 패, 손패, 패산(뒷면 벽패)은 적지 않습니다.

{NOTATION}

보이는 버림패 수를 먼저 세고 그 수만큼 정확히 나열하세요. 각 패의 확신도(0~1)를 적어 주세요."""

CENTER_PROMPT_V2 = f"""이 사진은 리치마작 자동탁의 가운데(또는 패산 끝의 왕패 부근)를 위에서 찍은 것입니다. {_ONLY_SEE}

{SIDE_NOTE}

- indicators: 패산(뒷면이 보이는 벽패 줄)·손패·울기 묶음·버림패 줄 어디에도 속하지 않고 따로 놓인 앞면 패 전부.
  보통 패산 끝에 1~2장 있습니다(도라 표시패, 뒤집은 뒷도라 표시패). 어느 것이 도라인지는 가리지 않습니다.
- riichi_sticks: 자리별로 1000점 리치봉(가늘고 긴 막대. 흰 막대에 빨간 점이거나 파란색 등 색깔 막대)이
  그 자리 강 앞에 놓여 있거나, 자동탁 가운데 기기의 그 자리 쪽 홈(슬롯)에 꽂혀 있으면 present=true. 빈 홈은 검은색입니다.

{NOTATION}

각 항목에 확신도(0~1)를 적어 주세요."""

# ---------------------------------------------------------------- 정리(순수 함수)


def _conf(v) -> float:
    try:
        return round(min(1.0, max(0.0, float(v))), 3)
    except (TypeError, ValueError):
        return 0.0


def _code(t) -> dict | None:
    if not isinstance(t, dict) or t.get("code") not in TILE_CODES:
        return None
    return {"tile": t["code"], "conf": _conf(t.get("confidence", 0))}


def _indexes(v, n: int) -> list[int]:
    return sorted({i for i in v or [] if isinstance(i, int) and 0 <= i < n})


def normalize_hand_v2(data: dict) -> dict:
    """→ {tiles:[{tile,conf}], separate_tile:{tile,conf}|None, melds:[{tiles,sideways,face_down,conf}], notes}"""
    melds = []
    for m in data.get("melds") or []:
        if not isinstance(m, dict):
            continue
        tiles = [t for t in m.get("tiles") or [] if t in TILE_CODES]
        if len(tiles) not in (3, 4):
            continue
        melds.append({
            "tiles": tiles,
            "sideways": _indexes(m.get("sideways"), len(tiles)),
            "face_down": _indexes(m.get("face_down"), len(tiles)),
            "conf": _conf(m.get("confidence", 0)),
        })
    return {
        "tiles": [t for t in (_code(x) for x in data.get("tiles") or []) if t],
        "separate_tile": _code(data.get("separate_tile")),
        "melds": melds,
        "notes": str(data.get("notes") or ""),
    }


def normalize_river_v2(data: dict) -> dict:
    """→ {tiles:[{tile,sideways,conf}], riichi_stick, notes}"""
    tiles = []
    for t in data.get("tiles") or []:
        c = _code(t)
        if c:
            tiles.append({**c, "sideways": bool(t.get("sideways", False))})
    return {"tiles": tiles, "riichi_stick": bool(data.get("riichi_stick", False)), "notes": str(data.get("notes") or "")}


def normalize_center_v2(data: dict) -> dict:
    """→ {indicators:[{tile,conf}], riichi_sticks:{side:bool}, riichi_sticks_conf:{side:float}, notes}"""
    sticks = {s: False for s in SIDES}
    conf = {s: 0.0 for s in SIDES}
    for it in data.get("riichi_sticks") or []:
        if isinstance(it, dict) and it.get("side") in SIDES:
            sticks[it["side"]] = bool(it.get("present", False))
            conf[it["side"]] = _conf(it.get("confidence", 0))
    return {
        "indicators": [t for t in (_code(x) for x in data.get("indicators") or []) if t],
        "riichi_sticks": sticks,
        "riichi_sticks_conf": conf,
        "notes": str(data.get("notes") or ""),
    }
