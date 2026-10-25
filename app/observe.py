"""관찰 전용 판독(/v2/*). 모델은 보이는 패만 고정된 JSON 으로 적고, 판단은 앱 코드가 한다.

- 손패 크롭 → 깐 손패, 떨어뜨려 놓은 한 장, 울기 묶음(패·옆으로 눕힌 위치·뒤집힌 위치)
- 울기 크롭(locate 의 melds 상자, 그 자리 방향으로 세운 것) → 묶음 하나
- 강 크롭(그 자리 사람이 보는 방향으로 세운 것) → 버린 순서대로 패와 가로 여부, 리치봉
- 가운데·왕패 크롭 → 앞면 표시패 전부, 자리별 리치봉
- 모서리 크롭(미리 그은 격자의 한 칸, 그 자리 방향으로 세운 것) → 울기 묶음 전부와 묶음 아닌 앞면 패
- 강 끝 크롭(가운데 칸에서 한 자리 쪽, 그 자리 방향으로 세운 것) → 그 자리 강의 마지막 패 한 장
- 리치봉 크롭(자동탁 가운데 기기) → 자리별 리치봉
- 도라 크롭(격자 가운데 칸) → 강 밖 펠트 위에 따로 놓인 앞면 패(도라 표시패)

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

MELD_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "tiles": {
            "type": "array",
            "description": "울기 묶음의 패를 왼쪽부터(3~4장). 뒤집혀 뒷면만 보이는 패는 같은 묶음의 앞면 패와 같은 것으로 적음",
            "items": {"type": "string", "enum": TILE_CODES},
        },
        "sideways": {**_INDEXES, "description": "옆으로(90도) 눕혀 놓은 패의 위치(0부터)"},
        "face_down": {**_INDEXES, "description": "뒤집혀 뒷면만 보이는 패의 위치(0부터)"},
        "confidence": {"type": "number"},
        "notes": _NOTES,
    },
    "required": ["tiles", "sideways", "face_down", "confidence", "notes"],
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

CORNER_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "melds": {
            "type": "array",
            "description": "앞면 3~4장 울기 묶음 전부. 없으면 빈 배열",
            "items": {
                "type": "object",
                "properties": {k: MELD_SCHEMA_V2["properties"][k] for k in ("tiles", "sideways", "face_down", "confidence")},
                "required": ["tiles", "sideways", "face_down", "confidence"],
            },
        },
        "loose": {
            "type": "array",
            "description": "묶음에 속하지 않고 1~2장만 따로 놓인 앞면 패(도라 표시패 등). 없으면 빈 배열",
            "items": _tile(),
        },
        "notes": _NOTES,
    },
    "required": ["melds", "loose", "notes"],
}

LAST_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "last_tile": {**_tile("아래쪽 플레이어 강의 마지막(가장 최근에 버린) 패. 강이 없으면 null"), "nullable": True},
        "tile_count": {"type": "integer", "description": "아래쪽 플레이어 강의 패 수(대략)"},
        "notes": _NOTES,
    },
    "required": ["last_tile", "tile_count", "notes"],
}

DORA_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "indicators": {
            "type": "array",
            "description": "강(버림패 줄) 밖, 펠트 위에 따로 떨어져 놓인 앞면 패 전부. 없으면 빈 배열",
            "items": _tile(),
        },
        "notes": _NOTES,
    },
    "required": ["indicators", "notes"],
}

STICKS_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "riichi_sticks": CENTER_SCHEMA_V2["properties"]["riichi_sticks"],
        "notes": _NOTES,
    },
    "required": ["riichi_sticks", "notes"],
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

MELD_PROMPT_V2 = f"""이 사진은 리치마작 한 플레이어의 울기 묶음(퐁·치·깡) 하나를 그 사람 쪽에서 본 방향으로 세운 것입니다. {_ONLY_SEE}

- tiles: 묶음의 패를 왼쪽부터 3~4장. 뒤집혀 뒷면만 보이는 패는 같은 묶음의 앞면 패와 같은 것으로 적습니다.
- sideways: 옆으로(90도) 눕혀 놓은 패의 위치(0부터). face_down: 뒤집힌 패의 위치(0부터).
- 사진 가장자리에 걸친 다른 패(손패, 도라 표시패, 버림패)는 적지 않습니다.

{NOTATION}

확신도(0~1)를 적어 주세요."""

CENTER_PROMPT_V2 = f"""이 사진은 리치마작 자동탁의 가운데(또는 패산 끝의 왕패 부근)를 위에서 찍은 것입니다. {_ONLY_SEE}

{SIDE_NOTE}

- indicators: 패산(뒷면이 보이는 벽패 줄)·손패·울기 묶음·버림패 줄 어디에도 속하지 않고 따로 놓인 앞면 패 전부.
  보통 패산 끝에 1~2장 있습니다(도라 표시패, 뒤집은 뒷도라 표시패). 어느 것이 도라인지는 가리지 않습니다.
- riichi_sticks: 자리별로 1000점 리치봉(가늘고 긴 막대. 흰 막대에 빨간 점이거나 파란색 등 색깔 막대)이
  그 자리 강 앞에 놓여 있거나, 자동탁 가운데 기기의 그 자리 쪽 홈(슬롯)에 꽂혀 있으면 present=true.
  자동탁 가운데 기기에는 네 변마다 길쭉한 홈이 하나씩 있고, 각 홈은 가장 가까운 변의 자리 것입니다.
  홈에 막대(예: 가운데 점과 양끝 화살표 무늬가 있는 파란 막대)가 들어 있으면 그 자리는 true, 빈 홈은 검은색이라 false 입니다.
  기기의 일부처럼 보여도 홈 안의 색깔 막대는 리치봉입니다. 여러 자리가 동시에 리치일 수 있으니 네 홈을 하나씩 따로 봅니다.

{NOTATION}

각 항목에 확신도(0~1)를 적어 주세요."""

CORNER_PROMPT_V2 = f"""이 사진은 리치마작 테이블의 한 모서리를, 그 모서리에 울기를 놓는 플레이어 쪽에서 본 방향으로 세운 것입니다. {_ONLY_SEE}

- melds: 앞면 3~4장 울기 묶음(퐁·치·깡) 전부. 묶음마다 패를 왼쪽부터 적고, 옆으로(90도) 눕힌 패의 위치(sideways)와
  뒤집혀 뒷면만 보이는 패의 위치(face_down)를 0부터 적습니다. 뒤집힌 패는 같은 묶음의 앞면 패와 같은 것으로 적습니다.
- loose: 묶음에 속하지 않고 1~2장만 따로 놓인 앞면 패(도라 표시패 등).
- 세워진 손패, 뒷면이 보이는 패산, 사진 가장자리에 걸려 잘린 패는 적지 않습니다. 아무것도 없으면 빈 배열입니다.

{NOTATION}

확신도(0~1)를 적어 주세요."""

LAST_PROMPT_V2 = f"""이 사진은 리치마작 테이블 가운데의 버림패(강)를 아래쪽 플레이어 쪽에서 본 방향으로 세운 것입니다. {_ONLY_SEE}

아래쪽 플레이어의 강만 봅니다. 글자가 바로 서 보이는(위아래가 맞는) 패들이 그 사람의 강입니다.
강은 6장씩 줄을 이루고, 첫 줄은 테이블 가운데 쪽(사진 위쪽)이며 다음 줄은 아래쪽 플레이어 쪽(사진 아래쪽)으로 쌓입니다.
각 줄은 왼쪽에서 오른쪽으로 버립니다. 따라서 마지막 패는 **가장 아래 줄의 가장 오른쪽 패**입니다.
그 패가 옆으로 눕혀져 있어도 그대로 마지막 패입니다.

- last_tile: 그 마지막 패 한 장. 아래쪽 플레이어의 강이 보이지 않으면 null.
- tile_count: 아래쪽 플레이어 강의 패 수.
- 옆으로(90도) 또는 거꾸로(180도) 돌아가 보이는 패는 다른 자리의 강이니 보지 않습니다. 가운데 자동탁 기기·패산도 보지 않습니다.

{NOTATION}

확신도(0~1)를 적어 주세요."""

DORA_PROMPT_V2 = f"""이 사진은 리치마작 자동탁 가운데를 위에서 찍은 것입니다. 가운데 기기 둘레에 네 자리의 버림패(강)가 6장씩 줄지어 있습니다. {_ONLY_SEE}

- indicators: 강 줄에 붙어 있지 않고 펠트 위에 따로 떨어져 놓인 앞면 패 전부(도라·뒷도라 표시패). 보통 1~4장이 한 줄로,
  강에서 한 장 이상 떨어진 곳(사진 가장자리 쪽일 수 있음)에 놓입니다. 패의 방향은 상관없습니다.
- 강 줄에 붙은 패, 뒷면(연보라)이 보이는 패, 사진 가장자리에 걸려 반 이상 잘린 패는 적지 않습니다. 없으면 빈 배열입니다.

{NOTATION}

각 패의 확신도(0~1)를 적어 주세요."""

STICKS_PROMPT_V2 = f"""이 사진은 리치마작 자동탁 가운데 기기(주사위 창이 있는 판)를 위에서 찍은 것입니다. 리치봉만 봅니다. {_ONLY_SEE}

{SIDE_NOTE}

- riichi_sticks: 자리별로 1000점 리치봉(가늘고 긴 막대. 흰 막대에 빨간 점이거나 파란색 등 색깔 막대)이 있으면 present=true.
  가운데 기기에는 네 변마다 길쭉한 홈이 하나씩 있고, 각 홈은 가장 가까운 변의 자리 것입니다.
  홈에 막대(예: 가운데 점과 양끝 화살표 무늬가 있는 파란 막대)가 들어 있으면 그 자리는 true, 빈 홈은 검은색이라 false 입니다.
  기기 밖, 그 자리 쪽 가장자리에 막대가 놓여 있어도 true 입니다.
  기기의 일부처럼 보여도 홈 안의 색깔 막대는 리치봉입니다. 여러 자리가 동시에 리치일 수 있으니 네 홈을 하나씩 따로 봅니다.
- 패(마작패)는 보지 않습니다.

각 자리에 확신도(0~1)를 적어 주세요."""

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


def normalize_meld_v2(data: dict) -> dict:
    """→ {tiles:[code], sideways:[int], face_down:[int], conf, notes}. 3~4장이 아니면 tiles 를 비운다."""
    tiles = [t for t in data.get("tiles") or [] if t in TILE_CODES]
    if len(tiles) not in (3, 4):
        tiles = []
    return {
        "tiles": tiles,
        "sideways": _indexes(data.get("sideways"), len(tiles)),
        "face_down": _indexes(data.get("face_down"), len(tiles)),
        "conf": _conf(data.get("confidence", 0)),
        "notes": str(data.get("notes") or ""),
    }


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


def normalize_corner_v2(data: dict) -> dict:
    """→ {melds:[{tiles,sideways,face_down,conf}], loose:[{tile,conf}], notes}. 3~4장이 아닌 묶음은 버린다."""
    melds = [m for m in (normalize_meld_v2(x) for x in data.get("melds") or [] if isinstance(x, dict)) if m["tiles"]]
    for m in melds:
        m.pop("notes", None)
    return {
        "melds": melds,
        "loose": [t for t in (_code(x) for x in data.get("loose") or []) if t],
        "notes": str(data.get("notes") or ""),
    }


def normalize_last_v2(data: dict) -> dict:
    """→ {last_tile:{tile,conf}|None, tile_count, notes}"""
    try:
        count = max(0, int(data.get("tile_count") or 0))
    except (TypeError, ValueError):
        count = 0
    return {"last_tile": _code(data.get("last_tile")), "tile_count": count, "notes": str(data.get("notes") or "")}


def normalize_sticks_v2(data: dict) -> dict:
    """→ {riichi_sticks:{side:bool}, riichi_sticks_conf:{side:float}, notes}"""
    c = normalize_center_v2({"riichi_sticks": data.get("riichi_sticks"), "notes": data.get("notes")})
    c.pop("indicators")
    return c


def normalize_dora_v2(data: dict) -> dict:
    """→ {indicators:[{tile,conf}], notes}"""
    return {
        "indicators": [t for t in (_code(x) for x in data.get("indicators") or []) if t],
        "notes": str(data.get("notes") or ""),
    }
