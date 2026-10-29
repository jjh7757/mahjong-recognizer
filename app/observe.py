"""관찰 전용 판독(/v2/*). 모델은 보이는 패만 고정된 JSON 으로 적고, 판단은 앱 코드가 한다.

- 손패 크롭 → 깐 손패, 떨어뜨려 놓은 한 장, 울기 묶음(패·옆으로 눕힌 위치·뒤집힌 위치)
- 울기 크롭(locate 의 melds 상자, 그 자리 방향으로 세운 것) → 묶음 하나
- 강 크롭(그 자리 사람이 보는 방향으로 세운 것) → 버린 순서대로 패와 가로 여부, 리치봉
- 가운데·왕패 크롭 → 앞면 표시패 전부, 자리별 리치봉
- 모서리 크롭(미리 그은 격자의 한 칸, 그 자리 방향으로 세운 것) → 울기 묶음 전부와 묶음 아닌 앞면 패
- 강 끝 크롭(가운데 칸에서 한 자리 쪽, 그 자리 방향으로 세운 것) → 그 자리 강의 마지막 패 한 장
- 리치봉 크롭(자동탁 가운데 기기) → 자리별 리치봉
- 도라 크롭(격자 가운데 칸) → 강 밖 펠트 위에 따로 놓인 앞면 패(도라 표시패)
- 자리 크롭(격자 5구역의 한 자리 = 그 사람 앞 손패 + 오른쪽 모서리, 그 사람 방향으로 세운 것)
  → 손패가 앞면인지, 깐 손패·떨어진 한 장·울기 묶음 전부·따로 놓인 앞면 패

울기 종류(치·퐁·깡), 리치, 강 끝 패, 도라·뒷도라 구분, 론/쯔모는 여기서 정하지 않는다.
"""

from __future__ import annotations

from collections import Counter

from .prompts import NOTATION, SIDE_NOTE
from .normalize import normalize_box
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

LAST_MULTI_MAX = 4


def last_multi_schema(n: int) -> dict:
    """사진 n 장의 강 끝을 한 번에: photo1..photoN 마다 last_tile·tile_count."""
    per = {"type": "object", "properties": {k: v for k, v in LAST_SCHEMA_V2["properties"].items() if k != "notes"},
           "required": ["last_tile", "tile_count"]}
    keys = [f"photo{i + 1}" for i in range(n)]
    return {"type": "object", "properties": {**{k: per for k in keys}, "notes": _NOTES}, "required": [*keys, "notes"]}


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

SEAT_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "face_up": {"type": "boolean", "description": "손패 줄의 그림이 보이면 true"},
        "tiles": {**HAND_SCHEMA_V2["properties"]["tiles"], "description": "face_up 일 때 앞면으로 깐 손패 줄의 패를 왼쪽부터. 아니면 빈 배열"},
        "separate_tile": HAND_SCHEMA_V2["properties"]["separate_tile"],
        "melds": {**HAND_SCHEMA_V2["properties"]["melds"], "description": "손패 줄 오른쪽(테이블 모서리까지)에 놓인 앞면 3~4장 울기 묶음 전부. 없으면 빈 배열"},
        "loose": {
            "type": "array",
            "description": "손패 줄·울기 묶음에 속하지 않고 1~3장만 따로 놓인 앞면 패(도라 표시패 등). 없으면 빈 배열",
            "items": _tile(),
        },
        "notes": _NOTES,
    },
    "required": ["face_up", "tiles", "separate_tile", "melds", "loose", "notes"],
}

WALL_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "box_2d": {
            "type": "array",
            "description": "왕패 7줄을 감싸는 상자 [ymin, xmin, ymax, xmax], 0~1000 정수. 못 찾으면 빈 배열",
            "items": {"type": "integer"},
        },
        "side": {"type": "string", "enum": list(SIDES), "description": "왕패가 어느 자리 앞의 패산인지(사진 기준)"},
        "notes": _NOTES,
    },
    "required": ["found", "box_2d", "notes"],
}

_BOX_INT = {"type": "array", "items": {"type": "integer"}}
SETUP_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "seats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"side": {"type": "string", "enum": list(SIDES)}, "hand": _BOX_INT, "melds": _BOX_INT},
                "required": ["side", "hand", "melds"],
            },
        },
        "device": _BOX_INT,
        "notes": _NOTES,
    },
    "required": ["seats", "device", "notes"],
}

WALLDORA_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "indicators": {
            "type": "array",
            "description": "패산 위에 앞면으로 뒤집힌 패. 왼쪽부터. 없으면 빈 배열",
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

# 강 끝을 자리마다 따로(3호출) 부르면 그중 하나가 생각을 수천 토큰 해 장면을 30초대로 늦췄다. 같은 크롭들을 한 호출에 넣으면
# 출력 칸이 여럿이라 모델이 생각을 건너뛴다(B·C·D 장면 60개 × 2조건, mahjong_calc/experiments/speed/last3.py, 2026-10-26):
# 론 패 54 → 60/60, 자리별 151 → 150/180, 시간 중앙 8.5 → 5.9초·p90 32.7 → 14.4초. 대신 같은 패를 매번 같은 쪽으로 틀린다(6통 → 8삭 등).
# 자리 판독과 같은 통수·삭수 참고표를 마지막 사진으로 붙인다(B·C·D 120장면씩 두 회차, experiments/speed/last3_opt.py):
# 자리별 300 → 319/360, 론 패 120/120 그대로, 시간 같음, 입력 토큰 +31%. 무늬 배치 문구·"6장씩 끊어 세기" 문구는 효과 없음.
_ORDINALS = ["첫째", "둘째", "셋째", "넷째", "다섯째"]

# 강 틀은 YOLO11 검출로 잡고(app/river.py) 마지막 패만 잘라 읽는다. 위 방식이 "어느 패가 마지막인가" 를 처진 패에서 틀리던 것을
# 위치는 검출 + 규칙, 종류는 모델로 나눠 해결(B·C·D 60장면 10회: 자리별 158 → 180/180, 론 패 60/60, 호출 중앙 4.4초,
# mahjong_calc/experiments/speed/river_yolo.py read --model nikmomo-s --conf 0.04). 문구는 측정한 그대로.
LAST_TILE_PROMPT = """사진 {n}장은 각각 리치마작 패 한 장을 위에서 찍은 것입니다(버림패 줄에서 잘라 냄, 가장자리에 이웃 패가 조금 걸릴 수 있음).
사진마다 가운데에 온전히 보이는 패 한 장의 종류를 적습니다.

- {keys}: 첫째부터 순서대로 그 사진 패의 code 와 확신도.

{notation}

마지막 사진은 이 패 세트의 통수(1p~9p)·삭수(1s~9s) 실물 참고표입니다(이름표가 위에 있음). 통수·삭수는 참고표와 무늬 개수·배치를 비교해 정합니다.
참고표는 답에 넣지 않습니다."""


def last_tile_prompt(n: int) -> str:
    return LAST_TILE_PROMPT.format(n=n, keys=", ".join(f"photo{i + 1}" for i in range(n)), notation=NOTATION)


def last_tile_schema(n: int) -> dict:
    per = {"type": "object", "properties": {"tile": LAST_SCHEMA_V2["properties"]["last_tile"]}, "required": ["tile"]}
    keys = [f"photo{i + 1}" for i in range(n)]
    return {"type": "object", "properties": {**{k: per for k in keys}, "notes": {"type": "string"}}, "required": keys}


def normalize_last_tiles(data: dict, n: int) -> list[dict | None]:
    """→ 사진 순서대로 {tile, conf} 또는 None."""
    return [_code((data.get(f"photo{i + 1}") or {}).get("tile")) if isinstance(data.get(f"photo{i + 1}"), dict) else None
            for i in range(n)]


# 자리 판독도 틀은 YOLO11 검출 + 규칙(app/seat.py)로 잡고, 조각을 번호 칸에 붙인 그림 한 장의 종류만 읽는다.
# 조각 13장을 따로 보내면 입력 17.9k 토큰, 한 장 그림이면 3.7k — C·D 40회: 정답 같음, Gemini 중앙 6.5 → 4.9초, p90 11.4 → 7.7초
# (mahjong_calc/experiments/speed/seat_yolo.py --det mosaic --layout sheet, NOTES 23). 문구는 측정한 것에서 뒷면 색 이름만 뺌.
PIECE_PROMPT = """첫 사진은 리치마작 테이블 위의 패 {n}장을 한 장씩 잘라 번호(1~{n}) 칸에 순서대로 붙인 것입니다(칸 가장자리에 이웃 패가 조금 걸릴 수 있음).

- tiles: 번호 순서대로 정확히 {n}개. 칸 가운데에 온전히 보이는 패 한 장의 종류(code)와 확신도.
  옆으로 눕힌 패는 그대로 읽습니다. 무늬 없는 단색 뒷면만 보이면 code 를 "back" 으로 적습니다.
- 장수·순서·묶음은 이미 정해져 있습니다. 패를 더하거나 빼지 않습니다.

{notation}

마지막 사진은 이 패 세트의 통수(1p~9p)·삭수(1s~9s) 실물 참고표입니다(이름표가 위에 있음). 통수·삭수는 참고표와 무늬 개수·배치를 비교해 정합니다.
참고표는 답에 넣지 않습니다."""


def piece_prompt(n: int) -> str:
    return PIECE_PROMPT.format(n=n, notation=NOTATION)


def piece_schema(n: int) -> dict:
    item = {"type": "object", "properties": {"code": {"type": "string", "enum": TILE_CODES + ["back"]},
                                             "confidence": {"type": "number"}}, "required": ["code", "confidence"]}
    return {"type": "object", "properties": {"tiles": {"type": "array", "items": item, "minItems": n, "maxItems": n},
                                             "notes": {"type": "string"}}, "required": ["tiles"]}


def normalize_pieces(data: dict, n: int) -> list[dict | None]:
    """→ 칸 순서대로 {tile, conf} 또는 None(뒷면·못 읽음). 장수가 n 과 다르면 모자란 칸은 None."""
    tiles = data.get("tiles") if isinstance(data.get("tiles"), list) else []
    out = [_code(t) for t in tiles[:n]]
    return out + [None] * (n - len(out))


def last_multi_prompt(n: int) -> str:
    """강 끝 n 장 + 마지막에 참고표 한 장(recognizer 가 붙인다)."""
    keys = ", ".join(f"photo{i + 1}" for i in range(n))
    return f"""사진 {n}장은 리치마작 테이블 가운데의 버림패(강)를 찍은 것으로, 사진마다 서로 다른 플레이어 쪽에서 본 방향으로 세웠습니다. {_ONLY_SEE}

사진마다 그 사진의 아래쪽 플레이어 강만 봅니다. 글자가 바로 서 보이는(위아래가 맞는) 패들이 그 사람의 강입니다.
강은 6장씩 줄을 이루고, 첫 줄은 테이블 가운데 쪽(사진 위쪽)이며 다음 줄은 아래쪽 플레이어 쪽(사진 아래쪽)으로 쌓입니다.
각 줄은 왼쪽에서 오른쪽으로 버립니다. 따라서 마지막 패는 **가장 아래 줄의 가장 오른쪽 패**입니다.
그 패가 옆으로 눕혀져 있어도 그대로 마지막 패입니다.

- {keys}: {'·'.join(_ORDINALS[:n])} 사진의 결과. 사진마다 따로 봅니다.
  - last_tile: 그 사진 아래쪽 플레이어 강의 마지막 패 한 장. 그 강이 보이지 않으면 null.
  - tile_count: 그 강의 패 수.
- 옆으로(90도) 또는 거꾸로(180도) 돌아가 보이는 패는 다른 자리의 강이니 보지 않습니다. 가운데 자동탁 기기·패산도 보지 않습니다.

{NOTATION}

{_ORDINALS[n]} 사진은 이 패 세트의 통수(1p~9p)·삭수(1s~9s) 실물 참고표입니다(이름표가 위에 있음). 통수·삭수는 참고표와 무늬 개수·배치를 비교해 정합니다.
참고표는 강이 아니므로 답(photo1~photo{n})은 앞의 사진 {n}장에 대해서만 적습니다.

확신도(0~1)를 적어 주세요."""

# 자리 판독은 사진 5장을 한 번에 보낸다: 자리 전체, 같은 사진을 왼쪽부터 세 토막(폭 40%, 30% 간격, 원본 해상도),
# 통수·삭수 1~9 실물 참고표(assets/ref_sheet_ps.jpg). 갤러리 장면 5개 × 60회 측정(mahjong_calc/experiments/speed, 2026-10-25):
# 사진 한 장 50% → 토막 83% → 토막 + 참고표 93%. 참고표는 매장과 다른 패 세트(집 패 정면 사진) 것이어도 같은 효과였다.
# face_up 을 "그림이 보이면 true" 로 둔 것: "세워져 있으면 false" 로 쓰면 비스듬한 사진에서 눕힌 패의 연보라 두께를 보고
# 세웠다며 손패를 비웠다(300회 중 12회 → 0/120). loose 는 놓인 곳과 관계없이 모두(도라 표시패를 적을지 고민하던 것).
# "손패 + 울기 = 13장, 화료패 더해 14장" 장수 규칙은 뺐다: B·C·D 60회씩 같은 시각에 있음 55 / 없음 59, 시간 같음(2026-10-26).
# 뒷면 색 이름("연보라")은 뺐다: 매장마다 뒷면 색이 다르다. B·C·D 뒷면을 파랑·초록·노랑·주황·검정으로 바꾼 합성 사진 네 자리 × 6색에서
# face_up 144/144·화료자 손패 36/36(색 이름 있을 때 144/144·35/36) — mahjong_calc/experiments/backcolor(2026-10-27).
SEAT_ZOOM_STRIPS = 3
SEAT_PROMPT_V2 = f"""이 사진은 리치마작 테이블을 위에서 찍은 것으로, 한 플레이어 자리(앞의 손패와 그 오른쪽 테이블 모서리까지)를 그 사람 쪽에서 본 방향으로 세운 것입니다. {_ONLY_SEE}

- face_up: 이 사람의 손패 줄(보통 10~14장)의 그림(글자·동그라미·막대)이 사진에 보이면 true.
  비스듬히 찍혀 패 윗변이나 옆에 뒷면 색 두께가 함께 보여도 그림이 보이면 true 입니다(위에서 찍은 사진에서 그림이 보이는 손패는 공개한 손패입니다).
  손패 줄의 그림이 하나도 안 보이고 무늬 없는 단색 뒷면(세트마다 색이 다름: 연보라·파랑·초록·노랑·주황·검정 등)만 보이면 false 이고, 그때 tiles·separate_tile 은 비웁니다.
- tiles: 앞면 손패 줄. 왼쪽부터. 떨어진 한 장과 울기 묶음은 제외합니다.
- separate_tile: 손패 줄 끝에서 조금 떨어뜨려 놓은 한 장(없으면 null).
- melds: 손패 줄 오른쪽, 테이블 모서리까지 따로 놓인 앞면 3~4장 묶음 전부(퐁·치·깡). 묶음마다 패를 왼쪽부터 적고,
  옆으로 눕힌 패의 위치(sideways)와 뒤집힌 패의 위치(face_down)를 0부터 적습니다. 뒤집힌 패는 같은 묶음의 앞면 패와 같은 것으로 적습니다.
- loose: 손패 줄·떨어진 한 장·울기 묶음이 아닌 앞면 패는 놓인 곳(패산 옆, 펠트 위, 모서리)과 상관없이 모두 적습니다.
  도라 표시패인지 버림패인지는 가리지 않습니다(앱이 정합니다). 무늬 없는 단색 뒷면만 보이는 패와 사진 가장자리에서 반 이상 잘린 패만 적지 않습니다.

{NOTATION}

보이는 패 수를 먼저 세고 그 수만큼 정확히 나열하세요. 각 패의 확신도(0~1)를 적어 주세요.

사진이 여러 장입니다. 첫 사진은 자리 전체이고, 그 다음 {SEAT_ZOOM_STRIPS}장은 같은 사진을 왼쪽부터 {SEAT_ZOOM_STRIPS}토막으로 나눠(이웃 토막과 조금씩 겹침)
원본 해상도로 크게 보여 준 것입니다. 패 종류(특히 통수·삭수의 무늬 개수)는 확대 토막에서 직접 세어 읽고,
장수·순서·앞면 여부·울기 묶음의 위치는 전체 사진으로 정합니다. 겹친 부분의 패를 두 번 세지 않습니다.

마지막 사진은 이 패 세트의 통수(1p~9p)·삭수(1s~9s) 실물 참고표입니다(이름표가 위에 있음). 통수·삭수는 참고표와 무늬 개수·배치를 비교해 정합니다.
참고표의 패는 이 자리의 패가 아닙니다."""



def _must_replace(s: str, old: str, new: str) -> str:
    """SEAT_PROMPT_V2 가 바뀌어 문구를 못 찾으면 조용히 어긋나지 않게 바로 실패."""
    if s.count(old) != 1:
        raise RuntimeError(f"HANDONLY_PROMPT_V2: 바꿀 문구를 못 찾음: {old[:40]!r}")
    return s.replace(old, new)


# 판 상태 추적(울기를 게임 중에 미리 읽어 둔 뒤의 화료 사진): 손패 칸만 자른 사진용.
# SEAT_PROMPT_V2 에서 울기 관련 줄만 바꾸고 나머지(앞면 정의·loose·표기법·토막·참고표)는 그대로 둔다 —
# 사진을 바꾸면 문구도 같이 바꿔야 공정하다(experiments/speed/seat_nomeld.py (c): B·D 120회 정답 같음, p90 14.1 → 9.2초).
HANDONLY_PROMPT_V2 = SEAT_PROMPT_V2
HANDONLY_PROMPT_V2 = _must_replace(
    HANDONLY_PROMPT_V2,
    "한 플레이어 자리(앞의 손패와 그 오른쪽 테이블 모서리까지)를 그 사람 쪽에서 본 방향으로 세운 것입니다.",
    "한 플레이어 앞의 손패 줄을 그 사람 쪽에서 본 방향으로 세운 것입니다. 이 사람의 울기 묶음(퐁·치·깡)은 이미 따로 읽었고,"
    " 놓여 있던 오른쪽 모서리는 사진에서 잘라 냈습니다. 그래서 손패 줄은 13장보다 적을 수 있습니다(울기 묶음 하나당 3장 적음).")
HANDONLY_PROMPT_V2 = _must_replace(HANDONLY_PROMPT_V2, "이 사람의 손패 줄(보통 10~14장)의", "이 사람의 손패 줄의")
_i = HANDONLY_PROMPT_V2.index("- melds:")
_j = HANDONLY_PROMPT_V2.index("- loose:")
HANDONLY_PROMPT_V2 = HANDONLY_PROMPT_V2[:_i] + HANDONLY_PROMPT_V2[_j:]
HANDONLY_PROMPT_V2 = _must_replace(HANDONLY_PROMPT_V2, "- loose: 손패 줄·떨어진 한 장·울기 묶음이 아닌", "- loose: 손패 줄·떨어진 한 장이 아닌")
HANDONLY_PROMPT_V2 = _must_replace(HANDONLY_PROMPT_V2, "떨어진 한 장과 울기 묶음은 제외합니다.", "떨어진 한 장은 제외합니다.")
HANDONLY_PROMPT_V2 = _must_replace(HANDONLY_PROMPT_V2, "장수·순서·앞면 여부·울기 묶음의 위치는", "장수·순서·앞면 여부는")
HANDONLY_PROMPT_V2 = _must_replace(HANDONLY_PROMPT_V2, "첫 사진은 자리 전체이고", "첫 사진은 손패 줄 전체이고")
assert HANDONLY_PROMPT_V2.count("울기") == 2 and "- melds:" not in HANDONLY_PROMPT_V2

HANDONLY_SCHEMA_V2 = {
    **SEAT_SCHEMA_V2,
    "properties": {k: v for k, v in SEAT_SCHEMA_V2["properties"].items() if k != "melds"},
    "required": [k for k in SEAT_SCHEMA_V2["required"] if k != "melds"],
}


DORA_PROMPT_V2 = f"""이 사진은 리치마작 자동탁 가운데를 위에서 찍은 것입니다. 가운데 기기 둘레에 네 자리의 버림패(강)가 6장씩 줄지어 있습니다. {_ONLY_SEE}

- indicators: 강 줄에 붙어 있지 않고 펠트 위에 따로 떨어져 놓인 앞면 패 전부(도라·뒷도라 표시패). 보통 1~4장이 한 줄로,
  강에서 한 장 이상 떨어진 곳(사진 가장자리 쪽일 수 있음)에 놓입니다. 패의 방향은 상관없습니다.
- 강 줄에 붙은 패, 뒷면(연보라)이 보이는 패, 사진 가장자리에 걸려 반 이상 잘린 패는 적지 않습니다. 없으면 빈 배열입니다.

{NOTATION}

각 패의 확신도(0~1)를 적어 주세요."""

# 도라 산(왕패) 찾기: 판 시작 때 테이블 전체(줄인 것)로 한 번. 앱이 이 상자를 그 판 동안 들고 있다가 리치 아닌 화료 때
# 그 상자만 잘라 /v2/walldora 로 읽는다.
WALL_PROMPT_V2 = """이 사진은 판이 시작된 리치마작 자동탁 전체를 위에서 찍은 것입니다. 왕패를 찾습니다.

- 네 변에 패산(뒷면이 보이게 2단으로 쌓인 패 줄)이 있고, 그중 한 곳에 도라 표시패가 앞면으로 뒤집혀 있습니다(깡을 했으면 그 옆에 더).
- 왕패는 그 도라 표시패가 있는 변의 패산에서 도라 표시패를 포함한 일자 구간(보통 7줄)입니다.
- box_2d: 그 변의 일자 구간만 감싸는 상자 [ymin, xmin, ymax, xmax], 사진 크기를 0~1000 으로 본 정수.
  모서리를 돌아 이웃 변으로 이어진 줄은 넣지 않습니다.
  찾지 못하면 found=false, box_2d 는 빈 배열.
- side: 왕패가 어느 자리 앞의 패산인지(bottom/right/top/left, 사진 기준)."""

# 처음 설정(카메라를 놓을 때 한 번): 네 자리 손패·울기 틀과 가운데 기기 틀. 앱이 이것으로 손패+울기 바깥 띠(격자 선)와
# 리치봉 상자를 넉넉하게 정하고 사용자가 끌어 고친다(사용자 그림 2026-10-29: 바깥 띠 · 리치봉 상자 · 나머지 패산 구역).
SETUP_PROMPT_V2 = """이 사진은 리치마작 자동탁을 위에서 찍은 것입니다. 구역의 틀(상자)만 잡습니다. 패의 종류는 읽지 않습니다.

사진 기준 네 자리(bottom = 사진 아래쪽 변, right = 오른쪽 변, top = 위쪽 변, left = 왼쪽 변)마다:
- hand: 그 자리 사람 앞의 손패 줄(세워져 있든 앞면으로 눕혀져 있든)을 감싸는 상자.
- melds: 그 사람 오른쪽 테이블 모서리 쪽에 놓인 울기 묶음 전부를 감싸는 상자. 없으면 빈 배열.
그리고 device: 가운데 자동탁 기기(주사위 창이 있는 사각 판)를 감싸는 상자.

상자는 [ymin, xmin, ymax, xmax], 사진 크기를 0~1000 으로 본 정수입니다."""

WALLDORA_PROMPT_V2 = f"""이 사진은 리치마작 테이블에서 왕패(도라 표시패를 뒤집어 놓는 패산 7줄) 부분만 잘라, 그 패산 앞 사람 쪽에서 본 방향으로
세운 것입니다. {_ONLY_SEE}

- indicators: 패산 위에 앞면(그림이 보이게)으로 뒤집혀 있는 패 전부(도라 표시패, 깡도라 표시패). 왼쪽부터.
- 뒷면(단색)만 보이는 패, 사진 가장자리에 걸려 반 이상 잘린 패는 적지 않습니다. 없으면 빈 배열입니다.

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


def normalize_last_multi(data: dict, n: int) -> dict:
    """→ {rivers:[{last_tile, tile_count}] (보낸 사진 순서), notes}. 빠진 사진은 last_tile None."""
    rivers = []
    for i in range(n):
        r = normalize_last_v2(data.get(f"photo{i + 1}") if isinstance(data.get(f"photo{i + 1}"), dict) else {})
        rivers.append({"last_tile": r["last_tile"], "tile_count": r["tile_count"]})
    return {"rivers": rivers, "notes": str(data.get("notes") or "")}


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


def normalize_wall_v2(data: dict) -> dict:
    """→ {found, box:[ymin,xmin,ymax,xmax]|None (0~1000), side|None, notes}. 상자가 뒤집혔거나 4개가 아니면 못 찾은 것으로."""
    box = normalize_box(data.get("box_2d")) if data.get("found", True) else None
    side = data.get("side")
    return {
        "found": box is not None,
        "box": box,
        "side": side if box is not None and side in SIDES else None,
        "notes": str(data.get("notes") or ""),
    }


def normalize_setup_v2(data: dict) -> dict:
    """→ {seats: {side: {hand: box|None, melds: box|None}}, device: box|None, notes}. 상자는 0~1000 [ymin, xmin, ymax, xmax]."""
    seats = {}
    for s in data.get("seats") or []:
        if isinstance(s, dict) and s.get("side") in SIDES:
            seats[s["side"]] = {"hand": normalize_box(s.get("hand")), "melds": normalize_box(s.get("melds"))}
    return {"seats": seats, "device": normalize_box(data.get("device")), "notes": str(data.get("notes") or "")}


def normalize_seat_v2(data: dict) -> dict:
    """→ {face_up, tiles:[{tile,conf}], separate_tile, melds:[...], loose:[{tile,conf}], notes}. 앞면이 아니면 손패를 비운다."""
    h = normalize_hand_v2(data)
    face_up = bool(data.get("face_up", False))
    return {
        "face_up": face_up,
        "tiles": h["tiles"] if face_up else [],
        "separate_tile": h["separate_tile"] if face_up else None,
        "melds": h["melds"],
        "loose": [t for t in (_code(x) for x in data.get("loose") or []) if t],
        "notes": h["notes"],
    }


def normalize_handonly_v2(data: dict) -> dict:
    """손패 전용 판독 → normalize_seat_v2 와 같은 모양(melds 는 빈 배열). 앱은 추적해 둔 울기를 채운다."""
    return normalize_seat_v2({**data, "melds": []})


def seat_key(seat: dict) -> tuple:
    """두 자리 판독이 같은 손패인지 비교하는 키: 앞면 여부, 손패 + 떨어진 한 장(순서 무시), 울기 묶음(순서 무시).
    따로 놓인 패(loose)는 도라 후보일 뿐이라 비교하지 않는다."""
    hand = [t["tile"] for t in seat["tiles"]] + ([seat["separate_tile"]["tile"]] if seat["separate_tile"] else [])
    return seat["face_up"], tuple(sorted(hand)), tuple(sorted(tuple(sorted(m["tiles"])) for m in seat["melds"]))


def agree_seats(reads: list[dict]) -> dict:
    """같은 자리를 여러 번 읽은 결과(normalize_seat_v2, 먼저 온 순서)를 하나로.

    모두 같으면 그 답(패 확신도는 읽은 것 중 가장 높은 값). 다르면 가장 많이 나온 답(같으면 먼저 온 것)을 고르고,
    다른 답에 없는 손패(장수 기준)와 다른 답에 없는 울기 묶음의 확신도를 0 으로 낮춰 앱이 확인을 묻게 한다.
    → {...고른 답, "agreement": {"reads": n, "agree": 고른 답과 같은 수, "unanimous": bool, "alternatives": [다른 답 요약]}}
    """
    keys = [seat_key(r) for r in reads]
    counts = Counter(keys)
    best = max(range(len(reads)), key=lambda i: (counts[keys[i]], -i))
    chosen = reads[best]
    same = [r for r, k in zip(reads, keys) if k == keys[best]]
    others = [r for r, k in zip(reads, keys) if k != keys[best]]
    unanimous = not others

    def tile_list(r: dict) -> list[dict]:
        return list(r["tiles"]) + ([r["separate_tile"]] if r["separate_tile"] else [])

    # 같은 답끼리는 위치가 같다고 보고 가장 높은 확신도를 쓴다.
    best_conf = [max(tile_list(r)[i]["conf"] for r in same) for i in range(len(tile_list(chosen)))]
    if others:
        # 다른 답 모두에 있는 만큼만 믿는다(장수 기준). 넘치는 패는 확신도 0.
        trusted = Counter(t["tile"] for t in tile_list(chosen))
        for r in others:
            trusted &= Counter(t["tile"] for t in tile_list(r))
        for i, t in enumerate(tile_list(chosen)):
            if trusted[t["tile"]] > 0:
                trusted[t["tile"]] -= 1
            else:
                best_conf[i] = 0.0
    n_tiles = len(chosen["tiles"])
    tiles = [{**t, "conf": c} for t, c in zip(chosen["tiles"], best_conf[:n_tiles])]
    separate = {**chosen["separate_tile"], "conf": best_conf[n_tiles]} if chosen["separate_tile"] else None
    other_melds = [Counter(tuple(sorted(m["tiles"])) for m in r["melds"]) for r in others]
    melds = []
    for m in chosen["melds"]:
        k = tuple(sorted(m["tiles"]))
        confs = [x["conf"] for r in same for x in r["melds"] if tuple(sorted(x["tiles"])) == k]
        melds.append({**m, "conf": max(confs) if all(c[k] > 0 for c in other_melds) else 0.0})

    def summary(r: dict) -> str:
        return " ".join(t["tile"] for t in tile_list(r)) + "".join(f" [{' '.join(m['tiles'])}]" for m in r["melds"]) \
            + ("" if r["face_up"] else " (앞면 아님)")

    return {
        **chosen,
        "tiles": tiles,
        "separate_tile": separate,
        "melds": melds,
        "agreement": {"reads": len(reads), "agree": len(same), "unanimous": unanimous,
                      "alternatives": sorted({summary(r) for r in others})},
    }
