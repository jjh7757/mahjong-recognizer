"""Gemini 구조화 출력 스키마 3종 (손패 / 위치 / 테이블 중앙).

좌표(box)는 Gemini 경계 상자 관례대로 [ymin, xmin, ymax, xmax], 0~1000 정규화.
side 는 **이미지 기준** 아래·오른쪽·위·왼쪽이다. 자리(동남서북)로 바꾸는 것은 앱이 대국 기록으로 한다.
"""

from __future__ import annotations

TILE_CODES = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
    + ["0m", "0p", "0s"]
)

SIDES = ["bottom", "right", "top", "left"]
MELD_TYPES = ["chi", "pon", "kan_open", "kan_closed"]


def _tile(description: str = "") -> dict:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "code": {"type": "string", "enum": TILE_CODES},
            "confidence": {"type": "number", "description": "0~1 확신도"},
        },
        "required": ["code", "confidence"],
    }


_BOX = {
    "type": "array",
    "description": "[ymin, xmin, ymax, xmax] 0~1000 정규화 좌표. 정수 4개",
    "items": {"type": "integer"},
}

_NOTES = {"type": "string", "description": "가려짐, 흐림 등 특이사항. 없으면 빈 문자열"}


# ---------------------------------------------------------------- 손패 (화료자 손패 크롭)
HAND_SCHEMA = {
    "type": "object",
    "properties": {
        "tiles": {
            "type": "array",
            "description": "손에 들고 있던(멘젠 부분) 패를 왼쪽에서 오른쪽 순서로. 울기 그룹과 화료패는 제외",
            "items": _tile(),
        },
        "melds": {
            "type": "array",
            "description": "손패 오른쪽에 따로 모아 놓은 울기(치·퐁·깡) 그룹. 없으면 빈 배열",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": MELD_TYPES},
                    "tiles": {
                        "type": "array",
                        "description": "그룹의 패. 치·퐁 3장, 깡 4장. 안깡은 뒤집힌 끝 2장을 가운데 패와 같은 것으로 적음",
                        "items": {"type": "string", "enum": TILE_CODES},
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["type", "tiles", "confidence"],
            },
        },
        "win_tile": {
            **_tile("손패에서 조금 떨어뜨려 따로 놓인 화료패 한 장. 없으면 null"),
            "nullable": True,
        },
        "notes": _NOTES,
    },
    "required": ["tiles", "melds", "win_tile", "notes"],
}


# ---------------------------------------------------------------- 위치 (테이블 전경)
LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "hands": {
            "type": "array",
            "description": "네 플레이어의 손패 영역. 보이는 자리만",
            "items": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "enum": SIDES},
                    "box": _BOX,
                    "face_up": {
                        "type": "boolean",
                        "description": "패가 앞면으로 눕혀져 그림이 보이면 true. 세워져 있거나 뒷면이면 false",
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["side", "box", "face_up", "confidence"],
            },
        },
        "rivers": {
            "type": "array",
            "description": "각 자리 앞 버림패(강) 영역",
            "items": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "enum": SIDES},
                    "box": _BOX,
                    "confidence": {"type": "number"},
                },
                "required": ["side", "box", "confidence"],
            },
        },
        "dead_wall": {
            "type": "object",
            "description": "뒤집힌 도라 표시패가 있는 왕패(벽패 더미) 영역. 없으면 null",
            "nullable": True,
            "properties": {"box": _BOX, "confidence": {"type": "number"}},
            "required": ["box", "confidence"],
        },
        "center": {
            "type": "object",
            "description": "강 4개와 왕패를 모두 포함하는 가운데 영역",
            "nullable": True,
            "properties": {"box": _BOX},
            "required": ["box"],
        },
        "notes": _NOTES,
    },
    "required": ["hands", "rivers", "dead_wall", "center", "notes"],
}


# ---------------------------------------------------------------- 테이블 중앙 (강 4개 + 왕패 크롭)
TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "dora_indicators": {
            "type": "array",
            "description": "왕패 위에 뒤집혀 앞면이 보이는 도라 표시패. 왼쪽부터. 보통 1장",
            "items": _tile(),
        },
        "ura_dora_indicators": {
            "type": "array",
            "description": "리치 화료 뒤 공개한 뒷도라 표시패(도라 표시패 아랫단을 뒤집은 것). 공개 안 됐으면 빈 배열",
            "items": _tile(),
        },
        "riichi": {
            "type": "array",
            "description": "자리별 리치 여부. 강에 옆으로 눕힌 패가 있거나 리치봉 표시등이 켜져 있으면 true",
            "items": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "enum": SIDES},
                    "riichi": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": ["side", "riichi", "confidence"],
            },
        },
        "last_discards": {
            "type": "array",
            "description": "각 강의 마지막 버림패(줄 끝). 모르는 자리는 생략",
            "items": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "enum": SIDES},
                    "code": {"type": "string", "enum": TILE_CODES},
                    "confidence": {"type": "number"},
                },
                "required": ["side", "code", "confidence"],
            },
        },
        "notes": _NOTES,
    },
    "required": ["dora_indicators", "ura_dora_indicators", "riichi", "last_discards", "notes"],
}
