"""Gemini 비전 모델로 마작 손패 사진에서 패를 읽는다.

입력은 "눕힌 손패를 위에서 찍은 사진" 을 전제로 한다. 테이블 전경처럼 버림패·벽패가 섞인
사진은 정확도가 크게 떨어진다 (experiments/vlm/RESULTS.md).
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass

from PIL import Image, ImageOps

TILE_CODES = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
    + ["0m", "0p", "0s"]
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tiles": {
            "type": "array",
            "description": "사진에 보이는 패를 왼쪽에서 오른쪽 순서로",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "enum": TILE_CODES},
                    "confidence": {"type": "number", "description": "0~1 확신도"},
                },
                "required": ["code", "confidence"],
            },
        },
        "notes": {"type": "string", "description": "가려짐, 흐림 등 특이사항. 없으면 빈 문자열"},
    },
    "required": ["tiles", "notes"],
}

PROMPT = """이 사진은 리치마작(일본식 마작) 손패를 찍은 것입니다. 사진에 보이는 패를 전부, 왼쪽에서 오른쪽 순서로 읽어 주세요. 두 줄이면 윗줄부터 읽습니다.

표기법:
- 만수(萬): 1m~9m. 한자 숫자(一二三四五六七八九) 위에 萬 이 있는 패.
- 통수(筒): 1p~9p. 동그라미 개수.
- 삭수(索): 1s~9s. 대나무 막대 개수. 1s 는 새 그림.
- 자패: 동 1z, 남 2z, 서 3z, 북 4z, 백(빈 패 또는 테두리만) 5z, 발(綠 發) 6z, 중(붉은 中) 7z.
- 적도라(붉은 5): 0m, 0p, 0s. 5인데 글자나 무늬 전체가 붉게 그려진 패만 해당합니다.

주의:
- 뒤집힌 패, 옆으로 눕힌 패, 멀리 있는 다른 사람의 패나 버림패는 읽지 않습니다.
- 같은 패가 여러 장이면 장수만큼 반복해 적습니다.
- 각 패마다 확신도(0~1)를 적어 주세요. 헷갈리는 패는 낮게 주세요.
- 보이는 패 수를 먼저 세고, 그 수만큼 정확히 나열하세요. 없는 패를 만들지 마세요."""


@dataclass
class RecognizeResult:
    tiles: list[dict]
    notes: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model: str


def prepare_image(data: bytes, max_side: int = 1568) -> bytes:
    """EXIF 회전 반영, 긴 변 max_side 이하, JPEG 로 통일."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class GeminiRecognizer:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", thinking: str = "default",
                 timeout_s: float = 90.0):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.thinking = thinking
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def _config(self):
        types = self._types
        kwargs = dict(response_mime_type="application/json", response_schema=RESPONSE_SCHEMA)
        if self.thinking == "off":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif self.thinking == "low":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        return types.GenerateContentConfig(**kwargs)

    def recognize(self, jpeg: bytes) -> RecognizeResult:
        types = self._types
        t0 = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT],
            config=self._config(),
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)
        um = response.usage_metadata
        inp = (um.prompt_token_count or 0) if um else 0
        out = ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
        if not response.text:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            raise RuntimeError(f"empty response ({reason})")
        data = json.loads(response.text)
        tiles = [
            {"tile": t["code"], "conf": round(float(t.get("confidence", 0)), 3)}
            for t in data.get("tiles", []) if t.get("code") in TILE_CODES
        ]
        return RecognizeResult(tiles, data.get("notes", ""), latency_ms, inp, out, self.model)
