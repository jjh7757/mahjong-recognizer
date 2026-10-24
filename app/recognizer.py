"""Gemini 비전 모델 호출.

세 가지 읽기가 있다 (설계: mahjong_calc/docs/table_reading_design.md).
- recognize():  눕힌 손패 크롭 → 손패·울기·화료패
- locate():     테이블 전경(줄인 것) → 손패·강·왕패 영역 좌표
- read_table(): 테이블 중앙 크롭 → 도라 표시패·리치·마지막 버림패

패 한 장이 100px 이상 찍혀야 읽힌다 (experiments/vlm/RESULTS.md). 전경 사진을 그대로 읽히면 안 된다.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from .normalize import normalize_hand, normalize_locate, normalize_table
from .prompts import HAND_PROMPT, LOCATE_PROMPT, TABLE_PROMPT
from .schemas import HAND_SCHEMA, LOCATE_SCHEMA, TABLE_SCHEMA, TILE_CODES  # noqa: F401  (TILE_CODES 재수출)


@dataclass
class GeminiCall:
    """한 번의 호출 결과. data 는 normalize 를 거친 dict."""

    data: dict
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model: str


@dataclass
class RecognizeResult:
    tiles: list[dict]
    notes: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model: str
    melds: list[dict] = field(default_factory=list)
    win_tile: dict | None = None


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
                 timeout_s: float = 90.0, locate_thinking: str | None = None):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.thinking = thinking
        self.locate_thinking = locate_thinking or thinking
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def _config(self, schema: dict, thinking: str | None = None):
        types = self._types
        thinking = thinking or self.thinking
        kwargs = dict(response_mime_type="application/json", response_schema=schema)
        if thinking == "off":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif thinking == "low":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        return types.GenerateContentConfig(**kwargs)

    def call(self, jpeg: bytes, prompt: str, schema: dict, thinking: str | None = None) -> GeminiCall:
        """이미지 + 프롬프트 → 스키마대로 파싱한 dict (normalize 전)."""
        types = self._types
        t0 = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            config=self._config(schema, thinking),
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)
        um = response.usage_metadata
        inp = (um.prompt_token_count or 0) if um else 0
        out = ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
        if not response.text:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            raise RuntimeError(f"empty response ({reason})")
        return GeminiCall(json.loads(response.text), latency_ms, inp, out, self.model)

    def recognize(self, jpeg: bytes) -> RecognizeResult:
        c = self.call(jpeg, HAND_PROMPT, HAND_SCHEMA)
        h = normalize_hand(c.data)
        return RecognizeResult(h["tiles"], h["notes"], c.latency_ms, c.input_tokens, c.output_tokens, self.model,
                               melds=h["melds"], win_tile=h["win_tile"])

    def locate(self, jpeg: bytes) -> GeminiCall:
        c = self.call(jpeg, LOCATE_PROMPT, LOCATE_SCHEMA, thinking=self.locate_thinking)
        c.data = normalize_locate(c.data)
        return c

    def read_table(self, jpeg: bytes) -> GeminiCall:
        c = self.call(jpeg, TABLE_PROMPT, TABLE_SCHEMA)
        c.data = normalize_table(c.data)
        return c

    def observe(self, jpeg: bytes, kind: str) -> GeminiCall:
        """/v2 관찰 전용 판독. kind = hand | river | meld | center."""
        from . import observe as ob

        prompt, schema, norm = {
            "hand": (ob.HAND_PROMPT_V2, ob.HAND_SCHEMA_V2, ob.normalize_hand_v2),
            "river": (ob.RIVER_PROMPT_V2, ob.RIVER_SCHEMA_V2, ob.normalize_river_v2),
            "meld": (ob.MELD_PROMPT_V2, ob.MELD_SCHEMA_V2, ob.normalize_meld_v2),
            "center": (ob.CENTER_PROMPT_V2, ob.CENTER_SCHEMA_V2, ob.normalize_center_v2),
        }[kind]
        c = self.call(jpeg, prompt, schema)
        c.data = norm(c.data)
        return c
