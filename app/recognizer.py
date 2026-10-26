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
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .normalize import normalize_hand, normalize_locate, normalize_table
from .prompts import HAND_PROMPT, LOCATE_PROMPT, TABLE_PROMPT
from .schemas import HAND_SCHEMA, LOCATE_SCHEMA, TABLE_SCHEMA, TILE_CODES  # noqa: F401  (TILE_CODES 재수출)

log = logging.getLogger("recognizer")

# 늦은 요청은 버리지 않고 클라이언트 시간 제한(90초)까지 스레드를 잡고 있으므로 넉넉히 둔다.
_hedge_pool = ThreadPoolExecutor(max_workers=48, thread_name_prefix="gemini")
# 자리 판독은 같은 요청을 여러 번 동시에 보내 답을 맞춰 본다(각 요청 안의 헤지는 _hedge_pool).
_fanout_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="seat")

# 통수·삭수 1~9 실물 참고표(자리 판독에 붙인다). 집 패 정면 사진에서 자름: mahjong_calc/experiments/speed/tileset2.
REF_SHEET = Path(__file__).parent / "assets" / "ref_sheet_ps.jpg"


def _retryable(e: Exception) -> bool:
    """다시 보내 볼 만한 실패: 시간 초과, Gemini 5xx(504 DEADLINE_EXCEEDED 등)."""
    import httpx
    from google.genai import errors

    return isinstance(e, (httpx.TimeoutException, errors.ServerError, TimeoutError))


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


def zoom_strips(jpeg: bytes, n: int, width: float = 0.4) -> list[bytes]:
    """사진을 왼쪽부터 n 토막(각각 폭 [width], 이웃과 겹침)으로 잘라 원본 해상도 그대로 JPEG 로."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    step = (1 - width) / (n - 1) if n > 1 else 0
    out = []
    for i in range(n):
        buf = io.BytesIO()
        img.crop((round(i * step * w), 0, round((i * step + width) * w), h)).save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return out


class GeminiRecognizer:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", thinking: str = "default",
                 timeout_s: float = 90.0, locate_thinking: str | None = None,
                 hedge_s: float = 12.0, locate_hedge_s: float = 35.0, attempts: int = 2, seat_reads: int = 3):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.thinking = thinking
        self.locate_thinking = locate_thinking or thinking
        # Gemini 호출 하나가 가끔 응답 없이 90초 넘게 멈춘다(2026-10-24 매장 실험, 평소 4~9초).
        # [hedge_s] 초 안에 답이 없으면 같은 요청을 하나 더 보내 먼저 온 답을 쓴다. 요청에 기한(HttpOptions timeout)을
        # 주면 Gemini 가 그 기한에 504 DEADLINE_EXCEEDED 로 끊어 재시도까지 둘 다 실패하는 일이 잦았다(b2d1861).
        # locate 는 생각이 길어 평소 15~30초라 따로 둔다.
        self.hedge_s = hedge_s
        self.locate_hedge_s = locate_hedge_s
        self.attempts = max(1, attempts)
        # 자리 판독은 [seat_reads] 번 동시에 읽어 모두 같을 때만 확신한다(observe.agree_seats). 같은 사진을 여러 번 읽으면
        # 남은 오독(먼 자리 통수·삭수)은 셋 중 하나꼴로만 나서, 만장일치면 틀린 채 반영되는 일이 0 이 된다(시뮬레이션).
        self.seat_reads = max(1, seat_reads)
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

    def call(self, jpeg: bytes, prompt: str, schema: dict, thinking: str | None = None,
             hedge_s: float | None = None, media_resolution: str | None = None,
             extra_images: list[tuple[bytes, str | None]] = ()) -> GeminiCall:
        """이미지 + 프롬프트 → 스키마대로 파싱한 dict (normalize 전).

        [hedge_s] 초 안에 답이 없거나 시간 초과·5xx 로 실패하면 같은 요청을 하나 더 보내(모두 [attempts] 개까지)
        먼저 성공한 답을 쓴다. 늦은 요청은 끊지 않는다(토큰은 들지만 늦게라도 이길 수 있다).
        [extra_images] 는 첫 사진 뒤에 붙일 (JPEG, 이미지 해상도 수준) 들.
        """
        types = self._types
        hedge_s = hedge_s or self.hedge_s
        t0 = time.perf_counter()

        def image(data: bytes, resolution: str | None):
            part = types.Part.from_bytes(data=data, mime_type="image/jpeg")
            if resolution:
                # Gemini 는 보낸 해상도와 관계없이 이미지를 정해진 토큰 수로 줄인다. ULTRA_HIGH 는 약 2배.
                part.media_resolution = types.PartMediaResolution(level=resolution)
            return part

        parts = [image(jpeg, media_resolution), *(image(d, r) for d, r in extra_images)]

        def once():
            return self.client.models.generate_content(
                model=self.model,
                contents=[*parts, prompt],
                config=self._config(schema, thinking),
            )

        pending = {_hedge_pool.submit(once)}
        sent = 1
        response = None
        last_error: Exception | None = None
        while pending and response is None:
            done, pending = wait(pending, timeout=hedge_s if sent < self.attempts else None,
                                 return_when=FIRST_COMPLETED)
            failed = False
            for f in done:
                try:
                    response = f.result()
                    break
                except Exception as e:  # noqa: BLE001
                    if not _retryable(e):
                        raise
                    last_error, failed = e, True
            if response is None and sent < self.attempts and (failed or not done):
                log.warning("gemini call hedge %d/%d after %.1fs: %s", sent + 1, self.attempts,
                            time.perf_counter() - t0, type(last_error).__name__ if failed else "slow")
                pending.add(_hedge_pool.submit(once))
                sent += 1
        if response is None:
            raise last_error or RuntimeError("no response")
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
        c = self.call(jpeg, LOCATE_PROMPT, LOCATE_SCHEMA, thinking=self.locate_thinking,
                      hedge_s=self.locate_hedge_s)
        c.data = normalize_locate(c.data)
        return c

    def read_table(self, jpeg: bytes) -> GeminiCall:
        c = self.call(jpeg, TABLE_PROMPT, TABLE_SCHEMA)
        c.data = normalize_table(c.data)
        return c

    def observe(self, jpeg: bytes, kind: str, ultra_high: bool = False) -> GeminiCall:
        """/v2 관찰 전용 판독. kind = hand | river | meld | center | corner | last | sticks | dora | seat."""
        from . import observe as ob

        prompt, schema, norm = {
            "hand": (ob.HAND_PROMPT_V2, ob.HAND_SCHEMA_V2, ob.normalize_hand_v2),
            "river": (ob.RIVER_PROMPT_V2, ob.RIVER_SCHEMA_V2, ob.normalize_river_v2),
            "meld": (ob.MELD_PROMPT_V2, ob.MELD_SCHEMA_V2, ob.normalize_meld_v2),
            "center": (ob.CENTER_PROMPT_V2, ob.CENTER_SCHEMA_V2, ob.normalize_center_v2),
            "corner": (ob.CORNER_PROMPT_V2, ob.CORNER_SCHEMA_V2, ob.normalize_corner_v2),
            "last": (ob.LAST_PROMPT_V2, ob.LAST_SCHEMA_V2, ob.normalize_last_v2),
            "sticks": (ob.STICKS_PROMPT_V2, ob.STICKS_SCHEMA_V2, ob.normalize_sticks_v2),
            "dora": (ob.DORA_PROMPT_V2, ob.DORA_SCHEMA_V2, ob.normalize_dora_v2),
            "seat": (ob.SEAT_PROMPT_V2, ob.SEAT_SCHEMA_V2, ob.normalize_seat_v2),
        }[kind]
        resolution = "MEDIA_RESOLUTION_ULTRA_HIGH" if ultra_high else None
        if kind == "seat":
            return self._observe_seat(jpeg, resolution)
        c = self.call(jpeg, prompt, schema, media_resolution=resolution)
        c.data = norm(c.data)
        return c

    def observe_last_multi(self, jpegs: list[bytes]) -> GeminiCall:
        """강 끝 크롭 여러 장(각 자리 쪽으로 세운 가운데 칸)을 한 호출로. data = {rivers:[...] (보낸 순서), notes}."""
        from . import observe as ob

        n = len(jpegs)
        c = self.call(jpegs[0], ob.last_multi_prompt(n), ob.last_multi_schema(n),
                      extra_images=[(j, None) for j in jpegs[1:]])
        c.data = ob.normalize_last_multi(c.data, n)
        return c

    def _observe_seat(self, jpeg: bytes, resolution: str | None) -> GeminiCall:
        """자리 전체 + 확대 토막 + 참고표를 [seat_reads] 번 동시에 읽어 맞춰 본다. 실패한 읽기는 빼고 맞춘다."""
        from . import observe as ob

        extra = [(s, resolution) for s in zoom_strips(jpeg, ob.SEAT_ZOOM_STRIPS)] + [(REF_SHEET.read_bytes(), None)]
        t0 = time.perf_counter()
        futures = [_fanout_pool.submit(self.call, jpeg, ob.SEAT_PROMPT_V2, ob.SEAT_SCHEMA_V2,
                                       media_resolution=resolution, extra_images=extra)
                   for _ in range(self.seat_reads)]
        calls, last_error = [], None
        for f in as_completed(futures):  # 먼저 온 순서(동률이면 먼저 온 답을 고른다)
            try:
                calls.append(f.result())
            except Exception as e:  # noqa: BLE001
                log.warning("seat read failed: %s", type(e).__name__)
                last_error = e
        if not calls:
            raise last_error or RuntimeError("no seat read")
        data = ob.agree_seats([ob.normalize_seat_v2(c.data) for c in calls])
        data["agreement"]["requested"] = self.seat_reads
        if len(calls) < self.seat_reads:
            data["agreement"]["unanimous"] = False  # 읽기가 빠지면 만장일치로 치지 않는다
        return GeminiCall(data, round((time.perf_counter() - t0) * 1000), sum(c.input_tokens for c in calls),
                          sum(c.output_tokens for c in calls), self.model)
