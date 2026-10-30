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
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .normalize import normalize_hand, normalize_locate, normalize_table
from .prompts import HAND_PROMPT, LOCATE_PROMPT, TABLE_PROMPT
from .schemas import HAND_SCHEMA, LOCATE_SCHEMA, TABLE_SCHEMA, TILE_CODES  # noqa: F401  (TILE_CODES 재수출)

log = logging.getLogger("recognizer")


def pool_sizes(max_concurrent: int, seat_reads: int, attempts: int = 2) -> tuple[int, int]:
    """(헤지 풀, 자리 읽기 풀) 스레드 수 = 수요. 동시 요청 × 자리 읽기 수 × 헤지(attempts) / 동시 요청 × 자리 읽기 수."""
    return max_concurrent * seat_reads * attempts, max_concurrent * seat_reads


# 늦은 요청은 버리지 않고 클라이언트 시간 제한(90초)까지 스레드를 잡고 있으므로 수요(MAX_CONCURRENT 16 × 자리 3 × 헤지 2 = 96)만큼
# 둔다. 48·16 고정이던 때는 자리 판독이 몰리면 헤지가 줄을 섰다(2026-10-30). main.py 와 같은 환경 변수를 본다.
_HEDGE_WORKERS, _FANOUT_WORKERS = pool_sizes(int(os.environ.get("MAX_CONCURRENT", 16)),
                                             int(os.environ.get("SEAT_READS", 3)))
_hedge_pool = ThreadPoolExecutor(max_workers=_HEDGE_WORKERS, thread_name_prefix="gemini")
# 자리 판독은 같은 요청을 여러 번 동시에 보내 답을 맞춰 본다(각 요청 안의 헤지는 _hedge_pool).
_fanout_pool = ThreadPoolExecutor(max_workers=_FANOUT_WORKERS, thread_name_prefix="seat")

# 통수·삭수 1~9 실물 참고표(자리 판독에 붙인다). 집 패 정면 사진에서 자름: mahjong_calc/experiments/speed/tileset2.
REF_SHEET = Path(__file__).parent / "assets" / "ref_sheet_ps.jpg"

# 디코딩 전에 거르는 픽셀 수 상한(Image.open 은 헤더만 읽으므로 그 자리에서 거른다). 4000×3000 = 12M, 8000×6000 = 48M.
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 40_000_000))


def rate_limited(e: Exception) -> bool:
    """Gemini 429(RESOURCE_EXHAUSTED: 분당 요청·토큰 과다). 2026-10-29 측정에서 실제로 났다."""
    from google.genai import errors

    return isinstance(e, errors.ClientError) and getattr(e, "code", None) == 429


def retryable(e: Exception) -> bool:
    """다시 보내 볼 만한 실패: 시간 초과, Gemini 5xx(504 DEADLINE_EXCEEDED 등), Gemini 429(1~3초 쉬고)."""
    import httpx
    from google.genai import errors

    return isinstance(e, (httpx.TimeoutException, errors.ServerError, TimeoutError)) or rate_limited(e)


_retryable = retryable  # 옛 이름


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


@dataclass
class GeminiCall:
    """한 번의 호출 결과. data 는 normalize 를 거친 dict.

    latency_ms 는 판독 함수에 들어와서 나갈 때까지(틀 잡기·토막 내기·Gemini·맞추기 모두), prep_ms 는 그중 첫 Gemini 요청을
    보내기 전까지(틀 잡기·토막 내기). 어느 경로든 같은 뜻이다(2026-10-30, 전엔 경로마다 전처리 포함 여부가 달랐다).
    요청 받기·디코딩·크기 줄이기(main._read_image)는 둘 다에 들어가지 않는다."""

    data: dict
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model: str
    prep_ms: int = 0


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
    """EXIF 회전 반영, 긴 변 max_side 이하, JPEG 로 통일.

    헤더의 픽셀 수가 MAX_PIXELS 를 넘으면 디코딩하지 않고 ValueError. JPEG 는 draft 로 DCT 단계에서 1/2·1/4·1/8 로 줄여
    디코딩해 큰 사진의 메모리·시간을 아낀다(결과 크기는 같다: draft 는 요청한 크기 이상으로만 줄인다)."""
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if w * h > MAX_PIXELS:
        raise ValueError(f"too many pixels ({w}x{h} > {MAX_PIXELS})")
    if img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))
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
                 hedge_s: float = 12.0, locate_hedge_s: float = 35.0, attempts: int = 2, seat_reads: int = 3,
                 river_locator=None, seat_locator=None, handonly_frame: bool = True, budget_s: float = 60.0,
                 rate_backoff_s: tuple[float, float] = (1.0, 3.0)):
        from google import genai
        from google.genai import types

        self._types = types
        # 강 끝 판독에서 강 틀(위치)을 잡는 검출기(app/river.RiverLocator). None 이면 강 크롭을 통째로 읽는다.
        self.river_locator = river_locator
        # 자리 판독에서 손패 줄·울기 틀을 잡는 검출기(app/seat.SeatLocator). None 이거나 틀을 못 잡으면 자리 크롭을 통째로 읽는다.
        self.seat_locator = seat_locator
        # 손패 전용(/v2/handonly)도 seat_locator 로 손패 틀 + 조각 읽기를 할지. 장수 검사에 요청의 울기 수를 더한다.
        self.handonly_frame = handonly_frame
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
        # 호출 하나(헤지 포함)의 전체 예산. 넘으면 늦은 요청을 기다리지 않고 TimeoutError(앱은 503 + Retry-After 를 받는다).
        # 전엔 마지막 헤지를 기한 없이 기다려 클라이언트 시간 제한(90초)까지 걸렸다(2026-10-30).
        self.budget_s = budget_s
        # Gemini 429 뒤 다시 보내기 전에 쉬는 시간(초, 이 범위에서 무작위).
        self.rate_backoff_s = rate_backoff_s
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
        먼저 성공한 답을 쓴다. 429 면 [rate_backoff_s] 만큼 쉬고 보낸다. 늦은 요청은 끊지 않는다(토큰은 들지만 늦게라도
        이길 수 있다) — 다만 [budget_s] 를 넘으면 더 기다리지 않고 TimeoutError.
        [extra_images] 는 첫 사진 뒤에 붙일 (JPEG, 이미지 해상도 수준) 들.
        """
        types = self._types
        hedge_s = hedge_s or self.hedge_s
        t0 = time.perf_counter()
        deadline = t0 + self.budget_s

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
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            done, pending = wait(pending, timeout=min(hedge_s, remaining) if sent < self.attempts else remaining,
                                 return_when=FIRST_COMPLETED)
            failed = False
            for f in done:
                try:
                    response = f.result()
                    break
                except Exception as e:  # noqa: BLE001
                    if not retryable(e):
                        raise
                    last_error, failed = e, True
            if response is None and sent < self.attempts and (failed or not done):
                if failed and rate_limited(last_error):
                    pause = random.uniform(*self.rate_backoff_s)
                    log.warning("gemini 429 (요청 과다): %.1fs 쉬고 다시 보낸다", pause)
                    time.sleep(pause)
                log.warning("gemini call hedge %d/%d after %.1fs: %s", sent + 1, self.attempts,
                            time.perf_counter() - t0, type(last_error).__name__ if failed else "slow")
                pending.add(_hedge_pool.submit(once))
                sent += 1
        if response is None:
            if pending:  # 예산 초과: 아직 답이 없는 요청은 그대로 두고(끊을 수 없다) 포기한다
                for f in pending:
                    f.cancel()
                raise TimeoutError(f"gemini call budget {self.budget_s:.0f}s exceeded (sent {sent}, last error "
                                   f"{type(last_error).__name__ if last_error else None})")
            raise last_error or RuntimeError("no response")
        latency_ms = _ms(t0)
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

    def observe(self, jpeg: bytes, kind: str, ultra_high: bool = False, melds: int = 0,
                hand_frac: float | None = None) -> GeminiCall:
        """/v2 관찰 전용 판독. kind = hand | river | meld | center | corner | last | sticks | dora | seat | handonly | wall | walldora.
        latency_ms 는 이 함수 전체(틀 잡기가 실패해 통째 읽기로 넘어간 시간도 포함), prep_ms 는 첫 Gemini 요청 전까지."""
        from . import observe as ob

        t0 = time.perf_counter()
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
            "handonly": (ob.HANDONLY_PROMPT_V2, ob.HANDONLY_SCHEMA_V2, ob.normalize_handonly_v2),
            "wall": (ob.WALL_PROMPT_V2, ob.WALL_SCHEMA_V2, ob.normalize_wall_v2),
            "walldora": (ob.WALLDORA_PROMPT_V2, ob.WALLDORA_SCHEMA_V2, ob.normalize_dora_v2),
            "setup": (ob.SETUP_PROMPT_V2, ob.SETUP_SCHEMA_V2, ob.normalize_setup_v2),
        }[kind]
        resolution = "MEDIA_RESOLUTION_ULTRA_HIGH" if ultra_high else None
        # 손패 전용도 손패 틀 + 조각 읽기로(handonly_frame). 조각 읽기는 패 종류만 묻고 울기는 틀 규칙에서 나오는데,
        # 울기를 잘라 낸 사진이라 melds 는 빈 배열로 둔다. 틀의 장수 검사(13·14)에는 앱이 이미 읽은 울기 수([melds])를 더한다.
        handonly_frame = kind == "handonly" and getattr(self, "handonly_frame", False)
        if (kind == "seat" or handonly_frame) and getattr(self, "seat_locator", None) is not None:
            # [hand_frac] 은 자리 판독만: 격자 손패 칸 폭 비율로 손패 칸·모서리 칸을 나눠 한 번 더 잡아 본다(seat.frame_split).
            c = self._observe_seat_frame(jpeg, resolution, extra_melds=melds if kind == "handonly" else 0,
                                         hand_frac=hand_frac if kind == "seat" else None, t0=t0)
            if c is not None:
                if kind == "handonly":
                    c.data["melds"] = []
                return c
        if kind in ("seat", "handonly"):
            c = self._observe_seat(jpeg, resolution, prompt, schema, norm, t0=t0)
            c.data["method"] = "whole"
            return c
        # 왕패 찾기(판 시작 때 한 번)·처음 설정 틀은 상자만 묻는다. 생각 수준은 WALL_THINKING(low).
        thinking = getattr(self, "wall_thinking", None) if kind in ("wall", "setup") else None
        c = self.call(jpeg, prompt, schema, thinking=thinking, media_resolution=resolution)
        c.data = norm(c.data)
        c.latency_ms = _ms(t0)
        return c

    def observe_last_multi(self, jpegs: list[bytes]) -> GeminiCall:
        """강 끝 크롭 여러 장(각 자리 쪽으로 세운 가운데 칸)을 한 호출로. data = {rivers:[...] (보낸 순서), notes, method}.

        검출기가 있으면 강마다 틀을 잡아 마지막 패만 잘라 읽는다(method "frame", tile_count 는 틀의 장수).
        한 자리라도 틀을 못 잡으면 크롭을 통째로 읽는다(method "whole", 강 크롭 + 참고표).
        latency_ms 는 틀 잡기부터 전체, prep_ms 는 틀 잡기(첫 Gemini 요청 전까지)."""
        from . import observe as ob

        t0 = time.perf_counter()
        n = len(jpegs)
        frames = self._river_frames(jpegs)
        prep_ms = _ms(t0)
        if frames is not None:
            tiles = [crop for _, crop in frames]
            c = self.call(tiles[0], ob.last_tile_prompt(n), ob.last_tile_schema(n),
                          extra_images=[(t, None) for t in tiles[1:]] + [(REF_SHEET.read_bytes(), None)])
            reads = ob.normalize_last_tiles(c.data, n)
            c.data = {
                "rivers": [{"last_tile": r, "tile_count": f.count, "rows": [len(row) for row in f.rows],
                            "sideways": [i for i, s in enumerate(f.sideways) if s]}
                           for r, (f, _) in zip(reads, frames)],
                "notes": str(c.data.get("notes") or ""),
                "method": "frame",
            }
        else:
            c = self.call(jpegs[0], ob.last_multi_prompt(n), ob.last_multi_schema(n),
                          extra_images=[(j, None) for j in jpegs[1:]] + [(REF_SHEET.read_bytes(), None)])
            c.data = {**ob.normalize_last_multi(c.data, n), "method": "whole"}
        c.latency_ms, c.prep_ms = _ms(t0), prep_ms
        return c

    def _river_frames(self, jpegs: list[bytes]):
        """[(RiverFrame, 마지막 패 JPEG)] — 검출기가 없거나 한 자리라도 틀을 못 잡으면 None."""
        if self.river_locator is None:
            return None
        from . import river

        out = []
        t0 = time.perf_counter()
        for j in jpegs:
            try:
                f, rgb = self.river_locator.locate(j)
            except Exception:  # noqa: BLE001 — 검출이 깨져도 판독은 통째 읽기로 계속
                log.exception("river locate failed")
                return None
            if f is None:
                log.info("river frame: not found -> whole read")
                return None
            out.append((f, river.crop(rgb, f.last)))
        log.info("river frames %s in %d ms", [f.rows and [len(r) for r in f.rows] for f, _ in out], _ms(t0))
        return out

    def _observe_seat_frame(self, jpeg: bytes, resolution: str | None, extra_melds: int = 0,
                            hand_frac: float | None = None, t0: float | None = None) -> GeminiCall | None:
        """자리 틀(YOLO11 + 규칙) → 조각을 번호 칸에 붙인 그림 한 장 + 참고표를 [seat_reads] 번 동시에 읽어 맞춰 본다.
        틀을 못 잡거나(장수·모양 검사), 읽은 조각에 뒷면·빈 칸이 있으면 None(통째 읽기로). data["method"] = "frame".
        [t0] 는 판독 시작 시각(observe) — latency_ms·prep_ms 의 기준."""
        from . import observe as ob
        from . import seat

        t0 = t0 or time.perf_counter()
        t_frame = time.perf_counter()
        try:
            opts = {k: v for k, v in (("extra_melds", extra_melds), ("hand_frac", hand_frac)) if v}
            fr, rgb = self.seat_locator.locate(jpeg, **opts)
        except Exception:  # noqa: BLE001 — 검출이 깨져도 판독은 통째 읽기로 계속
            log.exception("seat locate failed")
            return None
        frame_ms = _ms(t_frame)
        if fr is None:
            log.info("seat frame: not found in %d ms -> whole read", frame_ms)
            return None
        n = len(fr.pieces())
        log.info("seat frame %s in %d ms", fr.summary(), frame_ms)
        sheet = seat.sheet(rgb, fr)
        prompt, schema = ob.piece_prompt(n), ob.piece_schema(n)
        extra = [(REF_SHEET.read_bytes(), None)]
        prep_ms = _ms(t0)
        futures = [_fanout_pool.submit(self.call, sheet, prompt, schema, media_resolution=resolution, extra_images=extra)
                   for _ in range(self.seat_reads)]
        calls, reads, last_error = [], [], None
        for f in as_completed(futures):
            try:
                c = f.result()
            except Exception as e:  # noqa: BLE001
                log.warning("seat piece read failed: %s", type(e).__name__)
                last_error = e
                continue
            calls.append(c)
            reads.append(seat.to_seat(fr, ob.normalize_pieces(c.data, n)))
        if not calls:
            raise last_error or RuntimeError("no seat read")
        if any(r is None for r in reads):
            log.info("seat frame: piece read had back/empty cells -> whole read")
            return None
        data = ob.agree_seats(reads)
        data["agreement"]["requested"] = self.seat_reads
        if len(calls) < self.seat_reads:
            data["agreement"]["unanimous"] = False
        data["method"] = "frame"
        data["frame"] = {**fr.summary(), "ms": frame_ms}
        return GeminiCall(data, _ms(t0), sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls),
                          self.model, prep_ms=prep_ms)

    def _observe_seat(self, jpeg: bytes, resolution: str | None, prompt: str | None = None, schema: dict | None = None,
                      norm=None, t0: float | None = None) -> GeminiCall:
        """자리 전체(또는 손패 칸만) + 확대 토막 + 참고표를 [seat_reads] 번 동시에 읽어 맞춰 본다. 실패한 읽기는 빼고 맞춘다.
        [t0] 는 판독 시작 시각(observe; 틀 잡기에 실패한 시간도 들어간다) — latency_ms·prep_ms 의 기준."""
        from . import observe as ob

        t0 = t0 or time.perf_counter()
        prompt = prompt or ob.SEAT_PROMPT_V2
        schema = schema or ob.SEAT_SCHEMA_V2
        norm = norm or ob.normalize_seat_v2
        extra = [(s, resolution) for s in zoom_strips(jpeg, ob.SEAT_ZOOM_STRIPS)] + [(REF_SHEET.read_bytes(), None)]
        prep_ms = _ms(t0)
        futures = [_fanout_pool.submit(self.call, jpeg, prompt, schema, media_resolution=resolution, extra_images=extra)
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
        data = ob.agree_seats([norm(c.data) for c in calls])
        data["agreement"]["requested"] = self.seat_reads
        if len(calls) < self.seat_reads:
            data["agreement"]["unanimous"] = False  # 읽기가 빠지면 만장일치로 치지 않는다
        return GeminiCall(data, _ms(t0), sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls),
                          self.model, prep_ms=prep_ms)
