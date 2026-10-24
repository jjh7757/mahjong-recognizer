"""마작 패 인식 서버 (Gemini 비전).

모든 엔드포인트: multipart `image`, 헤더 `X-Api-Key`, 응답에 `notes`·`model`·`latency_ms` 공통.

POST /v1/recognize  눕힌 손패 크롭
  → {"tiles":[{"tile":"5m","conf":0.93},...], "count":13,
     "melds":[{"type":"pon","tiles":["7z","7z","7z"],"conf":0.8}], "win_tile":{"tile":"7s","conf":0.7}|null, ...}
POST /v1/locate     테이블 전경(긴 변 1024 정도로 줄인 것)
  → {"hands":[{"side":"bottom","box":[ymin,xmin,ymax,xmax],"face_up":true,"conf":0.9},...],
     "rivers":[...], "dead_wall":{"box":[...],"conf":..}|null, "center":{"box":[...]}|null,
     "face_up_sides":["bottom"], ...}
POST /v1/table      테이블 중앙 크롭(강 4개 + 왕패)
  → {"dora_indicators":[{"tile":"3p","conf":0.9}], "ura_dora_indicators":[...](리치 화료 뒤 공개했을 때만),
     "riichi":{"bottom":true,...}, "riichi_conf":{...},
     "last_discards":{"bottom":"7s","top":null,...}, "last_discards_conf":{...}, ...}
GET  /              헬스체크.

좌표는 0~1000 정규화, side 는 이미지 기준(bottom/right/top/left). 설계: mahjong_calc/docs/table_reading_design.md
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from .recognizer import GeminiCall, GeminiRecognizer, prepare_image

log = logging.getLogger("recognizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_KEY = os.environ.get("RECOGNIZER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "default")  # default | low | off
# 구역 나누기(locate)는 낮은 생각에서 모서리 울기·왕패를 자주 놓친다(2026-10-24 매장 실험). 따로 정한다.
LOCATE_THINKING = os.environ.get("LOCATE_THINKING", "default")
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", 4 * 1024 * 1024))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", 40))  # 화료 1회 = 3호출
# 화료 1회 = locate 1 + 판독 최대 10호출(손패·울기·강 4·가운데·왕패)을 동시에 보낸다. 4 로 두면 판독이 줄을 서 2~3배 느려졌다(2026-10-24).
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 16))

app = FastAPI(title="mahjong-recognizer", version="0.3.0")
recognizer: GeminiRecognizer | None = None
_sem = asyncio.Semaphore(MAX_CONCURRENT)


class RateLimiter:
    """분당 요청 수 제한(프로세스 전역). 비용 사고 방지용."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self.lock:
            self.calls = [t for t in self.calls if now - t < 60]
            if len(self.calls) >= self.per_min:
                return False
            self.calls.append(now)
            return True


_limiter = RateLimiter(RATE_PER_MIN)


@app.on_event("startup")
def startup() -> None:
    global recognizer
    if GEMINI_API_KEY:
        recognizer = GeminiRecognizer(GEMINI_API_KEY, model=GEMINI_MODEL, thinking=GEMINI_THINKING,
                                      locate_thinking=LOCATE_THINKING)
        log.info("gemini recognizer ready: model=%s thinking=%s locate_thinking=%s max_concurrent=%d",
                 GEMINI_MODEL, GEMINI_THINKING, LOCATE_THINKING, MAX_CONCURRENT)
    else:
        log.warning("GEMINI_API_KEY 가 없다 — /v1/* 는 503")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "backend": "gemini", "model": GEMINI_MODEL if recognizer else None,
            "endpoints": ["/v1/recognize", "/v1/locate", "/v1/table", "/v2/hand", "/v2/river", "/v2/meld", "/v2/center"]}


def _check_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


async def _accept(image: UploadFile, x_api_key: str | None) -> bytes:
    """키·설정·속도 제한·크기·디코딩 검사를 거친 JPEG 바이트."""
    _check_key(x_api_key)
    if recognizer is None:
        raise HTTPException(status_code=503, detail="recognizer not configured")
    if not _limiter.allow():
        raise HTTPException(status_code=429, detail="too many requests")
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"image too large (> {MAX_UPLOAD} bytes)")
    try:
        return await asyncio.to_thread(prepare_image, data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}") from e


def _usage(input_tokens: int, output_tokens: int) -> dict:
    """토큰 수. 앱이 촬영 기록에 남겨 비용을 집계한다."""
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


async def _call(fn: Callable[[bytes], object], jpeg: bytes):
    async with _sem:
        try:
            return await asyncio.to_thread(fn, jpeg)
        except Exception as e:  # noqa: BLE001
            log.exception("gemini call failed")
            raise HTTPException(status_code=502, detail=f"recognition failed: {type(e).__name__}") from e


@app.post("/v1/recognize")
async def recognize(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    jpeg = await _accept(image, x_api_key)
    result = await _call(recognizer.recognize, jpeg)
    log.info("recognize %s -> %d tiles, %d melds, win=%s in %d ms (tokens %d+%d) tiles=%s notes=%r",
             image.filename, len(result.tiles), len(result.melds),
             result.win_tile["tile"] if result.win_tile else None,
             result.latency_ms, result.input_tokens, result.output_tokens,
             " ".join(t["tile"] for t in result.tiles), result.notes[:200])
    return {
        "tiles": result.tiles,
        "count": len(result.tiles),
        "melds": result.melds,
        "win_tile": result.win_tile,
        "notes": result.notes,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "usage": _usage(result.input_tokens, result.output_tokens),
    }


@app.post("/v1/locate")
async def locate(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    jpeg = await _accept(image, x_api_key)
    c: GeminiCall = await _call(recognizer.locate, jpeg)
    d = c.data
    log.info("locate %s -> hands=%s face_up=%s rivers=%d dead_wall=%s in %d ms (tokens %d+%d) notes=%r",
             image.filename, [h["side"] for h in d["hands"]], d["face_up_sides"], len(d["rivers"]),
             d["dead_wall"] is not None, c.latency_ms, c.input_tokens, c.output_tokens, d["notes"][:200])
    return {**d, "model": c.model, "latency_ms": c.latency_ms, "usage": _usage(c.input_tokens, c.output_tokens)}


@app.post("/v1/table")
async def table(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    jpeg = await _accept(image, x_api_key)
    c: GeminiCall = await _call(recognizer.read_table, jpeg)
    d = c.data
    log.info("table %s -> dora=%s ura=%s riichi=%s last=%s in %d ms (tokens %d+%d) notes=%r",
             image.filename, " ".join(t["tile"] for t in d["dora_indicators"]),
             " ".join(t["tile"] for t in d["ura_dora_indicators"]),
             [s for s, v in d["riichi"].items() if v], d["last_discards"],
             c.latency_ms, c.input_tokens, c.output_tokens, d["notes"][:200])
    return {**d, "model": c.model, "latency_ms": c.latency_ms, "usage": _usage(c.input_tokens, c.output_tokens)}


async def _observe(kind: str, image: UploadFile, x_api_key: str | None) -> dict:
    jpeg = await _accept(image, x_api_key)
    c: GeminiCall = await _call(lambda j: recognizer.observe(j, kind), jpeg)
    log.info("observe/%s %s -> %s in %d ms (tokens %d+%d)", kind, image.filename,
             str({k: v for k, v in c.data.items() if k != "notes"})[:300], c.latency_ms, c.input_tokens, c.output_tokens)
    return {**c.data, "model": c.model, "latency_ms": c.latency_ms, "usage": _usage(c.input_tokens, c.output_tokens)}


@app.post("/v2/hand")
async def observe_hand(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """화료자 손패 크롭(그 자리 사람 방향으로 세운 것) → 깐 손패·떨어진 한 장·울기 묶음."""
    return await _observe("hand", image, x_api_key)


@app.post("/v2/river")
async def observe_river(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """강 크롭 하나(그 자리 사람 방향으로 세운 것) → 버린 순서대로 패·가로 여부, 리치봉."""
    return await _observe("river", image, x_api_key)


@app.post("/v2/meld")
async def observe_meld(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """울기 묶음 크롭 하나(그 자리 사람 방향으로 세운 것) → 패·가로 위치·뒤집힌 위치."""
    return await _observe("meld", image, x_api_key)


@app.post("/v2/center")
async def observe_center(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """가운데 또는 왕패 크롭 → 앞면 표시패 전부, 자리별 리치봉."""
    return await _observe("center", image, x_api_key)
