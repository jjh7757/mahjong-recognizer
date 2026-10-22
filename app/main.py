"""마작 패 인식 서버 (Gemini 비전).

POST /v1/recognize  (multipart: image, 헤더 X-Api-Key)
  → {"tiles":[{"tile":"5m","conf":0.93},...], "count":13, "notes":"", "model":"gemini-3.8-flash", "latency_ms":15000}
GET  /              헬스체크.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from .recognizer import GeminiRecognizer, prepare_image

log = logging.getLogger("recognizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_KEY = os.environ.get("RECOGNIZER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "default")  # default | low | off
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", 4 * 1024 * 1024))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", 20))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 2))

app = FastAPI(title="mahjong-recognizer", version="0.2.0")
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
        recognizer = GeminiRecognizer(GEMINI_API_KEY, model=GEMINI_MODEL, thinking=GEMINI_THINKING)
        log.info("gemini recognizer ready: model=%s thinking=%s", GEMINI_MODEL, GEMINI_THINKING)
    else:
        log.warning("GEMINI_API_KEY 가 없다 — /v1/recognize 는 503")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "backend": "gemini", "model": GEMINI_MODEL if recognizer else None}


def _check_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


@app.post("/v1/recognize")
async def recognize(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    _check_key(x_api_key)
    if recognizer is None:
        raise HTTPException(status_code=503, detail="recognizer not configured")
    if not _limiter.allow():
        raise HTTPException(status_code=429, detail="too many requests")

    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"image too large (> {MAX_UPLOAD} bytes)")
    try:
        jpeg = prepare_image(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}") from e

    async with _sem:
        try:
            result = await asyncio.to_thread(recognizer.recognize, jpeg)
        except Exception as e:  # noqa: BLE001
            log.exception("gemini call failed")
            raise HTTPException(status_code=502, detail=f"recognition failed: {type(e).__name__}") from e

    log.info("recognize %s -> %d tiles in %d ms (tokens %d+%d)", image.filename, len(result.tiles),
             result.latency_ms, result.input_tokens, result.output_tokens)
    return {
        "tiles": result.tiles,
        "count": len(result.tiles),
        "notes": result.notes,
        "model": result.model,
        "latency_ms": result.latency_ms,
    }
