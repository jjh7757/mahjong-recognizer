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

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile

from .observe import LAST_MULTI_MAX
from .recognizer import GeminiCall, GeminiRecognizer, prepare_image

log = logging.getLogger("recognizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_KEY = os.environ.get("RECOGNIZER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "default")  # default | low | off
# 구역 나누기(locate)는 낮은 생각에서 모서리 울기·왕패를 자주 놓친다(2026-10-24 매장 실험). 따로 정한다.
LOCATE_THINKING = os.environ.get("LOCATE_THINKING", "default")
# 이미지 해상도 수준을 ULTRA_HIGH 로 올려 읽을 /v2 판독. 자리 구역(손패 + 모서리)은 크고 대부분 빈 곳이라 패가 작다.
ULTRA_HIGH_KINDS = {k for k in os.environ.get("ULTRA_HIGH_KINDS", "seat,handonly").split(",") if k}
# 왕패 찾기(/v2/wall, 판 시작 때 한 번)의 생각 수준. minimal 은 gemini-3.8-flash 가 지원하지 않음(400).
WALL_THINKING = os.environ.get("WALL_THINKING", "low")
# 자리 판독(/v2/seat)을 몇 번 동시에 읽어 맞춰 볼지. 모두 같아야 확신(다르면 다른 패의 확신도를 0 으로).
SEAT_READS = int(os.environ.get("SEAT_READS", 3))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", 4 * 1024 * 1024))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", 40))  # 화료 1회 = 3호출
# 화료 1회 = locate 1 + 판독 최대 10호출(손패·울기·강 4·가운데·왕패)을 동시에 보낸다. 4 로 두면 판독이 줄을 서 2~3배 느려졌다(2026-10-24).
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 16))
# 강 끝(/v2/last3)의 강 틀을 YOLO11 검출로 잡을지(app/river.py). 0 이면 강 크롭을 통째로 읽는다. 검출 기준값은 RIVER_CONF.
RIVER_FRAME = os.environ.get("RIVER_FRAME", "1") == "1"
RIVER_CONF = float(os.environ.get("RIVER_CONF", 0.04))
# 자리 판독(/v2/seat)의 손패 틀을 YOLO11 검출로 잡아 조각 그림 한 장만 읽을지(app/seat.py). 틀을 못 잡으면 통째 읽기.
SEAT_FRAME = os.environ.get("SEAT_FRAME", "1") == "1"
# 손패 전용(/v2/handonly, 울기를 잘라 낸 사진)도 같은 손패 틀 + 조각 읽기로. 틀의 장수 검사에는 요청의 ?melds=(앱이 게임 중에
# 확정한 울기 수)를 더한다. SEAT_FRAME 이 꺼져 있으면(검출기 없음) 이것도 통째 읽기.
# B·D 60장면 같은 순간: 통째 p90 10.7 → 틀 8.0초, 정답 60/60 둘 다(mahjong_calc docs/metrics.md 표 2-1, 2026-10-28).
HANDONLY_FRAME = os.environ.get("HANDONLY_FRAME", "1") == "1"
# 자리 판독에 ?hand_frac= 가 오면, 한 장 규칙이 손패 틀을 못 잡은 자리를 손패 칸·모서리 칸으로 나눠 한 번 더 잡을지(seat.frame_split).
# 오프라인 B·C·D 틀 4/6 → 6/6(먼 자리 B), 검사 통과한 틀린 틀 0(mahjong_calc docs/metrics.md 표 2-3, 2026-10-28).
SEAT_SPLIT = os.environ.get("SEAT_SPLIT", "1") == "1"
# 자리 틀을 잡기 전에 탁자 끝(펠트 경계 + 패 1.7장 여유) 밖을 지울지(seat.table_only). 기본 끔: 오프라인 틀은 같거나 나았지만
# 같은 순간 Gemini 비교에서 C·D 가 +0.87초(p=0.006)·D 엇갈림 7/40(기준 1/40)이었다(metrics 표 2-1, 2026-10-28).
# 탁자 밖은 앱이 처음 사진을 펼 때(rectify.dart) 한 번에 자르는 쪽으로 간다.
SEAT_TABLE = os.environ.get("SEAT_TABLE", "0") == "1"

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
        locator = seat_locator = None
        if RIVER_FRAME:
            from .river import RiverLocator

            locator = RiverLocator(conf=RIVER_CONF)
        if SEAT_FRAME:
            from .seat import SeatLocator

            seat_locator = SeatLocator(shared=locator, conf=RIVER_CONF, table=SEAT_TABLE)  # 검출 모델은 강 틀과 하나
        recognizer = GeminiRecognizer(GEMINI_API_KEY, model=GEMINI_MODEL, thinking=GEMINI_THINKING,
                                      locate_thinking=LOCATE_THINKING, seat_reads=SEAT_READS, river_locator=locator,
                                      seat_locator=seat_locator, handonly_frame=HANDONLY_FRAME)
        recognizer.wall_thinking = WALL_THINKING
        log.info("gemini recognizer ready: model=%s thinking=%s locate_thinking=%s max_concurrent=%d seat_reads=%d "
                 "river_frame=%s(conf %.2f) seat_frame=%s handonly_frame=%s seat_split=%s seat_table=%s", GEMINI_MODEL,
                 GEMINI_THINKING, LOCATE_THINKING, MAX_CONCURRENT, SEAT_READS, RIVER_FRAME, RIVER_CONF, SEAT_FRAME,
                 HANDONLY_FRAME, SEAT_SPLIT, SEAT_TABLE)
    else:
        log.warning("GEMINI_API_KEY 가 없다 — /v1/* 는 503")


@app.get("/")
def health() -> dict:
    # endpoints 는 등록된 POST 라우트에서 뽑는다(손으로 적던 목록에 /v2/handonly 가 빠졌던 적이 있다).
    # git_sha 는 Dockerfile 이 넣는 GIT_SHA — 배포 워크플로가 이 값으로 새 이미지가 떴는지 확인한다.
    endpoints = sorted(r.path for r in app.routes if "POST" in getattr(r, "methods", ()))
    return {"status": "ok", "backend": "gemini", "model": GEMINI_MODEL if recognizer else None,
            "git_sha": os.environ.get("GIT_SHA", "unknown"), "endpoints": endpoints}


def _check_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


async def _accept(image: UploadFile, x_api_key: str | None, max_side: int = 1568) -> bytes:
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
        return await asyncio.to_thread(prepare_image, data, max_side)
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


async def _observe(kind: str, image: UploadFile, x_api_key: str | None, melds: int = 0,
                   hand_frac: float | None = None) -> dict:
    # 이미지 해상도를 ULTRA_HIGH 로 읽는 판독은 원본에 가깝게 받는다(1568 로 줄이면 올린 효과가 줄어든다).
    jpeg = await _accept(image, x_api_key, max_side=3072 if kind in ULTRA_HIGH_KINDS else 1568)
    opts: dict = {"melds": melds} if melds else {}  # 손패 전용에서 앱이 이미 읽은 울기 수(있을 때만)
    if hand_frac is not None and SEAT_SPLIT:
        opts["hand_frac"] = hand_frac  # 자리 판독에서 앱이 보낸 격자 손패 칸 폭 비율(있을 때만)
    c: GeminiCall = await _call(lambda j: recognizer.observe(j, kind, ultra_high=kind in ULTRA_HIGH_KINDS, **opts), jpeg)
    log.info("observe/%s%s %s -> %s in %d ms (tokens %d+%d)", kind,
             f"[{c.data['method']}]" if "method" in c.data else "", image.filename,
             str({k: v for k, v in c.data.items() if k not in ("notes", "method")})[:300],
             c.latency_ms, c.input_tokens, c.output_tokens)
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


@app.post("/v2/corner")
async def observe_corner(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """미리 그은 격자의 모서리 칸(그 자리 방향으로 세운 것) → 울기 묶음 전부, 묶음 아닌 앞면 패."""
    return await _observe("corner", image, x_api_key)


@app.post("/v2/last")
async def observe_last(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """가운데 칸의 한 자리 쪽(그 자리 방향으로 세운 것) → 그 자리 강의 마지막 패 한 장."""
    return await _observe("last", image, x_api_key)


@app.post("/v2/last3")
async def observe_last_multi(images: list[UploadFile] = File(...),
                             x_api_key: str | None = Header(default=None)) -> dict:
    """가운데 칸을 자리마다 세운 크롭 여러 장(보통 화료자 빼고 3장, 최대 4장) → 사진 순서대로 강의 마지막 패. 한 번의 Gemini 호출.
    강 틀은 YOLO11 검출로 잡아 마지막 패만 잘라 읽고(method "frame", rows·sideways 도 줌), 못 잡으면 통째로 읽는다(method "whole")."""
    if not 1 <= len(images) <= LAST_MULTI_MAX:
        raise HTTPException(status_code=400, detail=f"images must be 1..{LAST_MULTI_MAX}")
    jpegs = [await _accept(im, x_api_key) for im in images]
    c: GeminiCall = await _call(recognizer.observe_last_multi, jpegs)
    log.info("observe/last3[%s] %s -> %s in %d ms (tokens %d+%d)", c.data.get("method"), [im.filename for im in images],
             [r["last_tile"]["tile"] if r["last_tile"] else None for r in c.data["rivers"]],
             c.latency_ms, c.input_tokens, c.output_tokens)
    return {**c.data, "model": c.model, "latency_ms": c.latency_ms, "usage": _usage(c.input_tokens, c.output_tokens)}


@app.post("/v2/sticks")
async def observe_sticks(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """자동탁 가운데 기기 크롭 → 자리별 리치봉."""
    return await _observe("sticks", image, x_api_key)


@app.post("/v2/dora")
async def observe_dora(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """격자 가운데 칸 → 강 밖에 따로 놓인 앞면 패(도라 표시패)."""
    return await _observe("dora", image, x_api_key)


@app.post("/v2/wall")
async def observe_wall(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """판 시작 때 테이블 전체(줄인 것) → 왕패(도라 산) 상자 {found, box, side, notes}. 앱이 그 판 동안 들고 있다."""
    return await _observe("wall", image, x_api_key)


@app.post("/v2/setup")
async def observe_setup(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """처음 설정 때 테이블 전체(줄인 것) → 네 자리 손패·울기 틀과 가운데 기기 틀. 앱이 격자 선·리치봉 상자를 정한다."""
    return await _observe("setup", image, x_api_key)


@app.post("/v2/walldora")
async def observe_walldora(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    """왕패 상자 크롭(그 패산 앞 사람 쪽으로 세운 것) → 패산 위·옆 앞면 패(도라·깡도라 표시패)."""
    return await _observe("walldora", image, x_api_key)


@app.post("/v2/handonly")
async def observe_handonly(image: UploadFile = File(...), x_api_key: str | None = Header(default=None),
                           melds: int = Query(0, ge=0, le=4)) -> dict:
    """판 상태 추적용: 격자 자리의 손패 칸만(울기 모서리는 잘라 낸 것, 그 자리 방향으로 세운 것) → 앞면 여부·손패·따로 놓인 패.
    울기는 앱이 게임 중에 미리 읽어 둔 것을 쓴다(melds 는 늘 빈 배열). /v2/seat 와 같이 SEAT_READS 번 동시에 읽어 `agreement`.
    [melds] 는 그 울기 묶음 수 — 손패 틀(HANDONLY_FRAME)의 장수 검사(손패 + 떨어진 한 장 + 울기×3 = 13·14)에 더한다."""
    return await _observe("handonly", image, x_api_key, melds)


@app.post("/v2/seat")
async def observe_seat(image: UploadFile = File(...), x_api_key: str | None = Header(default=None),
                       hand_frac: float | None = Query(None, gt=0, lt=1)) -> dict:
    """격자 5구역의 한 자리(손패 + 오른쪽 모서리, 그 자리 방향으로 세운 것) → 앞면 여부·손패·울기 전부·따로 놓인 패.
    SEAT_READS 번 동시에 읽어 맞춘 결과와 `agreement`({reads, agree, unanimous, alternatives, requested}).
    [hand_frac] 은 손패 칸(격자 선 사이 변)이 이 크롭에서 차지하는 폭 비율(왼쪽부터) — 손패 틀을 못 잡으면 손패 칸·모서리 칸으로
    나눠 한 번 더 잡는다(SEAT_SPLIT). 없으면 지금처럼."""
    return await _observe("seat", image, x_api_key, hand_frac=hand_frac)
