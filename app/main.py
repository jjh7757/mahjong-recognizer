"""마작 패 인식 서버.

POST /v1/recognize  (multipart: image)  →  {"tiles": [...], "model_version": "...", "infer_ms": ...}
GET  /              헬스체크. 모델이 없으면 "model": null 로 응답하되 200.
"""

from __future__ import annotations

import io
import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps

from .detector import Detector, arrange

log = logging.getLogger("recognizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", 2 * 1024 * 1024))
API_KEY = os.environ.get("RECOGNIZER_API_KEY", "")
CONF = float(os.environ.get("CONF_THRESHOLD", 0.5))

app = FastAPI(title="mahjong-recognizer", version="0.1.0")
detector: Detector | None = None


@app.on_event("startup")
def load_model() -> None:
    global detector
    model = MODEL_DIR / "model.onnx"
    classes = MODEL_DIR / "classes.json"
    if model.exists() and classes.exists():
        detector = Detector(model, classes, conf=CONF)
        log.info("model loaded: %s (%d classes, input %d)", model.name, len(detector.classes), detector.size)
    else:
        log.warning("model not found in %s — /v1/recognize 는 503 을 돌려준다", MODEL_DIR)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model": detector.version if detector else None}


def _check_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


@app.post("/v1/recognize")
async def recognize(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    _check_key(x_api_key)
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"image too large (> {MAX_UPLOAD} bytes)")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}") from e

    dets, infer_s = detector.detect(img)
    tiles = arrange(dets)
    log.info("recognize %s %dx%d -> %d tiles in %.0f ms", image.filename, img.width, img.height,
             len(tiles), infer_s * 1000)
    return {
        "tiles": tiles,
        "count": len(tiles),
        "model_version": detector.version,
        "image_size": [img.width, img.height],
        "infer_ms": round(infer_s * 1000),
    }
