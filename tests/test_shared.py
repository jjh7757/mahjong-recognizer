import io
import threading
import time

import numpy as np
from PIL import Image

from app import local_read as lr
from app import main as server
from app.shared import Shared
from tests.test_api import _client, _jpeg


def test_shared_computes_once_for_concurrent_callers():
    sh, calls, gate = Shared(size=2, ttl=10), [], threading.Event()

    def slow():
        calls.append(1)
        gate.wait(2)
        return "x"

    out = []
    ts = [threading.Thread(target=lambda: out.append(sh.get("k", slow))) for _ in range(3)]
    for t in ts:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in ts:
        t.join()
    assert out == ["x"] * 3 and len(calls) == 1


def test_shared_drops_failures_and_expires():
    now = [0.0]
    sh = Shared(size=1, ttl=5, clock=lambda: now[0])
    try:
        sh.get("k", lambda: 1 / 0)
    except ZeroDivisionError:
        pass
    assert sh.get("k", lambda: 1) == 1  # 실패는 남기지 않는다
    assert sh.get("k", lambda: 2) == 1
    now[0] = 6
    assert sh.get("k", lambda: 3) == 3  # ttl 지나면 새로
    assert sh.get("j", lambda: 4) == 4 and sh.get("k", lambda: 5) == 5  # size 1


def test_decode_rgb_same_pixels_and_read_only():
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(buf, "JPEG")
    j = buf.getvalue()
    a, b = lr.decode_rgb(j), lr.decode_rgb(j)
    assert a is b and not a.flags.writeable
    assert np.array_equal(a, np.asarray(Image.open(io.BytesIO(j)).convert("RGB")))


def test_big_upload_prepared_once(monkeypatch):
    calls = []
    real = server.prepare_image
    monkeypatch.setattr(server, "prepare_image", lambda d, m: calls.append(m) or real(d, m))
    monkeypatch.setattr(server, "SHARE_MIN_BYTES", 0)
    server._prepared.clear()
    data = _jpeg((80, 80))
    assert server._prepare_shared(data, 4080) == server._prepare_shared(data, 4080)
    assert calls == [4080]
    server._prepare_shared(data, 3072)  # 크기가 다르면 따로
    assert calls == [4080, 3072]
    server._prepared.clear()


def test_locallast3_costs_one_token(monkeypatch):
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(per_min=60, burst=2))

    class Reader:
        def read_rivers_app(self, jpegs):
            return {"n": len(jpegs)}

    monkeypatch.setattr(server, "_get_local_reader", lambda: Reader())
    with _client(monkeypatch) as client:
        files = [("images", (f"{i}.jpg", _jpeg(), "image/jpeg")) for i in range(4)]
        r = client.post("/v2/locallast3", files=files)
        assert r.status_code == 200 and r.json() == {"n": 4}  # 사진 4장 = 요청 1
        assert client.post("/v2/locallast3", files=files[:1]).status_code == 200
        assert client.post("/v2/locallast3", files=files[:1]).status_code == 429
