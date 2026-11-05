"""같은 사진을 여러 요청이 받을 때 무거운 계산을 한 번만(2026-11-05, mahjong_calc metrics 표 1-84·1-86).

앱은 화료 사진 원본을 /v2/localroles 와 /v2/localdora 에 거의 동시에 보낸다. 둘 다 받을 때 12MP 디코딩 + JPEG 재인코딩
(`prepare_image`)을 하고 판독 함수가 그것을 다시 디코딩해, 2 vCPU 운영 서버에서 같은 일을 두 번씩 했다. 같은 바이트면
결과도 같으므로 먼저 온 요청이 계산하고, 동시에 들어온 요청은 그것을 기다려 받는다. 답은 바뀌지 않는다.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from typing import Any


def digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


class Shared:
    """열쇠별 계산을 한 번만. 같은 열쇠가 계산 중에 들어오면 그 결과를 기다린다. 최근 [size] 개를 [ttl] 초 동안 들고 있다
    (사진 원본 배열은 36MB 라 작게). 계산이 실패하면 남기지 않고 기다리던 쪽에도 같은 예외를 던진다."""

    def __init__(self, size: int, ttl: float, clock: Callable[[], float] = time.monotonic):
        self.size, self.ttl, self.clock = size, ttl, clock
        self.lock = threading.Lock()
        self.items: OrderedDict[Hashable, tuple[float, Future]] = OrderedDict()
        self.hits = 0

    def get(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        now = self.clock()
        with self.lock:
            for k in [k for k, (t, _) in self.items.items() if now - t > self.ttl]:
                del self.items[k]
            hit = self.items.get(key)
            if hit is not None:
                self.items.move_to_end(key)
                self.hits += 1
                fut, own = hit[1], False
            else:
                fut, own = Future(), True
                self.items[key] = (now, fut)
                while len(self.items) > self.size:
                    self.items.popitem(last=False)
        if own:
            try:
                fut.set_result(fn())
            except BaseException as e:
                fut.set_exception(e)
                with self.lock:
                    if key in self.items and self.items[key][1] is fut:
                        del self.items[key]
        return fut.result()

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
