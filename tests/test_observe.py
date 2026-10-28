from fastapi.testclient import TestClient

from app import main as server
from app.normalize import normalize_locate
from app.observe import (
    agree_seats,
    last_multi_prompt,
    last_multi_schema,
    normalize_center_v2,
    normalize_corner_v2,
    normalize_dora_v2,
    normalize_hand_v2,
    normalize_last_multi,
    normalize_last_v2,
    normalize_meld_v2,
    normalize_river_v2,
    normalize_seat_v2,
    normalize_sticks_v2,
)
from app.recognizer import GeminiCall
from tests.test_api import FakeRecognizer, _upload


def test_hand_v2_keeps_raw_melds_and_separate_tile():
    out = normalize_hand_v2({
        "tiles": [{"code": "3s", "confidence": 0.9}, {"code": "xx", "confidence": 1}],
        "separate_tile": {"code": "2s", "confidence": 0.8},
        "melds": [
            {"tiles": ["6z", "6z", "6z"], "sideways": [1, 9], "face_down": [], "confidence": 0.9},
            {"tiles": ["1m", "1m"], "sideways": [], "face_down": [], "confidence": 0.9},  # 2장: 제외
            {"tiles": ["4p", "4p", "4p", "4p"], "sideways": [], "face_down": [0, 3], "confidence": 0.9},
        ],
        "notes": "",
    })
    assert out["tiles"] == [{"tile": "3s", "conf": 0.9}]
    assert out["separate_tile"] == {"tile": "2s", "conf": 0.8}
    assert [m["tiles"] for m in out["melds"]] == [["6z", "6z", "6z"], ["4p", "4p", "4p", "4p"]]
    assert out["melds"][0]["sideways"] == [1]          # 범위 밖 9 는 버림
    assert out["melds"][1]["face_down"] == [0, 3]
    assert normalize_hand_v2({"tiles": []})["separate_tile"] is None


def test_river_v2_order_and_sideways():
    out = normalize_river_v2({
        "tiles": [{"code": "1z", "sideways": False, "confidence": 0.9},
                  {"code": "8s", "sideways": True, "confidence": 0.8},
                  {"code": "bad", "sideways": True}],
        "riichi_stick": True,
    })
    assert [(t["tile"], t["sideways"]) for t in out["tiles"]] == [("1z", False), ("8s", True)]
    assert out["riichi_stick"] is True


def test_center_v2_all_sides():
    out = normalize_center_v2({
        "indicators": [{"code": "9s", "confidence": 0.9}, {"code": "5s", "confidence": 0.9}],
        "riichi_sticks": [{"side": "right", "present": True, "confidence": 0.9}, {"side": "nope", "present": True}],
    })
    assert [t["tile"] for t in out["indicators"]] == ["9s", "5s"]
    assert out["riichi_sticks"] == {"bottom": False, "right": True, "top": False, "left": False}


class FakeObserver(FakeRecognizer):
    def __init__(self):
        self.ultra = {}

    def observe(self, jpeg: bytes, kind: str, ultra_high: bool = False, melds: int = 0) -> GeminiCall:
        self.ultra[kind] = ultra_high
        self.melds = melds
        data = {"river": {"tiles": [{"tile": "2s", "conf": 0.9, "sideways": False}], "riichi_stick": False, "notes": ""}}
        return GeminiCall(data.get(kind, {"notes": ""}), 20, 7, 3, "fake")


def test_v2_endpoints_return_usage(monkeypatch):
    monkeypatch.setattr(server, "recognizer", FakeObserver())
    monkeypatch.setattr(server, "API_KEY", "")
    client = TestClient(server.app)
    r = client.post("/v2/river", files=_upload())
    assert r.status_code == 200
    j = r.json()
    assert j["tiles"][0]["tile"] == "2s"
    assert j["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_meld_v2_and_locate_melds():
    assert normalize_meld_v2({"tiles": ["6z", "6z", "6z"], "sideways": [2], "face_down": [], "confidence": 0.9})["tiles"] == ["6z"] * 3
    assert normalize_meld_v2({"tiles": ["6z", "6z"], "sideways": [], "face_down": []})["tiles"] == []
    loc = normalize_locate({
        "hands": [], "rivers": [], "dead_wall": None, "center": None, "notes": "",
        "melds": [{"side": "top", "box": [20, 300, 60, 380], "confidence": 0.9},
                  {"side": "top", "box": [20, 400, 60, 480], "confidence": 0.8},
                  {"side": "x", "box": [0, 0, 1, 1]}],
    })
    assert [m["box"] for m in loc["melds"]] == [[20, 300, 60, 380], [20, 400, 60, 480]]


def test_corner_v2_melds_and_loose():
    out = normalize_corner_v2({
        "melds": [{"tiles": ["6m", "7m", "8m"], "sideways": [0], "face_down": [], "confidence": 0.9},
                  {"tiles": ["1z"], "sideways": [], "face_down": []}],  # 1장: 버림
        "loose": [{"code": "2p", "confidence": 0.8}, {"code": "?"}],
        "notes": "",
    })
    assert [m["tiles"] for m in out["melds"]] == [["6m", "7m", "8m"]]
    assert "notes" not in out["melds"][0]
    assert out["loose"] == [{"tile": "2p", "conf": 0.8}]


def test_last_v2_tile_or_none():
    assert normalize_last_v2({"last_tile": {"code": "7m", "confidence": 0.9}, "tile_count": 14})["last_tile"] == {
        "tile": "7m", "conf": 0.9}
    out = normalize_last_v2({"last_tile": None, "tile_count": "x"})
    assert out["last_tile"] is None and out["tile_count"] == 0


def test_sticks_v2_has_no_indicators():
    out = normalize_sticks_v2({"riichi_sticks": [{"side": "top", "present": True, "confidence": 0.8}]})
    assert out["riichi_sticks"] == {"bottom": False, "right": False, "top": True, "left": False}
    assert "indicators" not in out


def test_new_v2_routes(monkeypatch):
    monkeypatch.setattr(server, "recognizer", FakeObserver())
    monkeypatch.setattr(server, "API_KEY", "")
    client = TestClient(server.app)
    for path in ("/v2/corner", "/v2/last", "/v2/sticks", "/v2/dora", "/v2/seat", "/v2/handonly"):
        assert client.post(path, files=_upload()).status_code == 200
    # 자리 구역만 이미지 해상도를 ULTRA_HIGH 로 읽는다.
    assert server.recognizer.ultra == {"corner": False, "last": False, "sticks": False, "dora": False, "seat": True,
                                       "handonly": True}


def test_last_multi_order_and_missing():
    out = normalize_last_multi({"photo1": {"last_tile": {"code": "6s", "confidence": 0.9}, "tile_count": 17},
                                "photo3": {"last_tile": None, "tile_count": 5}, "notes": "n"}, 3)
    assert [r["last_tile"] and r["last_tile"]["tile"] for r in out["rivers"]] == ["6s", None, None]
    assert [r["tile_count"] for r in out["rivers"]] == [17, 0, 5]
    assert out["notes"] == "n"


def test_last_multi_schema_and_prompt_follow_count():
    s = last_multi_schema(3)
    assert s["required"] == ["photo1", "photo2", "photo3", "notes"]
    assert "photo4" not in s["properties"]
    p = last_multi_prompt(3)
    assert "사진 3장" in p and "첫째·둘째·셋째 사진의 결과" in p
    # 참고표는 강 크롭 뒤 마지막 사진
    assert "넷째 사진은 이 패 세트의" in p and "photo1~photo3" in p
    assert "다섯째 사진은 이 패 세트의" in last_multi_prompt(4)


class FakeLastMulti(FakeObserver):
    def observe_last_multi(self, jpegs: list[bytes]) -> GeminiCall:
        self.n = len(jpegs)
        rivers = [{"last_tile": {"tile": t, "conf": 0.9}, "tile_count": 10} for t in ("6s", "3s", "7z", "1z")[:len(jpegs)]]
        return GeminiCall({"rivers": rivers, "notes": ""}, 30, 9, 4, "fake")


def test_last3_route(monkeypatch):
    monkeypatch.setattr(server, "recognizer", FakeLastMulti())
    monkeypatch.setattr(server, "API_KEY", "")
    client = TestClient(server.app)
    up = [("images", (f"last_{s}.jpg", _upload()["image"][1], "image/jpeg")) for s in ("bottom", "left", "top")]
    r = client.post("/v2/last3", files=up)
    assert r.status_code == 200
    j = r.json()
    assert server.recognizer.n == 3
    assert [x["last_tile"]["tile"] for x in j["rivers"]] == ["6s", "3s", "7z"]
    assert j["usage"] == {"input_tokens": 9, "output_tokens": 4}
    # 5장 이상은 거절
    assert client.post("/v2/last3", files=up + up[:2]).status_code == 400


def test_dora_v2_indicators():
    out = normalize_dora_v2({"indicators": [{"code": "4p", "confidence": 0.9}, {"code": "7m", "confidence": 0.8}, {}]})
    assert [t["tile"] for t in out["indicators"]] == ["4p", "7m"]


def test_seat_v2_face_down_clears_hand():
    data = {
        "face_up": True,
        "tiles": [{"code": "1m", "confidence": 0.9}],
        "separate_tile": {"code": "2z", "confidence": 0.9},
        "melds": [{"tiles": ["6m", "7m", "8m"], "sideways": [0], "face_down": [], "confidence": 0.9}],
        "loose": [{"code": "7m", "confidence": 0.8}],
    }
    out = normalize_seat_v2(data)
    assert out["face_up"] and out["tiles"][0]["tile"] == "1m" and out["separate_tile"]["tile"] == "2z"
    assert out["melds"][0]["tiles"] == ["6m", "7m", "8m"] and out["loose"] == [{"tile": "7m", "conf": 0.8}]
    down = normalize_seat_v2({**data, "face_up": False})
    assert down["tiles"] == [] and down["separate_tile"] is None and down["melds"]


def _seat(hand: str, melds: tuple[str, ...] = (), face_up: bool = True, conf: float = 0.8) -> dict:
    return normalize_seat_v2({
        "face_up": face_up,
        "tiles": [{"code": t, "confidence": conf} for t in hand.split()],
        "separate_tile": None,
        "melds": [{"tiles": m.split(), "sideways": [0], "face_down": [], "confidence": conf} for m in melds],
        "loose": [],
    })


def test_agree_seats_unanimous_keeps_highest_conf():
    out = agree_seats([_seat("1s 2s 3s", ("6z 6z 6z",), conf=0.7), _seat("3s 2s 1s", ("6z 6z 6z",), conf=0.95),
                       _seat("1s 2s 3s", ("6z 6z 6z",), conf=0.8)])
    assert out["agreement"] == {"reads": 3, "agree": 3, "unanimous": True, "alternatives": []}
    assert [t["tile"] for t in out["tiles"]] == ["1s", "2s", "3s"]   # 먼저 온 답의 순서
    assert {t["conf"] for t in out["tiles"]} == {0.95} and out["melds"][0]["conf"] == 0.95


def test_agree_seats_majority_zeroes_disputed_tiles():
    # E 장면처럼 셋 중 하나가 3삭을 2삭으로: 다수 답을 고르고, 다른 답에 없는 3삭만 확신도 0.
    out = agree_seats([_seat("1s 2s 2s 7s"), _seat("1s 2s 3s 7s"), _seat("1s 2s 3s 7s")])
    assert [t["tile"] for t in out["tiles"]] == ["1s", "2s", "3s", "7s"]
    assert [t["conf"] for t in out["tiles"]] == [0.8, 0.8, 0.0, 0.8]
    assert out["agreement"]["agree"] == 2 and not out["agreement"]["unanimous"]
    assert out["agreement"]["alternatives"] == ["1s 2s 2s 7s"]


def test_agree_seats_disputed_meld_and_face_down():
    out = agree_seats([_seat("7z 1m", ("9s 9s 9s",)), _seat("7z 1m", ("6s 6s 6s",)), _seat("", face_up=False)])
    assert out["melds"][0]["tiles"] == ["9s", "9s", "9s"] and out["melds"][0]["conf"] == 0.0   # 셋 다 달라 먼저 온 답
    assert [t["conf"] for t in out["tiles"]] == [0.0, 0.0]   # 앞면 아님 답에는 손패가 없다
    assert "(앞면 아님)" in " ".join(out["agreement"]["alternatives"])


def test_zoom_strips_overlap_at_full_resolution():
    import io

    from PIL import Image

    from app.recognizer import zoom_strips
    buf = io.BytesIO()
    Image.new("RGB", (1000, 300), "white").save(buf, format="JPEG")
    sizes = [Image.open(io.BytesIO(s)).size for s in zoom_strips(buf.getvalue(), 3)]
    assert sizes == [(400, 300), (400, 300), (400, 300)]   # 0~400, 300~700, 600~1000


def test_seat_reads_fan_out_and_agree(monkeypatch):
    from app import recognizer as rec
    r = object.__new__(rec.GeminiRecognizer)
    r.seat_reads, r.model = 3, "fake"
    answers = iter(["1s 2s 3s", "1s 2s 3s", "1s 2s 2s"])
    seen = []

    def fake_call(jpeg, prompt, schema, media_resolution=None, extra_images=()):
        seen.append((media_resolution, [res for _, res in extra_images]))
        hand = next(answers)
        return GeminiCall({"face_up": True, "tiles": [{"code": t, "confidence": 0.9} for t in hand.split()],
                           "separate_tile": None, "melds": [], "loose": []}, 10, 100, 20, "fake")

    monkeypatch.setattr(r, "call", fake_call, raising=False)
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (900, 300), "white").save(buf, format="JPEG")
    c = r.observe(buf.getvalue(), "seat", ultra_high=True)
    # 확대 토막 3장은 ULTRA_HIGH, 참고표는 기본 해상도. 세 번 보냄.
    assert seen == [("MEDIA_RESOLUTION_ULTRA_HIGH", ["MEDIA_RESOLUTION_ULTRA_HIGH"] * 3 + [None])] * 3
    assert c.input_tokens == 300 and c.output_tokens == 60
    assert c.data["agreement"]["agree"] == 2 and not c.data["agreement"]["unanimous"]


def test_handonly_passes_meld_count(monkeypatch):
    fake = FakeObserver()
    monkeypatch.setattr(server, "recognizer", fake)
    monkeypatch.setattr(server, "API_KEY", "")
    client = TestClient(server.app)
    assert client.post("/v2/handonly?melds=2", files=_upload()).status_code == 200
    assert fake.melds == 2
    assert client.post("/v2/handonly?melds=9", files=_upload()).status_code == 422  # 울기는 최대 4묶음


def test_handonly_prompt_and_schema_drop_melds():
    """손패 전용 문구: 울기 줄이 빠지고 '이미 따로 읽었다' 는 설명만 남는다. 스키마에도 melds 가 없다."""
    from app import observe as ob
    assert "- melds:" not in ob.HANDONLY_PROMPT_V2 and "- melds:" in ob.SEAT_PROMPT_V2
    assert ob.HANDONLY_PROMPT_V2.count("울기") == 2
    assert "잘라 냈습니다" in ob.HANDONLY_PROMPT_V2
    assert "참고표" in ob.HANDONLY_PROMPT_V2  # 토막·참고표 안내는 그대로
    assert "melds" not in ob.HANDONLY_SCHEMA_V2["properties"] and "melds" not in ob.HANDONLY_SCHEMA_V2["required"]
    out = ob.normalize_handonly_v2({"face_up": True, "tiles": [{"code": "1s", "confidence": 0.9}],
                                    "separate_tile": None, "loose": []})
    assert out["melds"] == [] and [t["tile"] for t in out["tiles"]] == ["1s"]


def test_handonly_fans_out_like_seat(monkeypatch):
    from app import recognizer as rec
    r = object.__new__(rec.GeminiRecognizer)
    r.seat_reads, r.model = 3, "fake"
    answers = iter(["1s 2s 3s", "1s 2s 3s", "1s 2s 3s"])
    prompts = []

    def fake_call(jpeg, prompt, schema, media_resolution=None, extra_images=()):
        prompts.append((prompt, "melds" in schema["properties"]))
        hand = next(answers)
        return GeminiCall({"face_up": True, "tiles": [{"code": t, "confidence": 0.9} for t in hand.split()],
                           "separate_tile": None, "loose": []}, 10, 100, 20, "fake")

    monkeypatch.setattr(r, "call", fake_call, raising=False)
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (900, 300), "white").save(buf, format="JPEG")
    c = r.observe(buf.getvalue(), "handonly", ultra_high=True)
    from app import observe as ob
    assert prompts == [(ob.HANDONLY_PROMPT_V2, False)] * 3
    assert c.data["agreement"]["unanimous"] and c.data["agreement"]["requested"] == 3
    assert c.data["melds"] == [] and [t["tile"] for t in c.data["tiles"]] == ["1s", "2s", "3s"]
