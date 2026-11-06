"""탁자 기준점 잡기·구역 자르기(app/table_locate.py)와 WALL_FRAME 경로(개선 계획 8b "할 일 3")."""
import io
import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import local_read as lr
from app import main as server
from app import roles
from app import table_frame as tf
from app import table_locate as tl


def _scene(c=(1000.0, 800.0), theta=math.radians(12), u=40.0, D=3.2, river=True, wall=True):
    """기기(반폭 D) 둘레에 강 첫 줄(변마다 5장, 세운 앞면)과 패산 줄(눕힌 뒷면, 반폭 3.85·D)을 놓은 회전 상자들 — 사진 좌표."""
    cx, cy = c
    ex, ey = np.array([math.cos(theta), math.sin(theta)]), np.array([-math.sin(theta), math.cos(theta)])
    ang = math.degrees(theta)

    def at(qx, qy):
        p = np.array([cx, cy]) + u * (qx * ex + qy * ey)
        return float(p[0]), float(p[1])

    faces, backs = [], []

    def put(lst, qx, qy, w, h):
        px, py = at(qx, qy)
        rect = ((px, py), (w * u, h * u), ang)
        box = (int(px - w * u / 2), int(py - h * u / 2), int(w * u), int(h * u))
        lst.append((box, rect))

    if river:
        for k in range(5):
            a = -D + 0.5 + k
            put(faces, a, D + 0.7, 1.0, 1.35)  # 아래
            put(faces, -a, -D - 0.7, 1.0, 1.35)  # 위
            put(faces, D + 0.7, -a, 1.35, 1.0)  # 오른쪽
            put(faces, -D - 0.7, a, 1.35, 1.0)  # 왼쪽
    if wall:
        W, b = 3.85 * D, 1 / 0.87  # 눕힌 뒷면 짧은 변 = u / 0.87(프로필 amos)
        for k in range(-5, 6):
            put(backs, k * b, W, b, 1.6)
            put(backs, k * b, -W, b, 1.6)
            put(backs, W, k * b, 1.6, b)
            put(backs, -W, k * b, 1.6, b)
    return faces, backs


def test_locate_finds_device_center_theta_u():
    faces, backs = _scene()
    fr, info = tl.locate(faces, backs, (2000, 1600))
    assert fr is not None and info["source"] == "device", info
    assert math.hypot(fr.cx - 1000, fr.cy - 800) / fr.u < 0.5
    assert abs(math.degrees(fr.theta) - 12) < 0.5 and abs(fr.u - 40) < 1 and 2.7 < fr.D < 3.7
    assert info["wall"]["rows"] >= 1


def test_locate_falls_back_to_wall_then_fails():
    faces, backs = _scene(river=False)
    fr, info = tl.locate(faces, backs, (2000, 1600))
    assert fr is not None and fr.source == "wall" and math.hypot(fr.cx - 1000, fr.cy - 800) / fr.u < 1.0, info
    fr, info = tl.locate([], [], (2000, 1600))
    assert fr is None and info["error"]


def test_locate_rejects_center_outside_photo():
    faces, backs = _scene(c=(1000.0, 800.0))
    fr, info = tl.locate(faces, backs, (900, 700))
    assert fr is None and "밖" in info["error"]


def test_rects_array_accepts_axis_boxes():
    a = tl.rects_array([((10, 20, 30, 40), None), (1, 2, 3, 4)])
    assert a.tolist() == [[25, 40, 30, 40, 0], [2.5, 4, 3, 4, 0]]


def _tz(theta=math.radians(15), W=2000, H=1600):
    return tf.TableZones(tl.table(), tf.TableFrame(1000.0, 800.0, theta, 40.0, 3.2, (float(W), float(H))), W, H)


@pytest.mark.parametrize("side", tf.SIDES)
def test_warp_zone_maps_points_both_ways_and_puts_person_at_bottom(side):
    tz = _tz()
    rgb = np.zeros((1600, 2000, 3), np.uint8)
    zc = tl.warp_zone(rgb, tz, side, (-5.0, 5.0), (2.0, 8.0), clip=False)
    assert zc.rgb.shape[:2] == (round(6 * 40), round(10 * 40))
    # 그 자리 기준 (a, r) 점이 크롭에서 (a − a0, r − r0)·u 로, 되돌리면 원래 점
    for a, r in ((-5, 2), (5, 2), (0, 8), (1.5, 4.5)):
        p = tz.from_frame(*tz.side_to_frame(side, a, r))
        q = zc.to_crop([p])[0]
        assert abs(q[0] - (a + 5) * 40) < 1e-3 and abs(q[1] - (r - 2) * 40) < 1e-3  # getAffineTransform 이 float32
        assert np.allclose(zc.to_orig([q])[0], p, atol=1e-3)
    # 크롭 아래쪽(r 큼) = 그 사람 쪽 = 가운데에서 먼 쪽
    near, far = zc.to_orig([(200, 0)])[0], zc.to_orig([(200, 240)])[0]
    assert np.hypot(*(far - [1000, 800])) > np.hypot(*(near - [1000, 800]))


def test_warp_zone_copies_pixels():
    tz = _tz(theta=0.0)
    rgb = np.zeros((1600, 2000, 3), np.uint8)
    rgb[790:811, 1190:1211] = 255  # 가운데 오른쪽 5u 의 흰 점
    zc = tl.warp_zone(rgb, tz, "right", (-2.0, 2.0), (3.0, 7.0), clip=False)
    ys, xs = np.nonzero(zc.rgb[..., 0] > 128)
    assert abs(xs.mean() - 80) < 2 and abs(ys.mean() - 80) < 2  # (a, r) = (0, 5) → ((0 + 2)·40, (5 − 3)·40)


def test_warp_zone_clips_to_photo():
    tz = _tz(theta=0.0)
    zc = tl.warp_zone(np.zeros((1600, 2000, 3), np.uint8), tz, "bottom", (-100.0, 100.0), (0.0, 100.0))
    assert zc.a == (-25.0, 25.0) and zc.r == (0.0, 20.0)


def test_grid_from_frame_is_center_cell_bounds():
    tz = _tz(theta=0.0)
    g = tl.grid_from_frame(tz)
    R = 3.2 + 6
    assert g["x1"] == round((1000 - R * 40) / 2000 * 1000) and g["x2"] == round((1000 + R * 40) / 2000 * 1000)
    assert g["y1"] == round((800 - R * 40) / 1600 * 1000) and len(g["sticks"]) == 4
    big = tf.TableZones(tl.table(), tf.TableFrame(1000.0, 800.0, 0.0, 200.0, 3.2, (2000.0, 1600.0)), 2000, 1600)
    assert tl.grid_from_frame(big) is None, "가운데 칸이 사진보다 크면 기존 격자로"


def test_tokens_frame_equals_tokens_for_upright_frame():
    """θ = 0 이고 격자가 가운데 칸 그대로면 탁자 좌표 토큰 = 지금 토큰(같은 식)."""
    W, H = 2000, 1600
    tz = _tz(theta=0.0, W=W, H=H)
    R = tz.R
    g = {"x1": (1000 - R * 40) / W * 1000, "x2": (1000 + R * 40) / W * 1000, "y1": (800 - R * 40) / H * 1000, "y2": (800 + R * 40) / H * 1000}
    boxes = [(1100, 1250, 40, 54), (700, 300, 54, 40)]
    pts = [None, np.array([[700, 300], [754, 300], [754, 340], [700, 340]], float)]
    for side in tf.SIDES:
        for feat in (roles.FEAT, roles.FEAT_ROT):
            a = roles.tokens(boxes, W, H, g, side, feat, pts)
            b = roles.tokens_frame(boxes, tz, side, feat, pts)
            assert np.allclose(a, b, atol=1e-4), (side, feat, a, b)


def test_tokens_frame_undoes_photo_rotation():
    """사진이 θ 돌아가도 탁자 좌표 토큰의 크기·기울기는 돌아가지 않은 것과 같다."""
    th = math.radians(20)
    tz0, tz1 = _tz(theta=0.0), _tz(theta=th)
    c = np.array([1000.0, 800.0])
    sq = np.array([[-20, -27], [20, -27], [20, 27], [-20, 27]], float) + [0, 300]
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    p0, p1 = sq + c, sq @ rot.T + c
    box = lambda p: (p[:, 0].min(), p[:, 1].min(), np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    a = roles.tokens_frame([box(p0)], tz0, "bottom", roles.FEAT_ROT, [p0])
    b = roles.tokens_frame([box(p1)], tz1, "bottom", roles.FEAT_ROT, [p1])
    assert np.allclose(a, b, atol=1e-4)


def test_in_dora_zone_and_reach():
    tz = _tz(theta=0.0)
    assert tl.in_dora_zone(tz, (990, 790, 20, 20))  # 가운데
    assert tl.in_dora_zone(tz, (990, 800 + 10 * 40, 20, 20))  # 왕패 칸(10u)
    assert not tl.in_dora_zone(tz, (990, 800 + 15 * 40 - 10, 20, 20))  # 13u 넘음
    assert tl.dora_reach(tz) == 13


def test_mode_reads_env(monkeypatch):
    for v, want in (("", None), ("0", None), ("1", "axis"), ("axis", "axis"), ("ROT", "rot"), ("rot2", "rot2"), ("Hybrid", "hybrid"), ("rot3", None)):
        monkeypatch.setenv("WALL_FRAME", v)
        assert tl.mode() == want


def test_parse_frame_round_trip():
    f = tf.TableFrame(1.0, 2.0, 0.1, 30.0, 3.0, (100.0, 80.0), source="wall")
    assert tl.parse_frame(json.dumps(f.to_json())) == f and tl.parse_frame(None) is None
    with pytest.raises(ValueError):
        tl.parse_frame('{"cx": 1}')


# ---------------------------------------------------------------- 검출 공유 + WALL_FRAME 경로(가짜 검출기)

def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _photo(faces: list, size=(2000, 1600)) -> Image.Image:
    im = Image.new("RGB", size, (30, 120, 60))
    d = ImageDraw.Draw(im)
    for (x, y, w, h), _ in faces:
        d.rectangle((x, y, x + w, y + h), fill=(240, 240, 235))
        d.rectangle((x + w // 4, y + h // 4, x + 3 * w // 4, y + 3 * h // 4), fill=(20, 20, 20))
    return im


class _Net:
    feat = roles.FEAT

    def predict(self, x):
        return ["hand"] * len(x), [0.9] * len(x)


def _fake_reader(monkeypatch, faces, backs, W, H):
    """자리 판독 검출기(회전 상자)를 흉내: 사진 전체면 주어진 상자, 크롭이면 빈 것."""
    r = lr.LocalReader(seat_model=None)
    r._seat = object()
    calls = []

    def raw(rgb):
        calls.append(rgb.shape)
        return (faces, backs, [], {}) if rgb.shape[:2] == (H, W) else ([], [], [], {})

    monkeypatch.setattr(r, "_sessions", lambda: (None, None))
    monkeypatch.setattr(r, "_seat_raw", raw)
    monkeypatch.setattr(r, "detect", lambda rgb, low=True: [])
    return r, calls


def test_read_roles_wall_frame_shares_full_detection(monkeypatch):
    W, H = 2000, 1600
    faces, backs = _scene()
    r, calls = _fake_reader(monkeypatch, faces, backs, W, H)
    jpeg = _jpeg(_photo(faces, (W, H)))
    grid = {"x1": 280, "x2": 745, "y1": 173, "y2": 713}
    for m in ("", "axis", "rot", "rot2"):
        monkeypatch.setenv("WALL_FRAME", m)
        lr._decoded.clear()
        calls.clear()
        out = roles.read_roles(r, jpeg, "bottom", grid, "model", net=_Net())
        assert sum(c[:2] == (H, W) for c in calls) <= 1, f"{m}: 사진 전체 검출은 한 번(공유 캐시)"
        if not m:
            assert "frame" not in out and "wall_frame" not in out
            continue
        assert out["wall_frame"] == m and out["cut"] == ("rot" if m == "rot2" else m) and out["frame"]["source"] == "device", out.get("frame_error")
        assert math.hypot(out["frame"]["cx"] - 1000, out["frame"]["cy"] - 800) < 20 and out["ms"]["frame"] >= 0
        if m == "axis":
            assert out["grid"] != grid and out["grid"]["x1"] < 500 < out["grid"]["x2"]
        # localdora 도 같은 검출을 나눠 쓴다(이미 계산됨)
        n = len(calls)
        d = r.read_wall(jpeg)
        assert len(calls) == n and d["wall_frame"] == m and d["frame"] == out["frame"]
        assert d["cut"] == ("whole" if m == "rot2" else m)  # rot2 표시패는 세워 자르지 않고 사진 전체에서 후보만 구역으로


def test_seat_read_overlaps_neighbor_by_3u_only_on_the_left():
    """rot2 자리 읽기 크롭(구역표 seat_read) = seat 를 그 사람 왼쪽으로 3u 넓힌 것, 나머지 끝·바람개비 구역(zone_of)은 그대로(8b-3b 후보 1)."""
    tz = tf.TableZones(tf.ZoneTable.load(), tf.TableFrame(1000.0, 800.0, 0.0, 40.0, 3.2, (2000.0, 1600.0)), 2000, 1600)
    for side in tf.SIDES:
        s, r = tz.bounds("seat", side), tz.bounds("seat_read", side)
        assert r[0] == pytest.approx(s[0] - 3) and r[1:] == pytest.approx(s[1:])
    assert tz.zone_of(1000 + 40 * (-tz.R - 1), 800 + 40 * (tz.R + 2)) == "left"  # 겹친 띠는 구역으로는 이웃 자리


def test_loose_inside_crop_is_separate_candidate_only_when_given(monkeypatch):
    """크롭 안 loose 상자: inside(후보가 될 수 있는 상자 번호)가 있으면 크롭 밖 loose 와 같은 조건으로 떼어 둔 패 후보(8b-3b 후보 2), 없으면 지금처럼 무시."""
    hand = [{"box": [100 + 30 * i, 500, 28, 40], "role": "hand", "conf": 0.9, "in_seat": True} for i in range(13)]
    loose = {"box": [200, 430, 28, 40], "role": "loose", "conf": 0.9, "in_seat": True}  # 손패 앞(가운데 쪽)에 놓인 쯔모패

    def out():
        return {"framed": True, "separate_tile": None, "tiles": [{}] * 13, "melds": [], "roles": [*hand, dict(loose)]}

    class R:
        def read_any2(self, rgb, b):
            return ("7s", 0.9, "8s", 0.05)

    monkeypatch.setattr(lr, "decode_rgb", lambda jpeg: np.zeros((800, 800, 3), np.uint8))
    o = out()
    roles._loose_outside(R(), b"", o)
    assert o["separate_tile"] is None
    o = out()
    roles._loose_outside(R(), b"", o, inside={13})
    assert o["separate_tile"]["tile"] == "7s" and o["separate_tile"]["from"] == "loose_inside"
    o = out()
    roles._loose_outside(R(), b"", o, inside=set())  # 틀에 썼거나 혼자가 아니면(_inside_ok 에 없음) 후보 아님
    assert o["separate_tile"] is None
    o = out()
    o["roles"].append({"box": [500, 470, 28, 40], "role": "loose", "conf": 0.9, "in_seat": False})  # 크롭 밖 것과 합쳐 둘 → 안 고름
    roles._loose_outside(R(), b"", o, inside={13})
    assert o["separate_tile"] is None


def test_read_roles_wall_frame_uses_given_frame_and_falls_back(monkeypatch):
    W, H = 2000, 1600
    r, calls = _fake_reader(monkeypatch, [], [], W, H)  # 검출 상자 없음 → 기준점 못 잡음
    jpeg = _jpeg(_photo([], (W, H)))
    grid = {"x1": 280, "x2": 745, "y1": 173, "y2": 713}
    monkeypatch.setenv("WALL_FRAME", "rot")
    out = roles.read_roles(r, jpeg, "bottom", grid, "model", net=_Net())
    assert out["cut"] == "grid" and out["frame_error"] and "frame" not in out
    given = tf.TableFrame(1000.0, 800.0, 0.1, 40.0, 3.2, (4000.0, 3200.0), source="wall")  # 다른 크기에서 잰 것 → 옮겨 씀
    out = roles.read_roles(r, jpeg, "bottom", grid, "model", net=_Net(), frame=given)
    assert out["cut"] == "rot" and out["frame_info"]["given"] and out["frame"]["cx"] == 500 and out["frame"]["u"] == 20


def test_routes_accept_frame_and_full_photo_last3(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "GEMINI_API_KEY", "")
    monkeypatch.setattr(server, "recognizer", object())
    monkeypatch.setattr(roles, "_role_net", _Net())
    W, H = 800, 600
    r = lr.LocalReader(seat_model=None)
    monkeypatch.setattr(r, "detect", lambda rgb, low=True: [])
    monkeypatch.setattr(server, "_local_reader", r)
    client = TestClient(server.app)
    files = {"image": ("t.jpg", _jpeg(Image.new("RGB", (W, H), (30, 120, 60))), "image/jpeg")}
    base = {"side": "bottom", "x1": 280, "x2": 745, "y1": 173, "y2": 713}
    assert client.post("/v2/localroles", files=files, params={**base, "frame": "{bad"}).status_code == 422
    fj = json.dumps(tf.TableFrame(400.0, 300.0, 0.0, 15.0, 3.2, (W, H)).to_json())
    monkeypatch.setenv("WALL_FRAME", "rot")
    j = client.post("/v2/localroles", files=files, params={**base, "frame": fj}).json()
    assert j["cut"] == "rot" and j["frame"]["cx"] == 400
    j = client.post("/v2/localdora", files=files, params={"frame": fj}).json()
    assert j["cut"] == "rot" and j["indicators"] == []
    many = [("images", ("a.jpg", files["image"][1], "image/jpeg"))] * 2
    assert client.post("/v2/locallast3", files=many, params={"sides": "right"}).status_code == 422
    assert client.post("/v2/locallast3", files=[many[0]], params={"sides": "north"}).status_code == 422
    j = client.post("/v2/locallast3", files=[many[0]], params={"sides": "right,top,left", "frame": fj}).json()
    assert j["cut"] == "rot" and len(j["rivers"]) == 3 and j["sides"] == ["right", "top", "left"]
    monkeypatch.setenv("WALL_FRAME", "")
    j = client.post("/v2/locallast3", files=[many[0]], params={"sides": "left", "x1": 280, "x2": 745, "y1": 173, "y2": 713}).json()
    assert len(j["rivers"]) == 1 and "cut" not in j


# ---------------------------------------------------------------- WALL_FRAME=hybrid(8b-3c): 앱이 격자를 가졌는지(grid_src)로 경로

def _tz1000(theta_deg=0.0, u=10.0):
    return tf.TableZones(tl.table(), tf.TableFrame(500.0, 500.0, math.radians(theta_deg), u, 3.2, (1000.0, 1000.0)), 1000, 1000)


def test_grid_check_is_diagnostic_distance_and_theta():
    """grid_check 는 진단만(경로를 안 정함): 격자 가운데 칸 중심 ↔ 기준점 거리(기준점 u)·dx·dy·θ(90° 주기)."""
    c = tl.grid_check(_tz1000(), {"x1": 400, "x2": 622, "y1": 400, "y2": 600})
    assert c == {"dist_u": pytest.approx(1.1), "dx_u": pytest.approx(1.1), "dy_u": pytest.approx(0), "theta_deg": pytest.approx(0)}
    assert tl.grid_check(_tz1000(u=20.0), {"x1": 400, "x2": 622, "y1": 400, "y2": 600})["dist_u"] == pytest.approx(0.55)
    for th, want in ((2.9, 2.9), (-3.1, 3.1), (87.1, 2.9), (92.9, 2.9)):
        assert tl.grid_check(_tz1000(th), {"x1": 400, "x2": 600, "y1": 400, "y2": 600})["theta_deg"] == pytest.approx(want)
    assert "match" not in c


GOOD = {"x1": 280, "x2": 720, "y1": 173, "y2": 827}


@pytest.mark.parametrize("query, env, grid, want, src", [
    ("saved", "", GOOD, "saved", ["saved", "query"]),
    ("manual", "", GOOD, "saved", ["manual", "query"]),
    ("felt", "", GOOD, "saved", ["felt", "query"]),
    ("none", "", GOOD, "rot2", ["none", "query"]),
    ("default", "", GOOD, "rot2", ["default", "query"]),
    (None, "saved", GOOD, "saved", ["saved", "env"]),
    (None, "none", GOOD, "rot2", ["none", "env"]),
    ("none", "saved", GOOD, "rot2", ["none", "query"]),  # 요청이 env 보다 먼저
    (None, "", GOOD, "saved", ["saved", "grid"]),  # grid_src 없이 격자만(지금 앱) → 지금 경로
    (None, "", None, "saved", ["saved", "no_grid"]),  # 8b-4a: 격자 질의도 grid_src 도 없음(옛 앱 localdora) → 지금 경로
    (None, "bogus", None, "saved", ["saved", "no_grid"]),  # 모르는 env 는 무시
    ("none", "", None, "rot2", ["none", "query"]),  # 격자 없이 명시적 none(갤러리 사진) → rot2
    ("default", "saved", None, "rot2", ["default", "query"]),  # 명시적 default 가 env saved 보다 먼저
    ("felt", "none", None, "saved", ["felt", "query"]),  # 명시적 felt 가 env none 보다 먼저
])
def test_route_hybrid_by_grid_src(monkeypatch, query, env, grid, want, src):
    W, H = 2000, 1600
    faces, backs = _scene()
    r, _ = _fake_reader(monkeypatch, faces, backs, W, H)
    jpeg = _jpeg(_photo(faces, (W, H)))
    monkeypatch.setenv("WALL_FRAME", "hybrid")
    monkeypatch.setenv("WALL_FRAME_GRID_SRC", env)
    lr._decoded.clear()
    m, tz, meta = tl.route(r, jpeg, lr.decode_rgb(jpeg), None, grid, query)
    assert meta["wall_frame"] == want and meta["wall_frame_src"] == src and meta["wall_frame_mode"] == "hybrid"
    assert meta["frame"]["source"] == "device", "기준점은 지금 경로에서도 잡아 응답에 넣는다"
    assert (m, tz is None) == ((None, True) if want == "saved" else ("rot2", False))
    assert ("wall_frame_check" in meta) is (grid is not None)


def test_read_roles_hybrid_saved_equals_off_and_rot2_otherwise(monkeypatch):
    """hybrid saved = WALL_FRAME 없음과 같은 답(cut grid, 응답에 기준점), none → rot2(cut rot). localdora 도 같은 판정(격자 없으면 rot2 = whole)."""
    W, H = 2000, 1600
    strip = lambda o: {k: v for k, v in o.items() if k in ("tiles", "melds", "separate_tile", "framed", "face_up", "roles", "counts", "filled")}
    faces, backs = _scene()
    r, _ = _fake_reader(monkeypatch, faces, backs, W, H)
    jpeg = _jpeg(_photo(faces, (W, H)))
    monkeypatch.setenv("WALL_FRAME", "")
    lr._decoded.clear()
    base = roles.read_roles(r, jpeg, "bottom", GOOD, "model", net=_Net())
    base_d = r.read_wall(jpeg)
    monkeypatch.setenv("WALL_FRAME", "hybrid")
    lr._decoded.clear()
    out = roles.read_roles(r, jpeg, "bottom", GOOD, "model", net=_Net(), grid_src="saved")
    assert out["wall_frame"] == "saved" and out["cut"] == "grid" and strip(out) == strip(base) and out["frame"]["source"] == "device"
    assert out["wall_frame_check"]["theta_deg"] == pytest.approx(12, abs=0.5)
    out = roles.read_roles(r, jpeg, "bottom", GOOD, "model", net=_Net(), grid_src="none")
    assert out["wall_frame"] == "rot2" and out["cut"] == "rot"
    d = r.read_wall(jpeg)  # 옛 앱은 localdora 에 격자도 grid_src 도 안 보낸다 → 8b-4a 부터 지금 경로(답 = WALL_FRAME 없음)
    assert d["wall_frame"] == "saved" and d["cut"] == "grid" and d["wall_frame_src"] == ["saved", "no_grid"]
    assert d["indicators"] == base_d["indicators"] and d["pairs"] == base_d["pairs"]
    d = r.read_wall(jpeg, grid=GOOD, grid_src="none")  # 새 앱이 격자 없음을 알림 → rot2
    assert d["wall_frame"] == "rot2" and d["cut"] == "whole" and d["wall_frame_src"] == ["none", "query"]
    monkeypatch.setenv("WALL_FRAME_GRID_SRC", "saved")  # 측정용 강제
    d = r.read_wall(jpeg)
    assert d["wall_frame"] == "saved" and d["cut"] == "grid" and d["indicators"] == base_d["indicators"] and d["pairs"] == base_d["pairs"]


def test_hybrid_falls_back_to_grid_when_no_frame_and_last3_without_grid_is_rot2(monkeypatch):
    W, H = 2000, 1600
    r, _ = _fake_reader(monkeypatch, [], [], W, H)  # 기준점 못 잡음
    jpeg = _jpeg(_photo([], (W, H)))
    monkeypatch.setenv("WALL_FRAME", "hybrid")
    out = roles.read_roles(r, jpeg, "bottom", GOOD, "model", net=_Net(), grid_src="none")
    assert out["wall_frame"] == "none" and out["cut"] == "grid" and out["frame_error"]
    faces, backs = _scene()
    r, _ = _fake_reader(monkeypatch, faces, backs, W, H)
    jpeg = _jpeg(_photo(faces, (W, H)))
    j = r.read_rivers_full(jpeg, ["right"])  # 8b-4a: 출처 모름 → 지금 경로(격자도 없으면 강이 빔 = WALL_FRAME 없음과 같음)
    assert j["wall_frame"] == "saved" and j["cut"] == "grid" and j["rivers"] == [{"last_tile": None, "tile_count": 0, "sideways": []}]
    j = r.read_rivers_full(jpeg, ["right"], grid_src="none")
    assert j["wall_frame"] == "rot2" and j["cut"] == "rot"
    j = r.read_rivers_full(jpeg, ["right"], GOOD, grid_src="none")  # 격자를 같이 보내도 none 이면 rot2
    assert j["wall_frame"] == "rot2" and j["cut"] == "rot"
    j = r.read_rivers_full(jpeg, ["right"], GOOD, grid_src="felt")
    assert j["wall_frame"] == "saved" and j["cut"] == "grid"


def test_routes_accept_grid_src(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "GEMINI_API_KEY", "")
    monkeypatch.setattr(server, "recognizer", object())
    monkeypatch.setattr(roles, "_role_net", _Net())
    r = lr.LocalReader(seat_model=None)
    monkeypatch.setattr(r, "detect", lambda rgb, low=True: [])
    monkeypatch.setattr(server, "_local_reader", r)
    client = TestClient(server.app)
    files = {"image": ("t.jpg", _jpeg(Image.new("RGB", (800, 600), (30, 120, 60))), "image/jpeg")}
    base = {"side": "bottom", **GOOD}
    monkeypatch.setenv("WALL_FRAME", "hybrid")
    assert client.post("/v2/localroles", files=files, params={**base, "grid_src": "maybe"}).status_code == 422
    assert client.post("/v2/localdora", files=files, params={"grid_src": "maybe"}).status_code == 422
    j = client.post("/v2/localroles", files=files, params={**base, "grid_src": "Manual"}).json()
    assert j["wall_frame"] == "saved" and j["wall_frame_src"] == ["manual", "query"]
    j = client.post("/v2/localdora", files=files, params={**GOOD, "grid_src": "default"}).json()
    assert j["wall_frame_src"] == ["default", "query"] and j["wall_frame"] == "none"  # 빈 사진 = 기준점 못 잡음 → 지금 경로


def test_last_zone_matches_app_grid_zones():
    g = {"x1": 280, "x2": 745, "y1": 173, "y2": 713}  # GridZones.initial
    # Dart: cx = 1025 ~/ 2 = 512, cy = 443, hw = round(465·0.3) = 140, hh = round(540·0.3) = 162
    assert lr.last_zone(g, "bottom") == (443, 372, 713, 652)
    assert lr.last_zone(g, "top") == (173, 372, 443, 652)
    assert lr.last_zone(g, "left") == (281, 280, 605, 512)
    assert lr.last_zone(g, "right") == (281, 512, 605, 745)
    im = Image.new("RGB", (1000, 1000))
    assert lr.last_zone_crop(im, g, "right").size == (605 - 281, 745 - 512)
