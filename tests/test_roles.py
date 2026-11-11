import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import local_read as lr
from app import main as server
from app import roles
from app import seat as st


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _table(boxes: list, size=(2000, 1500)) -> Image.Image:
    im = Image.new("RGB", size, (30, 120, 60))
    d = ImageDraw.Draw(im)
    for x, y, w, h in boxes:
        d.rectangle((x, y, x + w, y + h), fill=(240, 240, 235))
        d.rectangle((x + 15, y + 20, x + w - 15, y + h - 20), fill=(20, 20, 20))  # 흰 비율 0.75(keep_faces 0.5 이상)
    return im


GRID = {"x1": 280, "x2": 745, "y1": 173, "y2": 713}


def test_unrotate_matches_pil_rotation():
    a = np.arange(6 * 4).reshape(4, 6).astype(np.uint8)
    for rot in (90, 180, 270):
        r = np.asarray(Image.fromarray(a).rotate(-rot, expand=True))
        for yy in range(r.shape[0]):
            for xx in range(r.shape[1]):
                x0, y0, _, _ = roles.unrotate((xx, yy, 1, 1), rot, 6, 4)
                assert a[int(y0), int(x0)] == r[yy, xx]


def test_crop_seat_boxes_map_back_to_full_photo():
    """자리 크롭에서 검출한 상자를 원본 좌표로 되돌리면 원래 상자와 같다(네 자리 모두)."""
    W, H = 2000, 1500
    for side in roles.SIDES:
        ymin, xmin, ymax, xmax = roles.seat_box(GRID, side)
        bx = (round((xmin + 30) * W / 1000), round((ymin + 30) * H / 1000), 60, 80)
        im = Image.new("RGB", (W, H))
        crop, (x0, y0, cw, ch) = roles.crop_seat(im, GRID, side)
        # 원본 상자 → 크롭(돌린) 좌표: PIL 로 실제 돌려 찾는다
        mask = Image.new("L", (W, H))
        ImageDraw.Draw(mask).rectangle((bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]), fill=255)
        m = mask.crop((x0, y0, x0 + cw, y0 + ch))
        if roles.ROT_CW[side]:
            m = m.rotate(-roles.ROT_CW[side], expand=True)
        ys, xs = np.nonzero(np.asarray(m))
        cb = (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())
        x, y, w, h = roles.unrotate(cb, roles.ROT_CW[side], cw, ch)
        assert abs(x + x0 - bx[0]) <= 2 and abs(y + y0 - bx[1]) <= 2 and abs(w - bx[2]) <= 2 and abs(h - bx[3]) <= 2, side


def test_tokens_turn_winner_to_bottom():
    W, H = 2000, 1500
    for side in roles.SIDES:
        ymin, xmin, ymax, xmax = roles.seat_box(GRID, side, 0)
        # 자리 띠 한가운데 패: 돌리면 아래쪽(v ≈ 1 보다 큼), 가로 폭이 세로보다 작다(세운 패)
        cx, cy = (xmin + xmax) / 2 * W / 1000, (ymin + ymax) / 2 * H / 1000
        w, h = (60, 80) if side in ("bottom", "top") else (80, 60)
        t = roles.tokens([(cx - w / 2, cy - h / 2, w, h)], W, H, GRID, side)[0]
        assert t[1] > 1.0 and 0 < t[0] < 1.3 and t[2] < t[3] and t[5] == 0.0 and t[4] == 0.0 and t[8] == 1.0, side
    assert roles.tokens([], W, H, GRID, "bottom").shape == (0, roles.FEAT)


def test_frame_from_roles_model_mode_groups_hand_sep_and_two_row_melds():
    hand = [(100 + 66 * i, 400, 60, 80) for i in range(7)]
    sep = (100 + 66 * 7 + 50, 402, 60, 80)
    melds = [(1500 + 66 * i, 400, 60, 80) for i in range(3)] + [(1500 + 66 * i, 490, 60, 80) for i in range(3)]
    other = [(600, 10, 60, 80)]
    faces = hand + [sep] + melds + other
    rl = ["hand"] * 7 + ["loose"] + ["meld"] * 6 + ["river"]
    rgb = np.asarray(_table(faces, (2000, 800)))
    fr, added = roles.frame_from_roles(rgb, faces, rl, 1300, "model")
    assert fr is not None and [b[0] for b in fr.hand] == sorted(b[0] for b in hand) and fr.separate == sep
    assert len(fr.melds) == 2 and fr.melds[0][0][1] == 400 and fr.melds[1][0][1] == 490, "위아래 두 줄은 y 로 나눈다"
    assert fr.count == 14 and added == []


def test_split_meld_group_stacked_rows_with_sagging_sideways_tiles():
    """11-07 135420 왼쪽 자리 크롭 상자 그대로: 울기 셋을 3×3 으로 붙여 쌓음, 눕힌 패가 줄마다 아래로 처짐 → 줄별 3장씩."""
    rows = [[(1814, 391, 122, 89), (1933, 331, 93, 113), (2023, 333, 88, 109)],
            [(1824, 482, 119, 86), (1940, 449, 87, 110), (2025, 447, 88, 110)],
            [(1822, 590, 118, 84), (1936, 564, 89, 109), (2024, 559, 87, 110)]]
    assert st.split_meld_group([b for r in rows for b in r], 110.0) == rows
    line9 = [(100 + 66 * i, 400, 60, 80) for i in range(9)]
    assert st.split_meld_group(line9, 80.0) == [line9[0:3], line9[3:6], line9[6:9]], "한 줄 9장은 왼쪽부터 3장씩"
    stack12 = [(1500 + 66 * i, 400 + 90 * j, 60, 80) for j in range(4) for i in range(3)]
    assert len(st.split_meld_group(stack12, 80.0)) == 4, "3장 네 줄"
    assert st.split_meld_group([(100 + 66 * i, 400, 60, 80) for i in range(12)], 80.0) is None, "한 줄 12장은 못 가림"
    assert st.split_meld_group([(100 + 66 * i, 400, 60, 80) for i in range(7)], 80.0) is None, "한 줄 7장은 3+4·4+3 을 못 가림"
    # 11-07 115802 오른쪽 자리 크롭 상자 그대로: 3m 퐁(위 줄 3장) + 1m 명깡(아래 줄 4장)을 붙여 둠
    pon = [(1987, 621, 124, 87), (2109, 586, 96, 117), (2198, 589, 92, 118)]
    kan = [(1890, 708, 97, 111), (1983, 710, 96, 113), (2073, 714, 95, 113), (2165, 742, 129, 90)]
    assert st.split_meld_group(pon + kan, 113.0) == [pon, kan], "퐁 + 깡 위아래 → 3장·4장"


def _rd(tiles, conf=0.95, alts=None):
    alts = alts or [None] * len(tiles)
    return [(t, conf, a, 0.03 if a else None) for t, a in zip(tiles, alts)]


def test_choose_melds_line_kan_and_pon_by_tiles():
    """한 줄로 붙은 7장(모양 규칙으로는 못 나눔): 읽은 패로 퐁 + 깡을 가림 — 순서가 어느 쪽이든."""
    line7 = [(100 + 66 * i, 400, 60, 80) for i in range(7)]
    parts, _, flag = st.choose_melds(line7, _rd(["3m"] * 3 + ["1m"] * 4), 60, 80, 2)
    assert flag == "ok" and parts == [[0, 1, 2], [3, 4, 5, 6]]
    parts, _, flag = st.choose_melds(line7, _rd(["1m"] * 4 + ["2s", "3s", "4s"]), 60, 80, 2)
    assert flag == "ok" and parts == [[0, 1, 2, 3], [4, 5, 6]]
    assert st.meld_parts(line7, 80) is not None, "틀 잡기에서는 임시 나눔으로 남긴다"


def test_choose_melds_guardrails():
    line6 = [(100 + 66 * i, 400, 60, 80) for i in range(6)]
    # 1순위로는 울기 모양이 안 되고(4m 자리에 9p), 2순위(4m)로 바꾸면 됨 → alt, 그 패는 2순위 값·확신
    rd = _rd(["2m", "3m", "9p", "5s", "5s", "5s"], alts=[None, None, "4m", None, None, None])
    rd[2] = ("9p", 0.4, "4m", 0.3)
    parts, used, flag = st.choose_melds(line6, rd, 60, 80, 2)
    assert flag == "alt" and used[2][:2] == ("4m", 0.3)
    # 2순위로도 안 됨 → 모양 규칙(3장씩) 그대로 + fallback
    parts, _, flag = st.choose_melds(line6, _rd(["2m", "9p", "7s", "5s", "1z", "5s"]), 60, 80, 2)
    assert flag == "fallback" and parts == [[0, 1, 2], [3, 4, 5]]
    # 퐁 + 깡 위아래, 깡 한 장을 퐁 패로 잘못 읽음: 줄 한가운데를 자르는 나눔(3z×4 + 6z×3)은 받지 않는다
    pon = [(1500 + 66 * i, 400, 60, 80) for i in range(3)]
    kan = [(1480 + 66 * i, 490, 60, 80) for i in range(4)]
    _, _, flag = st.choose_melds(pon + kan, _rd(["3z"] * 3 + ["3z", "6z", "6z", "6z"]), 60, 80, 2)
    assert flag != "ok"


def test_resplit_melds_lowers_conf_when_guarded():
    hand = [(100 + 66 * i, 400, 60, 80) for i in range(7)]
    line7 = [(1500 + 66 * i, 400, 60, 80) for i in range(7)]
    fr = st.SeatFrame(hand, None, st.meld_parts(line7, 80), [], 60, 80, 14, split=True)
    tiles = dict(zip(line7, ["3m"] * 3 + ["1m"] * 4))
    boxes, reads, flags = lr.resplit_melds(fr, lambda b: (tiles[tuple(b)], 0.95, None, None))
    assert [len(m) for m in boxes] == [3, 4] and flags == ["ok", "ok"] and fr.count == 13
    assert lr.meld_conf(reads[0], "ok") == 0.95 and lr.meld_conf(reads[0], "fallback") == st.MELD_FLAG_CONF


def test_frame_from_roles_needs_two_hand_tiles():
    faces = [(100, 400, 60, 80), (600, 10, 60, 80)]
    rgb = np.asarray(_table(faces, (2000, 800)))
    assert roles.frame_from_roles(rgb, faces, ["hand", "river"], 1300, "model") == (None, [])
    assert roles.frame_from_roles(rgb, faces, ["hand", "river"], 1300, "hybrid") == (None, [])


class _Net:
    """상자 위치로 역할을 정하는 가짜 모델: 아래 띠(v > 0.97)는 손패, 나머지는 강."""

    feat = roles.FEAT

    def predict(self, x):
        return ["hand" if r[1] > 0.97 else "river" for r in x], [0.9] * len(x)


def test_tokens_rot_features_use_tilted_rect_and_backs():
    W, H = 2000, 1500
    box = (1000, 1200, 100, 100)  # 기운 패를 감싸는 축 상자(정사각형)
    # 패 폭 60·높이 80 을 30° 기울인 회전 상자의 꼭짓점(obb.corners 와 같은 cv2 규약)
    import cv2

    pts = cv2.boxPoints(((1050.0, 1250.0), (60.0, 80.0), 30.0))
    t9 = roles.tokens([box], W, H, GRID, "bottom")[0]
    t11 = roles.tokens([box], W, H, GRID, "bottom", roles.FEAT_ROT, [pts])[0]
    assert len(t9) == 9 and len(t11) == 11
    # 폭·높이는 펠트 폭·높이로 따로 나누므로 비율만 본다: 축 상자(100×100) 대비 회전 상자(60×80)
    assert abs((t11[2] / t11[3]) / (t9[2] / t9[3]) - 60 / 80) < 0.05, "회전 상자 특징은 패 폭·높이"
    tilt = np.degrees(np.arctan2(t11[9], t11[10]) / 2)
    assert abs(abs(tilt) - 30) < 1.0, tilt
    axis = roles.tokens([box], W, H, GRID, "bottom", roles.FEAT_ROT, [None])[0]
    assert axis[9] == 0.0 and axis[10] == 1.0 and axis[2] == t9[2] and axis[3] == t9[3], "회전 상자 모르면 축·기울기 0"
    # 뒷면은 11칸일 때만 뒤에 붙고 플래그 1
    with_backs = roles.tokens([box], W, H, GRID, "bottom", roles.FEAT_ROT, [pts], [((500, 1200, 60, 80), None)])
    assert with_backs.shape == (2, 11) and with_backs[1][4] == 1.0 and with_backs[0][4] == 0.0
    assert roles.tokens([box], W, H, GRID, "bottom", roles.FEAT, None, [((500, 1200, 60, 80), None)]).shape == (1, 9)
    # 90° 돌린 자리에서도 기울기는 같은 값(접은 각)
    tr = roles.tokens([box], W, H, GRID, "right", roles.FEAT_ROT, [pts])[0]
    assert abs(abs(np.degrees(np.arctan2(tr[9], tr[10]) / 2)) - 30) < 1.0


def test_unrotate_pts_matches_unrotate_box():
    cw, ch = 600, 400
    b = (50, 60, 70, 30)
    for rot in (0, 90, 180, 270):
        x, y, w, h = roles.unrotate(b, rot, cw, ch)
        pts = roles.unrotate_pts([(b[0], b[1]), (b[0] + b[2], b[1]), (b[0] + b[2], b[1] + b[3]), (b[0], b[1] + b[3])], rot, cw, ch, 10, 20)
        assert abs(pts[:, 0].min() - (x + 10)) < 1e-6 and abs(pts[:, 1].min() - (y + 20)) < 1e-6
        assert abs(pts[:, 0].max() - pts[:, 0].min() - w) < 1e-6 and abs(pts[:, 1].max() - pts[:, 1].min() - h) < 1e-6


def test_read_roles_reads_winner_hand_from_full_photo(monkeypatch):
    W, H = 2000, 1500
    hand = [(600 + 66 * i, 1120, 60, 80) for i in range(13)]  # 아래 자리 손패 줄(격자 x1 = 280 → 560px 오른쪽, y2 = 713 아래)
    river = [(900 + 66 * i, 700, 60, 80) for i in range(6)]
    im = _table(hand + river, (W, H))
    r = lr.LocalReader(seat_model=None)  # 자리 검출도 detect(가짜)로
    calls = []

    def detect(rgb, low=True):
        calls.append(rgb.shape)
        if rgb.shape[0] == H:  # 사진 전체
            return hand + river
        # 자리 크롭(아래 자리는 돌리지 않음): 크롭 안의 손패
        ymin, xmin, _, _ = roles.seat_box(GRID, "bottom")
        x0, y0 = round(xmin * W / 1000), round(ymin * H / 1000)
        return [(x - x0, y - y0, w, h) for x, y, w, h in hand]

    monkeypatch.setattr(r, "detect", detect)
    monkeypatch.setattr(r, "read", lambda ims: [("1m", 0.9)] * len(ims))
    out = roles.read_roles(r, _jpeg(im), "bottom", GRID, "model", net=_Net())
    assert out["framed"] and len(out["tiles"]) == 13 and out["melds"] == [] and out["separate_tile"] is None
    assert len(calls) == 2, "사진 전체 한 번 + 자리 크롭 한 번"
    assert sum(x["in_seat"] for x in out["roles"]) == 13 and sum(x["role"] == "river" for x in out["roles"]) == 6
    assert out["counts"] == {"hand": 13} and set(out["ms"]) >= {"detect", "roles", "read", "total"}


def test_localroles_route_validates_and_answers_in_app_shape(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "GEMINI_API_KEY", "")
    monkeypatch.setattr(server, "recognizer", object())  # _gate 는 판독기가 있어야 통과(자체 모델 경로는 Gemini 를 안 쓴다)
    monkeypatch.setattr(server, "_local_reader", lr.LocalReader())
    monkeypatch.setattr(roles, "_role_net", _Net())
    client = TestClient(server.app)
    files = {"image": ("t.jpg", _jpeg(_table([], (800, 600))), "image/jpeg")}
    base = {"side": "bottom", "x1": 280, "x2": 745, "y1": 173, "y2": 713}
    assert client.post("/v2/localroles", files=files, params={**base, "side": "north"}).status_code == 422
    assert client.post("/v2/localroles", files=files, params={**base, "x2": 100}).status_code == 422
    assert client.post("/v2/localroles", files=files, params={**base, "sticks": "1,2"}).status_code == 422
    r = client.post("/v2/localroles", files=files, params={**base, "sticks": "391,445,603,639"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["face_up"] is False and j["tiles"] == [] and j["roles"] == [] and j["mode"] == "hybrid" and j["framed"] is False


@pytest.mark.skipif(not roles.ROLE_MODEL.exists(), reason="역할 모델 없음")
def test_role_model_onnx_runs():
    net = roles.RoleNet()
    assert net.feat in (roles.FEAT, roles.FEAT_ROT)
    x = np.zeros((5, net.feat), np.float32)
    x[:, 8] = 1
    r, c = net.predict(x)
    assert len(r) == 5 and all(v in roles.ROLES for v in r) and all(0 <= v <= 1 for v in c)


def test_read_roles_adds_classifier_second_choice(monkeypatch):
    """/v2/localroles 패마다 alt·conf2(분류기 2순위, 개선 계획 4단계). 답(tile·conf)은 그대로."""
    W, H = 2000, 1500
    hand = [(600 + 66 * i, 1120, 60, 80) for i in range(13)]
    im = _table(hand, (W, H))
    r = lr.LocalReader(seat_model=None)
    ymin, xmin, _, _ = roles.seat_box(GRID, "bottom")
    x0, y0 = round(xmin * W / 1000), round(ymin * H / 1000)
    monkeypatch.setattr(r, "detect", lambda rgb, low=True: hand if rgb.shape[0] == H else [(x - x0, y - y0, w, h) for x, y, w, h in hand])

    def probs(ims):
        p = np.zeros((len(ims), len(lr.CLASSES)))
        p[:, lr.CLASSES.index("3p")] = 0.7
        p[:, lr.CLASSES.index("0p")] = 0.2
        p[:, lr.CLASSES.index("1z")] = 0.1
        return p

    monkeypatch.setattr(r, "probs", probs)
    out = roles.read_roles(r, _jpeg(im), "bottom", GRID, "model", net=_Net())
    assert out["framed"] and len(out["tiles"]) == 13
    assert all(t == {"tile": "3p", "conf": 0.7, "alt": "0p", "conf2": 0.2} for t in out["tiles"])


def test_sep_near_and_lone_pair_rule(monkeypatch):
    """짝은 늘 2장: 크롭 밖 loose 후보가 없으면 혼자인 pair 를 떼어 둔 화료패 후보로(143104), 둘 붙은 pair(리치 짝)는 아님."""
    hand = [[100 + 50 * i, 500, 48, 64] for i in range(13)]
    roles_ = [{"box": b, "role": "hand", "in_seat": True} for b in hand]
    lone = {"box": [120, 380, 48, 64], "role": "pair", "in_seat": False}
    two = [{"box": [500, 360, 48, 64], "role": "pair", "in_seat": False}, {"box": [550, 360, 48, 64], "role": "pair", "in_seat": False}]
    near = roles._sep_near(roles_ + [lone] + two, hand, 64.0, None, "pair")
    assert lone in near and all(t in near for t in two)  # 거리 조건만 — 혼자인지는 _loose_outside 에서 거른다
    assert roles._sep_near(roles_ + [lone], hand, 64.0, None, "loose") == []

    calls = []

    class R:
        def read_any2(self, rgb, box):
            calls.append(box)
            return {"tile": "7s", "conf": 0.9}
    monkeypatch.setattr(roles.lr, "decode_rgb", lambda jpeg: None)
    monkeypatch.setattr(roles.lr, "tile_json", lambda d: d)
    for extra, want in (([lone] + two, "lone_pair"), (two, None)):
        out = {"framed": True, "separate_tile": None, "tiles": [{}] * 13, "melds": [], "roles": roles_ + extra}
        roles._loose_outside(R(), b"", out)
        assert (out["separate_tile"] or {}).get("from") == want
    monkeypatch.setattr(roles, "LONE_PAIR", False)
    out = {"framed": True, "separate_tile": None, "tiles": [{}] * 13, "melds": [], "roles": roles_ + [lone] + two}
    roles._loose_outside(R(), b"", out)
    assert out["separate_tile"] is None
