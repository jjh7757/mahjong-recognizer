"""판독 불확실성 신호(개선 계획 5단계, app/signals.py): 답은 그대로, 값만 더한다."""
import numpy as np
from PIL import Image, ImageDraw

from app import local_read as lr
from app import obb
from app import roles
from app import signals as sig

from test_roles import GRID, _jpeg, _Net, _table


def test_calibrate_keeps_argmax_and_sharpens_below_one():
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    for T in (0.5, 1.0, 2.0):
        q = sig.calibrate(p, T)
        assert np.allclose(q.sum(1), 1) and (q.argmax(1) == p.argmax(1)).all()
    assert np.allclose(sig.calibrate(p, 1.0), p)
    assert sig.calibrate(p, 0.5)[0, 0] > 0.6 > sig.calibrate(p, 2.0)[0, 0]


def _tile(color):
    im = Image.new("RGB", (100, 130), (245, 245, 240))
    d = ImageDraw.Draw(im)
    d.rectangle((35, 40, 65, 90), fill=color)
    return im


def test_red_ratio_separates_red_ink_from_black():
    assert sig.red_ratio(_tile((210, 20, 25))) > 0.9
    assert sig.red_ratio(_tile((20, 20, 20))) < 0.05
    assert sig.red_ratio(Image.new("RGB", (100, 130), (245, 245, 240))) is None  # 잉크 없음


def test_jitter_shrinks_and_reencodes():
    im = _tile((20, 20, 20))
    j = sig.jitter(im)
    assert j.size == (90, 117) and j.mode == "RGB"


def test_obb_decode_records_scores():
    # cx, cy, w, h, 앞면, 뒷면, 각도 × 상자 2개
    out = np.array([[50, 200], [50, 200], [20, 20], [30, 30], [0.9, 0.4], [0.0, 0.0], [0.0, 0.0]], np.float32)
    scores: dict = {}
    res = obb.decode(out, 1.0, 0, 0, 400, 400, scores=scores)
    assert sorted(scores.values()) == [0.4, 0.9] and set(scores) == {b for b, _ in res}
    rgb = np.zeros((400, 400, 3), np.uint8)
    with obb.session():
        obb.record(rgb, res, [], None, scores)
        assert obb.score_of(rgb, res[0][0]) == 0.9
    assert obb.score_of(rgb, res[0][0]) is None  # 요청 밖


class _SeedNet(_Net):
    """씨드 셋 중 하나가 첫 상자를 다르게 본다."""

    def predict_full(self, x):
        r, c = self.predict(x)
        other = list(r)
        if other:
            other[0] = "river"
        return r, c, [list(r), list(r), other]


def _read(monkeypatch, net, alt_differs: bool):
    W, H = 2000, 1500
    hand = [(600 + 66 * i, 1120, 60, 80) for i in range(13)]
    im = _table(hand, (W, H))
    r = lr.LocalReader(seat_model=None, alt_cls_model=None)
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

    class Alt:
        def get_inputs(self):
            return [type("I", (), {"name": "x"})()]

        def run(self, _, feed):
            y = np.zeros((len(feed["x"]), len(lr.CLASSES)), np.float32)
            y[:, lr.CLASSES.index("4p" if alt_differs else "3p")] = 5.0
            return [y]

    monkeypatch.setattr(sig.AltCls, "get", classmethod(lambda cls, path, threads=2: Alt()))
    return roles.read_roles(r, _jpeg(im), "bottom", GRID, "model", net=net)


def test_read_roles_adds_signals_without_changing_answer(monkeypatch):
    out = _read(monkeypatch, _SeedNet(), alt_differs=True)
    assert all(t == {"tile": "3p", "conf": 0.7, "alt": "0p", "conf2": 0.2} for t in out["tiles"])  # 답 그대로
    s = out["signals"]
    assert s["n"] == 13 and s["filled"] == 0 and len(s["tiles"]) == 13
    assert s["cls2_diff"] == 13 and s["jitter_diff"] == 0 and s["role_diff"] == 1
    q = sig.calibrate(np.array([0.7, 0.2, 0.1]))
    assert abs(s["margin_min"] - round(float(q[0] - q[1]), 3)) < 1e-6
    assert s["det_min"] is None  # 축 상자 검출기(seat_model 없음)는 점수 기록이 없다
    assert out["ms"]["signals"] == s["ms"] and "signals" in out["ms"]


def test_signals_off_switch(monkeypatch):
    monkeypatch.setattr(sig, "ON", False)
    out = _read(monkeypatch, _SeedNet(), alt_differs=False)
    assert "signals" not in out and "_sig_items" not in out and "_role_seeds" not in out


def test_plain_net_without_seeds_has_no_role_diff(monkeypatch):
    out = _read(monkeypatch, _Net(), alt_differs=False)
    assert "role_diff" not in out["signals"] and out["signals"]["cls2_diff"] == 0


def test_role_model_onnx_has_seed_output():
    if not roles.ROLE_MODEL.exists():
        return
    net = roles.RoleNet()
    x = np.zeros((5, net.feat), np.float32)
    x[:, 8] = 1
    r, c, seeds = net.predict_full(x)
    assert len(r) == 5 and seeds is not None and len(seeds) == 3 and all(len(s) == 5 for s in seeds)
    assert net.predict(x) == (r, c)

