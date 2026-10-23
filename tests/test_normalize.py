from app.normalize import normalize_box, normalize_hand, normalize_locate, normalize_table


def test_hand_drops_bad_meld_and_keeps_win_tile():
    out = normalize_hand({
        "tiles": [{"code": "1m", "confidence": 0.95}, {"code": "zz", "confidence": 1}, "garbage"],
        "melds": [
            {"type": "chi", "tiles": ["4p", "5p", "6p"], "confidence": 0.9},
            {"type": "pon", "tiles": ["7z", "7z"], "confidence": 0.9},          # 2장: 제외
            {"type": "kan_closed", "tiles": ["1s", "1s", "1s", "1s"], "confidence": 1.7},  # conf 는 1 로 자름
            {"type": "ron", "tiles": ["1m", "1m", "1m"], "confidence": 0.9},   # 모르는 type: 제외
        ],
        "win_tile": {"code": "7s", "confidence": 0.4},
        "notes": "흐림",
    })
    assert out["tiles"] == [{"tile": "1m", "conf": 0.95}]
    assert [m["type"] for m in out["melds"]] == ["chi", "kan_closed"]
    assert out["melds"][1]["conf"] == 1.0
    assert out["win_tile"] == {"tile": "7s", "conf": 0.4}
    assert "울기 2개" in out["notes"] and out["notes"].startswith("흐림")


def test_hand_without_optional_fields():
    out = normalize_hand({"tiles": [{"code": "5z", "confidence": "0.8"}]})
    assert out == {"tiles": [{"tile": "5z", "conf": 0.8}], "melds": [], "win_tile": None, "notes": ""}


def test_box_clamps_and_rejects():
    assert normalize_box([-5, 10.6, 1200, 900]) == [0, 11, 1000, 900]
    assert normalize_box([500, 100, 400, 900]) is None   # ymin >= ymax
    assert normalize_box([1, 2, 3]) is None
    assert normalize_box(["a", 1, 2, 3]) is None


def test_locate_one_per_side_and_center_fallback():
    out = normalize_locate({
        "hands": [
            {"side": "bottom", "box": [800, 100, 950, 900], "face_up": True, "confidence": 0.9},
            {"side": "bottom", "box": [0, 0, 10, 10], "face_up": False, "confidence": 0.1},  # 중복: 무시
            {"side": "left", "box": [100, 0, 900, 150], "face_up": False, "confidence": 0.7},
            {"side": "north", "box": [0, 0, 10, 10], "face_up": True, "confidence": 0.7},   # 모르는 side
            {"side": "top", "box": [50, 100, 40, 900], "face_up": True, "confidence": 0.7},  # 뒤집힌 상자
        ],
        "rivers": [
            {"side": "bottom", "box": [600, 300, 780, 700], "confidence": 0.8},
            {"side": "top", "box": [220, 300, 400, 700], "confidence": 0.8},
        ],
        "dead_wall": {"box": [300, 50, 400, 250], "confidence": 0.5},
        "center": None,
    })
    assert [h["side"] for h in out["hands"]] == ["bottom", "left"]
    assert out["face_up_sides"] == ["bottom"]
    assert out["center"] == {"box": [220, 300, 780, 700]}
    assert out["dead_wall"] == {"box": [300, 50, 400, 250], "conf": 0.5}
    assert out["notes"] == ""


def test_table_defaults_every_side():
    out = normalize_table({
        "dora_indicators": [{"code": "3p", "confidence": 0.9}, {"code": "bad", "confidence": 1}],
        "riichi": [{"side": "left", "riichi": True, "confidence": 0.8}],
        "last_discards": [{"side": "bottom", "code": "7s", "confidence": 0.7},
                          {"side": "right", "code": "??", "confidence": 0.7}],
    })
    assert out["dora_indicators"] == [{"tile": "3p", "conf": 0.9}]
    assert out["riichi"] == {"bottom": False, "right": False, "top": False, "left": True}
    assert out["riichi_conf"]["left"] == 0.8
    assert out["last_discards"] == {"bottom": "7s", "right": None, "top": None, "left": None}


def test_meld_consistency_flags_but_keeps():
    out = normalize_hand({
        "tiles": [],
        "melds": [
            {"type": "chi", "tiles": ["2s", "4s", "0s"], "confidence": 0.9},   # 연속 아님
            {"type": "chi", "tiles": ["3s", "4s", "0s"], "confidence": 0.9},   # 3-4-5(적) 정상
            {"type": "pon", "tiles": ["5m", "0m", "5m"], "confidence": 0.9},   # 적도라 섞인 퐁 정상
            {"type": "pon", "tiles": ["5m", "6m", "5m"], "confidence": 0.9},   # 같은 패 아님
        ],
        "win_tile": None,
    })
    assert [m["conf"] for m in out["melds"]] == [0.0, 0.9, 0.9, 0.0]
    assert "치 2s 4s 0s 는 연속이 아님" in out["notes"]
    assert "pon 5m 6m 5m 는 같은 패가 아님" in out["notes"]
