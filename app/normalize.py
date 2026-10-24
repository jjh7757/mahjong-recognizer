"""Gemini 가 돌려준 JSON 을 앱이 기대하는 모양으로 다듬는다. 순수 함수라 단위 테스트 대상."""

from __future__ import annotations

from .schemas import MELD_TYPES, SIDES, TILE_CODES


def _conf(v) -> float:
    try:
        return round(min(1.0, max(0.0, float(v))), 3)
    except (TypeError, ValueError):
        return 0.0


def _tile(t) -> dict | None:
    if not isinstance(t, dict) or t.get("code") not in TILE_CODES:
        return None
    return {"tile": t["code"], "conf": _conf(t.get("confidence", 0))}


def _tiles(items) -> list[dict]:
    out = []
    for t in items or []:
        n = _tile(t)
        if n:
            out.append(n)
    return out


def _append_note(notes: str, extra: str) -> str:
    return f"{notes} {extra}".strip() if notes else extra


# ---------------------------------------------------------------- 손패

def _meld_ok(m: dict) -> bool:
    if m.get("type") not in MELD_TYPES:
        return False
    tiles = m.get("tiles")
    if not isinstance(tiles, list) or any(t not in TILE_CODES for t in tiles):
        return False
    need = 4 if m["type"].startswith("kan") else 3
    return len(tiles) == need


def _chi_ok(tiles: list[str]) -> bool:
    """같은 수패 3장이 연속인가. 적도라 0 은 5 로 본다."""
    suits = {t[1] for t in tiles}
    if len(suits) != 1 or suits <= {"z"}:
        return False
    nums = sorted(5 if t[0] == "0" else int(t[0]) for t in tiles)
    return nums[1] == nums[0] + 1 and nums[2] == nums[1] + 1


def normalize_hand(data: dict) -> dict:
    """→ {tiles:[{tile,conf}], melds:[{type,tiles,conf}], win_tile:{tile,conf}|None, notes}"""
    notes = str(data.get("notes") or "")
    melds, dropped = [], 0
    for m in data.get("melds") or []:
        if isinstance(m, dict) and _meld_ok(m):
            entry = {"type": m["type"], "tiles": list(m["tiles"]), "conf": _conf(m.get("confidence", 0))}
            if m["type"] == "chi" and not _chi_ok(entry["tiles"]):
                # 치인데 연속이 아니면 울기는 있되 패를 못 읽은 것으로 본다: 확신 0, 앱이 고치게 한다.
                entry["conf"] = 0.0
                notes = _append_note(notes, f"치 {' '.join(entry['tiles'])} 는 연속이 아님")
            elif m["type"] != "chi" and len({t.replace("0", "5") for t in entry["tiles"]}) != 1:
                entry["conf"] = 0.0
                notes = _append_note(notes, f"{m['type']} {' '.join(entry['tiles'])} 는 같은 패가 아님")
            melds.append(entry)
        else:
            dropped += 1
    if dropped:
        notes = _append_note(notes, f"울기 {dropped}개는 형식이 맞지 않아 제외")
    return {
        "tiles": _tiles(data.get("tiles")),
        "melds": melds,
        "win_tile": _tile(data.get("win_tile")),
        "notes": notes,
    }


# ---------------------------------------------------------------- 위치

def normalize_box(b) -> list[int] | None:
    """[ymin,xmin,ymax,xmax] 정수 4개, 0~1000 으로 자르고 뒤집힌 상자는 버린다."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        v = [min(1000, max(0, int(round(float(x))))) for x in b]
    except (TypeError, ValueError):
        return None
    if v[0] >= v[2] or v[1] >= v[3]:
        return None
    return v


def _sided_boxes(items, extra_keys: tuple[str, ...] = ()) -> list[dict]:
    """side 별로 첫 유효 항목만. 순서는 SIDES 순."""
    by_side: dict[str, dict] = {}
    for it in items or []:
        if not isinstance(it, dict) or it.get("side") not in SIDES or it["side"] in by_side:
            continue
        box = normalize_box(it.get("box"))
        if box is None:
            continue
        entry = {"side": it["side"], "box": box, "conf": _conf(it.get("confidence", 0))}
        for k in extra_keys:
            entry[k] = bool(it.get(k, False))
        by_side[it["side"]] = entry
    return [by_side[s] for s in SIDES if s in by_side]


def _boxed(obj, with_conf: bool = True) -> dict | None:
    if not isinstance(obj, dict):
        return None
    box = normalize_box(obj.get("box"))
    if box is None:
        return None
    out = {"box": box}
    if with_conf:
        out["conf"] = _conf(obj.get("confidence", 0))
    return out


def normalize_locate(data: dict) -> dict:
    hands = _sided_boxes(data.get("hands"), ("face_up",))
    rivers = _sided_boxes(data.get("rivers"))
    center = _boxed(data.get("center"), with_conf=False)
    if center is None and rivers:
        # 강 상자들의 합집합으로 대신한다.
        ys = [r["box"][0] for r in rivers] + [r["box"][2] for r in rivers]
        xs = [r["box"][1] for r in rivers] + [r["box"][3] for r in rivers]
        center = {"box": [min(ys), min(xs), max(ys), max(xs)]}
    melds = []
    for it in data.get("melds") or []:
        if isinstance(it, dict) and it.get("side") in SIDES:
            box = normalize_box(it.get("box"))
            if box is not None:
                melds.append({"side": it["side"], "box": box, "conf": _conf(it.get("confidence", 0))})
    return {
        "hands": hands,
        "rivers": rivers,
        "melds": melds,
        "dead_wall": _boxed(data.get("dead_wall")),
        "center": center,
        "face_up_sides": [h["side"] for h in hands if h["face_up"]],
        "notes": str(data.get("notes") or ""),
    }


# ---------------------------------------------------------------- 테이블 중앙

def normalize_table(data: dict) -> dict:
    riichi = {s: False for s in SIDES}
    riichi_conf = {s: 0.0 for s in SIDES}
    for it in data.get("riichi") or []:
        if isinstance(it, dict) and it.get("side") in SIDES:
            riichi[it["side"]] = bool(it.get("riichi", False))
            riichi_conf[it["side"]] = _conf(it.get("confidence", 0))
    last: dict[str, str | None] = {s: None for s in SIDES}
    last_conf = {s: 0.0 for s in SIDES}
    for it in data.get("last_discards") or []:
        if isinstance(it, dict) and it.get("side") in SIDES and it.get("code") in TILE_CODES:
            last[it["side"]] = it["code"]
            last_conf[it["side"]] = _conf(it.get("confidence", 0))
    rivers: dict[str, list[str]] = {s: [] for s in SIDES}
    for it in data.get("rivers") or []:
        if isinstance(it, dict) and it.get("side") in SIDES:
            rivers[it["side"]] = [c for c in it.get("codes") or [] if c in TILE_CODES]
    return {
        "dora_indicators": _tiles(data.get("dora_indicators")),
        "ura_dora_indicators": _tiles(data.get("ura_dora_indicators")),
        "riichi": riichi,
        "riichi_conf": riichi_conf,
        "last_discards": last,
        "last_discards_conf": last_conf,
        "rivers": rivers,
        "notes": str(data.get("notes") or ""),
    }
