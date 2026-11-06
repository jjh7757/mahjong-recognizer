"""구역표·기준점 로더(app/table_frame.py)가 앱과 같은 답을 내는지(개선 계획 8b "할 일 3").

- 서버 복사본 `tests/fixtures/table_frame.json`(앱 `test/fixtures/table_frame.json`)을 서버 로더가 같은 허용 오차(점 0.5px)로 통과.
- 앱 저장소가 있으면 구역표·예제 두 파일이 앱 것과 같은지(없으면 skip). 앱 저장소 위치는 MAHJONG_CALC_REPO(기본 본 체크아웃),
  거기에 아직 없으면 그 아래 `.claude/worktrees/*`(8b-2 작업 브랜치)에서 찾는다.
"""
import json
import os
from pathlib import Path

import pytest

from app import table_frame as tf

FIXTURE = Path(__file__).parent / "fixtures" / "table_frame.json"
APP_REPO = Path(os.environ.get("MAHJONG_CALC_REPO", "C:/Users/jjh7757/dev/mahjong_calc"))


def _app_file(rel: str) -> Path | None:
    for root in [APP_REPO, *sorted(APP_REPO.glob(".claude/worktrees/*"))]:
        p = root / rel
        if p.is_file():
            return p
    return None


def test_fixture_passes_with_server_loader(capsys):
    assert tf.run_test(FIXTURE) == 0, capsys.readouterr().out


@pytest.mark.parametrize("rel,mine", [("assets/table_frame/zones.json", tf.ZONES), ("test/fixtures/table_frame.json", FIXTURE)])
def test_same_as_app_repo(rel, mine):
    app = _app_file(rel)
    if app is None:
        pytest.skip(f"앱 저장소에 {rel} 없음")
    assert json.loads(app.read_text(encoding="utf-8")) == json.loads(Path(mine).read_text(encoding="utf-8")), f"{app} 와 다름 — 복사해 맞출 것"


def test_zone_table_loads_and_rejects_bad_version():
    t = tf.ZoneTable.load()
    assert {"seat", "hand", "corner", "river", "dead_wall", "center", "device", "sticks", "dora_search"} <= set(t.zone_names)
    assert t.profile()["wall_half_per_device"] > 0
    with pytest.raises(ValueError):
        tf.ZoneTable.from_json({"version": 2, "rules": {"zones": {}}, "profiles": {}, "default_profile": "x"})
