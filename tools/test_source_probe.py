"""source_probe — 밀어 넣는 소스는 오래된 장만으로 막지 않는다 (2026-11-26)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_probe as sp  # noqa: E402


def test_push_source_with_stale_frame_is_unknown_not_down(monkeypatch):
    monkeypatch.setattr(sp, "_sender_running", lambda script: None)          # 보내는 쪽을 모르는 설치
    monkeypatch.setattr(sp, "_health", lambda port, timeout: None)            # 데이터 서버 꺼짐
    monkeypatch.setattr(sp, "_redis_age", lambda src, r: 79000.0)             # 어제 장만 남음
    out = sp.probe_one("cam360", {"cam360": 10001}, {}, None, 0.1)
    assert out["state"] == "unknown" and "밀어 넣는" in out["why"]


def test_pull_source_still_down_when_unreachable(monkeypatch):
    monkeypatch.setattr(sp, "_health", lambda port, timeout: None)
    monkeypatch.setattr(sp, "_redis_age", lambda src, r: 79000.0)
    monkeypatch.setattr(sp, "_tcp", lambda host, port, timeout: False)
    out = sp.probe_one("robot1", {"robot1": 10003}, {"robot1": "10.0.0.1"}, None, 0.1)
    assert out["state"] == "down"


def test_push_source_follows_its_sender_process(monkeypatch):
    # 2026-11-27 사용자: 360 카메라 런처가 꺼져 있는데 cam360 이 골라졌다 — 보내는 런처가 도는지로 가른다
    monkeypatch.setattr(sp, "_health", lambda port, timeout: None)
    monkeypatch.setattr(sp, "_redis_age", lambda src, r: 79000.0)
    monkeypatch.setattr(sp, "_sender_running", lambda script: False)
    out = sp.probe_one("cam360", {"cam360": 10001}, {}, None, 0.1)
    assert out["state"] == "down" and "360 카메라 실행" in out["why"]
    monkeypatch.setattr(sp, "_sender_running", lambda script: True)
    assert sp.probe_one("cam360", {"cam360": 10001}, {}, None, 0.1)["state"] == "ready"
    monkeypatch.setattr(sp, "_sender_running", lambda script: None)     # /proc 없음(모름) — 막지 않는다
    assert sp.probe_one("cam360", {"cam360": 10001}, {}, None, 0.1)["state"] == "unknown"
