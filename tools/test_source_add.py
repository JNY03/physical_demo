"""소스 더하기 시험 — 이름 · 포트 · 저장 규칙 (2026-11-26).

연결 시험(test_stream)은 진짜 기기가 있어야 하니 여기서는 그 자리를 대신 채워 넣고, **더하는 규칙**만 본다:
겹치는 이름을 막는지, 빈 포트를 고르는지, 시험에 실패하면 저장하지 않는지, 적은 것이 그대로 돌아오는지.
"""
import json

import pytest

from tools import source_add as sa


@pytest.fixture
def box(tmp_path, monkeypatch):
    """config/sources.json 을 임시 자리로 옮기고, 연결 시험은 성공한 것으로 둔다"""
    cfg = tmp_path / "sources.json"
    monkeypatch.setattr(sa, "CONFIG", cfg)
    monkeypatch.setattr(sa, "known_names", lambda: ({"pi12", "robot1", "cam360"}, {"pi12": 10007, "robot1": 10001}))
    monkeypatch.setattr(sa, "free_port", lambda used: next(p for p in range(10013, 10100) if p not in used))
    ok = {"ok": True, "url": "", "why": "", "bytes": 2048, "jpeg": 1024, "width": 1280, "height": 720, "ms": 90}
    monkeypatch.setattr(sa, "test_stream", lambda *a, **k: dict(ok))
    return cfg


def test_adds_and_reads_back(box):
    r = sa.add("pi14", "100.1.2.3", note="복도")
    assert r["ok"] and r["source"]["port"] == 10013
    assert r["source"]["host"] == "100.1.2.3" and r["source"]["cam_port"] == 8090
    saved = json.loads(box.read_text(encoding="utf-8"))
    assert saved["pi14"]["note"] == "복도"
    assert sa.load()["pi14"]["path"] == "/stream/1"


def test_refuses_names_in_use(box):
    assert not sa.add("pi12", "100.1.2.3")["ok"]            # 이미 쓰는 소스
    assert "pi12" in sa.add("pi12", "100.1.2.3")["why"]
    assert not box.exists()                                 # 아무것도 적지 않는다


def test_refuses_bad_name(box):
    for bad in ("", "P12", "1pi", "pi 12", "pi12!", "a"):
        assert not sa.add(bad, "100.1.2.3")["ok"], bad


def test_refuses_duplicate_add(box):
    assert sa.add("pi14", "100.1.2.3")["ok"]
    r = sa.add("pi14", "100.1.2.9")
    assert not r["ok"] and "--remove" in r["why"]


def test_port_skips_the_ones_in_use(box):
    assert sa.add("pi14", "100.1.2.3")["ok"]
    r = sa.add("pi15", "100.1.2.4")
    assert r["source"]["port"] == 10014                     # 10013 은 pi14 가 쓴다
    r2 = sa.add("pi16", "100.1.2.5", port=10007)            # pi12 의 포트
    assert not r2["ok"] and "포트" in r2["why"]


def test_failed_link_is_not_saved_unless_forced(box, monkeypatch):
    bad = {"ok": False, "why": "100.1.2.3:8090 에 닿지 않습니다 (TimeoutError)", "bytes": 0, "jpeg": 0,
           "width": None, "height": None, "ms": 3000, "url": ""}
    monkeypatch.setattr(sa, "test_stream", lambda *a, **k: dict(bad))
    r = sa.add("pi14", "100.1.2.3")
    assert not r["ok"] and r["test"]["why"] == bad["why"]
    assert not box.exists()
    r2 = sa.add("pi14", "100.1.2.3", force=True)            # 기기를 아직 안 켠 경우
    assert r2["ok"] and json.loads(box.read_text(encoding="utf-8"))["pi14"]["host"] == "100.1.2.3"


def test_jpeg_size_reads_the_header():
    # 가장 작은 JPEG 틀 — SOF0 에 적힌 480x640 을 읽어 낸다
    data = bytes([0xFF, 0xD8]) + bytes([0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0xE0, 0x02, 0x80]) + b"\x00" * 8
    assert sa._jpeg_size(data) == (640, 480)


def test_cli_list_is_json(box, capsys):
    assert sa.add("pi14", "100.1.2.3")["ok"]
    assert sa.main(["--list", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sources"]["pi14"]["port"] == 10013


def test_cli_remove(box, capsys):
    assert sa.add("pi14", "100.1.2.3")["ok"]
    assert sa.main(["--remove", "pi14", "--json"]) == 0
    assert sa.load() == {}
    assert sa.main(["--remove", "pi14", "--json"]) == 1     # 없는 것을 빼라면 1
    capsys.readouterr()
