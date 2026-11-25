import json
import threading

from ds_resources import desired as ds


class R:
    def __init__(self):
        self.kv = {}

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v):
        self.kv[k] = v


def doc(**kw):
    d = {"schema": ds.SCHEMA, "running": True, "server": {"sources": ["pi13", "a2"], "robot_cam": 1},
         "vision": {"enabled": True, "orchestrate": True, "models": ["yoloe"], "profile": {}}}
    d.update(kw)
    return d


def test_validate_matches_launcher_rules():
    assert ds.validate(doc()) == []
    assert "소스를 하나 이상 고르세요." in ds.validate(doc(server={"sources": []}))
    assert "비전 모델을 하나 이상 고르세요." in ds.validate(doc(vision={"enabled": True, "models": []}))
    assert any("cam360" in e for e in ds.validate(doc(slice={"enabled": True})))
    assert any("u 는 숫자" in e for e in ds.validate(doc(server={"sources": ["cam360"]}, slice={"enabled": True, "u": "x"})))
    t = doc(vision={"enabled": True, "models": ["targeted"], "profile": {"models": {"targeted": {"targets": []}}}})
    assert any("대상" in e for e in ds.validate(t))
    assert ds.validate(doc(running=False, server={"sources": []})) == []          # 끄는 문서는 소스가 없어도 된다


def test_write_bumps_version_and_read_prefers_newer(tmp_path):
    r, p = R(), tmp_path / "desired.json"
    a, errs = ds.write(r, p, doc(), clock=lambda: 1.0)
    assert errs == [] and a["version"] == 1
    b, _ = ds.write(r, p, doc(), clock=lambda: 2.0)
    assert b["version"] == 2 and json.loads(r.kv[ds.REDIS_KEY])["version"] == 2
    r.kv[ds.REDIS_KEY] = json.dumps(dict(b, version=1))                         # Redis 가 뒤처짐 (재시작 등)
    got, src = ds.read(r, p)
    assert got["version"] == 2 and src == "file"
    got, src = ds.read(None, tmp_path / "none.json")
    assert src == "default" and got["running"] is False


def test_invalid_is_not_written(tmp_path):
    r, p = R(), tmp_path / "d.json"
    out, errs = ds.write(r, p, doc(server={"sources": []}))
    assert out is None and errs and not p.exists() and ds.REDIS_KEY not in r.kv


def test_component_keys_follow_launcher_restart_rules():
    base = ds.component_keys(doc())
    prof = ds.component_keys(doc(vision={"enabled": True, "models": ["yoloe"], "profile": {"models": {"x": 1}}}))
    assert prof == base                                                         # 프로필만 바뀜 → 다시 띄우지 않음
    cam = ds.component_keys(doc(server={"sources": ["pi13", "a2"], "robot_cam": 2}))
    assert cam["server"] != base["server"]
    orch = ds.component_keys(doc(vision={"enabled": True, "models": ["yoloe"], "orchestrate": False}))
    assert orch["vision"] != base["vision"] and orch["server"] == base["server"]
    off = ds.component_keys(doc(running=False))
    assert off == {"server": None, "slice": None, "vision": None}


def test_vision_sources_add_slice_and_unknown_fields_survive():
    d = doc(server={"sources": ["cam360"]}, slice={"enabled": True}, extra={"keep": 1})
    assert ds.vision_sources(d) == ["cam360", "cam360_slice"]
    assert ds.normalize(d)["extra"] == {"keep": 1}


def test_concurrent_writes_do_not_share_a_version(tmp_path):
    # 탭이 둘 — 읽기~쓰기가 한 잠금 안에 있어야 같은 version 을 둘 다 +1 하지 않는다
    r, p = R(), tmp_path / "desired.json"
    n, got, start = 25, [], threading.Barrier(2)

    def run(by):
        start.wait()
        for _ in range(n):
            out, errs = ds.write(r, p, doc(), by=by)
            assert errs == []
            got.append(out["version"])
    ts = [threading.Thread(target=run, args=("a",)), threading.Thread(target=run, args=("b",))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(got) == list(range(1, 2 * n + 1))                             # 겹치지도 빠지지도 않는다
    assert ds.read(r, p)[0]["version"] == 2 * n
    assert json.loads(r.kv[ds.REDIS_KEY])["version"] == 2 * n


def test_write_gives_up_when_lock_is_held(tmp_path):
    p = tmp_path / "desired.json"
    lk = ds._lock(p)                                                            # 다른 프로세스가 쥐고 있는 셈
    try:
        out, errs = ds.write(None, p, doc(), wait_s=0.05)                       # 영원히 막히지 않는다
        assert out is None and errs and "다른 쪽" in errs[0] and not p.exists()
    finally:
        ds._unlock(lk)
    out, errs = ds.write(None, p, doc())                                        # Redis 가 없어도 파일만으로 된다
    assert errs == [] and out["version"] == 1 and json.loads(p.read_text(encoding="utf-8"))["version"] == 1
