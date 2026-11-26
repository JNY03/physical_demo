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


# ---------------------------------------------------------------- 기기별 모델 매칭 · 수동 배치 (2026-11-26)
def _v(**kw):
    return dict({"enabled": True, "orchestrate": True, "models": ["yoloe", "river"], "profile": {}}, **kw)


def test_assign_checks_devices_and_models():
    assert ds.validate(doc(vision=_v(assign={"pi13": ["yoloe"], "a2": []}))) == []      # a2 는 비전 없음 — 괜찮다
    assert any("고르지 않은 기기 pi9" in e for e in ds.validate(doc(vision=_v(assign={"pi9": ["yoloe"]}))))
    assert any("고르지 않은 모델 moge" in e for e in ds.validate(doc(vision=_v(assign={"pi13": ["moge"]}))))
    assert any("모든 기기를 비웠" in e for e in ds.validate(doc(vision=_v(assign={"pi13": [], "a2": []}))))
    # 직접 비전(자원 배분 끔)은 프로세스 하나가 모든 기기를 맡는다 — 기기마다 모델을 가를 수 없다
    assert any("자원 배분을 켠 채로" in e
               for e in ds.validate(doc(vision=_v(orchestrate=False, assign={"pi13": ["yoloe"]}))))
    assert ds.validate(doc(vision=_v(orchestrate=False))) == []                          # 매칭이 없으면 그대로 된다


def test_manual_place_only_with_orchestration():
    on = doc(vision=_v(manual={"enabled": True, "place": {"object_detect.device": "pi1"}}))
    assert ds.validate(on) == [] and ds.manual_place(on) == {"object_detect.device": "pi1"}
    off = doc(vision=_v(manual={"enabled": False, "place": {"object_detect.device": "pi1"}}))
    assert ds.manual_place(off) == {}                                   # 꺼 두면 자리를 넘기지 않는다
    bad = doc(vision=_v(orchestrate=False, manual={"enabled": True, "place": {}}))
    assert any("수동 배치는 자원 배분" in e for e in ds.validate(bad))


def test_component_keys_change_when_match_or_place_changes():
    a = ds.component_keys(doc(vision=_v()))
    b = ds.component_keys(doc(vision=_v(assign={"pi13": ["yoloe"]})))
    c = ds.component_keys(doc(vision=_v(manual={"enabled": True, "place": {"x.device": "pi1"}})))
    assert a["vision"] != b["vision"] and a["vision"] != c["vision"]     # 바뀌면 비전을 다시 띄운다
    assert a["server"] == b["server"] == c["server"]                     # 서버는 그대로 둔다
