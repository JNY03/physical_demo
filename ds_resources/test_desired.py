import json

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
