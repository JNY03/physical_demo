"""스캔 탐지 창구 — 비전 쪽(vision_infer) 시험: 판 대상 덮어쓰기 · 파일 변경 미루기 · 각도 표시 비우기 · 방위.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_scan_vision.py -q

모델 · Redis 없이 돈다(가짜 Redis · 가짜 targeted 실행기).
"""
import json
import math

import cue_catalog as cc
import vision_infer as vi


class FakeRedisError(Exception):
    pass


class FakeRedisMod:
    RedisError = FakeRedisError


class FakeRedis:
    def __init__(self):
        self.kv = {}

    def get(self, k):
        v = self.kv.get(k)
        return None if v is None else (v if isinstance(v, bytes) else str(v).encode())

    def set(self, k, v, ex=None):
        self.kv[k] = v

    def delete(self, *ks):
        for k in ks:
            self.kv.pop(k, None)


class FakeTargeted:
    def __init__(self, fail=False):
        self.fail, self.prewarmed, self.built, self.forgot = fail, 0, [], []

    def prewarm(self, resolved):
        self.prewarmed += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")

    def reconfigure(self, resolved):
        self.built.append([t["name"] for t in resolved["targets"]])

    def forget_device(self, dv):
        self.forgot.append(dv)


class FakeWatch:
    def __init__(self):
        self.next, self.st = None, []

    def poll(self):
        r, self.next = self.next, None
        return r

    def status(self, ok, msg, version=None):
        self.st.append((ok, msg, version))


CFG = {"models": {"targeted": {}}}


def resolve(prof):
    r = cc.resolve_profile(prof, CFG, ["targeted"])
    r["_raw"] = prof
    return r


def runner(fail=False, watch=None):
    lr = vi.LiveRunner.__new__(vi.LiveRunner)
    lr.watch, lr.r, lr.redis_mod = watch, FakeRedis(), FakeRedisMod
    lr.devices, lr.models = ["robot1"], ["targeted"]
    lr.runners = {"targeted": FakeTargeted(fail)}
    lr.cfg = {"models": {"targeted": {}}}
    lr.job = lr.scan = lr.deferred = lr.before = None
    lr.scan_t, lr.mark_seen, lr.scan_failed = 0.0, {}, None
    lr.depth_period = None
    vi.apply_profile(lr.cfg, resolve({"profile_version": 7, "models": {"targeted": {"targets": [{"name": "person"}]}}}))
    return lr


def tick(lr):
    """한 바퀴: check_scan(0.5초 간격 무시) → 적재 스레드가 있으면 끝까지 → 반영."""
    lr.scan_t = 0.0
    lr.check_scan()
    if lr.job is not None and lr.job[2] is not None:
        lr.job[2].join(5)
    lr._poll_job()
    lr.check_marks()


def put_scan(lr, scan_id="S1", targets=("demo_door",), devices=("robot1",)):
    lr.r.set("vision_scan", json.dumps({"scan_id": scan_id, "devices": list(devices),
                                        "targets": [{"name": t} for t in targets]}))


def ack(lr, scan_id="S1"):
    return json.loads(lr.r.kv["vision_scan_ack:%s" % scan_id])


# ---------------------------------------------------------------- 판 대상
def test_scan_swaps_targets_and_acks():
    lr = runner()
    put_scan(lr)
    tick(lr)
    tr = lr.runners["targeted"]
    assert tr.built[-1] == ["demo_door"]
    assert lr.cfg["_scan"] == "S1" and lr.cfg["_profile_version"] == 7     # 파일 번호는 그대로
    assert ack(lr) == dict(ack(lr), ok=True, error=None, version=7)
    assert lr.cfg["models"]["targeted"]["_resolved"]["distance"] is True


def test_scan_for_other_device_is_ignored():
    lr = runner()
    put_scan(lr, devices=("a2",))
    tick(lr)
    assert lr.scan is None and lr.runners["targeted"].built == []


def test_scan_end_restores_previous_targets():
    lr = runner()
    put_scan(lr)
    tick(lr)
    lr.r.delete("vision_scan")                       # 끝 · 취소 · TTL 만료 모두 같다
    tick(lr)
    assert lr.runners["targeted"].built[-1] == ["person"]
    assert "_scan" not in lr.cfg and lr.scan is None


def test_file_change_during_scan_is_deferred_then_applied():
    w = FakeWatch()
    lr = runner(watch=w)
    put_scan(lr)
    tick(lr)
    w.next = resolve({"profile_version": 8, "models": {"targeted": {"targets": [{"name": "chair"}]}}})
    lr.check_profile()
    assert lr.runners["targeted"].built[-1] == ["demo_door"]          # 판은 그대로
    assert w.st[-1][:2] == (True, "스캔 중 — 끝나면 반영")
    lr.r.delete("vision_scan")
    tick(lr)
    assert lr.runners["targeted"].built[-1] == ["chair"] and lr.cfg["_profile_version"] == 8
    assert w.st[-1] == (True, "반영됨", 8)


def test_scan_keeps_file_overlay_but_drops_it_if_it_does_not_fit():
    lr = runner()
    vi.apply_profile(lr.cfg, resolve({"profile_version": 9, "models": {"targeted": {
        "targets": [{"name": "demo_box"}], "overlay": {"show_rejected": True}}}}))
    put_scan(lr, targets=("demo_door",))
    tick(lr)
    assert ack(lr)["ok"] and lr.runners["targeted"].built[-1] == ["demo_door"]


def test_direct_input_query_is_a_target():
    lr = runner()
    put_scan(lr, targets=("fire extinguisher",))
    tick(lr)
    assert lr.runners["targeted"].built[-1] == ["fire extinguisher"]


def test_bad_query_is_rejected_once():
    lr = runner()
    put_scan(lr, targets=("소화기",))
    tick(lr)
    a = ack(lr)
    assert not a["ok"] and "대상 이름" in a["error"]
    lr.r.kv.pop("vision_scan_ack:S1")
    tick(lr)
    assert "vision_scan_ack:S1" not in lr.r.kv                         # 다시 하지 않는다


def test_prewarm_failure_is_acked_and_not_retried():
    lr = runner(fail=True)
    put_scan(lr)
    tick(lr)
    tick(lr)
    a = ack(lr)
    assert not a["ok"] and "out of memory" in a["error"]
    assert lr.runners["targeted"].prewarmed == 1 and lr.scan is None and lr.runners["targeted"].built == []


def test_depth_every_frame_during_scan():
    class Dp:
        period = 5
    lr = runner()
    lr.runners["targeted"].depth = Dp()
    put_scan(lr)
    tick(lr)
    assert lr.runners["targeted"].depth.period == 1
    lr.r.delete("vision_scan")
    tick(lr)
    assert lr.runners["targeted"].depth.period == 5


# ---------------------------------------------------------------- 각도 표시
def test_mark_clears_that_device_memory_only():
    lr = runner()
    lr.devices = ["robot1", "a2"]
    put_scan(lr)
    tick(lr)
    for store in ("_axes", "_stab", "_lab", "_asm"):
        lr.cfg[store] = {("robot1", "r", "targeted"): 1, ("a2", "r", "targeted"): 1, ("robot1", "r", "yoloe"): 1}
    lr.r.set("vision_scan_mark:robot1", 1)
    lr.check_marks()
    for store in ("_axes", "_stab", "_lab", "_asm"):
        assert set(lr.cfg[store]) == {("a2", "r", "targeted"), ("robot1", "r", "yoloe")}
    assert lr.runners["targeted"].forgot == ["robot1"] and lr.cfg["_scan_mark"] == {"robot1": 1}
    lr.cfg["_axes"][("robot1", "r", "targeted")] = 1
    lr.check_marks()                                  # 같은 번호면 다시 비우지 않는다
    assert ("robot1", "r", "targeted") in lr.cfg["_axes"]


def test_mark_ignored_without_scan():
    lr = runner()
    lr.r.set("vision_scan_mark:robot1", 1)
    lr.cfg["_axes"] = {("robot1", "r", "targeted"): 1}
    lr.check_marks()
    assert lr.cfg["_axes"]


def test_track_evidence_restarts_after_clear():
    """비운 뒤 같은 자리 상자는 앞 각도의 근거를 물려받지 않는다 — 첫 장부터 그 장 값으로 판정."""
    def d(sem):
        cv = {"class": {"pct": 80.0, "role": "score", "passed": None, "w": 0.5},
              "semantic": {"pct": sem, "role": "score", "passed": None, "w": 0.5}}
        return {"name": "demo_door", "track_id": None, "xyxy": [0, 0, 40, 80],
                "assess": {"suspect": None, "score": (80 + sem) / 2, "min_score": 50.0, "min_evidence": 0.45,
                           "cue_values": cv, "cards": {"demo_door": {"rows": [["class", 80.0, "door 0.6"]]}},
                           "axes": {"global": False, "local": None, "local_declared": False, "geometry": None}}}
    te = vi.TrackEvidence()
    for _ in range(5):
        te.update([d(90.0)])
    carried = te.update([d(0.0)])[0]["assess"]
    assert carried["score"] > 50                      # 물려받으면 점수가 남는다(그래서 비운다)
    fresh = vi.TrackEvidence().update([d(0.0)])[0]["assess"]
    assert fresh["score"] == 40.0 and fresh["track_frames"] == 1
    assert fresh["frame"]["cards"]["demo_door"]["rows"][0] == ["class", 80.0, "door 0.6"]


# ---------------------------------------------------------------- 방위
def test_bearing_go1_uses_box_focal():
    cfg = {"_go1_cam": ((464, 400), 300.0)}
    dets = [{"xyxy": [282, 0, 382, 10]}]              # 중심 332 → 332 − 232 = 100 px
    vi.add_bearing(dets, 464, 400, cfg, None)
    b = dets[0]["bearing"]
    assert b["f_source"] == "go1_box" and abs(b["offset_deg"] - math.degrees(math.atan(100 / 300))) < 0.01


def test_bearing_other_camera_uses_fov():
    cfg = {"_go1_cam": ((464, 400), 300.0)}
    dets = [{"xyxy": [1920, 0, 1920, 10]}]            # 오른쪽 끝 → 화각의 절반
    vi.add_bearing(dets, 1920, 1080, cfg, 90.0)
    assert dets[0]["bearing"]["f_source"] == "fov" and abs(dets[0]["bearing"]["offset_deg"] - 45.0) < 0.01


def test_bearing_unknown_focal_is_none():
    dets = [{"xyxy": [0, 0, 10, 10]}]
    vi.add_bearing(dets, 1920, 1080, {"_go1_cam": None}, None)
    assert dets[0]["bearing"]["offset_deg"] is None


def test_go1_focal_from_fit_file():
    from demo_box import go1_camera
    t = cc.load_targets()["demo_box"]
    try:
        size, f = go1_camera(t)
    except FileNotFoundError:
        import pytest
        pytest.skip("demo_box 거리 계수 파일이 없음")
    assert size == (464, 400) and 100 < f < 2000
