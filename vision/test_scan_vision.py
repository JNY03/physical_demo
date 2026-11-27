"""스캔 탐지 창구 — 비전 쪽(vision_infer) 시험: 판 대상 덮어쓰기 · 파일 변경 미루기 · 각도 표시 비우기 · 방위.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_scan_vision.py -q

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
        self.fail, self.prewarmed, self.built, self.forgot, self.reforget = fail, 0, [], [], []

    def prewarm(self, resolved):
        self.prewarmed += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")

    def reconfigure(self, resolved, forget=None):
        self.built.append([t["name"] for t in resolved["targets"]])
        self.reforget.append(forget)

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
    """판 동안 매 장 재는 기기 = 판 기기만 (2026-11-27). 주기는 기기가 바뀔 때 use_stream 이 건다."""
    class Dp:
        period = 5
    lr = runner2({"robot1": ["person"], "pi12": ["chair"]})
    tr = lr.runners["targeted"]
    tr.depth, tr.depth_period0 = Dp(), 5
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert tr.fast_devices == {"robot1"}
    tr.depth.period = 1                               # 판 기기 장을 처리하던 중에 판이 끝났다
    lr.r.delete("vision_scan")
    tick(lr)
    assert tr.fast_devices == set() and tr.depth.period == 5


def test_use_stream_sets_depth_period_per_device():
    class Dp:
        period, _mem, _n = 5, {}, 0
    tr = vi.TargetedRunner.__new__(vi.TargetedRunner)
    tr.depth, tr.depth_period0, tr._streams, tr._stream = Dp(), 5, {}, None
    tr.fast_devices = {"robot1"}
    tr.use_stream(("robot1", "r"))
    assert tr.depth.period == 1
    tr.use_stream(("pi12", "r"))
    assert tr.depth.period == 5
    tr.use_stream(("pi12", "r"))                      # 같은 기기 — 기억은 그대로, 주기도 그대로
    assert tr.depth.period == 5
    tr.fast_devices = set()
    tr.use_stream(("robot1", "r"))
    assert tr.depth.period == 5


# ---------------------------------------------------------------- 판 기기만 판 대상 (2026-11-27, 스캔판대상_장비별_요청_VZ_261126)
STORES = ("_asm", "_stab", "_lab", "_axes")


def runner2(dev_targets=None, distance=None):
    """robot1(go1) · pi12 가 한 프로세스 — 판 전 대상 person · chair, 기기별 대상 dev_targets."""
    lr = runner()
    lr.devices = ["robot1", "pi12"]
    tm = {"targets": [{"name": "person"}, {"name": "chair"}]}
    if distance is not None:
        tm["distance"] = distance
    prof = {"profile_version": 7, "models": {"targeted": tm}}
    if dev_targets:
        prof["devices"] = {dv: {"models": {"targeted": {"targets": t}}} for dv, t in dev_targets.items()}
    vi.apply_profile(lr.cfg, resolve(prof))
    return lr


def test_scan_targets_only_the_scan_device():
    lr = runner2({"robot1": ["person"], "pi12": ["chair"]})
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert ack(lr)["ok"]
    assert lr.runners["targeted"].built[-1] == ["person", "chair", "demo_box"]     # 최상위는 합친다
    assert lr.cfg["_targets_dev"] == {"robot1": ["demo_box"], "pi12": ["chair"]}  # pi12 는 판 전 그대로
    assert lr.cfg["_scan_devices"] == ["robot1"]
    lr.r.delete("vision_scan")
    tick(lr)
    assert lr.runners["targeted"].built[-1] == ["person", "chair"]
    assert lr.cfg["_targets_dev"] == {"robot1": ["person"], "pi12": ["chair"]}
    assert "_scan_devices" not in lr.cfg


def test_device_without_own_targets_keeps_before_targets():
    """장비별 대상이 없던 기기(= 전부 찾던 기기)는 판 전 대상을 명시 — 판 대상이 끼지 않는다."""
    lr = runner2({"robot1": ["person"]})
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert lr.cfg["_targets_dev"] == {"robot1": ["demo_box"], "pi12": ["person", "chair"]}


def test_device_without_own_targets_and_default_targets():
    """판 전 프로필에 대상을 적지 않았으면(기본 대상) 확정된 이름으로 명시한다."""
    lr = runner()
    lr.devices = ["robot1", "pi12"]
    vi.apply_profile(lr.cfg, resolve({"profile_version": 7}))
    before = [t["name"] for t in lr.cfg["models"]["targeted"]["_resolved"]["targets"]]
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert ack(lr)["ok"]
    assert lr.cfg["_targets_dev"]["pi12"] == [n for n in before]
    assert lr.cfg["_targets_dev"]["robot1"] == ["demo_box"]


def test_empty_device_targets_stay_empty_during_scan():
    lr = runner2({"robot1": ["person"], "pi12": []})
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert lr.cfg["_targets_dev"] == {"robot1": ["demo_box"], "pi12": []}


def test_scan_device_without_own_targets():
    lr = runner2({"pi12": ["chair"]})
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert lr.cfg["_targets_dev"] == {"robot1": ["demo_box"], "pi12": ["chair"]}


def test_before_targets_keep_before_distance():
    """판 전 거리 끔이면 판 전 대상은 거리 끔 그대로 — 판 대상만 거리를 잰다."""
    lr = runner2({"robot1": ["person"], "pi12": ["chair"]}, distance=False)
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    r = lr.cfg["models"]["targeted"]["_resolved"]
    assert r["distance"] is True
    assert {t["name"]: t["distance"] for t in r["targets"]} == {"person": False, "chair": False, "demo_box": True}


def test_same_name_uses_the_scan_target():
    lr = runner()
    lr.devices = ["robot1", "pi12"]
    vi.apply_profile(lr.cfg, resolve({"profile_version": 7, "models": {"targeted": {
        "targets": [{"name": "chair"}, {"name": "demo_box", "distance": False}]}}}))
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    r = lr.cfg["models"]["targeted"]["_resolved"]
    assert [t["name"] for t in r["targets"]] == ["chair", "demo_box"]
    assert {t["name"]: t["distance"] for t in r["targets"]}["demo_box"] is True
    assert lr.cfg["_targets_dev"] == {"robot1": ["demo_box"], "pi12": ["chair", "demo_box"]}


def test_device_not_running_targeted_is_not_merged():
    """--device-models 로 targeted 를 돌지 않는 기기만 함께 있으면 예전처럼 판 대상만."""
    lr = runner2()
    lr.cfg["_device_models"] = {"pi12": ["yoloe"]}
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert lr.runners["targeted"].built[-1] == ["demo_box"]


def test_single_device_process_is_unchanged():
    """대조군 — 판 기기가 프로세스의 유일한 기기면 최상위 = 판 대상 (합치지 않는다)."""
    lr = runner()
    vi.apply_profile(lr.cfg, resolve({"profile_version": 7, "models": {"targeted": {"targets": [{"name": "person"}]}},
                                      "devices": {"robot1": {"models": {"targeted": {"targets": ["person"]}}}}}))
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    assert ack(lr)["ok"] and lr.runners["targeted"].built[-1] == ["demo_box"]
    assert lr.cfg["_targets_dev"] == {"robot1": ["demo_box"]}
    assert lr.runners["targeted"].reforget[-1] == ["robot1"]


def test_scan_start_and_end_clear_only_scan_device_memory():
    lr = runner2({"robot1": ["person"], "pi12": ["chair"]})
    keys = {("robot1", "r", "targeted"), ("pi12", "r", "targeted"), ("pi12", "r", "yoloe")}
    for s in STORES:
        lr.cfg[s] = {k: 1 for k in keys}
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    for s in STORES:
        assert set(lr.cfg[s]) == keys - {("robot1", "r", "targeted")}             # pi12 트랙 번호가 이어진다
    assert lr.runners["targeted"].reforget[-1] == ["robot1"]
    lr.cfg["_stab"][("robot1", "r", "targeted")] = 1
    lr.r.delete("vision_scan")
    tick(lr)
    assert set(lr.cfg["_stab"]) == keys - {("robot1", "r", "targeted")}
    assert lr.runners["targeted"].reforget[-1] == ["robot1"]


def test_deferred_profile_after_scan_clears_all_memory():
    w = FakeWatch()
    lr = runner2({"robot1": ["person"], "pi12": ["chair"]})
    lr.watch = w
    put_scan(lr, targets=("demo_box",))
    tick(lr)
    lr.cfg["_stab"] = {("pi12", "r", "targeted"): 1}
    w.next = resolve({"profile_version": 8, "models": {"targeted": {"targets": [{"name": "chair"}]}}})
    lr.check_profile()
    lr.r.delete("vision_scan")
    tick(lr)
    assert lr.runners["targeted"].reforget[-1] is None and lr.cfg["_stab"] == {}


def test_runner_reconfigure_keeps_other_device_depth_memory():
    class Dp:
        _mem, _n = {}, 0
    tr = vi.TargetedRunner.__new__(vi.TargetedRunner)
    tr._build = lambda resolved: setattr(tr, "_streams", {}) or setattr(tr, "_stream", None)
    tr.depth = Dp()
    tr._streams = {("robot1", "r"): {"d_mem": {1: "a"}, "d_n": 3}}
    tr._stream = ("pi12", "r")
    tr.depth._mem, tr.depth._n = {7: "b"}, 2                                     # 지금 기기(pi12)의 기억
    tr.reconfigure({}, forget=["robot1"])
    assert tr._streams == {("pi12", "r"): {"d_mem": {7: "b"}, "d_n": 2}} and tr._stream is None
    tr.reconfigure({})
    assert tr._streams == {}


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


# ---------------------------------------------------------------- 문 후보 (2026-11-24)
def _runner_for_proposals():
    tr = vi.TargetedRunner.__new__(vi.TargetedRunner)
    tr.targets = [{"name": "demo_door", "ovd_propose": {"grounding_dino": {"min_score": 0.30, "max_area": 0.5}}}]
    tr.names, tr.alias = ["demo_door"], {}
    return tr


def test_gdino_proposes_far_door_that_yoloe_missed():
    """스캔 45°: YOLOE 0.12(후보 하한 0.15 미만) · GroundingDINO 0.52 — 후보가 하나 생기고, YOLOE 의 약한 상자가 근거로 붙는다."""
    tr = _runner_for_proposals()
    raw = [{"cls_name": "demo_door", "prompt": "door", "conf": 0.12, "xyxy": [104, 117, 153, 230], "arm": "yoloe"}]
    ovd = {"demo_door": [{"xyxy": [104, 118, 154, 231], "conf": 0.52, "arm": "grounding_dino", "prompt": "door"},
                         {"xyxy": [0, 0, 464, 399], "conf": 0.48, "arm": "grounding_dino", "prompt": "door"},
                         {"xyxy": [35, 168, 96, 267], "conf": 0.26, "arm": "grounding_dino", "prompt": "door"}]}
    out = tr._ovd_proposals([], raw, ovd, 464 * 400)
    assert len(out) == 1                              # 테두리 전체(넓이) · 0.26(점수) 는 거른다
    d = out[0]
    assert d["proposed_by"] == "grounding_dino" and d["conf"] == 0.52
    assert d["_by_arm"] == {"yoloe": {"door": (0.12, False)}} and d["arms"] == ["grounding_dino", "yoloe"]


def test_gdino_does_not_duplicate_an_existing_candidate():
    tr = _runner_for_proposals()
    dets = [{"name": "demo_door", "xyxy": [211, 116, 260, 228]}]
    ovd = {"demo_door": [{"xyxy": [210, 116, 261, 229], "conf": 0.78, "arm": "grounding_dino", "prompt": "door"}]}
    assert tr._ovd_proposals(dets, [], ovd, 464 * 400) == []


def test_proposed_candidate_counts_gdino_once_and_pairs_with_yoloe():
    import target_assess as ta
    d = {"name": "demo_door", "conf": 0.52, "xyxy": [104, 118, 154, 231], "lead": "door", "proposed_by": "grounding_dino",
         "_by_arm": {}, "_pair_with": []}
    ovd = [{"xyxy": [104, 118, 154, 231], "conf": 0.52, "arm": "grounding_dino"}]
    pct, txt, _ = ta.detector_support(d, ovd)
    assert abs(pct - 52.0) < 1e-6 and txt == "gdino 0.52"   # 0.52 를 두 번 세면 77
    d["_by_arm"] = {"yoloe": {"door": (0.12, False)}}
    assert abs(ta.detector_support(d, ovd)[0] - 100 * (1 - 0.88 * 0.48)) < 1e-6


def test_imgsz_floor_for_small_frames():
    import numpy as np

    class Arm:
        imgsz, follow_frame, min_size = 1280, True, 960
    assert vi._arm_imgsz(Arm, np.zeros((400, 464, 3), np.uint8)) == 960
    Arm.min_size = None
    assert vi._arm_imgsz(Arm, np.zeros((400, 464, 3), np.uint8)) == 480
    Arm.min_size = 960
    assert vi._arm_imgsz(Arm, np.zeros((1080, 1920, 3), np.uint8)) == 1280


# ---------------------------------------------------------------- 바닥선 거리 (2026-11-24)
G = {"frame": (464, 400), "crop": (114, 80, 351, 321), "a": 0.803, "b": 135.7, "k": 5214.4, "c": 5.1, "edge": 2.0,
     "range_cm": [50.0, 300.0]}


def test_ground_line_matches_box_height_on_the_fit():
    """바닥 관문 식 위에 정확히 놓인 박스면 바닥선 = 높이식."""
    from demo_box import ground_distance
    h = 26.87
    y2 = 0.803 * h + 135.7 + 80
    g = ground_distance([215.5, y2 - h, 254.3, y2], G)
    assert g["reason"] is None and abs(g["distance_cm"] - (5214.4 / h - 5.1)) < 0.5


def test_ground_line_on_saved_extinguisher_frames():
    """13:35 소화기 135° · 180° — 실제 1.5~2 m."""
    from demo_box import ground_distance
    assert 150 <= ground_distance([282.5, 210.0, 295.7, 241.7], G)["distance_cm"] <= 170
    assert 160 <= ground_distance([169.9, 212.0, 181.6, 239.8], G)["distance_cm"] <= 180


def test_ground_line_refusals():
    from demo_box import ground_distance
    assert "fisheye" in ground_distance([56.5, 214.3, 68.8, 247.6], G)["reason"]          # 225° — crop 밖
    assert ground_distance([200, 250, 240, 320], G)["reason"] == "bottom edge touches the frame"
    assert ground_distance([200, 150, 240, 200], G)["reason"] == "bottom edge above the floor line"
    far = ground_distance([200, 200, 240, 218], G)
    assert far["distance_cm"] > 300 and far["reason"] == "outside 50-300 cm"


def test_add_ground_distance_only_on_go1_frames():
    cfg = {"_go1_ground": G}
    dets = [{"xyxy": [169.9, 212.0, 181.6, 239.8]}]
    vi.add_ground_distance(dets, 464, 400, cfg)
    assert dets[0]["ground_distance"]["distance_cm"] is not None
    other = [{"xyxy": [169.9, 212.0, 181.6, 239.8]}]
    vi.add_ground_distance(other, 1280, 720, cfg)
    assert "ground_distance" not in other[0]


def test_go1_ground_from_fit_file():
    import cue_catalog as cc
    from demo_box import go1_ground
    g = go1_ground(cc.load_targets()["demo_box"])
    assert g["frame"] == (464, 400) and g["range_cm"] == [50.0, 300.0] and abs(g["k"] - 5214.4) < 0.1
