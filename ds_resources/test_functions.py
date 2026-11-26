"""기능 계약 — 어긴 계약은 ValueError, 23줄 표가 전부 읽히는지, 패턴별로 갈리는지, 쌍 규칙, 옛 cpu_tasks 호환
(functions/1, 2026-11-25).

계약을 고치면 여기서 드러난다. 값(cpu_m 같은 어림값)이 아니라 **모양과 규칙**을 지킨다 — 어림값은 실측이 대신한다.
"""
import copy

import pytest

from ds_resources import functions as F
from ds_resources import placement

#: 사용자가 정한 23줄 표의 기능 이름. 「경로 계획」만 둘로 쪼갰다(로컬 회피 safety 상주 · 전역 경로 mission 서버)
TABLE = {
    "motor_control": ("resident", "safety"),
    "emergency_stop": ("resident", "safety"),
    "sensor_collect": ("resident", "mission"),
    "camera_calibration": ("config", "mission"),
    "video_preprocess": ("device_only", "mission"),
    "object_detect": ("cascade", "mission"),
    "zeroshot_detect": ("cascade", "mission"),
    "targeted_detect": ("cascade", "mission"),           # 2026-11-26 사용자: 도메인 전용은 모델이 아니라 기능
    "river_detect": ("cascade", "mission"),
    "segmentation": ("cascade", "best_effort"),
    "depth_estimate": ("cascade", "mission"),
    "single_cam_track": ("device_only", "mission"),
    "multi_cam_track": ("split", "mission"),
    "reid": ("split", "mission"),
    "anomaly_detect": ("cascade", "mission"),
    "slam": ("split", "mission"),
    "vlm": ("server_only", "best_effort"),
    "grounding_explain": ("split", "best_effort"),
    "motion_forecast": ("cascade", "mission"),
    "local_avoidance": ("resident", "safety"),            # 경로 계획 ①
    "global_path_plan": ("server_only", "mission"),       # 경로 계획 ②
    "multi_robot_assign": ("server_only", "mission"),
    "digital_twin": ("split", "best_effort"),
    "model_train": ("server_only", "best_effort"),
    "model_update": ("pipeline", "mission"),
    "resource_status": ("pipeline", "mission"),
}

#: 어긴 계약 시험의 바탕 — 이 사전 하나는 반드시 읽혀야 한다
BASE = {"functions": {"demo": {
    "pattern": "cascade", "criticality": "mission", "trigger": "on_activity", "note": "시험용",
    "stages": {"device": {"optional": True, "cpu_m": 100, "mem_mb": 64, "egress_kb": 0.05,
                          "needs": {"sensors": ["camera"], "cores": 1.0}},
               "server": {"cpu_m": 200, "mem_mb": 256, "needs": {"accel": "gpu"}}}}}}


def reg(**patch):
    """BASE 를 복사하고 functions.demo 를 patch 로 덮는다 (중첩은 그대로 바꿔치기)"""
    r = copy.deepcopy(BASE)
    r["functions"]["demo"].update(copy.deepcopy(patch))
    return r


# ---------------------------------------------------------------- 계약 위반은 ValueError
def test_base_loads():
    f = F.load(reg())
    assert set(f) == {"demo"} and f["demo"]["pattern"] == "cascade"
    assert f["demo"]["priority_class"] == "ds-mission" and f["demo"]["forbid"] == []


@pytest.mark.parametrize("patch", [
    {"pattern": "sidecar"},                                   # 모르는 pattern
    {"criticality": "urgent"},                                # 모르는 criticality
    {"trigger": "on_boot"},                                   # 모르는 trigger
    {"forbid": ["edge"]},                                     # 모르는 쪽
    {"zone": "pi"},                                           # 모르는 기능 칸
    {"note": None},                                           # 있어야 하는 칸
    {"stages": {}},                                           # 단계가 없음
    {"stages": {"cloud": {"cpu_m": 1, "mem_mb": 1}}},         # 모르는 단계 이름
])
def test_function_level_violations(patch):
    with pytest.raises(ValueError):
        F.load(reg(**patch))


@pytest.mark.parametrize("stage", [
    {"pin": "cloud"},                                         # 모르는 pin
    {"pin": "server"},                                        # 그 단계 쪽이 아닌 pin (기기 단계에 server)
    {"state": "cached"},                                      # 모르는 state
    {"placement": "guess"},                                   # 모르는 placement
    {"roles": ["edge"]},                                      # 모르는 role
    {"needs": {"arch": "riscv"}},                             # 모르는 arch
    {"needs": {"sensors": ["radar"]}},                        # 모르는 센서
    {"needs": {"accel": "tpu"}},                              # 모르는 accel
    {"needs": {"zone": 1}},                                   # needs 에 모르는 칸
    {"deadline_ms": -5},                                      # 범위 밖
    {"egress_kb": "조금"},                                     # 수가 아님
    {"optional": "yes"},                                      # 참/거짓이 아님
    {"command": "python x.py"},                               # 글자 목록이어야 한다
    {"frames_kb": 10},                                        # 모르는 자원 칸 (frame_kb 오타)
])
def test_stage_level_violations(stage):
    r = reg()
    r["functions"]["demo"]["stages"]["device"].update(stage)
    with pytest.raises(ValueError):
        F.load(r)


def test_error_message_lists_what_is_allowed():
    """streams.check_code 방식 — 문구 비교로 되돌아가지 않게 가능한 값을 나열한다"""
    with pytest.raises(ValueError) as e:
        F.load(reg(pattern="sidecar"))
    for p in F.PATTERNS:
        assert p in str(e.value)
    with pytest.raises(ValueError) as e:
        F.load(reg(zone="pi"))
    assert "zone" in str(e.value) and "stages" in str(e.value)


def test_pattern_rules():
    with pytest.raises(ValueError):                           # cascade 는 기기 단계가 optional 이어야 한다
        r = reg()
        r["functions"]["demo"]["stages"]["device"]["optional"] = False
        F.load(r)
    with pytest.raises(ValueError):                           # cascade 는 두 단계가 다 있어야 한다
        r = reg()
        del r["functions"]["demo"]["stages"]["server"]
        F.load(r)
    with pytest.raises(ValueError):                           # split 은 어느 단계도 optional 일 수 없다
        F.load(reg(pattern="split"))
    with pytest.raises(ValueError):                           # device_only 의 서버 단계는 보조(optional)여야 한다
        r = reg(pattern="device_only")
        r["functions"]["demo"]["stages"]["device"]["optional"] = False
        F.load(r)
    with pytest.raises(ValueError):                           # server_only 의 기기 단계는 optional 이어야 한다
        r = reg(pattern="server_only")
        r["functions"]["demo"]["stages"]["device"]["optional"] = False
        F.load(r)
    with pytest.raises(ValueError):                           # resident 는 기기 단계 pin: device
        r = reg(pattern="resident")
        del r["functions"]["demo"]["stages"]["server"]
        r["functions"]["demo"]["stages"]["device"]["optional"] = False
        F.load(r)


def test_safety_cannot_leave_the_device():
    """안전은 서버로 옮기지 않는다 — 기기 단계가 있고 pin: device 여야 한다 (§10)"""
    r = reg(pattern="resident", criticality="safety")
    del r["functions"]["demo"]["stages"]["server"]
    r["functions"]["demo"]["stages"]["device"].update({"optional": False, "pin": "none"})
    with pytest.raises(ValueError):
        F.load(r)
    r["functions"]["demo"]["stages"]["device"]["pin"] = "device"
    assert F.load(r)["demo"]["criticality"] == "safety"


def test_forbidden_side_cannot_have_a_stage():
    r = reg(pattern="server_only", forbid=["device"])
    with pytest.raises(ValueError) as e:
        F.load(r)
    assert "device" in str(e.value)
    del r["functions"]["demo"]["stages"]["device"]
    assert F.load(r)["demo"]["forbid"] == ["device"]


def test_placeable_stage_needs_cpu_and_mem():
    r = reg()
    del r["functions"]["demo"]["stages"]["server"]["cpu_m"]
    with pytest.raises(ValueError) as e:
        F.load(r)
    assert "cpu_m" in str(e.value)


def test_bad_function_name():
    r = copy.deepcopy(BASE)
    r["functions"]["Object Detect"] = r["functions"].pop("demo")
    with pytest.raises(ValueError):
        F.load(r)


# ---------------------------------------------------------------- 레지스트리의 23줄 표
def test_registry_carries_the_whole_table():
    f = F.load()
    assert set(TABLE) <= set(f)
    assert len([k for k, v in f.items() if not v["legacy"]]) == len(TABLE) == 26   # 표 23줄, 경로 계획만 둘, 도메인 전용 기능 둘
    for name, (pattern, crit) in TABLE.items():
        assert (f[name]["pattern"], f[name]["criticality"]) == (pattern, crit), name
        assert f[name]["note"] and f[name]["stages"], name


def test_table_stage_sides():
    f = F.load()
    assert set(f["motor_control"]["stages"]) == {"device"}            # 서버 단계 없음
    assert set(f["global_path_plan"]["stages"]) == {"server"}
    assert set(f["model_train"]["stages"]) == {"server"} and f["model_train"]["forbid"] == ["device"]
    assert all(set(f[n]["stages"]) == {"device", "server"} for n in
               ("object_detect", "multi_cam_track", "reid", "slam", "digital_twin"))


def test_declared_details_that_the_table_asked_for():
    f = F.load()
    assert f["single_cam_track"]["stages"]["device"]["state"] == "sticky"
    assert f["multi_cam_track"]["stages"]["server"]["state"] == "sticky"
    slam = f["slam"]["stages"]["device"]
    assert slam["pin"] == "device" and slam["state"] == "sticky"
    assert f["video_preprocess"]["stages"]["device"]["pin"] == "device"              # 데이터 중력
    for n in ("motor_control", "emergency_stop", "local_avoidance"):                 # 안전은 기기 붙박이
        assert f[n]["stages"]["device"]["pin"] == "device"
        assert f[n]["priority_class"] == "ds-safety-high"
    for n in ("object_detect", "zeroshot_detect", "segmentation", "depth_estimate",
              "anomaly_detect", "motion_forecast"):                                  # cascade 의 기기 단계
        assert f[n]["stages"]["device"]["optional"] is True, n


def test_egress_is_not_frame_kb():
    """1차 거름은 받는 건 크고(압축 프레임) 보내는 건 수십 바이트다 — 두 칸은 다르다"""
    d = F.load()["object_detect"]["stages"]["device"]
    assert d["frame_kb"] == 120 and d["egress_kb"] < 0.1
    v = F.load()["video_preprocess"]["stages"]["device"]
    assert v["egress_kb"] < v["frame_kb"]                                            # 압축해서 내보낸다


def test_deadlines_are_tightest_on_safety():
    f = F.load()
    safety = max(f[n]["stages"]["device"]["deadline_ms"]
                 for n in ("motor_control", "emergency_stop", "local_avoidance"))
    assert safety <= 50
    assert f["global_path_plan"]["stages"]["server"]["deadline_ms"] > safety


# ---------------------------------------------------------------- expand
def test_expand_splits_by_pattern():
    f = F.load()
    placeable, residents, others = F.expand(f)
    assert {s["function"] for s in residents} == {"motor_control", "emergency_stop", "sensor_collect",
                                                  "local_avoidance"}
    assert {s["function"] for s in others} == {"camera_calibration", "model_update", "resource_status"}
    assert all(f[s["function"]]["pattern"] in F.PLACEABLE for s in placeable)
    # 상주 · 그밖은 배치 목록에 아예 없다 — 오케스트레이터는 그것을 놓지 않고 있는지만 본다
    assert not ({s["function"] for s in residents + others} & {s["function"] for s in placeable})


def test_expand_is_ordered_and_carries_the_whole_stage():
    f = F.load()
    placeable, _, _ = F.expand(f, ["reid", "object_detect"])
    assert [(s["function"], s["stage"]) for s in placeable] == [
        ("object_detect", "device"), ("object_detect", "server"), ("reid", "device"), ("reid", "server")]
    s = placeable[0]
    for k in ("function", "stage", "unit", "pin", "needs", "deadline_ms", "egress_kb", "state", "optional",
              "cpu_m", "mem_mb", "frame_kb", "fps", "proc_ms", "roles", "placement", "image", "command",
              "local_command", "pattern", "criticality", "trigger", "priority_class"):
        assert k in s, k


def test_expand_rejects_unknown_name():
    f = F.load()
    with pytest.raises(ValueError) as e:
        F.expand(f, ["object_detect", "teleport"])
    assert "teleport" in str(e.value) and "object_detect" in str(e.value)


def test_expand_of_nothing_is_empty():
    assert F.expand(F.load(), []) == ([], [], [])


# ---------------------------------------------------------------- pairs
def test_pairs_cascade_loses_quality_without_the_device_stage():
    f = F.load()
    stages, _, _ = F.expand(f, ["object_detect"])
    v = F.pairs(stages, [s for s in stages if s["stage"] == "server"])["object_detect"]
    assert v["verdict"] == "degraded" and v["missing"] == ["device"] and "품질" in v["reason"]
    assert F.pairs(stages)["object_detect"]["verdict"] == "ok"                       # placed 없음 = 전부 놓임


def test_pairs_split_is_refused_when_one_side_is_missing():
    f = F.load()
    stages, _, _ = F.expand(f, ["reid"])
    for keep in ("device", "server"):
        v = F.pairs(stages, [(("reid"), keep)])["reid"]
        assert v["verdict"] == "rejected" and "거절" in v["reason"]


def test_pairs_rejects_when_nothing_is_placed():
    stages, _, _ = F.expand(F.load(), ["object_detect"])
    assert F.pairs(stages, [])["object_detect"]["verdict"] == "rejected"


def test_pairs_accepts_stage_dicts_or_pairs_and_judges_every_pattern():
    f = F.load()
    stages, _, _ = F.expand(f, ["object_detect", "reid", "vlm", "global_path_plan"])
    only_server = [(s["function"], s["stage"]) for s in stages if s["stage"] == "server"]
    by_pair = F.pairs(stages, only_server)
    by_dict = F.pairs(stages, [s for s in stages if s["stage"] == "server"])
    assert by_pair == by_dict
    assert {k: v["verdict"] for k, v in by_pair.items()} == {
        "object_detect": "degraded",        # cascade — 기기 단계가 optional
        "reid": "rejected",                 # split — 한쪽이 없으면 기능 거절
        "vlm": "degraded",                  # server_only — 기기 단계는 선택적 경량
        "global_path_plan": "ok"}           # 서버 단계만 선언했고 그것이 놓였다
    with pytest.raises(ValueError):
        F.pairs(stages, ["reid"])


# ---------------------------------------------------------------- 옛 cpu_tasks 별칭
def test_old_cpu_task_is_read_as_a_single_stage_function():
    f = F.load()
    t = f["cam360_slice"]
    assert t["legacy"] is True and t["pattern"] == "device_only" and set(t["stages"]) == {"device"}
    assert t["criticality"] == "best_effort" and t["priority_class"] == placement.PRIORITY_CLASS
    assert t["stages"]["device"]["unit"] == "cam360_slice"                           # 배치표 이름이 바뀌지 않는다


def test_alias_keeps_every_field_place_cpu_reads():
    import yaml
    with open(F.REGISTRY, encoding="utf-8") as fh:
        old = (yaml.safe_load(fh) or {})["cpu_tasks"]["cam360_slice"]
    task = F.as_cpu_task(F.load()["cam360_slice"]["stages"]["device"])
    assert task["task"] == "cam360_slice"
    for k, v in old.items():
        assert task[k] == v, k                                                       # 칸 이름 · 뜻이 그대로다


def test_alias_task_still_places_with_placement_1():
    """옛 place_cpu 가 그대로 받는다 — 비교 측정 전이면 data_stream 기기에 둔다 (placement/1 규칙)"""
    task = F.as_cpu_task(F.load()["cam360_slice"]["stages"]["device"])
    out = {"cpu_units": [], "refused": []}
    placement.place_cpu(task, [], out, "book")
    assert out["cpu_units"][0]["task"] == "cam360_slice" and out["cpu_units"][0]["node"] == "book"
    assert out["cpu_units"][0]["requests"] == {"cpu_m": 1000, "mem_mb": 400}
    task["measured"] = {"book": 120.0, "pi13": 400.0}
    node = {"name": "pi13", "role": "sensor", "ready": True, "cpu_alloc_m": 4000.0, "mem_alloc_mb": 8000.0,
            "cpu_now_m": 100.0, "mem_now_mb": 500.0, "net": {"bw_mbps": 200.0}}
    out = {"cpu_units": [], "refused": []}
    placement.place_cpu(task, [node], out, "book")
    assert out["cpu_units"][0]["node"] == "pi13"                                     # 측정이 있으면 옮긴다


def test_functions_wins_over_a_cpu_task_with_the_same_name():
    r = copy.deepcopy(BASE)
    r["cpu_tasks"] = {"demo": {"cpu_m": 1, "mem_mb": 1, "roles": ["gpu"]}}
    f = F.load(r)
    assert f["demo"]["pattern"] == "cascade" and f["demo"]["legacy"] is False


def test_as_cpu_task_refuses_a_stage_without_resources():
    r = reg(pattern="config")
    r["functions"]["demo"]["stages"] = {"device": {"egress_kb": 1}}
    stages = F.expand(F.load(r), ["demo"])[2]
    with pytest.raises(ValueError):
        F.as_cpu_task(stages[0])


# ---------------------------------------------------------------- 소스별 기능 (2026-11-26)
def test_camera_stages_are_made_per_source():
    """카메라를 보는 기기 단계는 소스마다 하나다 — 카메라 3대면 셋. 서버 단계는 기능마다 하나"""
    f = F.load()
    st, _res, _oth = F.expand(f, ["object_detect"], sources=["pi12", "a2"])
    units = sorted(s["unit"] for s in st)
    assert units == ["object_detect.device@a2", "object_detect.device@pi12", "object_detect.server"]
    assert {s["unit"]: s.get("source") for s in st}["object_detect.device@pi12"] == "pi12"
    assert {s["unit"]: s.get("source") for s in st}["object_detect.server"] is None
    # per_source: false 인 단계는 늘리지 않는다 (지도는 기기에 하나다)
    st2, _r, _o = F.expand(f, ["slam"], sources=["pi12", "a2"])
    assert sorted(s["unit"] for s in st2) == ["slam.device", "slam.server"]


def test_functions_can_differ_per_source():
    """{소스: [기능...]} 이면 소스마다 고른 기능이 다르다"""
    f = F.load()
    st, _r, _o = F.expand(f, {"pi12": ["object_detect", "single_cam_track"], "a2": ["object_detect"]})
    assert sorted(s["unit"] for s in st) == ["object_detect.device@a2", "object_detect.device@pi12",
                                             "object_detect.server", "single_cam_track.device@pi12",
                                             "single_cam_track.server"]


def test_pairs_tells_which_source_is_missing():
    """한 소스에서만 빠지면 기능은 살고(degraded) 어느 소스인지 남는다"""
    f = F.load()
    st, _r, _o = F.expand(f, ["object_detect"], sources=["pi12", "a2"])
    placed = [s["unit"] for s in st if s["unit"] != "object_detect.device@a2"]
    g = F.pairs(st, placed)["object_detect"]
    assert g["verdict"] == "degraded" and g["sources_missing"] == ["a2(device)"]
    assert sorted(g["sources"]) == ["a2", "pi12"] and g["missing"] == []
    # 모든 소스에서 빠지면 예전처럼 단계가 통째로 없는 것이다
    g2 = F.pairs(st, [s["unit"] for s in st if s["stage"] == "server"])["object_detect"]
    assert g2["verdict"] == "degraded" and g2["missing"] == ["device"]


def test_functions_declare_which_models_they_can_use():
    """기능마다 쓸 수 있는 선택지(모델) — 런처가 「기능 → 모델」 로 고른다"""
    f = F.load()
    assert f["object_detect"]["options"] == ["yoloe"]              # 모델만 — 도메인 전용(targeted · river)은 기능
    assert f["zeroshot_detect"]["options"] == ["yoloe"]
    assert f["targeted_detect"]["options"] == ["targeted"] and f["river_detect"]["options"] == ["river"]
    assert f["depth_estimate"]["options"] == ["unidepth", "moge2_aerial"]
    assert f["single_cam_track"]["options"] == []        # 트래커는 모델이 없다 (ultralytics 안에 있다)
