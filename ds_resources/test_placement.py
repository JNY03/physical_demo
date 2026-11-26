import copy

import pytest

from ds_resources import placement as pl


def gpu_node(name="wsl", used=0.0, total=6141.0, **kw):
    return dict({"name": name, "role": "gpu", "ready": True, "gpus": [{"mem_total_mb": total, "mem_used_mb": used}],
                 "cpu_alloc_m": 20000, "mem_alloc_mb": 15000}, **kw)


def pi(name, role="robot", cpu_now=200, p95=300, bw=50.0, relay=False, **kw):
    return dict({"name": name, "role": role, "ready": True, "cpu_alloc_m": 4000, "mem_alloc_mb": 8000,
                 "cpu_now_m": cpu_now, "mem_now_mb": 600, "service": {"cpu_p95_m": p95, "mem_p95_mb": 700},
                 "net": {"bw_mbps": bw, "relay": relay}}, **kw)


COSTS = {"yoloe": {"need_mb": 1560, "basis": "estimate", "p90_ms": {"pi13": 100.0, "a2": 140.0, "ep-001": 60.0}},
         "river": {"need_mb": 3360, "basis": "estimate", "p90_ms": {"pi13": 600.0, "a2": 800.0}}}


#: 사람이 처리량 예산을 건 서버 — 서버 기본은 예산 없음 (2026-11-26)
BUDGET = {"gpu_budget": pl.GPU_BUDGET}


def req(*gpu, cpu=()):
    return {"request_id": "r1", "gpu": list(gpu), "cpu": list(cpu)}


def G(models, sources, fps=None):
    return {"models": models, "sources": sources, "target_fps": fps or {s: 5.0 for s in sources}}


def test_fits_and_records_inputs():
    out = pl.plan(req(G(["yoloe"], ["pi13", "ep-001"])), {"nodes": [gpu_node()], "costs": COSTS}, "reg1", "wsl")
    u = out["gpu_units"][0]
    assert out["algorithm"] == "placement/3" and out["inputs"]["registry"] == "reg1"
    assert u["node"] == "wsl" and u["fps_cap"] == {"ep-001": 5.0, "pi13": 5.0}
    assert u["gpu_util"] == round(5 * 0.1 + 5 * 0.06, 3) and not out["refused"]


def test_same_input_same_plan():
    snap = {"nodes": [gpu_node(), pi("pi7"), pi("pi1")], "costs": COSTS}
    r = req(G(["yoloe"], ["pi13"]), G(["river"], ["a2"]))
    a = pl.plan(r, snap, "h", "wsl")
    b = pl.plan(copy.deepcopy(r), copy.deepcopy({"costs": COSTS, "nodes": list(reversed(snap["nodes"]))}), "h", "wsl")
    assert a["gpu_units"] == b["gpu_units"] and a["refused"] == b["refused"]


def test_throughput_full_scales_fps_by_p90_then_refuses():
    out = pl.plan(req(G(["yoloe"], ["pi13", "a2"], {"pi13": 5.0, "a2": 5.0})), {"nodes": [gpu_node(limits=BUDGET)], "costs": COSTS},
                  "h", "wsl")                                        # 5×0.10 + 5×0.14 = 1.2 > 0.85 → 같은 비율로 낮춤
    u = out["gpu_units"][0]
    assert u["gpu_util"] <= pl.GPU_BUDGET + 1e-3 and not out["refused"]
    assert u["fps_cap"]["pi13"] == u["fps_cap"]["a2"] < 5.0
    assert {d["item"] for d in out["degraded"]} == {"gpu:yoloe:a2", "gpu:yoloe:pi13"}
    # 바닥 fps 로도 안 들어가는 경우 — p90 이 큰 소스부터 뺀다 (2026-11-25: 이름순이던 것을 바꿈)
    heavy = {"river": {"need_mb": 3360, "p90_ms": {"pi13": 600.0, "a2": 700.0}}}    # 둘 1.3 > 0.85, pi13 혼자 0.6
    out = pl.plan(req(G(["river"], ["pi13", "a2"], {"pi13": 1.0, "a2": 1.0})), {"nodes": [gpu_node(limits=BUDGET)], "costs": heavy},
                  "h", "wsl")
    assert [r["item"] for r in out["refused"]] == ["gpu:river:a2"]
    assert out["gpu_units"][0]["sources"] == ["pi13"]


def test_waiting_sources_are_dropped_before_live_ones_and_fps_rescaled():
    # 2026-11-25 실측: Go1(robot1, 영상 옴, s 420 ms) + drone · a2(꺼짐, 크기를 몰라 xl 1300 ms) → 예전에는 셋 다 거절
    costs = {"targeted": {"need_mb": 2000, "p90_ms": {"robot1": 420.0, "drone": 1300.0, "a2": 1300.0}}}
    g = G(["targeted"], ["robot1", "drone", "a2"], {"robot1": 30.0, "drone": 10.0, "a2": 10.0})
    g["waiting"] = ["drone", "a2"]
    out = pl.plan(req(g), {"nodes": [gpu_node(limits=BUDGET)], "costs": costs}, "h", "wsl")
    u = out["gpu_units"][0]
    assert u["sources"] == ["robot1"] and not out["refused"]
    assert u["fps_cap"]["robot1"] == 2.02                     # 남은 것끼리 다시 나눔: 0.85 / 0.42 내림
    assert sorted(w["item"] for w in out["waiting"]) == ["input:a2", "input:drone"]
    assert {w["code"] for w in out["waiting"]} == {"input_wait_unbudgeted"}


def test_estimate_that_fits_without_margin_is_placed_as_exclusive():
    # 추정 4650 × 1.2 = 5580 > 여유 5298 이지만 추정치 자체는 들어간다 — Executor 단독 측정과 같은 규칙
    node = gpu_node()
    node["gpus"][0]["mem_used_mb"] = node["gpus"][0]["mem_total_mb"] - 5298 - pl.SPARE_MB
    costs = {"targeted": {"need_mb": 5580, "estimate_mb": 4650, "basis": "estimate", "p90_ms": {"robot1": 420.0}}}
    out = pl.plan(req(G(["targeted"], ["robot1"], {"robot1": 2.0})), {"nodes": [node], "costs": costs}, "h", "wsl")
    assert out["gpu_units"] and out["gpu_units"][0].get("exclusive") is True
    costs["targeted"]["basis"] = "measured"                   # 실측이면 여유분까지 들어가야 한다
    out = pl.plan(req(G(["targeted"], ["robot1"], {"robot1": 2.0})), {"nodes": [node], "costs": costs}, "h", "wsl")
    assert not out["gpu_units"] and out["refused"][0]["code"] == "gpu_mem"


def test_scaling_never_rounds_over_budget():
    # 실측 버그: pi13 60 fps × p90 135 ms → 비율 0.1049… → 6.296 이 6.30 으로 올라가 0.8505 > 0.85 로 소스가 빠졌다
    costs = {"yoloe": {"need_mb": 1560, "p90_ms": {"pi13": 135.0}}}
    for f in (60.0, 10.0, 7.0, 33.3):
        out = pl.plan(req(G(["yoloe"], ["pi13"], {"pi13": f})), {"nodes": [gpu_node(limits=BUDGET)], "costs": costs}, "h", "wsl")
        assert not out["refused"], f
        assert out["gpu_units"][0]["gpu_util"] <= pl.GPU_BUDGET


def test_memory_full_refuses_with_reason_and_counts_earlier_units():
    out = pl.plan(req(G(["yoloe"], ["pi13"]), G(["river"], ["a2"])),
                  {"nodes": [gpu_node(used=1500)], "costs": COSTS}, "h", "wsl")
    assert [u["unit"] for u in out["gpu_units"]] == ["gpu:river"]       # river 3360 먼저(이름순) → yoloe 1560 이 안 들어감
    assert out["refused"][0]["item"] == "gpu:yoloe" and "GPU 메모리 부족" in out["refused"][0]["reason"]


def test_gpu_work_never_goes_to_pi():
    out = pl.plan(req(G(["yoloe"], ["pi13"])), {"nodes": [gpu_node(ready=False), pi("pi7")], "costs": COSTS}, "h", "wsl")
    assert not out["gpu_units"] and "파이로 옮기지 않는다" in out["refused"][0]["reason"]


def test_stale_nodes_get_no_new_work():
    out = pl.plan(req(G(["yoloe"], ["pi13"])), {"nodes": [gpu_node(stale=True)], "costs": COSTS}, "h", "wsl")
    assert not out["gpu_units"]
    t = {"task": "slice", "cpu_m": 500, "mem_mb": 200, "roles": ["sensor"]}
    out = pl.plan(req(cpu=[t]), {"nodes": [pi("pi6", role="sensor", stale=True)], "costs": {}}, "h", "wsl")
    assert out["refused"][0]["item"] == "cpu:slice"


def test_unseen_combo_is_refused_not_guessed():
    out = pl.plan(req(G(["moge2_aerial"], ["pi13"])), {"nodes": [gpu_node()], "costs": COSTS}, "h", "wsl")
    assert "비용표에 이 조합이 없음" in out["refused"][0]["reason"]


# ---------------------------------------------------------------- CPU
TASK = {"task": "relay", "cpu_m": 800, "mem_mb": 300, "roles": ["robot", "sensor"], "frame_kb": 150, "fps": 5,
        "proc_ms": 40}


def test_cpu_picks_most_room_and_sets_limits_and_priority():
    nodes = [pi("pi7", cpu_now=2000), pi("pi1", cpu_now=200)]
    out = pl.plan(req(cpu=[TASK]), {"nodes": nodes, "costs": {}}, "h", "wsl")
    u = out["cpu_units"][0]
    assert u["node"] == "pi1"
    assert u["limits"] == {"cpu_m": 1600, "mem_mb": 600} and u["priority_class"] == "ds-vision-low"
    assert u["net_ms_per_frame"] == 24.0                               # 150 KB × 8 ÷ 50 Mbit/s


def test_robot_service_p95_and_instant_load_are_reserved():
    busy = pi("pi7", cpu_now=100, p95=2600)                            # 4000 − 2600 − 800 = 600 < 800
    out = pl.plan(req(cpu=[TASK]), {"nodes": [busy], "costs": {}}, "h", "wsl")
    assert out["refused"]
    spike = pi("pi7", cpu_now=2600, p95=100)                           # 순간 부하도 같다
    assert pl.plan(req(cpu=[TASK]), {"nodes": [spike], "costs": {}}, "h", "wsl")["refused"]


def test_robot_with_unmeasured_active_load_keeps_half_free():
    idle_only = pi("pi7", cpu_now=100, p95=100)
    idle_only["service"]["active_measured"] = False                    # 로봇이 돌 때를 아직 못 잼
    big = dict(TASK, cpu_m=1500)                                       # 4000 − max(100, 2000) − 800 = 1200 < 1500
    assert pl.plan(req(cpu=[big]), {"nodes": [idle_only], "costs": {}}, "h", "wsl")["refused"]
    idle_only["service"]["active_measured"] = True                     # 재고 나면 실측으로
    assert pl.plan(req(cpu=[big]), {"nodes": [idle_only], "costs": {}}, "h", "wsl")["cpu_units"]


def test_cpu_ledger_within_one_plan():
    t2 = dict(TASK, task="relay2")
    out = pl.plan(req(cpu=[TASK, t2]), {"nodes": [pi("pi1", cpu_now=2000, p95=0)], "costs": {}}, "h", "wsl")
    assert [u["task"] for u in out["cpu_units"]] == ["relay"]           # 4000−2000−800 = 1200 → 첫 800 뒤 400 남음
    assert [r["item"] for r in out["refused"]] == ["cpu:relay2"]


def test_network_cost_excludes_slow_or_relayed_links():
    slow = pi("pi1", bw=10.0)                                          # 150 KB → 120 ms > 처리 40 ms
    assert pl.plan(req(cpu=[TASK]), {"nodes": [slow], "costs": {}}, "h", "wsl")["refused"]
    out = pl.plan(req(cpu=[TASK]), {"nodes": [pi("pi1", relay=True), pi("pi7")], "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] == "pi7"                         # 중계 벌점


def test_node_drop_moves_cpu_work():
    nodes = [pi("pi1"), pi("pi7", cpu_now=900)]
    assert pl.plan(req(cpu=[TASK]), {"nodes": nodes, "costs": {}}, "h", "wsl")["cpu_units"][0]["node"] == "pi1"
    nodes[0]["ready"] = False
    assert pl.plan(req(cpu=[TASK]), {"nodes": nodes, "costs": {}}, "h", "wsl")["cpu_units"][0]["node"] == "pi7"


def test_measured_placement_waits_for_comparison():
    t = dict(TASK, task="cam360_slice", placement="measured", roles=["sensor", "gpu"])
    out = pl.plan(req(cpu=[t]), {"nodes": [pi("pi6", role="sensor")], "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] == "wsl" and "비교 측정 전" in out["cpu_units"][0]["reason"]
    t["measured"] = {"pi6": 30.0, "wsl": 35.0}
    out = pl.plan(req(cpu=[t]), {"nodes": [pi("pi6", role="sensor"), gpu_node()], "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] in ("pi6", "wsl")


# ---------------------------------------------------------------- 이유 코드 (M11 R1 · R2a · R4)
def test_every_refusal_and_degrade_carries_a_known_code():
    from ds_resources.streams import REASON_CODES
    out = pl.plan(req(G(["yoloe"], ["pi13"]), G(["clip"], ["pi13"]), cpu=[TASK]),
                  {"nodes": [gpu_node(used=5000)], "costs": COSTS}, "h", "wsl")
    items = out["refused"] + out["degraded"]
    assert items and all(x["code"] in REASON_CODES for x in items)
    assert {x["code"] for x in out["refused"]} >= {"gpu_mem", "no_cost"}
    assert pl.plan(req(G(["yoloe"], ["pi13"])), {"nodes": [], "costs": COSTS}, "h", "wsl")["refused"][0]["code"] \
        == "no_gpu_node"


def test_cpu_refusal_tells_unknown_bandwidth_apart():
    out = pl.plan(req(cpu=[TASK]), {"nodes": [pi("pi1", bw=None), pi("pi7", bw=None)], "costs": {}}, "h", "wsl")
    r = out["refused"][0]
    assert r["code"] == "unknown_bandwidth" and "대역폭 아직 모름" in r["reason"]
    assert r["skipped"] == {"pi1": "unknown_bandwidth", "pi7": "unknown_bandwidth"}
    slow = pl.plan(req(cpu=[TASK]), {"nodes": [pi("pi1", bw=10.0)], "costs": {}}, "h", "wsl")["refused"][0]
    assert slow["code"] == "no_bandwidth"
    na = pi("pi1", bw=None)
    na["net"]["net_status"] = "unsupported"
    assert pl.plan(req(cpu=[TASK]), {"nodes": [na], "costs": {}}, "h", "wsl")["refused"][0]["code"] \
        == "bandwidth_unsupported"
    busy = pl.plan(req(cpu=[TASK]), {"nodes": [pi("pi7", p95=3500)], "costs": {}}, "h", "wsl")["refused"][0]
    assert busy["code"] == "no_node"
    # 한 노드는 바쁘고 한 노드는 대역폭을 아직 모른다 — 기다리면 놓일 수 있는 쪽(unknown_bandwidth)을 낸다
    mix = pl.plan(req(cpu=[TASK]), {"nodes": [pi("pi7", p95=3500), pi("pi1", bw=None)], "costs": {}}, "h", "wsl")
    assert mix["refused"][0]["code"] == "unknown_bandwidth"
    failed_keeps_old = pi("pi1", bw=80.0)
    failed_keeps_old["net"]["net_status"] = "failed"                      # 다시 재다 실패 — 앞서 잰 값으로 놓는다
    assert pl.plan(req(cpu=[TASK]), {"nodes": [failed_keeps_old], "costs": {}}, "h", "wsl")["cpu_units"]


def test_cpu_capability_off_blocks_only_cluster_nodes():
    nodes = [pi("pi1", cpu_tasks=False, cpu_why="PriorityClass ds-vision-low 없음"), gpu_node()]
    out = pl.plan(req(cpu=[TASK]), {"nodes": nodes, "costs": {}}, "h", "wsl")
    r = out["refused"][0]
    assert r["code"] == "no_cpu_capability" and "PriorityClass ds-vision-low 없음" in r["reason"]
    local_ok = dict(TASK, roles=["robot", "gpu"], frame_kb=None)
    out = pl.plan(req(cpu=[local_ok]), {"nodes": nodes, "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] == "wsl"                          # data_stream 기기에는 그대로 놓인다


# ---------------------------------------------------------------- 하드 조건 (placement/2, 작업 특성 × 노드 특성)
def hard(node, **kw):
    """노드 하나에 작업 하나 — 놓이면 노드 이름, 못 놓이면 거절 코드"""
    out = pl.plan(req(cpu=[dict(TASK, **kw)]), {"nodes": [node], "costs": {}}, "h", "wsl")
    return out["cpu_units"][0]["node"] if out["cpu_units"] else out["refused"][0]["code"]


def test_each_hard_condition_leaves_its_own_code():
    assert hard(pi("pi1", caps={"sensors": []}), needs={"sensors": ["camera"]}) == "no_sensor"
    assert hard(pi("pi1", caps={"sensors": ["imu"]}), needs={"sensors": ["camera", "imu"]}) == "no_sensor"
    assert hard(pi("pi1", caps={"arch": "amd64"}), needs={"arch": "arm64"}) == "arch_mismatch"
    assert hard(pi("pi1", caps={"accel": "none"}), needs={"accel": "npu"}) == "no_accel"
    assert hard(pi("pi1", thermal={"temp_c": 82.0}), ) == "thermal_guard"          # 지금 뜨겁다
    assert hard(pi("pi1", thermal={"temp_c": 40.0, "temp_max_c": 81.0})) == "thermal_guard"   # 최근 최고가 넘었다
    assert hard(pi("pi1", thermal={"temp_c": 50.0, "throttled": True})) == "thermal_guard"    # 이미 조여졌다
    assert hard(pi("pi1"), deadline_ms=10) == "deadline_miss"              # 전송 24 + 처리 40 > 마감 10
    assert hard(pi("pi1"), deadline_ms=200) == "pi1"                       # 마감 안에 든다
    # egress 도 전송에 든다 — 받는 150 KB + 내보내는 1850 KB = 320 ms > 마감 100
    assert hard(pi("pi1"), deadline_ms=100, egress_kb=1850) == "deadline_miss"
    assert hard(pi("pi1", thermal={"temp_c": 70.0})) == "pi1"              # 한계 아래면 그대로 놓인다


def test_unknown_node_caps_are_not_missing_caps():
    """가장 중요한 시험 — 모름(None)을 「없음」으로 다루면 placement/1 보다 퇴행한다"""
    needs = {"arch": "arm64", "sensors": ["camera", "imu"], "accel": "npu"}
    assert hard(pi("pi1"), needs=needs) == "pi1"                           # caps 가 아예 없는 노드
    assert hard(pi("pi1", caps={}), needs=needs) == "pi1"
    assert hard(pi("pi1", caps={"arch": None, "sensors": None, "accel": None}), needs=needs) == "pi1"
    assert hard(pi("pi1", caps={"model": "Raspberry Pi 5"}), needs=needs) == "pi1"
    assert hard(pi("pi1", thermal={}), needs=needs) == "pi1"               # 온도도 모르면 통과
    assert hard(pi("pi1", thermal={"temp_c": None, "throttled": None}), needs=needs) == "pi1"
    # 확인했는데 없음([]) 은 거부 — 모름과 가른다
    assert hard(pi("pi1", caps={"sensors": []}), needs=needs) == "no_sensor"
    ok, _code, why = pl.eligible(dict(TASK, needs=needs), pi("pi1"), "wsl")
    assert ok and "모름" in why                                            # 사유에 모름을 남긴다


def test_cores_reserved_is_taken_out_of_the_free_share():
    # 4000 − max(p95 300, 지금 200) − 여유 800 = 2900 → 코어 2개(2000 m) 를 빼면 900 ≥ 800 이라 아직 놓인다
    assert hard(pi("pi1", caps={"cores_reserved": 2})) == "pi1"
    assert hard(pi("pi1", caps={"cores_reserved": 3})) == "no_node"
    assert hard(pi("pi1", caps={"cores_reserved": None})) == "pi1"         # 모르면 0
    assert hard(pi("pi1", caps={"cores_reserved": 0})) == "pi1"


def test_pin_keeps_device_stages_off_the_server_and_server_stages_on_it():
    nodes = [pi("pi1"), gpu_node()]
    both = dict(TASK, roles=["robot", "gpu"])
    assert pl.plan(req(cpu=[dict(both, pin="device")]), {"nodes": nodes, "costs": {}}, "h",
                   "wsl")["cpu_units"][0]["node"] == "pi1"
    assert pl.plan(req(cpu=[dict(both, pin="server")]), {"nodes": nodes, "costs": {}}, "h",
                   "wsl")["cpu_units"][0]["node"] == "wsl"
    only_local = pl.plan(req(cpu=[dict(both, pin="device")]), {"nodes": [gpu_node()], "costs": {}}, "h", "wsl")
    assert only_local["refused"][0]["code"] == "no_node" and "pin: device" in only_local["refused"][0]["reason"]
    only_pi = pl.plan(req(cpu=[dict(both, pin="server")]), {"nodes": [pi("pi1")], "costs": {}}, "h", "wsl")
    assert only_pi["refused"][0]["code"] == "no_node" and "pin: server" in only_pi["refused"][0]["reason"]


def test_sticky_prefers_the_node_it_already_runs_on():
    nodes = [pi("pi1", cpu_now=200), pi("pi7", cpu_now=900)]               # 여유는 pi1 이 더 많다
    plain = pl.plan(req(cpu=[dict(TASK, current_node="pi7")]), {"nodes": nodes, "costs": {}}, "h", "wsl")
    assert plain["cpu_units"][0]["node"] == "pi1"                          # stateless 면 그냥 여유로 고른다
    stuck = pl.plan(req(cpu=[dict(TASK, state="sticky", current_node="pi7")]),
                    {"nodes": nodes, "costs": {}}, "h", "wsl")
    assert stuck["cpu_units"][0]["node"] == "pi7"                          # 옮기면 상태를 잃는다


def test_cool_node_wins_when_everything_else_is_equal():
    nodes = [pi("pi1", thermal={"temp_c": 70.0}), pi("pi7", thermal={"temp_c": 40.0})]
    out = pl.plan(req(cpu=[TASK]), {"nodes": nodes, "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] == "pi7"


def test_refusal_code_priority_puts_waitable_reasons_first():
    needs = {"arch": "arm64"}
    hot_and_wrong = [pi("pi1", caps={"arch": "amd64"}), pi("pi7", thermal={"temp_c": 90.0})]
    out = pl.plan(req(cpu=[dict(TASK, needs=needs)]), {"nodes": hot_and_wrong, "costs": {}}, "h", "wsl")
    assert out["refused"][0]["code"] == "thermal_guard"                    # 식으면 풀린다 → arch 보다 앞
    assert out["refused"][0]["skipped"] == {"pi1": "arch_mismatch", "pi7": "thermal_guard"}
    slow_and_late = [pi("pi1", bw=10.0), pi("pi7", bw=50.0)]               # pi1 대역폭 부족 · pi7 마감 초과
    out = pl.plan(req(cpu=[dict(TASK, deadline_ms=10)]), {"nodes": slow_and_late, "costs": {}}, "h", "wsl")
    assert out["refused"][0]["code"] == "deadline_miss"
    busy_and_sensorless = [pi("pi1", p95=3500), pi("pi7", caps={"sensors": []})]
    out = pl.plan(req(cpu=[dict(TASK, needs={"sensors": ["camera"]})]),
                  {"nodes": busy_and_sensorless, "costs": {}}, "h", "wsl")
    assert out["refused"][0]["code"] == "no_sensor"                        # 하드웨어가 바뀌어야 하는 쪽도 no_node 보다는 앞
    cap_off = [pi("pi1", cpu_tasks=False, thermal={"temp_c": 90.0})]
    assert pl.plan(req(cpu=[TASK]), {"nodes": cap_off, "costs": {}}, "h", "wsl")["refused"][0]["code"] \
        == "no_cpu_capability"


# ---------------------------------------------------------------- 기능 단위 배치 (functions 계약 × placement/2)
FN_REG = {"functions": {
    "grasp": {"pattern": "split", "criticality": "mission", "trigger": "on_request",
              "note": "집기 — 기기가 손을 쥐고 서버가 자세를 푼다",
              "stages": {"device": {"pin": "device", "needs": {"sensors": ["camera"]}, "cpu_m": 400, "mem_mb": 128,
                                    "roles": ["robot"], "state": "sticky"},
                         "server": {"pin": "server", "needs": {"accel": "gpu"}, "cpu_m": 400, "mem_mb": 128,
                                    "roles": ["gpu"]}}},
    "object_detect": {"pattern": "cascade", "criticality": "mission", "trigger": "on_activity",
                      "note": "탐지 — 기기에서 거르고 서버가 확인한다",
                      "stages": {"device": {"optional": True, "needs": {"sensors": ["camera"]}, "cpu_m": 600,
                                            "mem_mb": 256, "roles": ["robot"]},
                                 "server": {"pin": "server", "cpu_m": 500, "mem_mb": 256, "roles": ["gpu"]}}},
    "video_preprocess": {"pattern": "device_only", "criticality": "best_effort", "trigger": "always",
                         "note": "전처리 — 원본은 카메라 노드에만 있다. 서버 재인코딩은 보조다",
                         "stages": {"device": {"pin": "device", "needs": {"sensors": ["camera"]}, "cpu_m": 500,
                                               "mem_mb": 256, "roles": ["robot", "sensor"]},
                                    "server": {"pin": "server", "optional": True, "cpu_m": 400, "mem_mb": 256,
                                               "roles": ["gpu"]}}},
}}


def stages(*names):
    from ds_resources import functions
    placeable, _res, _other = functions.expand(functions.load(FN_REG), list(names) or None)
    return placeable


def fplan(names, nodes, measured=None):
    snap = {"nodes": nodes, "costs": {}, **({"measured": measured} if measured else {})}
    return pl.plan({"request_id": "r1", "functions": stages(*names)}, snap, "h", "wsl")


def test_split_rejection_cancels_the_stage_that_was_placed():
    out = fplan(["grasp"], [gpu_node(caps={"accel": "gpu"})])              # 로봇 노드가 없다 → 기기 단계를 못 놓는다
    assert out["cpu_units"] == []                                          # 놓였던 서버 단계까지 취소된다
    g = out["functions"]["grasp"]
    assert g["verdict"] == "rejected" and g["placed"] == ["server"] and g["missing"] == ["device"]
    assert "quality" not in g                                              # 거절이면 품질이 없다
    fn = [r for r in out["refused"] if r["item"] == "fn:grasp"][0]
    assert fn["code"] == "pair_incomplete" and fn["cancelled"] == ["grasp.server"] and fn["function"] == "grasp"
    assert [r["item"] for r in out["refused"] if r["code"] == "no_node"] == ["cpu:grasp.device"]


def test_split_with_both_sides_runs_full_and_carries_the_contract():
    out = fplan(["grasp"], [pi("pi1", caps={"sensors": ["camera", "imu"]}), gpu_node()])
    assert out["functions"]["grasp"] == {"verdict": "ok", "quality": "full", "placed": ["device", "server"],
                                         "missing": [], "reason": "선언한 단계가 모두 놓임 (split)"}
    units = {u["task"]: u for u in out["cpu_units"]}
    assert units["grasp.device"]["node"] == "pi1" and units["grasp.server"]["node"] == "wsl"
    assert units["grasp.device"]["function"] == "grasp" and units["grasp.device"]["stage"] == "device"
    assert units["grasp.device"]["priority_class"] == "ds-mission"          # criticality 가 고른 PriorityClass
    assert not out["refused"] and not out["degraded"]


def test_cascade_survives_without_its_device_stage_as_server_only():
    from ds_resources.streams import QUALITIES
    out = fplan(["object_detect"], [gpu_node()])                            # 로봇 노드가 없다
    g = out["functions"]["object_detect"]
    assert g["verdict"] == "degraded" and g["quality"] == "server_only" and g["quality"] in QUALITIES
    assert [u["task"] for u in out["cpu_units"]] == ["object_detect.server"]
    d = [x for x in out["degraded"] if x["item"] == "fn:object_detect"][0]
    assert d["code"] == "pair_incomplete" and d["quality"] == "server_only" and d["missing"] == ["device"]


def test_device_stage_alone_is_device_only():
    out = fplan(["video_preprocess"], [pi("pi1", caps={"sensors": ["camera"]})])   # 서버(보조) 단계를 놓을 곳이 없다
    g = out["functions"]["video_preprocess"]
    assert g["verdict"] == "degraded" and g["quality"] == "device_only"
    assert [u["task"] for u in out["cpu_units"]] == ["video_preprocess.device"]
    assert out["cpu_units"][0]["priority_class"] == "ds-vision-low"          # best_effort 는 그대로 낮음


def test_measured_stage_waits_on_the_server_but_a_pinned_one_never_does():
    def measured_stages(fn, side):                                           # 계약은 그대로 두고 그 단계만 measured 로
        return [dict(s, placement="measured") for s in stages(fn) if s["stage"] == side]

    nodes = [pi("pi1", caps={"sensors": ["camera"]}), gpu_node()]
    out = pl.plan({"request_id": "r1", "functions": measured_stages("object_detect", "device")},
                  {"nodes": nodes, "costs": {}}, "h", "wsl")
    device = out["cpu_units"][0]
    assert device["node"] == "wsl" and "비교 측정 전" in device["reason"]      # 비교 측정이 없으면 옮기지 않는다
    out = pl.plan({"request_id": "r1", "functions": measured_stages("object_detect", "device")},
                  {"nodes": nodes, "costs": {}, "measured": {"object_detect.device": {"pi1": 300.0, "wsl": 400.0}}},
                  "h", "wsl")
    assert out["cpu_units"][0]["node"] == "pi1"                              # 재고 나면 더 빠른 쪽으로
    out = pl.plan({"request_id": "r1", "functions": measured_stages("grasp", "device")},
                  {"nodes": nodes, "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] == "pi1"                              # pin: device 는 서버로 기다리지 않는다


def test_function_stages_see_their_needs_and_pin():
    # 뜨거운 노드라도 pin: device 단계는 **그 기기 말고 갈 곳이 없다** — placement/3 은 한도를 면제하고 둔다
    hot = pi("pi1", caps={"sensors": ["camera"]}, thermal={"temp_c": 90.0})
    out = fplan(["grasp"], [hot, gpu_node()])
    assert out["functions"]["grasp"]["verdict"] == "ok"
    u = [x for x in out["cpu_units"] if x["task"] == "grasp.device"][0]
    assert u["node"] == "pi1" and u["limit_mode"] == "none" and u["waived"] == ["thermal_guard"]
    assert "한도 면제" in u["reason"] and "90.0℃" in u["reason"]
    # 하드웨어 조건은 면제하지 않는다 — 없는 가속기는 면제로 생기지 않는다
    no_gpu = fplan(["grasp"], [pi("pi1", caps={"sensors": ["camera"]}), gpu_node(caps={"accel": "none"})])
    assert [r["code"] for r in no_gpu["refused"] if r["item"] == "cpu:grasp.server"] == ["no_accel"]


def test_old_cpu_path_is_byte_for_byte_what_placement_1_did():
    nodes = [pi("pi7", cpu_now=2000), pi("pi1", cpu_now=200)]
    out = pl.plan(req(cpu=[TASK]), {"nodes": nodes, "costs": {}}, "h", "wsl")
    assert pl.ALGORITHM == "placement/3" and out["algorithm"] == "placement/3"
    assert "functions" not in out                                           # 기능을 안 주면 칸이 없다
    assert out["cpu_units"] == [{"task": "relay", "node": "pi1", "net_ms_per_frame": 24.0,
                                 "requests": {"cpu_m": 800, "mem_mb": 300},
                                 "limits": {"cpu_m": 1600, "mem_mb": 600},
                                 "priority_class": "ds-vision-low",
                                 # placement/3 이 더한 칸 — 한도를 거는 쪽인가 · 누가 골랐나 · 우선순위
                                 "limit_mode": "capped", "placement_by": "auto", "rank": 2}]
    assert not out["refused"] and not out["degraded"]


# ---------------------------------------------------------------- 원본이 그 노드에 있을 때 (캐스케이드 1차 거름)
def _cam_node(name="pi7", bw=200.0, cpu=4000, mem=4000):
    """카메라가 달린(= 영상 소스를 맡은) 파이 노드"""
    return {"name": name, "role": "robot", "ready": True, "stale": False,
            "cpu_alloc_m": cpu, "cpu_now_m": 200, "mem_alloc_mb": mem, "mem_now_mb": 500,
            "caps": {"arch": "arm64", "sensors": ["camera", "i2c"], "accel": "none", "cores_reserved": None},
            "thermal": {"temp_c": 50.0}, "net": {"bw_mbps": bw, "relay": False},
            "service": {"cpu_p95_m": 0, "mem_p95_mb": 0, "active_measured": True}}


def _gate_task(**kw):
    """카메라를 요구하는 1차 거름 단계 — 원본 2.7 MB 를 받지 않고 결과 50 바이트만 내보낸다"""
    t = {"task": "gate.device", "cpu_m": 1600, "mem_mb": 512, "roles": ["robot"], "placement": "any",
         "frame_kb": 2700, "egress_kb": 0.05, "proc_ms": 120, "deadline_ms": 150,
         "needs": {"arch": "any", "sensors": ["camera"], "accel": "none"}}
    t.update(kw)
    return t


def test_frames_on_the_node_are_not_counted_as_transfer():
    """카메라가 그 노드에 있으면 받을 장은 전송이 아니다 — 내보내는 것만 센다"""
    node = _cam_node()
    per_frame, net_ms, code = pl._net(_gate_task(), node, "wsl")
    assert code is None
    assert per_frame == 0.0                                  # 받을 장이 거기 있다
    assert net_ms == pytest.approx(0.05 * 8 / 200.0)         # 내보내는 50 바이트만
    ok, why_code, _ = pl.eligible(_gate_task(), node, "wsl")
    assert (ok, why_code) == (True, None)                    # 마감 150 ms 안에 든다


def test_frames_from_elsewhere_still_count():
    """같은 작업이라도 카메라가 없는 노드에서는 원본을 받아 와야 한다 — 마감을 넘는다"""
    node = _cam_node(name="pi13")
    node["caps"]["sensors"] = []                             # 확인했는데 카메라 없음
    per_frame, net_ms, code = pl._net(_gate_task(), node, "wsl")
    assert code is None and per_frame == pytest.approx(2700 * 8 / 200.0)
    assert net_ms + 120 > 150.0                              # 전송 108 + 처리 120 → 마감 150 초과
    # 센서 조건에서 먼저 떨어진다(카메라를 요구하므로) — 전송 계산은 위에서 이미 확인했다
    ok, why_code, _ = pl.eligible(_gate_task(), node, "wsl")
    assert (ok, why_code) == (False, "no_sensor")


def test_unknown_sensors_are_conservative_about_transfer():
    """센서를 모르면 원본이 거기 있다고 단정하지 않는다(전송이 있다고 본다)"""
    node = _cam_node(name="pi1")
    node["caps"]["sensors"] = None                           # 모름
    per_frame, _net_ms, code = pl._net(_gate_task(), node, "wsl")
    assert code is None and per_frame == pytest.approx(2700 * 8 / 200.0)
    ok, why_code, _ = pl.eligible(_gate_task(), node, "wsl")
    assert (ok, why_code) == (False, "deadline_miss")        # 센서는 모름이라 통과, 마감에서 떨어진다


def test_measured_shortcut_does_not_put_a_camera_task_on_a_cameraless_server():
    """비교 측정 전 지름길이 하드웨어 조건을 넘지 않는다 — 카메라 없는 서버에 1차 거름을 얹지 않는다"""
    here = {"name": "wsl", "role": "gpu", "ready": True, "stale": False,
            "cpu_alloc_m": 24000, "cpu_now_m": 1000, "mem_alloc_mb": 31000, "mem_now_mb": 8000,
            "caps": {"arch": "amd64", "sensors": [], "accel": "gpu", "cores_reserved": 0},
            "thermal": {}, "net": {}}
    out = {"cpu_units": [], "refused": [], "degraded": []}
    pl.place_cpu(_gate_task(placement="measured", roles=["robot", "gpu"]), [here], out, "wsl")
    assert out["cpu_units"] == []                            # 지름길을 타지 않았다
    assert [x["code"] for x in out["refused"]] == ["no_sensor"]


def test_measured_shortcut_still_works_without_hardware_needs():
    """하드웨어 조건이 없는 옛 작업은 예전처럼 지름길로 data_stream 기기에 놓인다"""
    here = {"name": "wsl", "role": "gpu", "ready": True, "stale": False,
            "cpu_alloc_m": 24000, "cpu_now_m": 1000, "mem_alloc_mb": 31000, "mem_now_mb": 8000,
            "caps": {"arch": "amd64", "sensors": [], "accel": "gpu", "cores_reserved": 0},
            "thermal": {}, "net": {}}
    task = {"task": "cam360_slice", "cpu_m": 800, "mem_mb": 300, "roles": ["small", "gpu"],
            "placement": "measured", "frame_kb": 970, "proc_ms": 120}
    out = {"cpu_units": [], "refused": [], "degraded": []}
    pl.place_cpu(task, [here], out, "wsl")
    assert [(u["task"], u["node"]) for u in out["cpu_units"]] == [("cam360_slice", "wsl")]


# ================================================================ 한도 정책 · 우선순위 · 수동 배치 (placement/3)
def _safety_reg():
    """안전 기능 하나 — pin: device · criticality safety. 그 기기 말고 갈 곳이 없는 일이다"""
    return {"functions": {
        "safety_stop": {"pattern": "device_only", "criticality": "safety", "trigger": "always",
                        "note": "비상 정지 — 기기 안에서만",
                        "stages": {"device": {"pin": "device", "cpu_m": 1200, "mem_mb": 512, "roles": ["robot"]}}},
        "video_preprocess": {"pattern": "device_only", "criticality": "best_effort", "trigger": "always",
                             "note": "전처리",
                             "stages": {"device": {"pin": "device", "cpu_m": 900, "mem_mb": 400,
                                                   "roles": ["robot"]}}}}}


def _stages_of(reg, *names):
    from ds_resources import functions
    placeable, _r, _o = functions.expand(functions.load(reg), list(names) or None)
    return placeable


def test_server_has_no_limit_so_work_is_not_refused_for_policy_room():
    """서버(data_stream 기기)에는 한도를 걸지 않는다 — 정책으로 비워 둔 몫 때문에 일이 거절되던 것을 멈춘다"""
    # 헤드룸 · 지금 부하를 보면 들어갈 자리가 없는 서버 (쓰는 중 14000 / 15000 MB)
    full = gpu_node(mem_alloc_mb=15000, mem_now_mb=14000, cpu_alloc_m=20000, cpu_now_m=19000)
    t = {"task": "big", "cpu_m": 2000, "mem_mb": 900, "roles": ["gpu"], "proc_ms": 10}
    out = pl.plan(req(cpu=[t]), {"nodes": [full], "costs": {}}, "h", "wsl")
    u = out["cpu_units"][0]
    assert u["node"] == "wsl" and u["limit_mode"] == "none" and not out["refused"]
    assert u["limits"] == {"cpu_m": None, "mem_mb": 1800}        # CPU 상한은 없고 메모리 상한은 그대로
    assert "no_node" in u["waived"] and "한도 면제" in u["reason"]


def test_on_device_still_has_a_limit_for_work_that_could_move():
    """온디바이스는 한도가 있다 — 옮길 수 있는 일은 그 기기의 몫을 넘지 않는다"""
    busy = pi("pi1", cpu_now=3900)                               # 쓸 수 있는 몫이 거의 없다
    t = dict(TASK, cpu_m=2000)
    out = pl.plan(req(cpu=[t]), {"nodes": [busy]}, "h", "wsl")
    assert not out["cpu_units"] and out["refused"][0]["code"] == "no_node"


def test_work_that_must_stay_on_the_device_is_waived_there_too():
    """반드시 그 기기에서 해야 하는 일(pin: device · safety)은 온디바이스에서도 한도를 면제한다 —
    옮길 자리가 없어 거절은 「안 돈다」는 뜻뿐이다"""
    hot_busy = pi("pi1", cpu_now=3900, thermal={"temp_c": 85.0})
    out = pl.plan({"request_id": "r1", "functions": _stages_of(_safety_reg(), "safety_stop")},
                  {"nodes": [hot_busy, gpu_node()], "costs": {}}, "h", "wsl")
    u = out["cpu_units"][0]
    assert u["node"] == "pi1" and u["limit_mode"] == "none" and not out["refused"]
    assert sorted(u["waived"]) == ["no_node", "thermal_guard"]
    assert u["priority_class"] == "ds-safety-high" and u["limits"]["cpu_m"] is None


def test_safety_takes_the_room_first_and_best_effort_keeps_its_limit():
    """우선순위 — 안전이 먼저 자리를 잡는다. best_effort 는 pin 이 device 라도 한도를 지켜,
    남은 자리가 없으면 거절된다(없으면 그만인 일에 기기의 제어 몫을 내주지 않는다)"""
    tight = pi("pi1", cpu_now=3000, p95=0)                       # 물리적으로 남은 몫 1000 m
    out = pl.plan({"request_id": "r1",
                   "functions": _stages_of(_safety_reg(), "safety_stop", "video_preprocess")},
                  {"nodes": [tight], "costs": {}}, "h", "wsl")
    units = {u["task"]: u for u in out["cpu_units"]}
    assert list(units) == ["safety_stop.device"]                 # 안전만 놓인다 (1200 m > 남은 1000 m 이지만 면제)
    assert units["safety_stop.device"]["limit_mode"] == "none" and units["safety_stop.device"]["rank"] == 0
    ref = [r for r in out["refused"] if r["item"] == "cpu:video_preprocess.device"][0]
    assert ref["code"] == "no_node"                              # best_effort 는 한도 안에서만


def test_preempt_pulls_lower_priority_work_but_never_higher():
    """면제로 들어온 일이 노드를 넘치게 했으면 **그보다 낮은** 일이 배치표에서 빠진다 (k8s PriorityClass 와 같다).
    낮은 일이 높은 일을 쫓아내지는 않는다."""
    node = pi("pi1", cpu_now=0, p95=0)                           # 물리적으로 4000 m · 7300 MB
    def unit(task, node_name, cpu, rank, mode):
        return {"task": task, "node": node_name, "limit_mode": mode, "rank": rank,
                "requests": {"cpu_m": cpu, "mem_mb": 100}, "limits": {"cpu_m": None, "mem_mb": 200},
                "priority_class": "x"}
    out = {"cpu_units": [unit("low", "pi1", 2000, 2, "capped"), unit("mid", "pi1", 1000, 1, "capped"),
                         unit("safe", "pi1", 3000, 0, "none")], "refused": [], "degraded": []}
    pl.preempt(out, [node], "wsl")
    assert [u["task"] for u in out["cpu_units"]] == ["mid", "safe"]       # 2000+1000+3000 > 4000 → low 가 비킨다
    assert [r["code"] for r in out["refused"]] == ["preempted"]
    assert out["refused"][0]["item"] == "cpu:low" and "자리를 내주고" in out["refused"][0]["reason"]
    # 면제된 쪽이 낮은 우선순위면 높은 일을 쫓아내지 않는다 — 넘친 채로 두고 그대로 둔다
    out2 = {"cpu_units": [unit("mission", "pi1", 3000, 1, "capped"), unit("extra", "pi1", 3000, 2, "none")],
            "refused": [], "degraded": []}
    pl.preempt(out2, [node], "wsl")
    assert [u["task"] for u in out2["cpu_units"]] == ["mission", "extra"] and not out2["refused"]


def test_manual_placement_keeps_hardware_rules_but_waives_limits():
    """수동 배치 — 사람이 고른 자리로 간다(한도 면제). 없는 센서는 면제로 생기지 않는다"""
    hot = pi("pi1", cpu_now=3900, caps={"sensors": ["camera"]}, thermal={"temp_c": 90.0})
    other = pi("pi7", caps={"sensors": ["camera"]})
    snap = {"nodes": [hot, other, gpu_node()], "costs": {}}
    st = stages("object_detect")                                 # cascade — 기기 단계는 어느 로봇에나 갈 수 있다
    pick = {"request_id": "r1", "functions": st, "manual": {"object_detect.device": "pi1"}}
    out = pl.plan(pick, snap, "h", "wsl")
    u = [x for x in out["cpu_units"] if x["task"] == "object_detect.device"][0]
    assert u["node"] == "pi1" and u["placement_by"] == "manual" and u["limit_mode"] == "none"
    assert "수동 배치" in u["reason"]
    # 센서가 없는 노드를 고르면 그 하드웨어 이유 코드로 거절한다
    no_cam = {"request_id": "r1", "functions": st, "manual": {"object_detect.device": "pi9"}}
    out2 = pl.plan(no_cam, {"nodes": [hot, other, gpu_node(), pi("pi9", caps={"sensors": []})], "costs": {}}, "h", "wsl")
    ref = [r for r in out2["refused"] if r["item"] == "cpu:object_detect.device"][0]
    assert ref["code"] == "no_sensor" and "수동으로 고른 노드 pi9" in ref["reason"]


def test_manual_none_excludes_and_unknown_node_is_reported():
    out = pl.plan(req(cpu=[TASK]) | {"manual": {"relay": "none"}}, {"nodes": [pi("pi1")], "costs": {}}, "h", "wsl")
    assert not out["cpu_units"] and out["refused"][0]["code"] == "manual_excluded"
    out = pl.plan(req(cpu=[TASK]) | {"manual": {"relay": "pi404"}}, {"nodes": [pi("pi1")], "costs": {}}, "h", "wsl")
    assert out["refused"][0]["code"] == "manual_unavailable" and "pi404" in out["refused"][0]["reason"]


def test_gpu_budget_is_shared_between_bundles_but_keeps_a_heavy_one_alive():
    """기기별 모델 매칭으로 묶음이 둘이 되면 처리량 예산을 소스 수로 나눈다 —
    먼저 온 묶음이 다 쓰지 않게. 다만 제 몫으로 바닥 fps 도 못 도는 무거운 묶음은 남은 예산으로 살린다"""
    two = req(G(["yoloe"], ["pi13"], {"pi13": 10.0}), G(["yoloe"], ["a2"], {"a2": 10.0}))
    out = pl.plan(two, {"nodes": [gpu_node(limits=BUDGET)], "costs": COSTS}, "h", "wsl")
    caps = {s: u["fps_cap"][s] for u in out["gpu_units"] for s in u["fps_cap"]}
    assert set(caps) == {"pi13", "a2"}                           # 둘 다 산다 (예전에는 먼저 온 쪽이 다 먹었다)
    assert sum(u["gpu_util"] for u in out["gpu_units"]) <= pl.GPU_BUDGET + 1e-6


def test_limits_are_per_node_and_of_several_kinds():
    """한도는 한 가지가 아니다 — 비율(CPU · 메모리) · 코어 · 온도 · GPU 예비. 노드마다 다르게 걸 수 있다"""
    n = pi("pi1", cpu_now=0, p95=0)                               # 4000 m · 8000 MB
    cpu, _mem = pl._cpu_room(n)
    assert round(cpu) == round(4000 - 4000 * pl.CPU_HEADROOM)     # 기본 20%
    assert round(pl._cpu_room(dict(n, limits={"headroom_cpu": 0.0}))[0]) == 4000      # 0% 면 꽉 채운다
    assert round(pl._cpu_room(dict(n, limits={"headroom_cpu": 0.5}))[0]) == 2000
    assert round(pl._cpu_room(dict(n, limits={"headroom_mem": 0.5}))[1]) == round(8000 - 700 - 4000)
    assert round(pl._cpu_room(dict(n, limits={"cores_reserved": 2}))[0]) == round(4000 - 2000 - 800)
    assert round(pl._cpu_room(dict(n, limits={"headroom_cpu": 2.0}))[0]) == round(4000 * 0.1)   # 범위 밖은 잘라 쓴다
    # 온도 한계도 노드마다
    hot = pi("pi1", thermal={"temp_c": 78.0})
    task = dict(TASK, roles=["robot"])
    assert pl.eligible(task, hot, "wsl")[1] == "thermal_guard"                        # 기본 75℃
    assert pl.eligible(task, dict(hot, limits={"temp_max": 85.0}), "wsl")[0] is True  # 85℃ 로 올리면 통과


def test_server_has_no_limit_by_default_but_takes_one_when_set():
    """서버는 기본이 「없음」이다. 사람이 걸면 그때부터 서버도 그 한도를 지킨다 (2026-11-26)"""
    full = gpu_node(mem_alloc_mb=15000, mem_now_mb=14000, cpu_alloc_m=20000, cpu_now_m=19000)
    t = {"task": "big", "cpu_m": 2000, "mem_mb": 900, "roles": ["gpu"], "proc_ms": 10}
    assert pl.limit_mode(t, full, "wsl") == "none"                                     # 기본 면제
    out = pl.plan(req(cpu=[t]), {"nodes": [full], "costs": {}}, "h", "wsl")
    assert out["cpu_units"][0]["node"] == "wsl"
    # 한도를 걸면 서버도 지킨다 — 자리가 없으면 거절된다
    capped = dict(full, limits={"headroom_cpu": 0.1, "headroom_mem": 0.1})
    assert pl.limit_mode(t, capped, "wsl") == "capped"
    out2 = pl.plan(req(cpu=[t]), {"nodes": [capped], "costs": {}}, "h", "wsl")
    assert not out2["cpu_units"] and out2["refused"][0]["code"] == "no_node"
    # 반드시 그 기기에서 할 일은 한도를 걸어도 면제다 (옮길 자리가 없다)
    safety = dict(t, criticality="safety")
    assert pl.limit_mode(safety, capped, "wsl") == "none"


def test_gpu_spare_can_be_set_per_node():
    """GPU 에 남겨 둘 메모리도 노드마다 — 기본은 SPARE_MB"""
    node = gpu_node(used=4000.0)                                   # 6141 - 4000 = 2141 남음
    costs = {"yoloe": {"need_mb": 1800, "basis": "measured", "p90_ms": {"pi13": 100.0}}}
    snap = {"nodes": [node], "costs": costs}
    out = pl.plan(req(G(["yoloe"], ["pi13"])), snap, "h", "wsl")
    assert out["gpu_units"] and out["gpu_units"][0]["node"] == "wsl"        # 1800 ≤ 2141 - 300
    tight = dict(node, limits={"gpu_spare_mb": 1000.0})
    out2 = pl.plan(req(G(["yoloe"], ["pi13"])), {"nodes": [tight], "costs": costs}, "h", "wsl")
    assert not out2["gpu_units"] and out2["refused"][0]["code"] == "gpu_mem"


def test_server_without_gpu_budget_runs_heavy_sources_at_target_fps():
    # 2026-11-26 실측: cam360 targeted 가 1 fps 로도 예산 0.85 를 넘어 거절 — 서버는 처리량 예산이 정책 몫이라 면제
    costs = {"targeted": {"need_mb": 2000, "basis": "measured", "p90_ms": {"cam360": 1800.0, "robot1": 420.0}}}
    g = G(["targeted"], ["cam360", "robot1"], {"cam360": 5.0, "robot1": 20.0})
    out = pl.plan(req(g), {"nodes": [gpu_node()], "costs": costs}, "h", "wsl")
    u = out["gpu_units"][0]
    assert not out["refused"] and not out["degraded"] and u["limit_mode"] == "none" and "한도 면제" in u["reason"]
    assert u["sources"] == ["cam360", "robot1"] and u["fps_cap"] == {"cam360": 5.0, "robot1": 20.0}
    out = pl.plan(req(g), {"nodes": [gpu_node(limits=BUDGET)], "costs": costs}, "h", "wsl")
    assert out["degraded"] and "limit_mode" not in out["gpu_units"][0]     # 사람이 걸면 예전 규칙
    big = {"targeted": {"need_mb": 9000, "basis": "measured", "p90_ms": {"cam360": 1800.0}}}
    out = pl.plan(req(G(["targeted"], ["cam360"])), {"nodes": [gpu_node()], "costs": big}, "h", "wsl")
    assert out["refused"][0]["code"] == "gpu_mem"                           # 물리 한계(GPU 메모리)는 그대로
