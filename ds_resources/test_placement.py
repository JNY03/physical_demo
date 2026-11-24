import copy

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


def req(*gpu, cpu=()):
    return {"request_id": "r1", "gpu": list(gpu), "cpu": list(cpu)}


def G(models, sources, fps=None):
    return {"models": models, "sources": sources, "target_fps": fps or {s: 5.0 for s in sources}}


def test_fits_and_records_inputs():
    out = pl.plan(req(G(["yoloe"], ["pi13", "ep-001"])), {"nodes": [gpu_node()], "costs": COSTS}, "reg1", "wsl")
    u = out["gpu_units"][0]
    assert out["algorithm"] == "placement/1" and out["inputs"]["registry"] == "reg1"
    assert u["node"] == "wsl" and u["fps_cap"] == {"ep-001": 5.0, "pi13": 5.0}
    assert u["gpu_util"] == round(5 * 0.1 + 5 * 0.06, 3) and not out["refused"]


def test_same_input_same_plan():
    snap = {"nodes": [gpu_node(), pi("pi7"), pi("pi1")], "costs": COSTS}
    r = req(G(["yoloe"], ["pi13"]), G(["river"], ["a2"]))
    a = pl.plan(r, snap, "h", "wsl")
    b = pl.plan(copy.deepcopy(r), copy.deepcopy({"costs": COSTS, "nodes": list(reversed(snap["nodes"]))}), "h", "wsl")
    assert a["gpu_units"] == b["gpu_units"] and a["refused"] == b["refused"]


def test_throughput_full_scales_fps_by_p90_then_refuses():
    out = pl.plan(req(G(["yoloe"], ["pi13", "a2"], {"pi13": 5.0, "a2": 5.0})), {"nodes": [gpu_node()], "costs": COSTS},
                  "h", "wsl")                                        # 5×0.10 + 5×0.14 = 1.2 > 0.85 → 같은 비율로 낮춤
    u = out["gpu_units"][0]
    assert u["gpu_util"] <= pl.GPU_BUDGET + 1e-3 and not out["refused"]
    assert u["fps_cap"]["pi13"] == u["fps_cap"]["a2"] < 5.0
    assert {d["item"] for d in out["degraded"]} == {"gpu:yoloe:a2", "gpu:yoloe:pi13"}
    # 바닥 fps 로도 안 들어가는 경우 — 소스를 이름 역순으로 뺀다
    heavy = {"river": {"need_mb": 3360, "p90_ms": {"pi13": 600.0, "a2": 700.0}}}    # 둘 1.3 > 0.85, a2 혼자 0.7
    out = pl.plan(req(G(["river"], ["pi13", "a2"], {"pi13": 1.0, "a2": 1.0})), {"nodes": [gpu_node()], "costs": heavy},
                  "h", "wsl")
    assert [r["item"] for r in out["refused"]] == ["gpu:river:pi13"]
    assert out["gpu_units"][0]["sources"] == ["a2"]


def test_scaling_never_rounds_over_budget():
    # 실측 버그: pi13 60 fps × p90 135 ms → 비율 0.1049… → 6.296 이 6.30 으로 올라가 0.8505 > 0.85 로 소스가 빠졌다
    costs = {"yoloe": {"need_mb": 1560, "p90_ms": {"pi13": 135.0}}}
    for f in (60.0, 10.0, 7.0, 33.3):
        out = pl.plan(req(G(["yoloe"], ["pi13"], {"pi13": f})), {"nodes": [gpu_node()], "costs": costs}, "h", "wsl")
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
