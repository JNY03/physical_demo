import json

import pytest

from ds_resources import streams as st


def S(cpu=10.0, mem=20.0, ready=True, **kw):
    return dict({"role": "robot", "source": "k3s", "ready": ready, "cpu_pct": cpu, "mem_pct": mem}, **kw)


# ---------------------------------------------------------------- 계약
def test_orch_record_fields_are_strings():
    r = st.orch_record("fallback", "req1", mode="local", basis="saved", reason="k3s 안 됨",
                       request={"robots": ["ep-001"], "models": ["yoloe"]}, plan={"units": []}, t=0, execution_id="ex1")
    assert r["schema"] == "orch/1" and r["event"] == "fallback" and r["mode"] == "local" and r["basis"] == "saved"
    assert r["request_id"] == "req1" and r["execution_id"] == "ex1" and len(r["event_id"]) == 32
    assert json.loads(r["request"]) == {"models": ["yoloe"], "robots": ["ep-001"]}
    assert all(isinstance(v, str) for v in r.values())
    assert r["ts"].endswith("+09:00")


@pytest.mark.parametrize("kw", [dict(event="nope", request_id="r"), dict(event="planned", request_id=""),
                                dict(event="planned", request_id="r", mode="docker"),
                                dict(event="planned", request_id="r", basis="guess")])
def test_orch_record_rejects(kw):
    with pytest.raises(ValueError):
        st.orch_record(**kw)


def test_node_record_baseline_only_when_relevant():
    s = S(baseline={"cpu": 4})
    assert "baseline" in st.node_record("pi1", "added", s)
    assert "baseline" not in st.node_record("pi1", "usage", s)
    with pytest.raises(ValueError):
        st.node_record("pi1", "moved", s)


def test_latest_by_execution_dedupes_and_rebuilds():
    a = st.orch_record("applied", "q", execution_id="e1", event_id="1")
    b = st.orch_record("running", "q", execution_id="e1", event_id="2")
    c = st.orch_record("planned", "q2", event_id="3")
    last = st.latest_by_execution([a, b, b, a, c])                     # 겹친 기록 — 같은 event_id 는 처음 것만 센다
    assert last["e1"]["event"] == "running"                            # 늦게 다시 온 a(applied)가 running 을 덮지 않는다
    last = st.latest_by_execution([a, b, c])
    assert last["e1"]["event"] == "running" and last["request:q2"]["event"] == "planned"


# ---------------------------------------------------------------- 기록 억제
def test_first_sight_is_added_and_small_changes_are_dropped():
    f = st.NodeChangeFilter()
    assert [r["change"] for r in f.observe("pi1", S(), 0)] == ["added"]
    for t, cpu in enumerate([11, 13, 9, 12, 14, 8], start=1):      # 같은 구간 안 · 경계 근처 흔들림
        assert f.observe("pi1", S(cpu=float(cpu)), 100 + t) == []


def test_jitter_at_band_edge_is_suppressed():
    f = st.NodeChangeFilter()
    f.observe("pi1", S(cpu=19.0), 0)                                 # 구간 1 (10~20)
    for t, cpu in enumerate([20.5, 19.5, 21.0, 18.0], start=1):     # 경계 20 ± 2 안
        assert f.observe("pi1", S(cpu=cpu), 100 + t * 40) == []


def test_band_change_needs_confirmation_and_gap():
    f = st.NodeChangeFilter()
    f.observe("pi1", S(cpu=10.0), 0)
    assert f.observe("pi1", S(cpu=55.0), 10) == []                   # 한 번만 → 아직
    assert f.observe("pi1", S(cpu=55.0), 20) == []                   # 두 번째지만 마지막 기록에서 30초 안
    out = f.observe("pi1", S(cpu=56.0), 40)                           # 같은 구간 세 번째 · 30초 지남
    assert [r["change"] for r in out] == ["usage"] and out[0]["cpu_pct"] == "56.0"
    assert f.observe("pi1", S(cpu=57.0), 80) == []                    # 기록한 구간 안


def test_spike_that_returns_is_not_recorded():
    f = st.NodeChangeFilter()
    f.observe("pi1", S(cpu=10.0), 0)
    assert f.observe("pi1", S(cpu=90.0), 100) == []
    assert f.observe("pi1", S(cpu=11.0), 110) == []                   # 한 번 튀고 돌아옴
    assert f.observe("pi1", S(cpu=90.0), 120) == []                   # 다시 처음부터 센다


def test_ready_and_baseline_changes_are_immediate():
    f = st.NodeChangeFilter()
    f.observe("pi1", S(baseline={"boot": "a"}), 0)
    out = f.observe("pi1", S(ready=False, baseline={"boot": "a"}), 1)
    assert [r["change"] for r in out] == ["not_ready"]
    out = f.observe("pi1", S(ready=True, baseline={"boot": "b"}), 2)
    assert [r["change"] for r in out] == ["ready", "baseline"]
    assert json.loads(out[1]["baseline"]) == {"boot": "b"}


def test_service_mode_change_is_immediate():
    f = st.NodeChangeFilter()
    f.observe("pi7", S(service_mode="idle"), 0)
    out = f.observe("pi7", S(service_mode="active"), 1)                # 로봇이 돌기 시작
    assert [r["change"] for r in out] == ["service"] and out[0]["service_mode"] == "active"
    assert f.observe("pi7", S(service_mode="active"), 2) == []


def test_gpu_memory_band_counts_as_usage():
    f = st.NodeChangeFilter()
    g = lambda used: [{"name": "RTX", "mem_used_mb": used, "mem_total_mb": 6000}]  # noqa: E731
    f.observe("wsl", S(gpu=g(600)), 0)
    f.observe("wsl", S(gpu=g(4800)), 40)
    out = f.observe("wsl", S(gpu=g(4800)), 80)
    assert [r["change"] for r in out] == ["usage"]


def test_remove():
    f = st.NodeChangeFilter()
    assert f.remove("ghost") == []
    f.observe("pi1", S(), 0)
    out = f.remove("pi1", S(), 5)
    assert out[0]["change"] == "removed" and out[0]["ready"] == "0"
    assert [r["change"] for r in f.observe("pi1", S(), 6)] == ["added"]


# ---------------------------------------------------------------- 쓰기
class FakeRedis:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def xadd(self, stream, fields, maxlen=None, approximate=None):
        if self.fail:
            raise ConnectionError("down")
        self.rows.append((stream, fields, maxlen))


def test_writer_adds_seq_and_never_raises():
    r = FakeRedis()
    w = st.StreamWriter(r, maxlen=50)
    assert w.orch("requested", "run1", reason="x")
    assert w.nodes(st.NodeChangeFilter().observe("pi1", S(), 0)) == [True]
    assert [x[0] for x in r.rows] == [st.ORCH_STREAM, st.NODE_STREAM]
    assert r.rows[1][1]["seq"] == "2" and r.rows[0][2] == 50
    bad = st.StreamWriter(FakeRedis(fail=True))
    assert bad.orch("failed", "run2") is False and bad.fail == 1 and "down" in bad.err


# ---------------------------------------------------------------- 이유 코드 (M11 R4)
def test_reason_codes_are_checked_in_one_place():
    from ds_resources import streams as st
    assert {"gpu_mem", "no_gpu_node", "no_bandwidth", "unknown_bandwidth", "no_cpu_capability", "no_node",
            "input_wait"} <= set(st.REASON_CODES)
    item = st.reason_item("gpu:yoloe", "gpu_mem", "GPU 메모리 부족", stage="admit")
    assert item == {"item": "gpu:yoloe", "code": "gpu_mem", "reason": "GPU 메모리 부족", "stage": "admit"}
    with pytest.raises(ValueError):
        st.reason_item("gpu:yoloe", "gpu_memory", "오타")
    rec = st.orch_record("failed", "r1", reason="아무 문구", code="gpu_mem")
    assert rec["code"] == "gpu_mem"
    assert "code" not in st.orch_record("failed", "r1")                 # 코드가 없으면 칸도 없다 (칸 추가만)
    with pytest.raises(ValueError):
        st.orch_record("failed", "r1", code="nope")
    assert st.item_codes([item, {"item": "x", "reason": "옛 항목"}]) == {"gpu_mem"}


# ---------------------------------------------------------------- 작업 · 노드 특성 이유 코드와 기능 칸
def test_task_and_node_reason_codes_exist():
    assert {"no_sensor", "arch_mismatch", "no_accel", "thermal_guard", "deadline_miss",
            "pair_incomplete"} <= set(st.REASON_CODES)
    for code in ("no_sensor", "arch_mismatch", "no_accel", "thermal_guard", "deadline_miss", "pair_incomplete"):
        assert st.REASON_CODES[code]                                  # 한국어 설명이 있다
        assert st.check_code(code) == code
        assert st.orch_record("failed", "r1", code=code)["code"] == code
    with pytest.raises(ValueError):                                   # 모르는 코드는 여전히 거부
        st.check_code("no_sensors")
    with pytest.raises(ValueError):
        st.reason_item("det:ep-001", "thermal", "오타")


def test_function_stage_quality_only_when_given():
    r = st.orch_record("planned", "r1")
    assert "function" not in r and "stage" not in r and "quality" not in r   # 값이 없으면 칸도 없다
    r = st.orch_record("applied", "r1", function="follow_me", stage="device", quality="full")
    assert r["schema"] == "orch/1"                                    # 칸 추가는 계약 버전을 올리지 않는다
    assert r["function"] == "follow_me" and r["stage"] == "device" and r["quality"] == "full"
    assert all(isinstance(v, str) for v in r.values())
    r = st.orch_record("degraded", "r1", code="pair_incomplete", function="follow_me", quality="device_only")
    assert r["quality"] == "device_only" and "stage" not in r
    for stage in st.STAGES:
        assert st.orch_record("planned", "r1", stage=stage)["stage"] == stage
    for q in st.QUALITIES:
        assert st.orch_record("planned", "r1", quality=q)["quality"] == q


@pytest.mark.parametrize("kw", [dict(stage="edge"), dict(stage=""), dict(quality="half"), dict(quality="")])
def test_orch_record_rejects_bad_stage_and_quality(kw):
    with pytest.raises(ValueError):
        st.orch_record("planned", "r1", **kw)


def test_caps_is_a_node_change():
    assert "caps" in st.NODE_CHANGES
    rec = st.node_record("pi11", "caps", S())
    assert rec["change"] == "caps" and rec["schema"] == "node/1"
    assert "baseline" not in st.node_record("pi11", "caps", S(baseline={"arch": "arm64"}))
