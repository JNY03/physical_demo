import json

import pytest

from ds_resources import gpu_executor as ge
from ds_resources.state import StateFile


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeProc:
    n = 4000

    def __init__(self, *a, **kw):
        FakeProc.n += 1
        self.pid, self.rc, self.cmd = FakeProc.n, None, a[0]

    def poll(self):
        return self.rc


class FakeRedis:
    def __init__(self):
        self.kv = {}

    def scan_iter(self, pattern, count=None):
        suffix = pattern.split("*")[-1]
        return [k for k in self.kv if k.startswith(ge.TELEMETRY_PREFIX) and k.endswith(suffix)]

    def get(self, k):
        return self.kv.get(k)


def gpus(used):
    return lambda timeout=2: [{"name": "RTX", "mem_total_mb": 6141.0, "mem_used_mb": float(used)}]


def make(tmp_path, used=0.0, clock=None):
    r = FakeRedis()
    ex = ge.GpuExecutor(StateFile(tmp_path / "s.json"), r, gpus=gpus(used), popen=FakeProc,
                        clock=clock or Clock(), python="py", log=lambda *_: None)
    return ex, r


# ---------------------------------------------------------------- 추정 · 필요량
def test_estimate_counts_registry_dependencies():
    assert ge.estimate_mb(["yoloe"])[0] == 1300.0
    assert ge.estimate_mb(["river"])[0] == 2800.0
    mb, need = ge.estimate_mb(["targeted"], {"models": {"targeted": {"targets": ["demo_door"]}}, "features": {"reid": True}},
                              {"models": {"targeted": {"distance": True}}})
    assert {"reid", "depth", "yoloe_text"} <= set(need)
    assert "grounding_dino" in need                                   # demo_door 는 GDINO 근거를 선언한다
    assert mb >= 400 + 900 + 700 + 1300 + 1300


def test_required_prefers_trusted_measurement():
    assert ge.required_mb(1300) == (1300 * 1.2, "estimate")
    assert ge.required_mb(1300, ({"gpu_mb_peak": 2000}, "measured", False)) == (2000 * 1.2, "measured")
    assert ge.required_mb(5370, ({"gpu_mb_peak": 3000}, "measured", False)) == (3000 * 1.2, "measured")
    assert ge.required_mb(1300, ({"gpu_mb_peak": 2000}, "learning", False))[1] == "estimate"
    assert ge.required_mb(1300, ({"gpu_mb_peak": 2000}, "measured", True))[1] == "estimate"      # stale
    # torch+ctx 는 onnxruntime(CLIP) 몫이 빠져 있다 → 레지스트리 몫을 더한다
    assert ge.required_mb(0, ({"gpu_mb_peak": 2000, "gpu_basis": "torch+ctx"}, "measured", False), 350)[0] == 2350 * 1.2
    assert ge.required_mb(0, ({"gpu_mb_peak": 2000, "gpu_basis": "process"}, "measured", False), 350)[0] == 2000 * 1.2
    assert ge.untracked_mb(["clip", "yoloe_text", "depth"]) == 350.0


# ---------------------------------------------------------------- 들이기 · 예약
def test_admit_reads_live_gpu_and_counts_pending(tmp_path):
    ex, _ = make(tmp_path, used=1000)
    ok, why, calc = ex.admit(3000)
    assert ok and calc["room_mb"] == round(6141 - 1000 - 300)
    ex.launch({"devices": ["pi13"], "models": ["river"]}, 3000, calc)
    ok, why, calc = ex.admit(3000)                                    # 적재 중 예약 3000 을 뺀다
    assert not ok and "GPU 메모리 부족" in why and calc["pending_mb"] == 3000


def test_exclusive_learning_when_no_measurement_and_gpu_idle(tmp_path):
    ex, _ = make(tmp_path, used=0)
    ok, why, calc = ex.admit(5370 * 1.2, "estimate", 5370)            # 6444 > 여유 5841 이지만 추정치 5370 은 들어간다
    assert ok and calc["mode"] == "exclusive" and "단독 측정" in why
    uid = ex.launch({"devices": ["pi13"], "models": ["targeted"]}, 5370 * 1.2, calc)
    assert ex.state.units()[uid]["exclusive"]
    ok, why, _ = ex.admit(100)
    assert not ok and "단독 측정 실행 중" in why                        # 실측이 끝날 때까지 다른 작업 없음
    ex2, _ = make(tmp_path / "b", used=0)
    ok, _, _ = ex2.admit(5370 * 1.2, "measured", 5370)
    assert not ok                                                      # 실측이 있는데 안 들어가면 거절


def test_admit_refuses_without_live_reading(tmp_path):
    ex, _ = make(tmp_path)
    ex.gpus = lambda timeout=2: []
    ok, why, _ = ex.admit(100)
    assert not ok and "확인할 수 없어" in why


def test_launch_command_and_state(tmp_path):
    ex, _ = make(tmp_path)
    uid = ex.launch({"devices": ["pi13", "a2"], "models": ["yoloe"], "profile_path": "/p.json", "request_id": "r1"}, 1560)
    u = ex.state.units()[uid]
    assert u["status"] == "starting" and u["reserved_mb"] == 1560 and u["request_id"] == "r1"
    assert ex.procs[uid].cmd[2:] == ["--devices", "pi13", "a2", "--models", "yoloe", "--profile", "/p.json"]


# ---------------------------------------------------------------- 감시
def test_starting_becomes_loaded_when_vision_reports(tmp_path):
    c = Clock()
    ex, r = make(tmp_path, clock=c)
    uid = ex.launch({"devices": ["pi13"], "models": ["yoloe"]}, 1560)
    pid = ex.state.units()[uid]["pid"]
    assert ex.poll() == []                                             # 아직 보고 없음 (적재 중)
    r.kv["vision_telemetry:h:%d" % pid] = json.dumps({"streams": {"pi13": {"frames": 0}}})
    assert ex.poll() == [(uid, "loaded")]                              # 적재 끝 — 장이 0 이어도(로봇이 꺼져 있어도)
    assert ex.pending_mb() == 0                                        # loaded 는 기기 사용량에 이미 들어 있다


def test_load_timeout_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ge.os, "killpg", lambda *a: None)
    c = Clock()
    ex, _ = make(tmp_path, clock=c)
    uid = ex.launch({"devices": ["pi13"], "models": ["yoloe"]}, 1560)
    c.t += ge.LOAD_TIMEOUT_S + 1
    assert ex.poll() == [(uid, "failed")]
    assert ex.state.units()[uid]["status"] == "failed" and "모델 적재 실패" in ex.state.units()[uid]["reason"]


def test_dead_process_restarts_in_place_then_stops(tmp_path):
    ex, _ = make(tmp_path)
    uid = ex.launch({"unit_id": "u1", "devices": ["pi13"], "models": ["yoloe"]}, 1560)
    for i in range(ge.MAX_RESTARTS):
        ex.procs[uid].rc = 1
        assert ex.poll() == [(uid, "restarted")]
        assert ex.state.units()[uid]["restarts"] == i + 1
    ex.procs[uid].rc = 1
    assert ex.poll() == [(uid, "stopped")]
    assert ex.state.units()[uid]["status"] == "stopped"


# ---------------------------------------------------------------- 실행 상태 파일
def test_state_file_single_writer(tmp_path):
    a = StateFile(tmp_path / "s.json")
    a.put_unit("u", {"status": "starting"})
    b = StateFile(tmp_path / "s.json")
    assert a.writer and not b.writer
    assert b.units()["u"]["status"] == "starting"                      # 읽기는 된다
    with pytest.raises(PermissionError):
        b.put_unit("v", {})
    a.close()
    c = StateFile(tmp_path / "s.json")
    assert c.writer and c.units()["u"]["status"] == "starting"         # 다시 열어도 남아 있다
    assert not list(tmp_path.glob(".state-*"))


# ---------------------------------------------------------------- 다시 띄우기는 spec 으로 (M11 R3)
def test_restart_keeps_plan_conditions_from_spec(tmp_path):
    ex, _ = make(tmp_path, used=0)
    ok, why, calc = ex.admit(5370 * 1.2, "estimate", 5370)
    assert calc["mode"] == "exclusive"
    spec = {"devices": ["pi13"], "models": ["yoloe"], "profile_path": "/p.json", "fps_cap": {"pi13": 7.0},
            "gpu_util": 0.5, "request_id": "r1", "execution_id": "exe-1", "future_field": {"x": 1}}
    uid = ex.launch(dict(spec), 6444, calc)
    assert ex.state.units()[uid]["spec"] == spec                         # 배치표가 준 것 전부 (새 칸 포함)
    for i in range(ge.MAX_RESTARTS):
        ex.procs[uid].rc = 1
        assert ex.poll() == [(uid, "restarted")]
        u = ex.state.units()[uid]
        cmd = ex.procs[uid].cmd
        assert cmd[cmd.index("--max-fps") + 1] == "pi13=7" and "--profile" in cmd
        assert u["fps_cap"] == {"pi13": 7.0} and u["gpu_util"] == 0.5 and u["exclusive"] is True
        assert u["spec"] == spec and u["restarts"] == i + 1 and u["request_id"] == "r1"
        assert u["future_field"] == {"x": 1}


def test_restart_of_old_record_without_spec(tmp_path):
    ex, _ = make(tmp_path)
    old = {"unit_id": "u-old", "kind": "gpu", "pid": 999999, "status": "loaded", "reserved_mb": 1560,
           "admit": {"mode": "normal"}, "exclusive": False, "started_at": 1.0, "restarts": 0, "log": "x.log",
           "devices": ["pi13", "a2"], "models": ["yoloe"], "profile_path": "/p.json", "fps_cap": {"pi13": 6.29, "a2": 5.0},
           "gpu_util": 0.849, "request_id": "r0", "execution_id": "exe-0", "loaded_at": 2.0, "running_emitted": True}
    ex.state.put_unit("u-old", old)                                      # M11 전 상태 파일 — spec 칸이 없다
    assert ex.poll() == [("u-old", "restarted")]                         # pid 999999 는 없다 → 죽은 것으로 본다
    u = ex.state.units()["u-old"]
    cmd = ex.procs["u-old"].cmd
    assert cmd[cmd.index("--max-fps") + 1:cmd.index("--max-fps") + 3] == ["a2=5", "pi13=6.29"]
    assert u["fps_cap"] == old["fps_cap"] and u["gpu_util"] == 0.849 and u["status"] == "starting"
    assert u["spec"]["devices"] == ["pi13", "a2"] and "pid" not in u["spec"] and "running_emitted" not in u["spec"]


def test_admit_refusals_carry_codes(tmp_path):
    ex, _ = make(tmp_path, used=5800)
    assert ex.admit(3000, "measured")[2]["code"] == "gpu_mem"
    ex.gpus = lambda timeout=2: []
    assert ex.admit(100)[2] == {"code": "gpu_unreadable"}
