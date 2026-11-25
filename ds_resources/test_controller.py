import json

import cv2
import numpy as np

from ds_resources import controller as ctl
from ds_resources import gpu_executor as ge
from ds_resources import memory as mm
from ds_resources.state import StateFile

JPG = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))[1].tobytes()


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeProc:
    n = 7000

    def __init__(self, cmd, **kw):
        FakeProc.n += 1
        self.pid, self.cmd, self.rc = FakeProc.n, cmd, None

    def poll(self):
        return self.rc


class FakeRedis:
    def __init__(self, clock):
        self.clock, self.kv, self.rows, self.down, self.frames = clock, {}, [], False, {}

    def add_frames(self, src, n=50, age_s=0.2):
        now = int(self.clock() * 1000)
        self.frames[src] = [("%d-0" % (now - int(age_s * 1000) - i * 100), {b"image": JPG}) for i in range(n)]

    def xrevrange(self, key, hi, lo, count=None):
        src = key.split(":", 1)[1]
        rows = [(e.encode(), f) for e, f in self.frames.get(src, [])]
        if lo != "-":
            lo_ms = int(lo.split("-")[0])
            rows = [r for r in rows if int(r[0].split(b"-")[0]) >= lo_ms]
        return rows[:count]

    def xadd(self, stream, fields, maxlen=None, approximate=None):
        if self.down:
            raise ConnectionError("redis down")
        self.rows.append(fields)

    def scan_iter(self, pattern, count=None):
        suffix = pattern.split("*")[-1]
        return [k for k in self.kv if k.endswith(suffix)]

    def get(self, k):
        return self.kv.get(k)

    def events(self):
        return [r["event"] for r in self.rows]


def gpus(used=0.0):
    return lambda timeout=2: [{"name": "RTX", "mem_total_mb": 6141.0, "mem_used_mb": used}]


def make(tmp_path, used=0.0):
    c = Clock()
    r = FakeRedis(c)
    st = StateFile(tmp_path / "s.json")
    ex = ge.GpuExecutor(st, r, gpus=gpus(used), popen=FakeProc, clock=c, python="py", log=lambda *_: None)
    k = ctl.Controller(r, st, ex, mm.ResourceMemory(tmp_path / "m.json", clock=c), local_node="wsl", cluster=None,
                       cfg={}, clock=c, log=lambda *_: None, gpus=gpus(used))
    return k, r, c


def test_jpeg_size():
    assert ctl.jpeg_size(JPG) == (1280, 720)


def test_local_fallback_event_sequence_and_fps_cap(tmp_path):
    k, r, _ = make(tmp_path)
    r.add_frames("pi13")
    rid = k.submit(["pi13"], ["yoloe"])
    assert r.events() == ["requested", "cluster_checked", "fallback", "planned", "applied"]
    fb = [x for x in r.rows if x["event"] == "fallback"][0]
    assert fb["mode"] == "local" and fb["basis"] == "saved"
    plan = json.loads([x for x in r.rows if x["event"] == "planned"][0]["plan"])
    u = plan["gpu_units"][0]
    assert u["need_mb"] == round(1300 * 1.2) and plan["algorithm"] == "placement/2"
    unit = list(k.state.units().values())[0]
    assert unit["request_id"] == rid and unit["execution_id"].startswith("exe-")
    cmd = k.ex.procs[unit["unit_id"]].cmd
    assert "--max-fps" in cmd and cmd[cmd.index("--max-fps") + 1].startswith("pi13=")


def test_missing_input_waits_instead_of_failing(tmp_path):
    k, r, c = make(tmp_path)
    r.add_frames("pi13", age_s=30)                                    # 30초 멈춘 영상 (로봇이 꺼짐)
    k.submit(["pi13", "a2"], ["yoloe"])                               # a2 는 프레임이 아예 없다
    assert "applied" in r.events() and "failed" not in r.events()
    waits = [x["reason"] for x in r.rows if x["event"] == "degraded" and "입력 대기" in x["reason"]]
    assert len(waits) == 2
    u = list(k.state.units().values())[0]
    assert u["devices"] == ["a2", "pi13"]                             # 같은 프로세스에 미리 넣는다
    plan = json.loads([x for x in r.rows if x["event"] == "planned"][0]["plan"])
    assert set(plan["gpu_units"][0]["fps_cap"]) == {"a2", "pi13"}       # 예산을 미리 잡는다


def test_robot_comes_online_later_and_goes_away(tmp_path):
    k, r, c = make(tmp_path)
    k.submit(["pi13", "a2"], ["yoloe"])                               # 둘 다 꺼져 있다
    u = list(k.state.units().values())[0]
    key = "vision_telemetry:h:%d" % u["pid"]
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 0}, "a2": {"frames": 0}}})
    assert k.poll() == []                                              # loaded, 아직 running 아님
    r.add_frames("pi13")                                               # pi13 로봇이 켜졌다
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 40, "fps": 4.0}, "a2": {"frames": 0}}})
    ev = k.poll()
    assert [e["event"] for e in ev] == ["running"]
    assert json.loads(ev[0]["detail"])["waiting"] == ["a2"]
    r.add_frames("a2")                                                 # a2 도 켜졌다 — 다시 띄우지 않고 바로
    ev = k.poll()
    assert [e["event"] for e in ev] == ["running"] and "입력 들어옴: a2" in ev[0]["reason"]
    c.t += ctl.INPUT_DOWN_S + 1                                        # pi13 이 꺼졌다 (a2 는 계속)
    r.add_frames("a2")
    ev = k.poll()
    assert [e["event"] for e in ev] == ["degraded"] and "입력 끊김: pi13" in ev[0]["reason"]
    assert k.ex.procs[u["unit_id"]].rc is None                         # 프로세스는 그대로


def test_gpu_refusal_is_failed_with_reason(tmp_path):
    k, r, _ = make(tmp_path, used=5000)
    r.add_frames("pi13")
    k.submit(["pi13"], ["river"])
    assert "applied" not in r.events() and "GPU 메모리 부족" in r.rows[-1]["reason"]


def test_running_needs_frames_and_fresh_input(tmp_path):
    k, r, c = make(tmp_path)
    r.add_frames("pi13")
    k.submit(["pi13"], ["yoloe"])
    u = list(k.state.units().values())[0]
    r.kv["vision_telemetry:h:%d" % u["pid"]] = json.dumps({"streams": {"pi13": {"frames": 30, "fps": 10.0}}})
    r.add_frames("pi13", age_s=20)                                    # 처리한 장은 있지만 입력이 멈췄다
    assert "running" not in [e["event"] for e in k.poll()]            # starting → loaded, running 아님
    assert "running" not in [e["event"] for e in k.poll()]
    r.add_frames("pi13")
    assert [e["event"] for e in k.poll()] == ["running"]
    assert [e["event"] for e in k.poll()] == []                         # 한 번만


def test_degraded_after_sustained_low_fps(tmp_path):
    k, r, c = make(tmp_path)
    r.add_frames("pi13", n=100)                                       # 10 fps
    k.submit(["pi13"], ["yoloe"], target_fps={"pi13": 5.0})
    u = list(k.state.units().values())[0]
    key = "vision_telemetry:h:%d" % u["pid"]
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 30, "fps": 5.0}}})
    k.poll(), k.poll()                                                # loaded → running
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 10, "fps": 2.0}}})
    assert k.poll() == []                                             # 적재 직후(워밍업) — 낮아도 세지 않는다
    c.t += ctl.WARMUP_S + 1
    r.add_frames("pi13", n=100)
    assert k.poll() == []                                             # 워밍업 뒤 낮아지기 시작
    c.t += 31
    r.add_frames("pi13", n=100)
    ev = k.poll()
    assert [e["event"] for e in ev] == ["degraded"] and "미달" in ev[0]["reason"]
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 30, "fps": 5.0}}})   # 다시 목표 근처
    r.add_frames("pi13", n=100)
    ev = k.poll()
    assert [e["event"] for e in ev] == ["running"] and "fps 회복" in ev[0]["reason"]   # 마지막 상태가 degraded 로 남지 않음


def test_redis_down_events_go_to_outbox_then_flush(tmp_path):
    k, r, _ = make(tmp_path)
    r.add_frames("pi13")
    r.down = True
    k.submit(["pi13"], ["yoloe"])
    assert r.rows == [] and len(k.state.data["outbox"]) == 5
    assert len(StateFile(tmp_path / "s.json", writer=False).data["outbox"]) == 5    # 파일에 남아 있다
    r.down = False
    k.poll()
    assert r.events()[:5] == ["requested", "cluster_checked", "fallback", "planned", "applied"]
    assert k.state.data["outbox"] == []


def test_stop_emits_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(ge.os, "killpg", lambda *a: None)
    k, r, _ = make(tmp_path)
    r.add_frames("pi13")
    rid = k.submit(["pi13"], ["yoloe"])
    assert k.stop(rid) == 1 and r.events()[-1] == "stopped"


class Cluster:
    def __init__(self, ok):
        self.ok = ok

    def check(self):
        return self.ok, "" if self.ok else "API 응답 없음"

    def nodes(self):
        return [{"name": "pi7", "role": "robot", "ready": True}]


def test_cluster_ok_is_live_mode(tmp_path):
    k, r, _ = make(tmp_path)
    k.cluster = Cluster(True)
    r.add_frames("pi13")
    k.submit(["pi13"], ["yoloe"])
    assert "fallback" not in r.events()
    assert [x for x in r.rows if x["event"] == "planned"][0]["mode"] == "k3s"


# ---------------------------------------------------------------- CPU 작업 (M8)
class FakeCpu:
    def __init__(self):
        self.launched, self.stopped, self.st = [], [], {}

    def launch(self, unit):
        self.launched.append(unit)

    def status(self, unit):
        return self.st.get(unit["unit_id"], {"phase": "Running", "ready": True})

    def stop(self, unit):
        self.stopped.append(unit["unit_id"])


class Nodes:
    def __init__(self):
        self.ok = True
        self.n = {"pi6": dict(name="pi6", role="sensor", ready=True, cpu_alloc_m=4000, mem_alloc_mb=8000, cpu_now_m=100,
                              mem_now_mb=500, net={"bw_mbps": 80}),
                  "pi13": dict(name="pi13", role="small", ready=True, cpu_alloc_m=4000, mem_alloc_mb=900, cpu_now_m=600,
                               mem_now_mb=300, net={"bw_mbps": 30})}

    def check(self):
        return (True, "") if self.ok else (False, "API 응답 없음")

    def nodes(self):
        return [dict(v) for v in self.n.values()]


PROBE = {"probe": {"cpu_m": 200, "mem_mb": 64, "roles": ["sensor", "small"], "image": "busybox:1.36", "command": ["sh"]}}


def cpu_setup(tmp_path):
    k, r, c = make(tmp_path)
    k.cluster, k.cpu_ex = Nodes(), FakeCpu()
    rid = k.submit([], [], cpu_tasks=["probe"], task_specs=PROBE)
    return k, r, c, rid


def unit_of(k):
    return [u for u in k.state.units().values() if u.get("kind") == "cpu" and u["status"] in ("starting", "running")][0]


def test_cpu_task_goes_to_pi_with_limits_and_runs_on_heartbeat(tmp_path):
    k, r, c, rid = cpu_setup(tmp_path)
    u = unit_of(k)
    assert u["node"] == "pi6" and u["limits"] == {"cpu_m": 400, "mem_mb": 128} and u["priority_class"] == "ds-vision-low"
    assert k.poll() == []                                              # 하트비트 없음 → 아직
    r.kv["cpu_telemetry:" + u["unit_id"]] = json.dumps({"ts": c()})
    assert [e["event"] for e in k.poll()] == ["running"]


def test_node_drop_moves_only_cpu_work(tmp_path):
    k, r, c, rid = cpu_setup(tmp_path)
    u = unit_of(k)
    k.cluster.n["pi6"]["ready"] = False
    assert k.poll() == []                                              # 바로 옮기지 않는다 (유예)
    c.t += 10
    k.cluster.n["pi6"]["ready"] = True
    assert k.poll() == []                                              # 잠깐 끊겼다 돌아옴 → 그대로
    k.cluster.n["pi6"]["ready"] = False
    k.poll()
    c.t += ctl.NODE_BAD_GRACE_S + 1
    ev = k.poll()
    assert ev[-1]["event"] == "replanned" and "pi6 에서 pi13" in ev[-1]["reason"]
    assert k.cpu_ex.stopped == [u["unit_id"]] and unit_of(k)["node"] == "pi13"


def test_silent_pod_is_replaced(tmp_path):
    k, r, c, rid = cpu_setup(tmp_path)
    u = unit_of(k)
    r.kv["cpu_telemetry:" + u["unit_id"]] = json.dumps({"ts": c()})
    k.poll()                                                           # running
    c.t += ctl.CPU_REPLACE_AFTER_S + 5
    assert k.poll()[-1]["event"] == "replanned"


def test_crashloop_is_replaced(tmp_path):
    k, r, c, rid = cpu_setup(tmp_path)
    u = unit_of(k)
    k.cpu_ex.st[u["unit_id"]] = {"phase": "Running", "ready": False, "waiting": "ImagePullBackOff"}
    assert "ImagePullBackOff" in k.poll()[-1]["reason"]


def test_k3s_down_mid_run_is_degraded_once(tmp_path):
    k, r, c, rid = cpu_setup(tmp_path)
    k.cluster.ok = False
    ev = k.poll()
    assert [e["event"] for e in ev] == ["degraded"] and ev[0]["basis"] == "saved"
    assert k.poll() == []                                              # 한 번만
    assert k.cpu_ex.stopped == []                                      # 파드는 그대로 둔다


def test_no_k3s_keeps_cpu_work_local(tmp_path):
    k, r, c = make(tmp_path)
    k.cpu_ex = FakeCpu()                                               # 실행기는 있지만 클러스터가 없다
    k.submit([], [], cpu_tasks=["probe"], task_specs={"probe": dict(PROBE["probe"], roles=["sensor", "gpu"])})
    u = [x for x in k.state.units().values() if x.get("kind") == "cpu"][0]
    assert u["status"] == "local" and k.cpu_ex.launched == []


def test_image_less_task_never_goes_to_pi(tmp_path):
    k, r, c = make(tmp_path)
    k.cluster, k.cpu_ex = Nodes(), FakeCpu()
    k.submit([], [], cpu_tasks=["cam360_slice"])                       # 레지스트리 — image: null
    u = [x for x in k.state.units().values() if x.get("kind") == "cpu"][0]
    assert u["node"] == "wsl" and u["status"] == "local" and k.cpu_ex.launched == []


# ---------------------------------------------------------------- check() 셋째 상태 (M11 R2a)
class CapsNodes(Nodes):
    """새 플러그인 — check() 가 (ok, why, caps). 노드는 읽히지만 CPU 작업은 못 올린다(PriorityClass 없음)"""

    def __init__(self, caps):
        super().__init__()
        self.caps = caps

    def check(self):
        return True, "", dict(self.caps)


def test_third_state_keeps_k3s_mode_and_nodes_but_refuses_pi_cpu(tmp_path):
    k, r, c = make(tmp_path)
    k.cluster, k.cpu_ex = CapsNodes({"cpu_tasks": False, "why": "PriorityClass ds-vision-low 없음"}), FakeCpu()
    r.add_frames("pi13")
    p = k.prepare(["pi13"], ["yoloe"], cpu_tasks=["probe"], task_specs=PROBE)
    assert p["mode"] == "k3s" and p["basis"] == "live" and p["ok"]
    assert {n["name"] for n in p["snapshot"]["nodes"]} == {"wsl", "pi6", "pi13"}        # 노드 정보 유지
    rid = k.submit(["pi13"], ["yoloe"], cpu_tasks=["probe"], task_specs=PROBE)
    assert "fallback" not in r.events() and k.cpu_ex.launched == []
    assert [x for x in r.rows if x["event"] == "planned"][0]["mode"] == "k3s"
    cc = [x for x in r.rows if x["event"] == "cluster_checked"][0]
    assert json.loads(cc["detail"])["caps"]["cpu_tasks"] is False and "PriorityClass" in cc["reason"]
    fail = [x for x in r.rows if x["event"] == "failed"]
    assert [x["code"] for x in fail] == ["no_cpu_capability"] and "PriorityClass ds-vision-low 없음" in fail[0]["reason"]
    assert any(u.get("kind") == "gpu" for u in k.state.units().values())                # GPU 는 그대로 돈다
    assert k.state.data["requests"][rid]["refused"][0]["code"] == "no_cpu_capability"


def test_caps_ok_places_pi_cpu_and_old_two_tuple_still_works(tmp_path):
    k, r, c = make(tmp_path)
    k.cluster, k.cpu_ex = CapsNodes({"cpu_tasks": True}), FakeCpu()
    k.submit([], [], cpu_tasks=["probe"], task_specs=PROBE)
    assert [u["node"] for u in k.cpu_ex.launched] == ["pi6"]
    assert ctl.check_cluster(Cluster(True)) == (True, "", {})            # 옛 플러그인 (ok, why)
    assert ctl.check_cluster(Cluster(False))[:2] == (False, "API 응답 없음")
    assert ctl.check_cluster(None)[0] is False

    class Boom:
        def check(self):
            raise RuntimeError("x")
    assert ctl.check_cluster(Boom())[0] is False


# ---------------------------------------------------------------- 이유 코드 · admit 거절 → 재시도 (M11 R4)
def _admit_refusal(tmp_path, reason_text=None, monkeypatch=None):
    k, r, c = make(tmp_path)
    r.add_frames("pi13")
    k.state.put_unit("old", {"kind": "gpu", "status": "starting", "reserved_mb": 5000, "pid": 1})   # 앞 실행이 반납 중
    if reason_text is not None:
        orig = k.ex.admit

        def admit(*a, **kw):
            ok, why, calc = orig(*a, **kw)
            return ok, (why if ok else reason_text), calc
        k.ex.admit = admit
    rid = k.submit(["pi13"], ["yoloe"])
    return k, r, rid


def test_admit_refusal_is_recorded_with_code(tmp_path):
    k, r, rid = _admit_refusal(tmp_path)
    req = k.state.data["requests"][rid]
    assert req["plan"]["refused"] == []                                  # 배치표는 통과했다
    assert [(x["code"], x["stage"]) for x in req["refused"]] == [("gpu_mem", "admit")]
    fail = [x for x in r.rows if x["event"] == "failed"][-1]
    assert fail["code"] == "gpu_mem"


def test_cli_retries_on_admit_refusal_even_if_text_changes(tmp_path):
    from ds_resources import cli
    for text in (None, "Not enough VRAM (문구가 바뀌어도)"):
        k, r, rid = _admit_refusal(tmp_path / ("t%d" % (text is None)), text)
        assert cli.should_retry(k.state.data["requests"][rid])
    assert cli.should_retry({"plan": {"refused": [{"item": "gpu:x", "code": "gpu_mem", "reason": "?"}]}})   # 옛 기록
    assert not cli.should_retry({"refused": [{"item": "cpu:x", "code": "no_node", "reason": "GPU 메모리 부족"}]})


def test_submit_with_retry_resubmits_until_launched(tmp_path):
    from ds_resources import cli
    k, r, rid = _admit_refusal(tmp_path)
    sleeps, n = [], {"submits": 0}

    def submit():
        n["submits"] += 1
        if n["submits"] == 3:
            k.state.update_unit("old", status="stopped")                # 앞 프로세스가 GPU 를 돌려줬다
        return k.submit(["pi13"], ["yoloe"])
    mine = lambda: [u for uid, u in k.ex.active_units().items() if uid != "old"]  # noqa: E731
    rid = cli.submit_with_retry(k, submit, lambda rid: not mine(), sleep=sleeps.append, retries=5, gap=0)
    assert n["submits"] == 3 and len(sleeps) == 2
    assert k.state.data["requests"][rid]["refused"] == [] and mine()


def test_unbudgeted_waiting_source_triggers_replan_when_it_comes_online(tmp_path):
    # 2026-11-25 실측: 영상이 오는 robot1 은 바로 돌고, 꺼진 a2 · drone 은 예산 밖 대기 → a2 가 켜지면 다시 배치
    k, r, c = make(tmp_path)
    r.add_frames("robot1")

    def costs(models, sources, profile=None):                         # 영상이 없는 소스는 크기를 몰라 xl(무겁게)
        p90 = {s: (1300.0 if sources[s].get("waiting") else 300.0) for s in sources}
        return {"need_mb": 2000, "basis": "measured", "estimate_mb": 1700, "needs": [], "p90_ms": p90,
                "p90_basis": {s: "registry" for s in sources}, "cfg_hash": "h"}
    k.costs = costs
    rid = k.submit(["robot1", "a2", "drone"], ["targeted"])
    units = list(k.state.units().values())
    assert [u["devices"] for u in units] == [["robot1"]] and "failed" not in r.events()
    assert sorted(k.pending_sources(rid)) == ["a2", "drone"]
    assert k.replan_if_needed(rid) is None                             # 아직 아무도 안 켜짐
    r.add_frames("a2")                                                 # a2 로봇이 켜졌다
    new = k.replan_if_needed(rid)
    assert new and new != rid and "replanned" in r.events()
    live = [u for u in k.state.units().values() if u.get("status") in ("starting", "loaded")]
    assert [sorted(u["devices"]) for u in live] == [["a2", "robot1"]]   # 영상 오는 둘끼리 예산을 다시 나눔
    assert k.pending_sources(new) == ["drone"]


def test_snapshot_counts_loading_units_in_gpu_budget(tmp_path):
    # 적재 중(starting)인 단위도 처리량 몫을 쥐고 있다 — gpu_executor.active_units() 와 같은 기준
    k, _, _ = make(tmp_path)
    assert k.local_node_snapshot()["gpu_util_committed"] == 0.0
    k.state.put_unit("loading", {"kind": "gpu", "status": "starting", "gpu_util": 0.4})
    assert k.local_node_snapshot()["gpu_util_committed"] == 0.4          # 적재 창 안에서도 예산에 잡힌다
    k.state.put_unit("done", {"kind": "gpu", "status": "loaded", "gpu_util": 0.3})
    assert abs(k.local_node_snapshot()["gpu_util_committed"] - 0.7) < 1e-9
    k.state.update_unit("loading", status="stopped")                     # 끝난 단위는 빠진다
    assert abs(k.local_node_snapshot()["gpu_util_committed"] - 0.3) < 1e-9


# ---------------------------------------------------------------- 기능 단위 배치 (placement/2 · functions 계약)
FUNCS = {"functions": {
    "det": {"pattern": "cascade", "criticality": "mission", "trigger": "always", "note": "거름 → 서버 탐지",
            "stages": {"device": {"cpu_m": 1600, "mem_mb": 512, "roles": ["robot"], "optional": True,
                                  "egress_kb": 0.05, "note": "파이 1차 거름"},
                       "server": {"cpu_m": 500, "mem_mb": 2048, "roles": ["gpu"], "note": "서버 탐지"}}},
    "reid": {"pattern": "split", "criticality": "mission", "trigger": "on_request", "note": "특징 → 융합",
             "stages": {"device": {"cpu_m": 900, "mem_mb": 256, "roles": ["robot"], "note": "특징 추출"},
                        "server": {"cpu_m": 400, "mem_mb": 1024, "roles": ["gpu"], "note": "유사도 융합"}}}}}


def func_setup(tmp_path, names):
    """기능 계약을 가짜로 끼운다 — 실제 레지스트리가 바뀌어도 이 시험은 흔들리지 않게.
    Nodes() 에는 robot 역할 노드가 없으므로 device 단계는 어디에도 놓이지 않는다."""
    from ds_resources import functions as fnc
    k, r, c = make(tmp_path)
    k.cluster, k.cpu_ex = Nodes(), FakeCpu()
    k._funcs = fnc.load(FUNCS)
    rid = k.submit([], [], functions=names)
    return k, r, rid


def test_cascade_without_the_device_stage_runs_degraded_as_server_only(tmp_path):
    """캐스케이드의 기기 단계가 못 놓이면 기능은 살고 품질만 낮아진다 — 기록에 quality 가 남는다"""
    k, r, rid = func_setup(tmp_path, ["det"])
    plan = (k.state.data["requests"][rid]).get("plan")
    assert plan["algorithm"] == "placement/2"
    assert plan["functions"]["det"]["verdict"] == "degraded"
    assert plan["functions"]["det"]["quality"] == "server_only"
    # 서버 단계는 실제로 놓였고, 그 단위 기록 · applied 기록에 기능 · 단계 · 품질이 함께 남는다
    assert [u["task"] for u in plan["cpu_units"]] == ["det.server"]
    applied = [x for x in r.rows if x["event"] == "applied" and x.get("function") == "det"]
    assert applied and applied[0]["stage"] == "server" and applied[0]["quality"] == "server_only"
    low = [x for x in r.rows if x["event"] == "degraded" and x.get("code") == "pair_incomplete"]
    assert low and low[0]["function"] == "det"


def test_split_with_one_stage_missing_is_rejected_and_cancels_the_other(tmp_path):
    """split 은 한쪽이 없으면 기능을 거절하고, 이미 놓인 단계도 취소한다"""
    k, r, rid = func_setup(tmp_path, ["reid"])
    plan = (k.state.data["requests"][rid]).get("plan")
    assert plan["functions"]["reid"]["verdict"] == "rejected"
    assert "quality" not in plan["functions"]["reid"]
    assert plan["cpu_units"] == []                              # 놓였던 서버 단계까지 취소
    assert not k.cpu_ex.launched                                # 파드를 띄우지 않았다
    bad = [x for x in r.rows if x["event"] == "failed" and x.get("code") == "pair_incomplete"]
    assert bad and bad[0]["function"] == "reid"


def test_old_path_is_untouched_by_the_functions_contract(tmp_path):
    """기능을 주지 않으면 요청에 functions 칸이 없고 옛 경로와 똑같다"""
    k, _r, _c, rid = cpu_setup(tmp_path)
    plan = (k.state.data["requests"][rid]).get("plan")
    assert "functions" not in plan
    assert [u["task"] for u in plan["cpu_units"]] == ["probe"]
