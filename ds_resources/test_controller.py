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
    assert u["need_mb"] == round(1300 * 1.2) and plan["algorithm"] == "placement/1"
    unit = list(k.state.units().values())[0]
    assert unit["request_id"] == rid and unit["execution_id"].startswith("exe-")
    cmd = k.ex.procs[unit["unit_id"]].cmd
    assert "--max-fps" in cmd and cmd[cmd.index("--max-fps") + 1].startswith("pi13=")


def test_missing_input_is_failed_not_applied(tmp_path):
    k, r, _ = make(tmp_path)
    r.add_frames("pi13", age_s=30)                                    # 30초 멈춘 영상
    k.submit(["pi13"], ["yoloe"])
    assert "applied" not in r.events()
    f = [x for x in r.rows if x["event"] == "failed"][0]
    assert "입력 영상 없음" in f["reason"]


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
    assert k.poll() == []                                             # 낮아지기 시작
    c.t += 31
    r.add_frames("pi13", n=100)
    ev = k.poll()
    assert [e["event"] for e in ev] == ["degraded"] and "미달" in ev[0]["reason"]


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
